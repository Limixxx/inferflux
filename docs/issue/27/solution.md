---
title: "Issue #27 解决方案"
issue_number: 27
issue_type: Feature
created: 2026-08-31
updated: 2026-08-31
status: draft
review_round: 1
---

# Issue #27 解决方案

## 需求分析

### 问题描述

实现 `EPLBSimulator`（§10.4.4），即 EP 负载均衡仿真器，仿真 SGLang 的 `eplb` 模块——按运行时专家负载统计动态调整专家到 EP rank 的映射，以降低负载不均造成的性能损失。

### 能力目标

1. **周期性重平衡触发**：每 100 步检查一次是否需要重平衡（`global_step % 100 === 0` 时触发检查）
2. **方差阈值判定**：计算各 expert 负载的标准差与均值之比（`variance_ratio = stdev / avg`），若 `variance_ratio < load_variance_threshold`（默认 0.1，即 10%），则认为负载均匀，跳过重平衡以节省成本
3. **贪心重排计划**：需要重平衡时，构造重排计划（每个 expert → new_rank），目标使每 rank 负载接近 `_target_load_per_rank = avg × 1.02`；采用贪心策略从过载 rank 中搬走最多 expert
4. **重平衡成本**：累加 `rebalance_cost_fixed_ticks`（默认 50 ticks）到 `ParallelMetrics.epRebalanceCostTicks`
5. **映射缓存更新**：修改 `SimMoeBackend._expertToRankMap`（即 `expertToRankMap` 数组），使下一次 forward 生效
6. **集成点**：SimScheduler 的 `normal_tick` 末尾（scheduler_engine_forward 之后）调用 `eplbSim.maybe_rebalance(step++, moeBackend.expertLoadCounts)`

### 影响范围

- **新增文件**：`server/src/sglang/parallel/eplb.ts` — `EPLBSimulator` 类
- **修改文件**：
  - `server/src/sglang/parallel/index.ts` — 导出 `EPLBSimulator`
  - `server/src/sglang/parallel/moe.ts` — 新增 `expertLoadCounts` getter 属性（读取当前专家负载快照）
- **新增测试**：`server/src/test/sglang-p3b.test.ts`
- **不修改**：`ParallelMetrics`（已有 `epRebalanceCostTicks` 字段）、`SimulatorConfig`（已有 `enableEplb` 字段）、`validate.ts`（已有 EP 相关约束）

### 依赖分析

- **Issue #21 (ParallelTopology)**：已实现 `computeMoeRanks`，提供 EP rank 空间。EPLBSimulator 不直接依赖 topology，但重排后需保持 `expertToRankMap` 与拓扑的一致性约束
- **Issue #26 (SimMoeBackend)**：已实现 `expertToRankMap`（可写数组）、`expertLoadCounts`（需新增 getter，读取 `metrics.epExpertLoad` 快照）、`_expertToRank` 方法

## 改造方案

### 总体思路

新建 `EPLBSimulator` 类，实现 `maybe_rebalance` 方法。该方法接收当前全局步数和专家负载计数数组，执行以下流程：

1. 判断是否到达检查周期（`global_step % 100 !== 0` → 跳过）
2. 计算负载方差比率（`stdev / avg`），判断是否低于阈值
3. 若需要重平衡：贪心构造重排计划，更新 `moeBackend.expertToRankMap`，累加成本
4. 返回 `{ shouldRebalance, rebalanceTicks, movedExperts }` 结果对象

### 详细设计

#### 1. `EPLBSimulator` 类（新增 `parallel/eplb.ts`）

```typescript
// 构造选项
export interface EPLBSimulatorOpts {
  enabled: boolean;          // config.enableEplb
  numExperts: number;        // modelConfig.numExperts
  epSize: number;            // config.epSize
  metrics: ParallelMetrics;  // 指标写入目标
  rebalanceIntervalSteps?: number;   // 默认 100
  loadVarianceThreshold?: number;    // 默认 0.1
  rebalanceCostFixedTicks?: number;  // 默认 50
}

// maybe_rebalance 返回值
export interface RebalanceResult {
  shouldRebalance: boolean;
  rebalanceTicks: number;
  movedExperts: number;
}

export class EPLBSimulator {
  readonly enabled: boolean;
  readonly numExperts: number;
  readonly epSize: number;
  readonly metrics: ParallelMetrics;
  readonly rebalanceIntervalSteps: number;   // 100
  readonly loadVarianceThreshold: number;    // 0.1
  readonly rebalanceCostFixedTicks: number;  // 50

  constructor(opts: EPLBSimulatorOpts) { ... }

  /**
   * 检查并执行重平衡
   * @param globalStep 当前全局步数
   * @param expertLoadCounts 各 expert 累计负载（来自 moeBackend.expertLoadCounts）
   * @param moeBackend MoE 后端实例（用于更新 expertToRankMap）
   */
  maybe_rebalance(
    globalStep: number,
    expertLoadCounts: number[],
    moeBackend: SimMoeBackend,
  ): RebalanceResult;
}
```

#### 2. `maybe_rebalance` 核心逻辑

```
maybe_rebalance(globalStep, expertLoadCounts, moeBackend):
  1. enabled=false 或 epSize<=1 → return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 }
  2. globalStep % rebalanceIntervalSteps !== 0 → return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 }
  3. 计算 rank 负载:
     - rankLoads[r] = Σ expertLoadCounts[e] for e where expertToRankMap[e] === r
  4. avg = mean(rankLoads)
  5. 若 avg === 0 → return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 }
  6. stdev = sqrt( Σ(rankLoads[r] - avg)² / epSize )
  7. varianceRatio = stdev / avg
  8. 若 varianceRatio < loadVarianceThreshold → return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 }
  9. 执行贪心重排:
     - targetLoadPerRank = avg * 1.02
     - 收集所有 expert: { expertId, currentRank, load }
     - 按 rank 分组，计算每 rank 当前总负载
     - 贪心循环：找出过载最多 rank（currentLoad - targetLoad 最大的），
       从中选负载最小的 expert 搬到欠载最多 rank（targetLoad - currentLoad 最大的）
     - 搬迁条件：搬运该 expert 后，目标 rank 新负载不超过旧最大 rank 负载
     - 更新 expertToRankMap[e] = newRank
     - 计数 movedExperts
  10. metrics.epRebalanceCostTicks += rebalanceCostFixedTicks
  11. return { shouldRebalance: true, rebalanceTicks: rebalanceCostFixedTicks, movedExperts }
```

#### 3. 贪心重排策略详细说明

- 将所有 expert 按当前 rank 分组，计算每 rank 的负载总和
- `targetLoadPerRank = avg × 1.02`（2% buffer，避免因微小波动反复搬迁）
- 循环执行：
  1. 计算每 rank 的 `currentLoad`
  2. 找 `surplus = currentLoad - targetLoadPerRank` 最大的 rank（过载 rank）
  3. 若 surplus <= 0 → 所有 rank 不再过载，终止循环
  4. 找 `deficit = targetLoadPerRank - currentLoad` 最大的 rank（欠载 rank）
  5. 从过载 rank 的 expert 中选择负载最小的 expert（搬迁代价最小）
  6. **约束检查**：搬迁后目标 rank 新负载 ≤ 旧 `maxRankLoad`（搬迁前全局最大 rank 负载）。若违反则跳过该 expert，尝试次小；若无可搬迁则终止
  7. 执行搬迁：`expertToRankMap[expertId] = targetRank`，更新两 rank 负载
  8. `movedExperts++`

#### 4. `SimMoeBackend` 新增 getter（修改 `parallel/moe.ts`）

```typescript
/** 获取当前各 expert 的累计负载快照（来自 metrics.epExpertLoad） */
get expertLoadCounts(): number[] {
  return [...this.metrics.epExpertLoad];
}
```

> **设计说明**：`expertToRankMap` 已是 `public readonly` 数组（引用不可变但元素可写），EPLBSimulator 可直接修改其元素值。新增 `expertLoadCounts` getter 提供 `metrics.epExpertLoad` 的浅拷贝，防止外部意外修改指标数组。

#### 5. 导出更新（修改 `parallel/index.ts`）

```typescript
export {
  EPLBSimulator,
} from "./eplb";

export type {
  EPLBSimulatorOpts,
  RebalanceResult,
} from "./eplb";
```

### 修改点清单

1. **新增** `server/src/sglang/parallel/eplb.ts` — `EPLBSimulator` 类完整实现
2. **修改** `server/src/sglang/parallel/moe.ts` — 新增 `expertLoadCounts` getter
3. **修改** `server/src/sglang/parallel/index.ts` — 新增 `EPLBSimulator` 及相关类型导出
4. **新增** `server/src/test/sglang-p3b.test.ts` — 验收测试

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 描述 |
|------|----------|------|
| T1 | 构造 — enabled=false 时 maybe_rebalance 直接返回 false | 禁用 EPLB 时所有调用均返回不重平衡 |
| T2 | 构造 — epSize<=1 时 maybe_rebalance 直接返回 false | 单 rank EP 不需要重平衡 |
| T3 | 100 步周期 — global_step 非 100 倍数时跳过 | step=50 → shouldRebalance=false |
| T4 | 100 步周期 — global_step=100 时触发检查 | step=100 → 进入方差判定阶段 |
| T5 | 方差低跳过 — 负载均匀时不重平衡 | 所有 expert 负载相同 → variance_ratio=0 < 0.1 → 跳过 |
| T6 | 方差高触发重平衡 | 人为构造不均匀负载 → variance_ratio > 0.1 → 触发 |
| T7 | movedExperts 非负 | 任何情况下 movedExperts >= 0 |
| T8 | plan 不使新 rank max 负载超过旧 max | 贪心约束验证 |
| T9 | rebalanceTicks 等于 rebalanceCostFixedTicks | 触发重平衡时返回正确的固定成本 |
| T10 | metrics.epRebalanceCostTicks 累加正确 | 多次重平衡后指标累加 |
| T11 | expertToRankMap 更新后下一次 forward 生效 | 重排后路由结果反映新映射 |
| T12 | avg=0 时不重平衡（不除零） | 所有 expert 负载为 0 的边界条件 |
| T13 | 多次周期触发 | step=100, 200, 300 各触发一次检查 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|----------|----------|
| B1 | enabled=false | maybe_rebalance 始终返回 shouldRebalance=false |
| B2 | epSize=1 | 退化返回不重平衡 |
| B3 | 所有 expertLoadCounts=0 | avg=0 → 安全返回不重平衡 |
| B4 | 单 expert per rank | 无法搬迁（每 rank 只剩 1 expert）→ movedExperts=0 |
| B5 | numExperts < epSize | 不可能出现（受 divEven 约束），由 validateParallelConfig 拦截 |
| B6 | 负载极度不均（某 rank 负载为 0，另一 rank 全部负载） | 触发重平衡，贪心搬迁 expert |
| B7 | 连续多次 maybe_rebalance 调用 | 非检查周期返回 false，不重复累加成本 |
| B8 | 重排后每 rank 至少保留 1 个 expert | 搬迁时检查 source rank expert 数量 > 1 |

## 风险与注意事项

### 兼容性影响

- **expertToRankMap 可变性**：当前 `expertToRankMap` 声明为 `readonly number[]`，数组引用不可变但元素可写。EPLBSimulator 直接修改元素值是安全的，但需确保修改时机在 forward 之外（normal_tick 末尾），避免 forward 执行中途映射变化。
- **与 topology 一致性**：构造时 `SimMoeBackend` 有自检逻辑验证 `expertToRankMap` 与 `topology.computeMoeRanks` 一致。EPLB 重排后映射会偏离拓扑初始分配——这是预期行为（动态重平衡的意义），但自检仅在构造时执行一次，不受影响。

### 性能影响

- `maybe_rebalance` 每 100 步执行一次，内部计算 O(numExperts × epSize)，对于典型规模（8-256 experts, 2-8 ep ranks）开销可忽略。
- 贪心重排循环最坏 O(numExperts²)，但实际因约束提前终止，远低于理论上限。

### 回滚方案

- 设置 `enableEplb=false`（默认值），`EPLBSimulator` 所有方法退化 noop，无任何副作用。
- 重排后的 `expertToRankMap` 可通过重建 `SimMoeBackend` 实例恢复初始映射。
