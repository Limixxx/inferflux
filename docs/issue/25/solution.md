---
title: "Issue #25 解决方案"
issue_number: 25
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #25 解决方案

## 需求分析

### 问题描述

Issue #25 要求在 `server/src/sglang/parallel/**` 下实现 P2b 层 DP Attention Simulator（`DPAttentionSimulator`），这是 MLA（Multi-head Latent Attention）模型专用的 DP Attention 仿真组件。

核心功能：
1. 仅当 `model_config.use_mla && parallel.enable_dp_attention` 时启用（否则 attn 通信走 normal，P2a 标准 DP 不需要）
2. `simulate_mlp_forward(batch, local_batch_sizes_per_rank)` — 仿真 MLP 层的 all-gather → forward → slice 通信成本
3. `total_all_gather_bytes_per_step(batch)` — 纯预览函数，计算每步 all-gather 数据量
4. Integrate 点：MockEngine.forward_batch 内，对每一层 MLP 调用 dp_attn_sim.simulate_mlp_forward；attn 跳过；累加 ParallelMetrics.dp_attn_comm_ticks
5. 单元测试：dp_size=2，batch=2 分块 1+1；all_gather_ticks 随 dp_size 线性；mla=false 不启用，skip

### 能力目标

- **DP Attention 仿真**：忠实模拟 MLA 模型下 DP Attention 的 all-gather → forward → slice 数据流通信成本
- **条件启用**：仅在 `useMla=true && enableDpAttention=true && dpSize>1` 时激活，其余情况退化为 noop
- **通信成本计算**：基于已有的 `SimCommGroup("dp_attn")` 计算精确的 all-gather ticks
- **指标收集**：将 all-gather 通信成本累加到 `ParallelMetrics.dpAttnCommTicks`
- **预览功能**：`totalAllGatherBytesPerStep` 可在不执行通信的情况下预览数据量

### 影响范围

- **新增文件**：`server/src/sglang/parallel/dp_attn.ts` — DPAttentionSimulator 实现
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出）、`server/src/sglang/index.ts`（导出）
- **不修改**：业务源码、MockEngine（集成由后续 Issue 完成）、测试代码

### 依赖关系

- **依赖 #21 (P0)**：`SimCommGroup`（已实现，支持 `dp_attn` group_type）、`ParallelTopology`（已实现，含 `enableDpAttention`）、`ParallelMetrics`（已实现，含 `dpAttnCommTicks` 字段）、`SimulatorConfig`（已新增 `enableDpAttention`/`dpAttentionAllGatherCostPerByteTicks` 等字段）— **已实现**
- **阻塞**：P6 `init_parallel_groups`（条件创建，enable_dp_attention）— 后续 Issue

---

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建 `dp_attn.ts` 模块，实现 `DPAttentionSimulator` 类。该类内部持有 `SimCommGroup("dp_attn")` 实例，根据 `useMla` 和 `enableDpAttention` 条件决定是否启用。核心方法 `simulateMlpForward` 计算每层 MLP 的 all-gather 通信成本，`totalAllGatherBytesPerStep` 提供纯预览功能。

### 详细设计

#### 1. DPAttentionSimulator — DP Attention 仿真器

**文件**：`server/src/sglang/parallel/dp_attn.ts`

**核心机制**（对应 SGLang 的 `enable_dp_attention`）：

1. **Attention 层**：不通信（MLA 规则：KV cache 每 rank 自己维护，attn 结果按 dp 子 batch 独立）
2. **MLP 层**：
   - all-gather 输入：`sum(local_batch_sizes) × hidden_dim × dtype_size`，走 `SimCommGroup("dp_attn").allGather(local_batch_sizes.map(sz => sz * hiddenSize * dtypeSize))`
   - forward：不计时成本（由 MockEngine 负责）
   - slice 输出：把 global batch 切回 rank 本地 batch_sizes，通信 0，纯索引操作

**类设计**：

```typescript
export interface DPAttentionSimulatorOpts {
  dpSize: number;
  hiddenSize: number;
  dtypeSize: number;
  useMla: boolean;
  enableDpAttention: boolean;
  networkBandwidthGBps: number;
  networkLatencyUs: number;
}

export class DPAttentionSimulator {
  readonly dpSize: number;
  readonly hiddenSize: number;
  readonly dtypeSize: number;
  readonly useMla: boolean;
  readonly enableDpAttention: boolean;

  /** all-gather 通信组（仅当启用时非 null） */
  readonly commGroup: SimCommGroup | null;

  /** 是否启用（useMla && enableDpAttention && dpSize > 1） */
  readonly enabled: boolean;

  constructor(opts: DPAttentionSimulatorOpts);

  /**
   * 仿真 MLP 层的 all-gather → forward → slice 通信成本。
   * localBatchSizes: 各 DP rank 的本地 batch_size 列表。
   * 返回 { commTicks, allGatherBytes }。
   */
  simulateMlpForward(localBatchSizes: number[]): {
    commTicks: number;
    allGatherBytes: number;
  };

  /**
   * 纯预览函数：计算给定 batch 下每步 all-gather 数据量（字节数）。
   * batch: 总 batch 大小。
   * 返回每步 all-gather 总字节数。
   */
  totalAllGatherBytesPerStep(batch: number): number;
}
```

**启用条件**：

```typescript
this.enabled = opts.useMla && opts.enableDpAttention && opts.dpSize > 1;
```

仅当三个条件同时满足时才创建 `SimCommGroup("dp_attn")` 实例并启用仿真，否则 `commGroup = null`，所有方法退化为 noop。

**simulateMlpForward 实现**：

```typescript
simulateMlpForward(localBatchSizes: number[]): { commTicks: number; allGatherBytes: number } {
  if (!this.enabled || this.commGroup === null) {
    return { commTicks: 0, allGatherBytes: 0 };
  }

  // all-gather: 各 rank 的 hidden_states 汇聚
  // 数据量 = local_batch_size[i] × hiddenSize × dtypeSize（每个 rank）
  const gatherSizes = localBatchSizes.map(sz => sz * this.hiddenSize * this.dtypeSize);
  const allGatherBytes = gatherSizes.reduce((s, v) => s + v, 0);

  // all-gather 通信成本
  const commTicks = this.commGroup.allGather(gatherSizes);

  // slice 不产生通信成本（本地索引切片）
  return { commTicks, allGatherBytes };
}
```

**totalAllGatherBytesPerStep 实现**：

```typescript
totalAllGatherBytesPerStep(batch: number): number {
  if (!this.enabled) {
    return 0;
  }
  // 假设 batch 在 dp_size 个 rank 间均匀分布
  const localBatchSize = batch / this.dpSize;
  const perRankBytes = localBatchSize * this.hiddenSize * this.dtypeSize;
  return perRankBytes * this.dpSize; // = batch × hiddenSize × dtypeSize
}
```

**构造函数**：

```typescript
constructor(opts: DPAttentionSimulatorOpts) {
  this.dpSize = opts.dpSize;
  this.hiddenSize = opts.hiddenSize;
  this.dtypeSize = opts.dtypeSize;
  this.useMla = opts.useMla;
  this.enableDpAttention = opts.enableDpAttention;

  this.enabled = opts.useMla && opts.enableDpAttention && opts.dpSize > 1;

  this.commGroup = this.enabled
    ? new SimCommGroup({
        groupType: "dp_attn",
        size: opts.dpSize,
        networkBandwidthGBps: opts.networkBandwidthGBps,
        latencyUs: opts.networkLatencyUs,
        efficiency: 1.0, // DP Attention 无额外效率因子衰减
      })
    : null;
}
```

#### 2. parallel/index.ts 导出更新

新增 `DPAttentionSimulator` 和 `DPAttentionSimulatorOpts` 导出：

```typescript
export {
  DPAttentionSimulator,
  DPAttentionSimulatorOpts,
} from "./dp_attn";
```

#### 3. sglang/index.ts 导出更新

新增 `DPAttentionSimulator` 和 `DPAttentionSimulatorOpts` 导出到顶层：

```typescript
export {
  DPAttentionSimulator,
  DPAttentionSimulatorOpts,
} from "./parallel";
```

#### 4. Integrate 点（说明，不修改 MockEngine）

Issue 明确的 integrate 点是 MockEngine.forward_batch 内，对每一层 MLP 调用 `dpAttnSim.simulateMlpForward`，attn 跳过，累加 `ParallelMetrics.dpAttnCommTicks`。但本 Issue 仅负责 DPAttentionSimulator 组件实现，MockEngine 集成由后续 Issue（含 forward_batch 实现）完成。

集成伪代码参考：

```typescript
// 在 MockEngine.forward_batch 中（后续 Issue 实现）
for (let layer = 0; layer < numLayers; layer++) {
  // Attention 层：跳过（MLA 规则，每 rank 独立 KV cache）
  // 无通信成本

  // MLP 层：调用 DP Attention 仿真
  const result = this.dpAttnSim.simulateMlpForward(localBatchSizes);
  metrics.parallel.dpAttnCommTicks += result.commTicks;
}
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/dp_attn.ts` — DPAttentionSimulator 实现
2. **修改** `server/src/sglang/parallel/index.ts` — 新增导出 DPAttentionSimulator + DPAttentionSimulatorOpts
3. **修改** `server/src/sglang/index.ts` — 新增导出 DPAttentionSimulator + DPAttentionSimulatorOpts

---

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | 启用条件 — useMla=true && enableDpAttention=true && dpSize=2 | enabled=true，commGroup 非 null |
| T2 | 不启用 — useMla=false | enabled=false，commGroup=null，simulateMlpForward 返回 {0,0} |
| T3 | 不启用 — enableDpAttention=false | enabled=false，commGroup=null |
| T4 | 不启用 — dpSize=1 | enabled=false，commGroup=null |
| T5 | simulateMlpForward dpSize=2 batch=2 分块 1+1 | 返回正 commTicks 和正确 allGatherBytes |
| T6 | all_gather_ticks 随 dpSize 线性增长 | dpSize=2 vs dpSize=4，后者 commTicks 更大 |
| T7 | simulateMlpForward 未启用返回 0 | mla=false 不启用时返回 {commTicks:0, allGatherBytes:0} |
| T8 | totalAllGatherBytesPerStep 启用时返回正值 | batch × hiddenSize × dtypeSize |
| T9 | totalAllGatherBytesPerStep 未启用返回 0 | mla=false 时返回 0 |
| T10 | commGroup 类型为 dp_attn | commGroup.groupType === "dp_attn" |
| T11 | SimCommGroup size=1 退化 | 构造 dpSize=2 时 commGroup.size=2，但若传 dpSize=1 则不创建 |
| T12 | allGatherBytes 计算正确 | localBatchSizes=[1,1] → 2 × hiddenSize × dtypeSize |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | localBatchSizes 为空数组 | commTicks=0（SimCommGroup.allGather([]) 返回 0） |
| B2 | localBatchSizes 全为 0 | allGatherBytes=0，commTicks 仅含 latency |
| B3 | batch=0 传入 totalAllGatherBytesPerStep | 返回 0 |
| B4 | dpSize=2 但 localBatchSizes 长度不等于 dpSize | 由调用方保证一致性，不做额外校验 |
| B5 | useMla=undefined（falsy） | enabled=false，退化为 noop |

---

## 风险与注意事项

### 兼容性影响

- **纯新增模块**：`dp_attn.ts` 为全新文件，不修改任何现有逻辑
- **导出扩展**：仅在 `index.ts` 中新增导出，不影响现有导出项
- **不修改 SimulatorConfig / ParallelMetrics**：所需字段（`enableDpAttention`、`useMla`、`dpAttnCommTicks`）已在 #21 P0 Issue 中实现

### 性能影响

- `simulateMlpForward` 为 O(dpSize) 纯算术计算（遍历 localBatchSizes 求和 + 一次 allGather 调用），无性能风险
- `totalAllGatherBytesPerStep` 为 O(1) 纯算术，无性能风险
- 不引入任何副作用或状态变更（除 commGroup 内部计数外）

### 回滚方案

- 新增文件删除即可回滚
- `index.ts` 的修改通过 git revert 即可恢复
- 无数据库/持久化变更，回滚无数据风险
