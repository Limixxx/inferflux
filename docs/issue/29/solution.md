---
title: "Issue #29 解决方案"
issue_number: 29
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 2
---

# Issue #29 解决方案

## 需求分析

### 问题描述

Issue #29 要求在 `server/src/sglang/parallel/**` 下实现 P5 层 CP（Context Parallel）并行仿真组件，用于仿真长序列在 CP 维度的 KV all-gather 通信成本。核心内容包括：

1. **CPSimulator**（§10.8）：Context Parallel 仿真器，仅在 `cp_size > 1` 时启用
2. **simulate_attn_forward**：仿真 CP attention 的 all-gather 通信成本，计算 KV cache 全量同步开销
3. **MockEngine 集成**：在 `forward_batch` 每层 attention 结束时注入 CP 通信成本，MLP 层不通信
4. **ParallelMetrics 更新**：累加 `cp_comm_ticks / cp_all_gather_count / cp_seq_len_per_rank`

### 能力目标

- 实现 CPSimulator 类，封装 CP 通信组（SimCommGroup group_type="cp"）和 KV all-gather 数据量计算
- 长序列按 cp_size 切分：`seq_per_rank = ceil(seq_len, cp_size)`
- KV all-gather 数据量公式（单层，每 rank）：`seq_len_per_rank × num_kv_heads × head_dim × dtype_size × 2`（K+V 两份）
- 通过 SimCommGroup("cp").allGather() 计算通信 ticks
- cp_size=1 时 CPSimulator 不创建，simulate_attn_forward 返回 0（skip）
- 在 MockEngine.forward_batch 中每层 attn 结束时调用 cp_sim.simulate_attn_forward，累加到 ParallelMetrics

### 影响范围

- **新增文件**：`server/src/sglang/parallel/cp_simulator.ts` — CPSimulator 实现
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出 CPSimulator）、`server/src/sglang/engine/index.ts`（MockEngine 集成 CPSimulator）、`server/src/sglang/index.ts`（导出）
- **不修改**：comm_group.ts、topology.ts、metrics.ts（已有 CP 相关字段）、types.ts（已有 cpSize/cpAllGatherCostPerByteTicks/cpEfficiency 配置）、测试代码

### 依赖关系

- **依赖 #21 (P0)**：SimCommGroup（cp group_type 带宽/延迟参数）+ ParallelTopology（computeAttnRanks）+ ParallelMetrics（cp_comm_ticks/cp_all_gather_count/cp_seq_len_per_rank 字段）— **已实现**
- **被阻塞于 #30 (P6)**：init_parallel_groups 条件创建 cp_sim（cp_size>1 时创建 CPSimulator 实例）

---

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建 `cp_simulator.ts`，实现 CPSimulator 类。该类在构造时创建 `SimCommGroup("cp")` 通信组，提供 `simulateAttnForward` 方法计算 KV all-gather 通信成本。在 MockEngine 的 `forward_batch` 方法中，每层 attention 计算结束后调用 CPSimulator，将通信成本累加到 ParallelMetrics 的 CP 字段。MLP 层跳过 CP 通信。

### 详细设计

#### 1. CPSimulator 类

**文件**：`server/src/sglang/parallel/cp_simulator.ts`

**类设计**：

```typescript
export class CPSimulator {
  readonly cpSize: number;
  readonly commGroup: SimCommGroup | null;
  totalCommTicks: number = 0;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig);
  simulateAttnForward(seqLen: number): CPAttnResult;
}

export interface CPAttnResult {
  commTicks: number;
  allGatherBytes: number;
  seqLenPerRank: number;
}
```

**构造函数逻辑**：

- 保存 `config.cpSize`
- 当 `cpSize > 1` 时创建 `SimCommGroup({ groupType: "cp", size: cpSize, networkBandwidthGBps: config.networkBandwidthGBps, latencyUs: config.networkLatencyUs, efficiency: config.cpEfficiency })`
- 当 `cpSize <= 1` 时 `commGroup = null`

**simulateAttnForward 方法逻辑**：

```
输入: seqLen (完整序列长度)

1. 若 cpSize <= 1 或 commGroup === null → 返回 { commTicks: 0, allGatherBytes: 0, seqLenPerRank: seqLen }

2. 计算 seq_len_per_rank = divCeil(seqLen, cpSize)

3. 计算 KV all-gather 数据量（单层，每 rank 持有的 KV）:
   kv_bytes_per_rank = seqLenPerRank × modelConfig.numKvHeads × modelConfig.headDim × config.dtypeSize × 2
   （×2 是因为 K 和 V 两份）
   注意：不含 numLayers 因子，因为本方法在 engine 的层循环中逐层调用

4. 调用 commGroup.allGather([kv_bytes_per_rank]) 计算通信 ticks
   allGather 接收每 rank 的字节数数组

5. 累加 totalCommTicks += commTicks

6. 返回 { commTicks, allGatherBytes: kv_bytes_per_rank, seqLenPerRank }
```

**关键设计点**：

- `simulateAttnForward` 的参数为 `seqLen`（完整序列长度），内部自动计算 `seqLenPerRank`
- `seq_len_per_rank = divCeil(seqLen, cpSize)`：处理序列长度不能整除 cp_size 的情况，向上取整
- `kv_bytes_per_rank` 不含 `num_layers` 因子：因为本方法代表单层 attention 后的 KV all-gather，在 MockEngine 的层循环中逐层调用并累加。总通信量 = num_layers × 单层通信量
- `allGather` 接收的是每 rank 的字节数（基于 `seqLenPerRank`），而非完整序列的字节数
- 构造函数使用已有的 `SimulatorConfig` 和 `ModelConfig` 类型，保持类型一致性
- 返回 `CPAttnResult` 接口，包含通信 ticks、all-gather 字节数（每 rank）、每 rank 序列长度，便于 MockEngine 累加到 ParallelMetrics

#### 2. parallel/index.ts 导出更新

在 `parallel/index.ts` 中新增 CPSimulator 导出：

```typescript
export {
  CPSimulator,
  CPAttnResult,
} from "./cp_simulator";
```

#### 3. MockEngine 集成

**文件**：`server/src/sglang/engine/index.ts`

在 MockEngine 中新增：

- `cpSim: CPSimulator | null` 属性，构造时根据 `config.cpSize > 1` 条件创建
- 在 `forward_batch` 方法中，每层 attention 计算结束后调用 `cpSim.simulateAttnForward(batch.seqLen)`
- 将结果累加到 `metrics.parallel.cpCommTicks / cpAllGatherCount / cpSeqLenPerRank`
- MLP 层不执行 CP 通信

**集成伪代码**：

```typescript
class MockEngine {
  cpSim: CPSimulator | null;

  constructor(config, modelConfig) {
    // ... 已有代码
    this.cpSim = config.cpSize > 1
      ? new CPSimulator(config, modelConfig)
      : null;
  }

  forwardBatch(batch, samplingArgs) {
    // ... 已有逻辑

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      // 1. Attention 计算
      // ... 已有 attn 逻辑

      // 2. CP KV all-gather（仅 attn 层后）
      if (this.cpSim) {
        const cpResult = this.cpSim.simulateAttnForward(batch.seqLen);
        this.metrics.parallel.cpCommTicks += cpResult.commTicks;
        this.metrics.parallel.cpAllGatherCount += 1;
        this.metrics.parallel.cpSeqLenPerRank = cpResult.seqLenPerRank;
      }

      // 3. MLP 计算（不触发 CP 通信）
      // ... 已有 mlp 逻辑
    }
  }
}
```

#### 4. sglang/index.ts 导出更新

在顶层导出中新增 CPSimulator：

```typescript
export {
  CPSimulator,
  CPAttnResult,
} from "./parallel";
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/cp_simulator.ts` — CPSimulator + CPAttnResult 实现
2. **修改** `server/src/sglang/parallel/index.ts` — 导出 CPSimulator、CPAttnResult
3. **修改** `server/src/sglang/engine/index.ts` — MockEngine 集成 CPSimulator（条件创建 + forward_batch 注入）
4. **修改** `server/src/sglang/index.ts` — 顶层导出 CPSimulator、CPAttnResult

---

## 测试设计

### 验收测试用例清单

#### CPSimulator 单元测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | cp_size=1 时 simulateAttnForward 返回零 | cp_size=1 不创建通信组，返回 commTicks=0 |
| T2 | cp_size=4 时 comm_ticks 为 cp_size=1 的 ~4× | 验证 CP 通信成本随 cp_size 增大而增加（allGather 数据量不变但组变大） |
| T3 | seq_len 不能整除 cp_size 时 seq_per_rank 分布正确 | 如 seq_len=10, cp_size=4 → seq_per_rank=3 (divCeil) |
| T4 | cp_size=4 时 allGatherBytes 计算正确 | kv_bytes_per_rank = seq_len_per_rank × num_kv_heads × head_dim × dtype_size × 2 |
| T5 | cp_size=4 时 cpAllGatherCount 每层递增 | 每次 simulateAttnForward 调用 allGatherCount += 1 |
| T6 | totalCommTicks 累加正确 | 多次调用后 totalCommTicks 等于各次返回值之和 |
| T7 | cp_size=1 时 CPSimulator 为 null（skip） | MockEngine 中 cp_size=1 不创建 CPSimulator |

#### 集成测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T8 | cp_size=4 forward_batch 后 cpCommTicks > 0 | MockEngine 完整 forward 后 CP 通信指标非零 |
| T9 | cp_size=1 forward_batch 后 cpCommTicks = 0 | CP 未启用时无通信成本 |
| T10 | cp_size=4 时 cpSeqLenPerRank 正确 | seq_len=1024, cp_size=4 → seq_len_per_rank=256 |
| T11 | CP + TP 组合：commTicksTotal 包含 cp + tp | 验证 CP 与 TP 通信成本正确叠加 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | seq_len=0 | kv_bytes=0, allGather 返回 latency_us（仅延迟） |
| B2 | seq_len < cp_size（如 seq_len=2, cp_size=4） | seq_per_rank=1 (divCeil)，部分 rank 空转 |
| B3 | cp_size=tp_size（如 cp_size=4, tp_size=4） | attn_tp_size=1，所有 TP rank 参与不同 CP chunk |
| B4 | num_layers=1 | 总 cpCommTicks = 单层 commTicks，allGatherBytes 不含 numLayers 因子 |
| B5 | cpEfficiency=1.0 与默认 0.90 对比 | efficiency=1.0 时通信成本更低 |
| B6 | 极大 seq_len（如 128K） | kv_bytes 大，allGather 成本显著 |

---

## 风险与注意事项

### 兼容性影响

- **CPSimulator 新增类**：纯新增代码，不影响现有功能
- **MockEngine 修改**：仅在 `cp_size > 1` 时新增 CP 逻辑分支，`cp_size=1`（默认值）时行为完全不变
- **ParallelMetrics CP 字段**：已在 P0（#21）中预定义，本 Issue 仅填充数据，无需修改数据结构

### 性能影响

- CPSimulator.simulateAttnForward 为 O(1) 纯算术计算，每层调用一次，总计 num_layers 次，无性能风险
- allGather 计算基于 SimCommGroup 已有的成本模型，无额外开销

### 回滚方案

- 新增文件删除即可回滚（cp_simulator.ts）
- MockEngine 中的 CP 分支条件判断确保 cp_size=1 时无影响，回滚只需移除 cpSim 相关代码
- 无数据库/持久化变更，回滚无数据风险
