---
title: "Issue #28 解决方案"
issue_number: 28
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #28 解决方案

## 需求分析

### 问题描述

Issue #28 要求在 `server/src/sglang/parallel/**` + `engine/**` 下实现 PP（Pipeline Parallel）流水并行仿真器 `PPPipelineSimulator`（§10.5），包含：

1. **三种流水线调度**：`gpipe` / `1f1b` / `interleaved`，通过 `config.ppPipelineSchedule` 配置
2. **Bubble 公式仿真**（§10.5.2）：
   - gpipe：`bubble_ticks = (pp_size - 1) × micro_batch_ticks × num_micro_batches`
   - 1f1b：`bubble_ticks = (pp_size - 1) × micro_batch_ticks`（最优，推荐默认）
   - interleaved：`bubble_ticks = (pp_size - 1) × num_chunks × micro_batch_ticks`（num_chunks=2 典型）
3. **Stage 间通信成本**：`_stageSendRecvCost(bytesPerStage)` 通过 `SimCommGroup("pp").sendRecv(bytes)` 计两遍（send + recv）
4. **仅最后 stage 采样**：`is_pp_last_stage(stage_idx) = stage_idx === pp_size-1`；forward_batch 返回 `is_intermediate: boolean`（非 last stage）时，不触发采样、不增加 `sampling_counter`
5. **`simulate_pipeline_step(batch)`**：返回 `{total_ticks, bubble_ticks, send_recv_ticks, per_stage_ticks[]}`
6. **Integrate 点**：MockEngine.forward_batch 前先切 micro_batch；按 schedule 循环；last 走采样；中间 stage 返回 intermediate
7. **ParallelMetrics**：pp_bubble_ticks / pp_num_micro_batches / pp_send_recv_ticks 回填；size=1 时 bubble=0
8. **单元测试**：gpipe bubble 随 pp_size 平方上升；1f1b 与 size=1 退化成线性单测；intermediate 阶段 sampling_counter 不变化；last 阶段 token 生成

### 能力目标

- 纯算术仿真流水线调度时序和通信成本，不实际分割模型或传输数据
- 支持 `gpipe` / `1f1b` / `interleaved` 三种调度策略的 bubble 和通信成本计算
- 实现正确的 bubble 公式：gpipe bubble 随 pp_size 和 num_micro_batches 二次增长，1f1b 最优，interleaved 介于两者之间
- 仅最后一个 PP stage 执行采样逻辑，中间 stage 只传递 hidden_states
- `pp_size=1` 时所有方法 noop（退化为单 stage），不破坏现有单实例行为

### 影响范围

- **新建文件**：`server/src/sglang/parallel/pp.ts` — PPPipelineSimulator 实现
- **修改文件**：
  - `server/src/sglang/parallel/index.ts` — 导出 PPPipelineSimulator
  - `server/src/sglang/engine/index.ts` — MockEngine 集成 PP 通信成本注入 + is_pp_last 采样控制
  - `server/src/sglang/core/index.ts` — ForwardOutput 类型扩展（增加 is_intermediate 字段）
  - `server/src/test/sglang-pp.test.ts` — PP 单元测试
- **依赖**：Issue #21（P0: ParallelTopology + SimCommGroup）已完成，`pp_stage_layers` 和 `SimCommGroup("pp")` 可用

## 改造方案

### 总体思路

按照总体设计文档 §10.5 的规格，在 `server/src/sglang/parallel/pp.ts` 中实现 `PPPipelineSimulator` 类，支持三种调度策略的 bubble 计算和 stage 间通信成本计算。然后在 `MockEngine.forward_batch` 中集成 PP 通信成本注入和采样控制逻辑。

### 详细设计

#### 1. PPPipelineSimulator 类（`parallel/pp.ts`）

```typescript
export interface PipelineStepResult {
  totalTicks: number;       // send_recv_ticks + bubble_ticks
  bubbleTicks: number;      // 流水线气泡 ticks
  sendRecvTicks: number;    // stage 间 send/recv 总成本
  perStageTicks: number[];  // 每个 stage 的通信 ticks
}

export class PPPipelineSimulator {
  readonly ppSize: number;
  readonly schedule: "1f1b" | "gpipe" | "interleaved";
  readonly numMicroBatches: number;
  readonly commGroup: SimCommGroup | null;
  readonly stageLayers: Array<{ start: number; end: number }>;
  readonly numChunks: number;  // interleaved 专用，默认 2

  bubbleTicks: number;
  commTicksTotal: number;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig);

  /** 仿真整个 pipeline forward，返回 PipelineStepResult */
  simulatePipelineStep(batch: Batch): PipelineStepResult;

  /** 将 batch 按 numMicroBatches 分割 */
  _splitMicroBatches(batch: Batch): Array<{ size: number }>;

  /** 计算单个 stage 间 send/recv 的通信成本（双向） */
  _stageSendRecvCost(microBatchSize: number): number;

  /** 是否最后一个 PP stage */
  isPpLastStage(stageIdx: number): boolean;
}
```

**关键设计点**：

1. **Bubble 公式**（§10.5.2）：

   - `microBatchTicks = config.eagerForwardCostTicks`（单个 micro-batch 在单个 stage 的计算耗时）
   - gpipe：`bubble_ticks = (pp_size - 1) × microBatchTicks × numMicroBatches` — 所有 micro-batch 全部 forward 完后才能 backward，bubble 最大
   - 1f1b：`bubble_ticks = (pp_size - 1) × microBatchTicks` — 交替 forward/backward，bubble 最优
   - interleaved：`bubble_ticks = (pp_size - 1) × numChunks × microBatchTicks`（numChunks=2 典型）— 将模型层分成多 chunk 交错执行，bubble 介于 gpipe 和 1f1b 之间

2. **通信成本计算**：每个 micro-batch 在相邻 stage 间传递 hidden_states，产生 `sendRecv` 成本。数据量 = `microBatchSize × hiddenSize × dtypeSize`。对每个 micro-batch，需要 (pp_size - 1) 次 stage 间传输。

3. **`_stageSendRecvCost` 实现**：调用 `SimCommGroup("pp").sendRecv(dataBytes)` 计算单次点对点通信成本。每个 micro-batch 经过 pp_size-1 个 stage 边界，每次边界产生一次 send + recv（已包含在 sendRecv 计算中）。

4. **退化为 noop**：`pp_size=1` 时，`simulatePipelineStep` 返回全零结果，`commGroup=null`，所有通信计算跳过。

#### 2. ForwardOutput 类型扩展（`core/index.ts`）

在 Batch 相关输出类型中增加 `isIntermediate` 字段：

```typescript
export interface ForwardOutput {
  logits: number[] | null;
  sampledIds: number[] | null;
  isIntermediate: boolean;  // 新增：非最后 PP stage 时为 true
}
```

#### 3. MockEngine 集成（`engine/index.ts`）

```typescript
class MockEngine {
  ppRank: number;
  isPpLast: boolean;
  ppSim: PPPipelineSimulator;

  forwardBatch(batch: Batch, args: BatchSamplingArgs): ForwardOutput {
    // ... 原有 forward 逻辑 ...

    // PP 通信成本注入
    const stepResult = this.ppSim.simulatePipelineStep(batch);
    if (stepResult.totalTicks > 0) {
      this.metrics.parallel.ppSendRecvTicks += stepResult.sendRecvTicks;
      this.metrics.parallel.ppBubbleTicks += stepResult.bubbleTicks;
      this.metrics.parallel.ppNumMicroBatches += this.ppSim.numMicroBatches;
    }

    // 关键：只有最后一个 PP stage 做采样
    if (!this.isPpLast) {
      return { logits, sampledIds: null, isIntermediate: true };
    }
    // 最后 stage 正常采样
    const nextTokenIds = this.sampler.sample(logits, args);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }
}
```

**关键约束**：
- CUDA Graph replay 路径跳过 PP 通信仿真（graph 内已固化），与 §10.5.3 一致
- `isIntermediate=true` 时不调用 sampler、不增加 `sampling_counter`
- `pp_size=1` 时 `isPpLast=true`，行为与单实例完全一致

#### 4. ParallelMetrics 回填

PP 指标字段已在 P0（Issue #21）中预定义：
- `ppBubbleTicks` — 流水线气泡累计
- `ppNumMicroBatches` — 执行的 micro-batch 总数
- `ppSendRecvTicks` — PP send/recv 通信成本累计

本次只需在 `simulatePipelineStep` 和 `forwardBatch` 中正确累加即可。

### 修改点清单

1. **新建** `server/src/sglang/parallel/pp.ts` — PPPipelineSimulator 完整实现（构造函数、三种调度 bubble 计算、通信成本计算、micro-batch 分割、isPpLastStage）
2. **修改** `server/src/sglang/parallel/index.ts` — 增加 `PPPipelineSimulator` 及相关类型导出
3. **修改** `server/src/sglang/core/index.ts` — 新增 `ForwardOutput` 接口（含 `isIntermediate` 字段）
4. **修改** `server/src/sglang/engine/index.ts` — MockEngine 集成 PPPipelineSimulator：PP 通信成本注入 + is_pp_last 采样控制 + PP 指标回填
5. **新建** `server/src/test/sglang-pp.test.ts` — PP 单元测试（bubble 公式验证、退化测试、intermediate 采样控制测试）

## 测试设计

### 验收测试用例清单

| 编号 | 测试用例 | 预期结果 |
|------|---------|---------|
| T1 | gpipe bubble 公式验证 | `pp_size=4, num_micro_batches=4` → `bubble = 3 × eagerForwardCostTicks × 4` |
| T2 | gpipe bubble 随 pp_size 二次增长 | `pp_size=2,4,8, num_micro_batches=4` → bubble 随 pp_size 线性增长（但总延迟含 pp_size × num_micro_batches 因子，呈二次特征） |
| T3 | 1f1b bubble 最优 | `pp_size=4, num_micro_batches=4` → `bubble = 3 × eagerForwardCostTicks`（远小于 gpipe） |
| T4 | interleaved bubble 介于中间 | `pp_size=4, num_chunks=2` → `bubble = 3 × 2 × eagerForwardCostTicks` |
| T5 | pp_size=1 退化 | `simulatePipelineStep` 返回全零，无通信成本，无 bubble |
| T6 | send/recv 通信成本 | `pp_size=2, num_micro_batches=2` → 2 次 stage 间传输，每次 cost > 0 |
| T7 | isPpLastStage 验证 | `pp_size=4` → `isPpLastStage(3)=true`，其他为 false |
| T8 | intermediate stage 不采样 | `isPpLast=false` → `ForwardOutput.isIntermediate=true`，`sampledIds=null` |
| T9 | last stage 正常采样 | `isPpLast=true` → `ForwardOutput.isIntermediate=false`，`sampledIds` 非空 |
| T10 | sampling_counter 中间 stage 不变化 | intermediate stage 的 `samplingCounter` 保持不变 |
| T11 | ParallelMetrics PP 指标回填 | forward 后 `ppBubbleTicks > 0`、`ppSendRecvTicks > 0`、`ppNumMicroBatches > 0` |
| T12 | pp_size=1 时 ParallelMetrics 全零 | `ppBubbleTicks=0, ppSendRecvTicks=0, ppNumMicroBatches=0` |
| T13 | micro-batch 分割 | `batch.size=8, numMicroBatches=4` → 4 个 micro-batch，size 均为 2 |
| T14 | micro-batch 不整除 | `batch.size=7, numMicroBatches=3` → sizes=[3, 2, 2] |
| T15 | numMicroBatches=1 退化 | `_splitMicroBatches` 返回单个元素，等同于不切分 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `pp_size=0` | 构造函数抛出 Error（无效配置） |
| B2 | `numMicroBatches=0` | `_splitMicroBatches` 返回空数组，无通信成本 |
| B3 | `batch.size=0` | micro-batch size 为 0，通信数据量为 0，sendRecvCost 为 0 |
| B4 | 未知 schedule 值 | `simulatePipelineStep` 抛出 `ValueError` |
| B5 | `commGroup=null`（pp_size=1） | `_stageSendRecvCost` 返回 0 |
| B6 | bandwidth=0 | `SimCommGroup.sendRecv` 返回 Infinity |

## 风险与注意事项

### 兼容性影响

- **零破坏性**：`pp_size=1` 时所有新增逻辑 noop，不影响现有单实例行为
- **ForwardOutput 新增字段**：`isIntermediate` 默认为 `false`，现有代码无需修改
- **ParallelMetrics 字段已在 P0 预定义**：无需修改 metrics 结构

### 性能影响

- PPPipelineSimulator 是纯算术计算，无 IO/网络开销，性能影响可忽略
- 三种调度策略的计算复杂度均为 O(pp_size × numMicroBatches)，在仿真尺度下无性能风险

### 回滚方案

- 若需回滚，删除 `parallel/pp.ts`，还原 `parallel/index.ts` 和 `engine/index.ts` 的改动即可
- `ForwardOutput.isIntermediate` 字段为可选增量，不影响已有逻辑
