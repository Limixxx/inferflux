---
title: "Issue #28 解决方案"
issue_number: 28
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: revised
review_round: 2
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
3. **Stage 间通信成本**：`_stageSendRecvCost(microBatchSize)` 通过 `SimCommGroup("pp").sendRecv(bytes)` 计两遍（send + recv）
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
  - `server/src/sglang/parallel/index.ts` — 导出 PPPipelineSimulator 及相关类型
  - `server/src/sglang/engine/index.ts` — MockEngine 集成 PP 通信成本注入 + is_pp_last 采样控制
  - `server/src/sglang/core/index.ts` — 新增 `ForwardOutput` 接口（含 `isIntermediate` 字段）
  - `server/src/test/sglang-pp.test.ts` — PP 单元测试
- **依赖**：Issue #21（P0: ParallelTopology + SimCommGroup）已完成，`ppStageLayers` 和 `SimCommGroup("pp")` 可用

## 改造方案

### 总体思路

按照总体设计文档 §10.5 的规格，在 `server/src/sglang/parallel/pp.ts` 中实现 `PPPipelineSimulator` 类，支持三种调度策略的 bubble 计算和 stage 间通信成本计算。然后在 `MockEngine.forwardBatch` 中集成 PP 通信成本注入和采样控制逻辑。

### 上一轮评审问题回应（Review Round 1 → Round 2）

| 编号 | 评审问题 | 回应与改进 |
|------|---------|-----------|
| R1 | Micro-batch 分割时机不明确 | **明确契约**：micro-batch 分割仅发生在 PP 仿真层面（`PPPipelineSimulator._splitMicroBatches`），Scheduler 无需感知 micro-batch 存在。分割结果仅用于计算通信成本和 bubble，不影响 Scheduler 的调度逻辑。不整除时采用 ceil 分配策略（前 `remainder` 个 micro-batch 多分配 1），不进行 padding，允许末尾 micro-batch 较小。 |
| R2 | ForwardOutput.isIntermediate 行为定义模糊 | **明确定义行为合约**：`isIntermediate=true` 时：(1) logits 仍然生成（用于通信传给下一 stage）；(2) `sampledIds` 为 `null`；(3) Scheduler/MockEngine **不调用 sampler**；(4) `samplingCounter` 不增加。详见下方 ForwardOutput 类型定义。 |
| R3 | CUDA Graph 与 PP 互作关系 | **明确说明**：`canUseCudaGraph` 判断**不需要**考虑 PP stage 状态。CUDA Graph replay 路径已固化通信操作，因此 replay 时 PP 通信成本记为 0，这是正确行为（与 §10.5.3 一致）。PP 通信成本仅在 eager forward 路径中注入。 |
| R4 | 数据量计算缺少细节（TP × PP 互作） | **补充说明**：当 `tp_size > 1` 时，hidden_size 已按 TP 分割，因此实际传输数据量 = `microBatchSize × (hiddenSize / tpSize) × dtypeSize`。`_stageSendRecvCost` 中需引入 `tpSize` 参数进行修正。详见下方通信成本计算设计。 |
| R5 | pp_stage_layers 的获取路径 | **明确路径**：构造函数内创建 `ParallelTopology({ ppSize })` 实例，调用 `topology.ppStageLayers(modelConfig.numLayers)` 获取 `stageLayers`。不是外部注入，而是在构造器内部自动推导。 |

### 详细设计

#### 1. PPPipelineSimulator 类（`parallel/pp.ts`）

```typescript
export interface PipelineStepResult {
  totalTicks: number;       // bubble_ticks + send_recv_ticks
  bubbleTicks: number;      // 流水线气泡 ticks
  sendRecvTicks: number;    // stage 间 send/recv 总成本
  perStageTicks: number[];  // 每个 stage 边界的通信 ticks（长度 = pp_size - 1）
}

export class PPPipelineSimulator {
  readonly ppSize: number;
  readonly schedule: "1f1b" | "gpipe" | "interleaved";
  readonly numMicroBatches: number;
  readonly tpSize: number;       // R4: 用于修正通信数据量
  readonly commGroup: SimCommGroup | null;
  readonly stageLayers: Array<{ start: number; end: number }>;
  readonly numChunks: number;    // interleaved 专用，默认 2

  bubbleTicks: number;
  commTicksTotal: number;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig) {
    this.ppSize = config.ppSize;
    this.schedule = config.ppPipelineSchedule;
    this.numMicroBatches = config.ppNumMicroBatches;
    this.tpSize = config.tpSize;   // R4: 记录 tpSize
    this.numChunks = 2;             // interleaved 默认值

    // R5: 构造器内自动推导 stageLayers
    const topology = new ParallelTopology({ ppSize: this.ppSize });
    this.stageLayers = topology.ppStageLayers(modelConfig.numLayers);

    // 通信组：pp_size > 1 时创建，否则为 null
    this.commGroup = this.ppSize > 1
      ? new SimCommGroup({
          groupType: "pp",
          size: this.ppSize,
          networkBandwidthGBps: config.networkBandwidthGBps,
          latencyUs: config.networkLatencyUs,
        })
      : null;

    this.bubbleTicks = 0;
    this.commTicksTotal = 0;
  }

  /** 仿真整个 pipeline forward，返回 PipelineStepResult */
  simulatePipelineStep(batch: Batch): PipelineStepResult;

  /**
   * 将 batch 按 numMicroBatches 分割。
   * R1 契约：仅用于 PP 仿真层面计算通信成本和 bubble，
   * 不影响 Scheduler 的调度逻辑。
   * 不整除时采用 ceil 分配（前 remainder 个多 1），不 padding。
   */
  _splitMicroBatches(batchSize: number): Array<{ size: number }>;

  /**
   * 计算单个 stage 间 send/recv 的通信成本（双向）。
   * R4 修正：数据量 = microBatchSize × (hiddenSize / tpSize) × dtypeSize。
   * 当 tp_size > 1 时，hidden_size 已按 TP 分割到各 rank，
   * 实际传输的 hidden_size 为 hiddenSize / tpSize。
   */
  _stageSendRecvCost(microBatchSize: number, hiddenSize: number, dtypeSize: number): number;

  /** 是否最后一个 PP stage */
  isPpLastStage(stageIdx: number): boolean;
}
```

**关键设计点**：

1. **Bubble 公式**（§10.5.2，严格遵循 Issue 规格）：

   - `microBatchTicks = config.eagerForwardCostTicks`（单个 micro-batch 在单个 stage 的计算耗时）
   - gpipe：`bubble_ticks = (pp_size - 1) × microBatchTicks × numMicroBatches` — 所有 micro-batch 全部 forward 完后才能 backward，bubble 最大
   - 1f1b：`bubble_ticks = (pp_size - 1) × microBatchTicks` — 交替 forward/backward，bubble 最优
   - interleaved：`bubble_ticks = (pp_size - 1) × numChunks × microBatchTicks`（numChunks=2 典型）— 将模型层分成多 chunk 交错执行，bubble 介于 gpipe 和 1f1b 之间

2. **通信成本计算**（R4 修正）：

   每个 micro-batch 在相邻 stage 间传递 hidden_states，产生 `sendRecv` 成本。
   - 数据量 = `microBatchSize × effectiveHiddenSize × dtypeSize`
   - `effectiveHiddenSize = hiddenSize / tpSize`（R4：TP 分割修正）
   - 对每个 micro-batch，需要 (pp_size - 1) 次 stage 间传输
   - `perStageTicks[i]` 记录第 i 个 stage 边界（stage i → stage i+1）的累计通信成本

3. **`_splitMicroBatches` 契约**（R1 明确）：

   - 分割仅发生在 PP 仿真内部，Scheduler 层面不感知 micro-batch
   - 不整除时：前 `remainder` 个 micro-batch 分配 `ceil(batchSize / numMicroBatches)` 个样本，剩余分配 `floor(batchSize / numMicroBatches)` 个
   - 不进行 padding，允许末尾 micro-batch 较小（与 SGLang 实际行为一致）
   - `numMicroBatches=1` 时退化为不切分，返回单个 micro-batch
   - `batchSize=0` 时返回 `numMicroBatches` 个 size=0 的 micro-batch（通信数据量为 0）

4. **退化为 noop**：`pp_size=1` 时，`simulatePipelineStep` 返回全零结果（`totalTicks=0, bubbleTicks=0, sendRecvTicks=0, perStageTicks=[]`），`commGroup=null`，所有通信计算跳过。

5. **`_stageSendRecvCost` 实现**（R4 修正后）：

   调用 `SimCommGroup("pp").sendRecv(dataBytes)` 计算单次点对点通信成本。
   - `dataBytes = microBatchSize * (hiddenSize / this.tpSize) * dtypeSize`
   - 当 `commGroup=null`（pp_size=1）时返回 0
   - 当 `tpSize=1` 时退化为 `microBatchSize × hiddenSize × dtypeSize`（与原逻辑一致）

6. **`perStageTicks` 计算**：

   每个 stage 边界的通信成本 = 该边界上所有 micro-batch 的 sendRecv 之和。
   对于三种调度策略，通信模式相同（每个 micro-batch 都要经过所有 stage 边界），因此 `perStageTicks[i]` 的计算在三种调度中一致：
   ```
   perStageTicks[i] = sum(mb in microBatches, _stageSendRecvCost(mb.size, hiddenSize, dtypeSize))
   ```

#### 2. ForwardOutput 接口定义（`core/index.ts`）

```typescript
/**
 * Forward 输出（§10.5.3）
 *
 * 行为合约（R2 明确）：
 * - isIntermediate=false（最后 PP stage 或 pp_size=1）：
 *   logits 非空，sampledIds 非空，sampler 被调用，samplingCounter 增加
 * - isIntermediate=true（中间 PP stage）：
 *   logits 非空（用于通信传给下一 stage），sampledIds=null，
 *   sampler 不被调用，samplingCounter 不增加
 */
export interface ForwardOutput {
  logits: number[] | null;      // 模型输出 logits
  sampledIds: number[] | null;  // 采样结果（中间 stage 为 null）
  isIntermediate: boolean;      // true=中间 PP stage，不采样
}
```

**R2 行为合约完整说明**：

| 条件 | `logits` | `sampledIds` | 调用 sampler? | `samplingCounter` |
|------|---------|-------------|--------------|-------------------|
| `isIntermediate=false`（last stage） | 非空 | 非空 | 是 | +1 |
| `isIntermediate=true`（中间 stage） | 非空（传给下一 stage） | `null` | 否 | 不变 |
| `pp_size=1`（退化） | 非空 | 非空 | 是 | +1 |

#### 3. MockEngine 集成（`engine/index.ts`）

```typescript
class MockEngine {
  ppRank: number;
  isPpLast: boolean;
  ppSim: PPPipelineSimulator;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig, ppRank: number = 0) {
    this.ppRank = ppRank;
    this.isPpLast = (ppRank === config.ppSize - 1);
    this.ppSim = new PPPipelineSimulator(config, modelConfig);
    // ... 原有初始化 ...
  }

  forwardBatch(batch: Batch, args: BatchSamplingArgs): ForwardOutput {
    // ... 原有 forward 逻辑 ...

    // R3: CUDA Graph replay 路径跳过 PP 通信仿真
    // canUseCudaGraph 判断不需要考虑 PP stage 状态
    // 因为 Graph 内已固化通信操作，replay 时 PP 通信成本为 0
    let ppStepResult: PipelineStepResult | null = null;

    if (this.graphRunner.canUseCudaGraph(batch)) {
      logits = this.graphRunner.replay(batch);
      // CUDA Graph 内 PP 通信成本为 0，不调用 simulatePipelineStep
    } else {
      logits = this._mockModelForward(batch);
      // eager forward 路径：注入 PP 通信成本
      ppStepResult = this.ppSim.simulatePipelineStep(batch);
    }

    // PP 指标回填
    if (ppStepResult && ppStepResult.totalTicks > 0) {
      this.metrics.parallel.ppSendRecvTicks += ppStepResult.sendRecvTicks;
      this.metrics.parallel.ppBubbleTicks += ppStepResult.bubbleTicks;
      this.metrics.parallel.ppNumMicroBatches += this.ppSim.numMicroBatches;
    }

    // R2: 严格遵循行为合约
    if (!this.isPpLast) {
      // 中间 stage：logits 仍生成（通信用），但不采样
      return { logits, sampledIds: null, isIntermediate: true };
    }

    // 最后 stage（或 pp_size=1）：正常采样
    const nextTokenIds = this.sampler.sample(logits, args);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }
}
```

**关键约束**：
- `isIntermediate=true` 时**不调用** `sampler.sample()`，**不增加** `samplingCounter`
- `pp_size=1` 时 `isPpLast=true`，行为与单实例完全一致，`isIntermediate=false`
- CUDA Graph replay 路径跳过 PP 通信仿真（R3：与 §10.5.3 一致），`canUseCudaGraph` 无需考虑 PP stage

#### 4. ParallelMetrics 回填

PP 指标字段已在 P0（Issue #21）中预定义于 `ParallelMetrics`：
- `ppBubbleTicks` — 流水线气泡累计
- `ppNumMicroBatches` — 执行的 micro-batch 总数
- `ppSendRecvTicks` — PP send/recv 通信成本累计

本次只需在 `simulatePipelineStep` 和 `forwardBatch` 中正确累加即可。`pp_size=1` 时三个字段均保持默认值 0。

#### 5. SimulatorConfig 已有 PP 字段确认

`SimulatorConfig`（`types.ts`）中已包含完整的 PP 配置字段（评审建议补充的字段实际已存在）：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ppSize` | `number` | `1` | PP stage 数 |
| `ppNumMicroBatches` | `number` | `1` | micro-batch 数 |
| `ppSendRecvCostPerByteTicks` | `number` | `0.0005` | 旧兼容字段 |
| `ppPipelineSchedule` | `"1f1b" \| "gpipe" \| "interleaved"` | `"1f1b"` | 调度策略 |
| `networkBandwidthGBps` | `number` | `100` | 通信统一参数 |
| `networkLatencyUs` | `number` | `5` | 通信统一参数 |

**无需新增字段**。`PPPipelineSimulator` 将直接使用 `networkBandwidthGBps` + `networkLatencyUs` 创建 `SimCommGroup("pp")`，不再使用旧的 `ppSendRecvCostPerByteTicks`。

### 修改点清单

1. **新建** `server/src/sglang/parallel/pp.ts` — PPPipelineSimulator 完整实现（构造函数、三种调度 bubble 计算、通信成本计算含 TP 修正、micro-batch 分割、isPpLastStage）
2. **修改** `server/src/sglang/parallel/index.ts` — 增加 `PPPipelineSimulator`、`PipelineStepResult` 类型导出
3. **修改** `server/src/sglang/core/index.ts` — 新增 `ForwardOutput` 接口定义（含 `isIntermediate` 字段及行为合约注释）
4. **修改** `server/src/sglang/engine/index.ts` — MockEngine 集成 PPPipelineSimulator：PP 通信成本注入 + is_pp_last 采样控制 + PP 指标回填 + CUDA Graph 路径跳过 PP
5. **修改** `server/src/sglang/index.ts` — 增加导出 `PPPipelineSimulator`、`PipelineStepResult`、`ForwardOutput`
6. **新建** `server/src/test/sglang-pp.test.ts` — PP 单元测试（bubble 公式验证、退化测试、intermediate 采样控制测试、TP 修正测试）

## 测试设计

### 验收测试用例清单

| 编号 | 测试用例 | 预期结果 |
|------|---------|---------|
| T1 | gpipe bubble 公式验证 | `pp_size=4, num_micro_batches=4, eagerForward=10` → `bubble = 3 × 10 × 4 = 120` |
| T2 | gpipe bubble 随 pp_size 和 num_micro_batches 二次增长 | `pp_size=2,4,8 × num_micro_batches=4` → bubble 分别为 `1×10×4=40`, `3×10×4=120`, `7×10×4=280` |
| T3 | 1f1b bubble 最优 | `pp_size=4, eagerForward=10` → `bubble = 3 × 10 = 30`（远小于 gpipe 的 120） |
| T4 | interleaved bubble 介于中间 | `pp_size=4, num_chunks=2, eagerForward=10` → `bubble = 3 × 2 × 10 = 60` |
| T5 | pp_size=1 退化 | `simulatePipelineStep` 返回 `{totalTicks:0, bubbleTicks:0, sendRecvTicks:0, perStageTicks:[]}` |
| T6 | send/recv 通信成本 | `pp_size=2, num_micro_batches=2, tp_size=1` → 2 次 stage 间传输，每次 cost > 0 |
| T7 | isPpLastStage 验证 | `pp_size=4` → `isPpLastStage(3)=true`，`isPpLastStage(0/1/2)=false` |
| T8 | intermediate stage 不采样 | `isPpLast=false` → `ForwardOutput.isIntermediate=true`，`sampledIds=null`，sampler 未调用 |
| T9 | last stage 正常采样 | `isPpLast=true` → `ForwardOutput.isIntermediate=false`，`sampledIds` 非空，sampler 已调用 |
| T10 | sampling_counter 中间 stage 不变化 | intermediate stage 的 `samplingCounter` 保持不变 |
| T11 | ParallelMetrics PP 指标回填 | forward 后 `ppBubbleTicks > 0`、`ppSendRecvTicks > 0`、`ppNumMicroBatches > 0` |
| T12 | pp_size=1 时 ParallelMetrics 全零 | `ppBubbleTicks=0, ppSendRecvTicks=0, ppNumMicroBatches=0` |
| T13 | micro-batch 分割 | `batchSize=8, numMicroBatches=4` → 4 个 micro-batch，size 均为 2 |
| T14 | micro-batch 不整除（R1 契约） | `batchSize=7, numMicroBatches=3` → sizes=[3, 2, 2]（不 padding） |
| T15 | numMicroBatches=1 退化 | `_splitMicroBatches` 返回单个元素，等同于不切分 |
| T16 | TP × PP 通信数据量修正（R4） | `tp_size=2, hiddenSize=4096` → effectiveHiddenSize=2048，通信数据量减半 |
| T17 | TP × PP tp_size=1 退化 | `tp_size=1` → effectiveHiddenSize=hiddenSize，与无 TP 时一致 |
| T18 | CUDA Graph 路径跳过 PP（R3） | CUDA Graph replay 时 `ppStepResult=null`，PP 指标不累加 |
| T19 | pp_stage_layers 获取验证（R5） | `pp_size=4, numLayers=32` → stageLayers 长度 4，每 stage 8 层 |
| T20 | perStageTicks 计算 | `pp_size=3, numMicroBatches=2` → `perStageTicks` 长度 2，每个元素 > 0 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `pp_size=0` | 构造函数抛出 Error（无效配置，ParallelTopology 会拒绝） |
| B2 | `numMicroBatches=0` | `_splitMicroBatches` 返回空数组，`simulatePipelineStep` 中通信成本和 bubble 均为 0 |
| B3 | `batchSize=0` | micro-batch size 为 0，通信数据量为 0，sendRecvCost 为 0 |
| B4 | 未知 schedule 值 | `simulatePipelineStep` 抛出 `Error` |
| B5 | `commGroup=null`（pp_size=1） | `_stageSendRecvCost` 返回 0 |
| B6 | `networkBandwidthGBps=0` | `SimCommGroup.sendRecv` 返回 Infinity |

## 风险与注意事项

### 兼容性影响

- **零破坏性**：`pp_size=1` 时所有新增逻辑 noop，不影响现有单实例行为
- **ForwardOutput 新增接口**：`isIntermediate` 字段为新增接口定义，现有代码未使用 ForwardOutput，无破坏性
- **ParallelMetrics 字段已在 P0 预定义**：无需修改 metrics 结构

### 性能影响

- PPPipelineSimulator 是纯算术计算，无 IO/网络开销，性能影响可忽略
- 三种调度策略的计算复杂度均为 O(pp_size × numMicroBatches)，在仿真尺度下无性能风险

### TP × PP 互作风险（R4）

- 当 `tp_size > 1` 时，hidden_size 已按 TP 分割，实际传输数据量减少
- 本方案通过 `hiddenSize / tpSize` 修正数据量，确保通信成本计算准确
- 当 `tpSize` 不能整除 `hiddenSize` 时，使用 `Math.floor(hiddenSize / tpSize)` 向下取整（与 ParallelTopology 中 head 分配逻辑一致）

### 回滚方案

- 若需回滚，删除 `parallel/pp.ts`，还原 `parallel/index.ts`、`core/index.ts`、`engine/index.ts` 和 `index.ts` 的改动即可
- `ForwardOutput` 接口为新增定义，删除即可
- `ParallelMetrics` 中 PP 字段为 P0 预定义，不受影响
