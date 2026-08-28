---
title: "Issue #28 解决方案"
issue_number: 28
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: revised
review_round: 3
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
  - `server/src/sglang/index.ts` — 增加导出
  - `server/src/test/sglang-pp.test.ts` — PP 单元测试
- **依赖**：Issue #21（P0: ParallelTopology + SimCommGroup）已完成，`ppStageLayers` 和 `SimCommGroup("pp")` 可用

## 上一轮评审问题回应（Review Round 2 → Round 3）

| 编号 | 评审问题 | 回应与改进 |
|------|---------|-----------|
| R2-1 | **tick/单位与时序细节**：需要与技术报告对 tick 的定义逐字对齐 | **全面对齐技术报告 §4.2 + §10.5.2 的 tick 定义**：(1) `micro_batch_ticks` = `config.eagerForwardCostTicks`（§4.2 行1059，默认值 10，含义为「GPU ticks per eager forward」）；(2) 在 `PPPipelineSimulator` 构造函数中显式赋值 `this.microBatchTicks = config.eagerForwardCostTicks`，并在所有 bubble 公式中使用此字段而非直接引用 config；(3) 仿真内部约定 1 tick = 1 μs（与 `networkLatencyUs` 单位一致），`SimCommGroup.sendRecv` 返回值单位为 μs。详见下方 §Tick 定义对齐表。 |
| R2-2 | **通信/重叠模型**：是否允许 send/recv 与计算重叠（partial overlap）或必须串行计入 | **明确通信与计算的重叠策略**：技术报告 §4.2 行1101 已定义 `commOverlapWithCompute: bool = True`。本方案据此实现两种模式：(1) **重叠模式（默认）**：PP send/recv 与下一 stage 的计算可并行，`effectiveSendRecv = max(0, rawSendRecv - microBatchTicks)`；(2) **非重叠模式**：通信严格串行，`effectiveSendRecv = rawSendRecv`。由 `config.commOverlapWithCompute` 控制。重叠折算在 `PPPipelineSimulator` 层完成，不影响 `SimCommGroup` 的通用性。 |
| R2-3 | **CUDA Graph 路径处理**：需确认技术报告对此行为的明确准则 | **引用 §10.5.3 行4480-4482**：报告代码 `pp_comm = 0  # CUDA Graph 内不仿真 PP`。本方案在 `MockEngine.forwardBatch` 中实现：当 `canUseCudaGraph(batch)` 为 true 时，不调用 `ppSim.simulatePipelineStep()`，PP 指标不累加。实现注释中将引用 §10.5.3 行4480-4482。 |
| R2-4 | **interleaved 参数默认值**：方案提到 num_chunks（典型=2），应确认默认 | **在 `SimulatorConfig` 中新增 `ppInterleavedNumChunks: number` 字段**，默认值 `2`（与 Issue 规格一致）。当 `schedule !== "interleaved"` 时此字段被忽略。 |
| R2-5 | **TP × PP 数据量公式精确度**：需把公式与报告示例/推导一致 | **精确修正公式**：报告 §10.5.2 行4423 的 `_stage_send_recv_cost` 未包含 TP 分割修正。实际 SGLang 中 TP rank 各自持有 hidden_size/tp_size 分片。本方案修正为 `dataBytes = microBatchSize × Math.ceil(hiddenSize / tpSize) × dtypeSize`，使用 `Math.ceil` 确保不丢失数据。代码注释标注「参考 §10.5.2 行4423，增加 TP 分割修正」。当 `tpSize=1` 时公式退化为与报告一致。 |
| R2-6 | **微批次不整除场景的测试**：需覆盖极端 remainder | **扩展测试覆盖**：新增 (1) `batchSize=1, numMB=4` → sizes=[1,0,0,0]；(2) `batchSize=3, numMB=7` → sizes=[1,1,1,0,0,0,0]；(3) `batchSize=1000, numMB=999` → sizes=[2,1,1,...,1]，验证大量 micro-batch 通信成本累加精度。 |
| R2-7 | **指标粒度与可观测性**：per_stage_ticks 与 ParallelMetrics 字段精确映射 | **明确映射**：`ParallelMetrics.ppSendRecvTicks = sum(perStageTicks)`，`ppBubbleTicks = bubbleTicks`，`ppNumMicroBatches = numMicroBatches`。端到端测试中校验 `ppSendRecvTicks === perStageTicks.reduce((a,b) => a+b, 0)`。 |
| R2-8 | **边界行为测试**：量化断言 | **所有测试均使用量化断言**：例如 gpipe `pp=4, numMB=4, eagerForward=10` → 断言 `bubbleTicks === 120`（而非 `bubbleTicks > 0`）。详见测试设计。 |

## 改造方案

### 总体思路

按照总体设计文档 §10.5 的规格，在 `server/src/sglang/parallel/pp.ts` 中实现 `PPPipelineSimulator` 类，支持三种调度策略的 bubble 计算和 stage 间通信成本计算。然后在 `MockEngine.forwardBatch` 中集成 PP 通信成本注入和采样控制逻辑。

### Tick 定义对齐表（回应 R2-1）

| 变量名 | 含义 | 报告出处 | 单位 | 默认值 |
|--------|------|---------|------|--------|
| `microBatchTicks` | 单个 micro-batch 在单个 stage 的前向计算时间 | §4.2 行1059 `eager_forward_cost_ticks` | μs | 10 |
| `bubbleTicks` | 流水线气泡总时间 | §10.5.2 行4378 | μs | 0 |
| `commTicksTotal` | PP stage 间通信累计时间 | §10.5.2 行4379 | μs | 0 |
| `SimCommGroup.sendRecv()` | 单次点对点传输延迟 | §3.4.4 行853-856 | μs | 公式计算 |
| `perStageTicks[i]` | stage 边界 i→i+1 累计通信时间 | 本方案新增 | μs | — |
| `totalTicks` | bubbleTicks + sendRecvTicks | 本方案新增 | μs | — |

**换算关系**：仿真内部约定 1 tick = 1 μs。`config.eagerForwardCostTicks` 的语义为「单个 micro-batch 在单个 stage 的计算时间（μs）」，与 `SimCommGroup` 的 `networkLatencyUs` 和带宽换算（GB/s → B/μs）保持一致。

**关于技术报告 GPipe bubble 公式差异的说明**：报告 §10.5.2 行4435 的 GPipe 实现为 `bubble_ticks += (pp_size - 1) * eager_forward_cost_ticks`，缺少 `num_micro_batches` 乘数。但标准 GPipe 论文和 Issue 规格均为 `bubble = (p-1) × m × t`（其中 m=num_micro_batches, t=单 stage 时间）。本方案按 Issue 规格实现完整公式，报告代码为简化伪代码。同理，报告 interleaved 公式 `(pp_size-1)//2 * eagerForward` 与 Issue 规格 `(pp_size-1) × num_chunks × microBatch` 不同，本方案按 Issue 规格实现。

### 详细设计

#### 1. PPPipelineSimulator 类（`parallel/pp.ts`）

```typescript
export interface PipelineStepResult {
  totalTicks: number;       // bubbleTicks + sendRecvTicks
  bubbleTicks: number;      // 流水线气泡 ticks（μs）
  sendRecvTicks: number;    // stage 间 send/recv 总成本（μs）
  perStageTicks: number[];  // 每个 stage 边界的通信 ticks（长度 = pp_size - 1）
}

export class PPPipelineSimulator {
  readonly ppSize: number;
  readonly schedule: "1f1b" | "gpipe" | "interleaved";
  readonly numMicroBatches: number;
  readonly tpSize: number;
  readonly commGroup: SimCommGroup | null;
  readonly stageLayers: Array<{ start: number; end: number }>;
  readonly numChunks: number;                  // interleaved 专用，默认 2
  readonly microBatchTicks: number;            // R2-1: = config.eagerForwardCostTicks
  readonly commOverlapWithCompute: boolean;     // R2-2: 通信是否可与计算重叠
  readonly hiddenSize: number;
  readonly dtypeSize: number;

  bubbleTicks: number;
  commTicksTotal: number;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig) {
    this.ppSize = config.ppSize;
    this.schedule = config.ppPipelineSchedule;
    this.numMicroBatches = config.ppNumMicroBatches;
    this.tpSize = config.tpSize;
    this.numChunks = config.ppInterleavedNumChunks ?? 2; // R2-4
    this.microBatchTicks = config.eagerForwardCostTicks;  // R2-1
    this.commOverlapWithCompute = config.commOverlapWithCompute; // R2-2
    this.hiddenSize = modelConfig.hiddenSize;
    this.dtypeSize = config.dtypeSize;

    // 构造器内自动推导 stageLayers（Round 1 R5）
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
   * 将 batch 按 numMicroBatches 分割（R1 契约）。
   * 仅用于 PP 仿真层面，不影响 Scheduler。
   * 不整除时采用 ceil 分配，不 padding。
   */
  _splitMicroBatches(batchSize: number): Array<{ size: number }>;

  /**
   * 计算单个 stage 间 send/recv 通信成本。
   * R2-5: dataBytes = microBatchSize × ceil(hiddenSize / tpSize) × dtypeSize
   * 参考 §10.5.2 行4423，增加 TP 分割修正。
   */
  _stageSendRecvCost(microBatchSize: number): number;

  /** 是否最后一个 PP stage */
  isPpLastStage(stageIdx: number): boolean;
}
```

**关键设计点**：

1. **Bubble 公式**（严格遵循 Issue 规格）：

   - `microBatchTicks = config.eagerForwardCostTicks`（R2-1 对齐 §4.2 行1059）
   - gpipe：`bubble_ticks = (pp_size - 1) × microBatchTicks × numMicroBatches`
   - 1f1b：`bubble_ticks = (pp_size - 1) × microBatchTicks`
   - interleaved：`bubble_ticks = (pp_size - 1) × numChunks × microBatchTicks`（numChunks 默认 2，R2-4）

2. **通信成本计算**（R2-5 精确修正）：

   - `dataBytes = microBatchSize × Math.ceil(hiddenSize / tpSize) × dtypeSize`
   - 对每个 micro-batch，需要 (pp_size - 1) 次 stage 间传输
   - `perStageTicks[i]` = stage i → stage i+1 的累计通信成本

3. **通信与计算重叠**（R2-2 明确）：

   根据 `config.commOverlapWithCompute`（§4.2 行1101，默认 true）：
   - **重叠模式**（默认）：`effectiveSendRecv = max(0, rawSendRecv - microBatchTicks)`
   - **非重叠模式**：`effectiveSendRecv = rawSendRecv`
   - 重叠折算在 `PPPipelineSimulator` 层完成，不影响 `SimCommGroup` 通用性

4. **`_splitMicroBatches` 契约**（R1 明确，R2-6 扩展测试）：

   - 分割仅发生在 PP 仿真内部，Scheduler 层面不感知
   - 不整除时：前 `remainder` 个多 1，不 padding
   - `batchSize < numMicroBatches` 时：前 `batchSize` 个各 1，剩余为 0

5. **退化为 noop**：`pp_size=1` 时返回全零结果，`commGroup=null`

6. **`_stageSendRecvCost` 实现**（R2-5 修正后）：

   - `dataBytes = microBatchSize * Math.ceil(hiddenSize / this.tpSize) * dtypeSize`
   - `commGroup=null` 时返回 0
   - `tpSize=1` 时退化为 `microBatchSize × hiddenSize × dtypeSize`
   - `microBatchSize=0` 时返回 0

7. **`perStageTicks` 计算**（R2-7 明确语义）：

   长度 = `pp_size - 1`，三种调度策略通信模式相同：
   ```
   perStageTicks[i] = sum(mb in microBatches, effectiveSendRecv(mb.size))
   ```

#### 2. ForwardOutput 接口定义（`core/index.ts`）

```typescript
/**
 * Forward 输出（§10.5.3）
 *
 * 行为合约：
 * - isIntermediate=false（最后 PP stage 或 pp_size=1）：
 *   logits 非空，sampledIds 非空，sampler 被调用，samplingCounter 增加
 * - isIntermediate=true（中间 PP stage）：
 *   logits 非空（通信传给下一 stage），sampledIds=null，
 *   sampler 不被调用，samplingCounter 不增加
 */
export interface ForwardOutput {
  logits: number[] | null;      // 模型输出 logits
  sampledIds: number[] | null;  // 采样结果（中间 stage 为 null）
  isIntermediate: boolean;      // true=中间 PP stage，不采样
}
```

**行为合约完整说明**：

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
  }

  forwardBatch(batch: Batch, args: BatchSamplingArgs): ForwardOutput {
    // R2-3: CUDA Graph replay 路径跳过 PP 通信仿真
    // 引用 §10.5.3 行4480-4482
    let ppStepResult: PipelineStepResult | null = null;

    if (this.graphRunner.canUseCudaGraph(batch)) {
      logits = this.graphRunner.replay(batch);
      // CUDA Graph 内 PP 通信成本为 0
    } else {
      logits = this._mockModelForward(batch);
      ppStepResult = this.ppSim.simulatePipelineStep(batch);
    }

    // PP 指标回填（R2-7: 精确映射）
    if (ppStepResult && ppStepResult.totalTicks > 0) {
      this.metrics.parallel.ppSendRecvTicks += ppStepResult.sendRecvTicks;
      this.metrics.parallel.ppBubbleTicks += ppStepResult.bubbleTicks;
      this.metrics.parallel.ppNumMicroBatches += this.ppSim.numMicroBatches;
    }

    // 严格遵循行为合约
    if (!this.isPpLast) {
      return { logits, sampledIds: null, isIntermediate: true };
    }

    const nextTokenIds = this.sampler.sample(logits, args);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }
}
```

**关键约束**：
- `isIntermediate=true` 时**不调用** `sampler.sample()`，**不增加** `samplingCounter`
- `pp_size=1` 时 `isPpLast=true`，行为与单实例完全一致
- CUDA Graph replay 路径跳过 PP 通信仿真（R2-3：与 §10.5.3 行4480-4482 一致）

#### 4. SimulatorConfig 新增 PP 字段（回应 R2-4）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ppInterleavedNumChunks` | `number` | `2` | interleaved 调度的 chunk 数 |

现有 PP 字段无需变更（`ppSize`, `ppNumMicroBatches`, `ppSendRecvCostPerByteTicks`, `ppPipelineSchedule`, `networkBandwidthGBps`, `networkLatencyUs`, `commOverlapWithCompute` 均已存在）。

`PPPipelineSimulator` 将直接使用 `networkBandwidthGBps` + `networkLatencyUs` 创建 `SimCommGroup("pp")`，不再使用旧的 `ppSendRecvCostPerByteTicks`。

#### 5. ParallelMetrics 回填

PP 指标字段已在 P0（Issue #21）中预定义：
- `ppBubbleTicks` ← `PipelineStepResult.bubbleTicks`
- `ppSendRecvTicks` ← `PipelineStepResult.sendRecvTicks`（= `perStageTicks.reduce((a,b)=>a+b,0)`）
- `ppNumMicroBatches` ← `PPPipelineSimulator.numMicroBatches`

`pp_size=1` 时三个字段均保持默认值 0。

### 修改点清单

1. **新建** `server/src/sglang/parallel/pp.ts` — PPPipelineSimulator 完整实现
2. **修改** `server/src/sglang/parallel/index.ts` — 增加 `PPPipelineSimulator`、`PipelineStepResult` 导出
3. **修改** `server/src/sglang/core/index.ts` — 新增 `ForwardOutput` 接口
4. **修改** `server/src/sglang/engine/index.ts` — MockEngine 集成 PPPipelineSimulator
5. **修改** `server/src/sglang/types.ts` — SimulatorConfig 新增 `ppInterleavedNumChunks` 字段
6. **修改** `server/src/sglang/index.ts` — 增加导出
7. **新建** `server/src/test/sglang-pp.test.ts` — PP 单元测试

## 测试设计

### 验收测试用例清单

| 编号 | 测试用例 | 预期结果 |
|------|---------|---------|
| T1 | gpipe bubble 公式验证 | `pp=4, numMB=4, eagerForward=10` → `bubble === 120` |
| T2 | gpipe bubble 二次增长 | `pp=2,4,8 × numMB=4` → `bubble === 40, 120, 280` |
| T3 | 1f1b bubble 最优 | `pp=4, eagerForward=10` → `bubble === 30` |
| T4 | interleaved bubble | `pp=4, numChunks=2, eagerForward=10` → `bubble === 60` |
| T5 | pp_size=1 退化 | 返回 `{totalTicks:0, bubbleTicks:0, sendRecvTicks:0, perStageTicks:[]}` |
| T6 | send/recv 通信成本 | `pp=2, numMB=2, tp=1, batchSize=4` → `sendRecvTicks > 0, perStageTicks.length === 1` |
| T7 | isPpLastStage | `pp=4` → `isPpLastStage(3)=true, isPpLastStage(0/1/2)=false` |
| T8 | intermediate stage 不采样 | `isPpLast=false` → `isIntermediate=true, sampledIds=null` |
| T9 | last stage 正常采样 | `isPpLast=true` → `isIntermediate=false, sampledIds` 非空 |
| T10 | sampling_counter 中间不变 | 断言 `counter === 初始值` |
| T11 | ParallelMetrics 回填 | `ppBubbleTicks > 0, ppSendRecvTicks > 0, ppNumMicroBatches > 0`；`ppSendRecvTicks === perStageTicks.reduce((a,b)=>a+b,0)` |
| T12 | pp_size=1 时 Metrics 全零 | 断言 `===0` |
| T13 | micro-batch 分割 | `batchSize=8, numMB=4` → sizes=[2,2,2,2] |
| T14 | 不整除（R1 契约） | `batchSize=7, numMB=3` → sizes=[3,2,2] |
| T15 | numMicroBatches=1 | 返回单个元素 |
| T16 | TP×PP 修正（R2-5） | `tp=2, hiddenSize=4096` → `sendRecvCost(tp=2) === sendRecvCost(tp=1) / 2` |
| T17 | TP tp=1 退化 | 与无 TP 时一致 |
| T18 | CUDA Graph 跳过 PP（R2-3） | `ppBubbleTicks===0 && ppSendRecvTicks===0` |
| T19 | pp_stage_layers | `pp=4, numLayers=32` → 长度 4，每 stage 8 层 |
| T20 | perStageTicks 计算 | `pp=3, numMB=2` → `perStageTicks.length === 2`，每个 > 0 |
| T21 | 通信重叠模式（R2-2） | `overlap=true, sendRecv=8, microBatch=10` → `effectiveSendRecv === 0`；`overlap=false` → `===8` |
| T22 | 通信部分重叠（R2-2） | `overlap=true, sendRecv=15, microBatch=10` → `effectiveSendRecv === 5` |
| T23 | interleaved numChunks 可配（R2-4） | `pp=4, numChunks=3, eagerForward=10` → `bubble === 90` |
| T24 | 极端 micro-batch（R2-6） | `batchSize=1, numMB=4` → sizes=[1,0,0,0]，sendRecv 仅来自 size=1 |
| T25 | 大量 micro-batch（R2-6） | `batchSize=1000, numMB=999` → sizes=[2,1,1,...,1]，通信成本累加精确 |
| T26 | TP hiddenSize 不整除（R2-5） | `tp=3, hiddenSize=4096` → `ceil(4096/3) === 1366` |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `pp_size=0` | 抛出 Error |
| B2 | `numMicroBatches=0` | 返回全零 |
| B3 | `batchSize=0` | `sendRecvTicks=0` |
| B4 | 未知 schedule | 抛出 Error |
| B5 | `commGroup=null` | `_stageSendRecvCost` 返回 0 |
| B6 | `networkBandwidthGBps=0` | `sendRecv` 返回 Infinity |
| B7 | `ppInterleavedNumChunks=0` | interleaved bubble=0 |
| B8 | `ppInterleavedNumChunks=1` | 退化与 1f1b 相同 |

### 端到端集成验证

| 编号 | 验证场景 | 预期结果 |
|------|---------|---------|
| E2E-1 | gpipe 全流程 | `ppBubbleTicks === 计算值 && ppSendRecvTicks === perStageTicks求和` |
| E2E-2 | pp_size=1 全流程 | `isIntermediate=false && PP字段全零` |
| E2E-3 | CUDA Graph + PP | `ppStepResult=null && PP字段不变` |
| E2E-4 | 重叠模式对比 | `overlap=true 的 sendRecvTicks ≤ overlap=false` |

## 风险与注意事项

### 兼容性影响

- **零破坏性**：`pp_size=1` 时所有新增逻辑 noop
- **ForwardOutput 新增接口**：现有代码未使用 ForwardOutput，无破坏性
- **ParallelMetrics 字段已在 P0 预定义**：无需修改 metrics 结构
- **SimulatorConfig 新增字段**：`ppInterleavedNumChunks` 有默认值 2，不影响现有配置

### 性能影响

- PPPipelineSimulator 是纯算术计算，性能影响可忽略
- 三种调度策略计算复杂度均为 O(pp_size × numMicroBatches)
- 通信重叠折算仅增加一次 `max(0, x - y)` 运算

### TP × PP 互作风险（R2-5）

- 使用 `Math.ceil(hiddenSize / tpSize)` 修正数据量
- 使用 `Math.ceil` 而非 `Math.floor`：保证不丢失数据

### 技术报告差异风险

- GPipe 和 interleaved bubble 公式与报告 §10.5.2 简化伪代码存在差异，但与 Issue 规格及标准 PP 论文一致
- 实现阶段将在代码注释中明确标注公式来源和差异说明

### 回滚方案

- 删除 `parallel/pp.ts`，还原 `parallel/index.ts`、`core/index.ts`、`engine/index.ts`、`types.ts` 和 `index.ts` 的改动
- `ForwardOutput` 接口为新增定义，删除即可
- `ParallelMetrics` 中 PP 字段为 P0 预定义，不受影响
