---
title: "Issue #17 解决方案"
issue_number: 17
issue_type: Feature
created: 2026-08-31
updated: 2026-08-31
status: revised
review_round: 2
---

# Issue #17 解决方案

## 需求分析

- **问题描述**：Issue #17 要求实现 SGLang 仿真器 S3 阶段的核心调度循环组件：`MockEngine.forward_batch`（含 prefill/decode 时间模拟、chunked prefill 标识、CUDA Graph 标识、PP last 标识）、`MockSampler`（含 `prepare`/`sample` 方法及采样参数处理）、`MockAttnBackend`（含 `prepare_metadata`/`simulate_kv_recycle` stub）、`SimScheduler.normal_tick`（含完整的调度循环：消息处理 → prefill/decode 调度 → forward → 结果处理）、`SchedulerIOMixin`（含 `step()`/`_process_messages`/`_send_resp` 及消息队列管理）。Phase 2 目标：单实例可跑通一个短 prompt end-to-end。

- **能力目标**：
  1. **MockEngine.forward_batch**（§4.3/§9.11）：返回 `ForwardOutput`，包含 `nextTokensGpu`/`nextTokensCpu`/`copyDoneEvent`（对齐 §9.11 L3690-3694），支持时间模拟和标识字段；对 ChunkedReq 跳过 `completeOne()`；调用 MockSampler 采样
  2. **MockSampler**（§4.3/§9.11）：`prepare(batch)` 生成 `BatchSamplingArgs`；`sample(logits, args)` 按 greedy/random/fixed 模式采样；实现 `apply_temperature`/`apply_top_p_top_k`/`apply_logits_penalty` 采样管线
  3. **MockAttnBackend**（§4.3）：`prepare_metadata(batch)` 为桩实现；`simulate_kv_recycle` 为 stub 返回 0
  4. **SimScheduler.normal_tick**（§9.2/§9.10）：串行执行消息处理 → prefill 调度 → decode 调度 → forward → 结果处理；`_scheduleNextBatch` prefill 优先、否则 decode
  5. **SchedulerIOMixin**（§4.3/§9.6）：`step()` 推进 req_in/resp_token；`self.messages = deque[msg]`；`_process_messages`/`_send_resp`；支持 offline 模式
  6. Phase 2 验收：单实例可跑通一个短 prompt end-to-end；TS strict zero-any

- **影响范围**：仅修改 `server/src/sglang/` 目录下的 `engine/index.ts`（重构 MockEngine.forward_batch + 新增 MockSampler/MockAttnBackend/MockEvent）、`scheduler/index.ts`（新增 SimScheduler + SchedulerIOMixin）、`types.ts`（新增消息与接口类型）、`core/index.ts`（扩展 ForwardOutput/BatchSamplingArgs/Batch）、`index.ts`（新增 re-export），以及 `server/src/test/` 目录（新增 `sglang-s3.test.ts`）。不修改已有测试代码。

- **依赖 Issue**：
  - #16 S2: PrefillManager + PrefillAdder + DecodeManager（已完成）
  - #15 K4: RadixPrefixCache（已完成）

- **阻塞 Issue**：
  - S4: SimGraphRunner（bs 分桶 CUDA 图）
  - S5: Overlap Scheduling（last_data 延迟空 tick 刷新）

## 上一轮评审问题回应（Review Round 1 → 2）

### 1. _process_one_msg 消息类型简化问题

**评审意见**：方案仅处理 SimRequestMsg + abort，缺少 ExitMsg 和 BatchBackendMsg，需确认是否在 S3 范围内。

**修订**：`_processOneMsg` 需完整对齐 §9.11 L3027-3052 的四种消息类型：
- `BatchBackendMsg` → 递归展开调用 `_processOneMsg`
- `ExitMsg` → 抛出错误（仿真中等效于 KeyboardInterrupt，终止调度循环）
- `UserMsg`（对应 `SimRequestMsg` tag=`req_in`）→ maxNewTokens 调整后加入 prefillManager
- `AbortBackendMsg`（对应 abort 语义）→ 从 prefillManager/decodeManager 中 abort 并释放资源

S3 新增 `SchedulerMsg` 联合类型覆盖以上四种，`_processOneMsg` 按类型分发。

### 2. copy_done_event 缺失问题

**评审意见**：ForwardOutput 缺少 `copy_done_event` 字段，`_process_last_data` 中需要 `copy_done.synchronize()` 调用。

**修订**：新增 `MockEvent` 类（对齐 §9.11 L3693），`ForwardOutput` 新增 `copyDoneEvent: MockEvent` 字段。`_processLastData` 开头调用 `ongoingData[1].copyDoneEvent.synchronize()` 后再处理结果。

### 3. finished_reqs 集合更新逻辑

**评审意见**：方案提到 `finishedReqs` 但未详述更新逻辑。

**修订**：对齐 §9.11 L2959-2978，`_processLastData` 内部维护 `newFinishedReqs: Set<Req>` 收集本轮完成的请求，遍历结束后 `this.finishedReqs = newFinishedReqs`。S3 仅 `normal_tick` 模式下 `finishedReqs` 每轮更新，防止重复释放。

### 4. is_chunked 判断语义问题

**评审意见**：方案使用 `batch.hasIdleReqs` 判断 chunked prefill，但报告中 chunked 标识来自 batch 中是否存在 ChunkedReq 实例。

**修订**：`isChunkPrefill` 改为检查 batch 中是否存在 ChunkedReq 实例：`batch.reqs 中存在 instanceof ChunkedReq`。这与 §9.11 L3686 中 `isinstance(req, ChunkedReq)` 的判断语义一致。`batch.hasIdleReqs` 字段保留用于其他用途，不再用于 chunked prefill 判断。

### 5. dummy_req 填充问题

**评审意见**：方案 padBatch 用 `null` 填充，但报告使用 `dummy_req`（tableIdx=maxRunningReq）。

**修订**：对齐 §9.11 L3662-3674，MockEngine 构造时创建 `dummyReq`：`tableIdx = maxRunningReq`，`deviceLen = 1`，`maxDeviceLen = 1`。`pageTable[maxRunningReq]` 填充为 `numTokens`（标记所有页已使用）。`GraphRunner.padBatch` 使用 `dummyReq` 填充而非 `null`。

## 改造方案

### 总体思路

按照 §9.11 完整实现代码集，将 S3 阶段的五个组件分为三个层次实现：

1. **底层组件**（`engine/index.ts`）：新增 `MockSampler`（替换现有 Sampler 的功能扩展）、`MockAttnBackend`（新增）、`MockEvent`（新增）；`MockEngine` 新增 `forward_batch` 方法和 `dummyReq`
2. **中层调度器**（`scheduler/index.ts`）：新增 `SimScheduler` 类（含 `normal_tick` 完整循环）和 `SchedulerIOMixin` 类
3. **类型层**（`types.ts`/`core/index.ts`）：扩展消息类型、`ForwardOutput`、`BatchSamplingArgs`、`Batch` 等

核心设计决策：

1. **MockSampler 与现有 Sampler 并存**：当前 `Sampler` 是 P3a/P4 的简化桩，仅支持 `sample(logits, batchSize)`。S3 新增 `MockSampler` class，保留原 `Sampler` 不做改动（P3a/P4/P5 测试依赖），`MockEngine` 内部新增 `mockSampler` 属性，`forward_batch` 使用 `mockSampler`。
2. **MockAttnBackend 独立 class**：`prepare_metadata(batch)` 为空操作桩，`simulate_kv_recycle` 返回 0 ticks。
3. **SimScheduler 继承 SchedulerIOMixin**：`SchedulerIOMixin` 负责消息 I/O，`SimScheduler` 继承并实现调度逻辑。`offlineMode=true` 时 receive/send 为 noop。
4. **normal_tick 循环严格对齐 §9.11**：`_processOneMsg` → `_scheduleNextBatch` → `_forward` → `_processLastData`。
5. **ForwardOutput 对齐 §9.11**：包含 `nextTokensGpu`/`nextTokensCpu`/`copyDoneEvent`（L3690-3694），扩展时间模型和标识字段。
6. **Batch 扩展**：新增 `paddedReqs`/`inputIds`/`positions`/`outLoc`/`attnMetadata` 属性。
7. **dummyReq 对齐 §9.11**：MockEngine 构造时创建 dummyReq 用于 CUDA Graph padding（L3662-3674）。
8. **_processOneMsg 完整消息类型**：对齐 §9.11 L3027-3052 四种消息类型。
9. **copyDoneEvent 机制**：对齐 §9.11 L3693/L2956，`_processLastData` 开头调用 synchronize。

### 详细设计

#### 1. MockEvent 类

对齐 §9.11 L3693，仿真中 `copy_done_event` 的桩实现。

```typescript
/** 仿真事件同步桩（对齐 §9.11 copy_done_event） */
export class MockEvent {
  private _synchronized: boolean = false;

  /** 标记事件已完成（仿真中立即完成） */
  record(): void {
    this._synchronized = true;
  }

  /** 同步等待事件完成（仿真中为 noop，立即返回） */
  synchronize(): void {
    this._synchronized = true;
  }
}
```

#### 2. BatchSamplingArgs 类

```typescript
export class BatchSamplingArgs {
  readonly temperatures: number[] | null;
  readonly topK: number[];
  readonly topP: number[];

  constructor(opts: { temperatures: number[] | null; topK?: number[]; topP?: number[] }) {
    this.temperatures = opts.temperatures;
    this.topK = opts.topK ?? [];
    this.topP = opts.topP ?? [];
  }

  get isGreedy(): boolean {
    return this.temperatures === null;
  }
}
```

#### 3. MockSampler 类

核心职责：模拟采样器，支持 greedy/random/fixed 三种模式，实现采样参数处理管线。对齐 §9.11 L3558-3585。

**构造函数**：`vocabSize: number`, `mode: "random" | "greedy" | "fixed"`, `fixedToken: number`

**prepare(batch: Batch): BatchSamplingArgs**（对齐 §9.11 L3568-3577）：
- 遍历 batch 中所有 req 的 `samplingParams`
- 若所有 req 均为 greedy → `temperatures: null`
- 否则 → `temperatures/topK/topP` 数组

**sample(logits: number[][], args: BatchSamplingArgs): number[]**（对齐 §9.11 L3579-3584）：
- greedy/fixed/random 三种模式

**采样参数处理管线**（方法级，供 sample 内部调用）：
- `apply_temperature(logits, temperature)` — logits /= temperature
- `apply_top_p_top_k(logits, top_p, top_k)` — top-k 过滤 + top-p 核化
- `apply_logits_penalty(logits, token_ids, penalty)` — 重复惩罚
- `apply_logits_prob(logits)` — stub
- `apply_logits_bias(logits, bias)` — stub

#### 4. MockAttnBackend 类

对齐 §9.11 L876-883。

- `prepare_metadata(batch: Batch): void` — 空操作，设置 `batch.attnMetadata = {}`
- `simulate_kv_recycle(): number` — 返回 0（stub）

#### 5. MockEngine.forward_batch 新增方法

对齐 §9.11 L3676-3694。

```typescript
forward_batch(batch: Batch, sampleArgs: BatchSamplingArgs): ForwardOutput {
  // 1. CUDA Graph 判断
  const isGraphCapture = this.graphRunner.canUseCudaGraph(batch);

  // 2. mock forward
  let logits: number[][];
  if (isGraphCapture) {
    logits = this.graphRunner.replay(batch);
  } else {
    logits = this._mockModelForward(batch);
  }

  // 3. complete_one（跳过 ChunkedReq），对齐 §9.11 L3684-3686
  for (const req of batch.reqs.values()) {
    if (!(req instanceof ChunkedReq)) {
      req.completeOne();
    }
  }

  // 4. 采样，对齐 §9.11 L3689
  const nextTokens = this.mockSampler.sample(logits, sampleArgs);

  // 5. 时间模型
  const isChunkPrefill = [...batch.reqs.values()].some(r => r instanceof ChunkedReq);
  const prefillBatchTime = batch.extendInputTokens > 0 ? this._computePrefillTime(batch) : 0;
  const decodeBatchTime = batch.numDecodeTokens > 0 ? this._computeDecodeTime(batch) : 0;

  // 6. 构造 ForwardOutput，对齐 §9.11 L3690-3694
  const copyDoneEvent = new MockEvent();
  copyDoneEvent.record();

  return {
    nextTokensGpu: nextTokens,
    nextTokensCpu: [...nextTokens],
    copyDoneEvent,
    isIntermediate: false,
    logits,
    prefillBatchTime,
    decodeBatchTime,
    isChunkPrefill,
    isGraphCapture,
    isPpLast: true,
  };
}
```

**时间模型**（简化版，对齐 §4.3 L1272-1281）：
- `_computePrefillTime(batch)` — 基于 `batch.extendInputTokens` 和 `config.eagerForwardCostTicks` 计算
- `_computeDecodeTime(batch)` — 基于 `batch.numDecodeTokens` 和 `config.graphReplayCostTicks`/`config.eagerForwardCostTicks` 计算

**MockEngine 新增属性**：
- `mockSampler: MockSampler`
- `mockAttnBackend: MockAttnBackend`
- `dummyReq: Req`（对齐 §9.11 L3662-3671）
- `pageTable: number[][]`（已存在，需确保初始化）
- `numPages: number`
- `maxSeqLen: number`

**dummyReq 创建**（对齐 §9.11 L3662-3674）：
```typescript
// dummyReq 用于 CUDA Graph padding，tableIdx = maxRunningReq
this.dummyReq = new Req({ rid: -1, inputIds: [0], samplingParams: new SamplingParams() });
this.dummyReq.deviceLen = 1;
this.dummyReq.maxDeviceLen = 1;
// pageTable 最后一行预留给 dummyReq，填充 numTokens 标记所有页已使用
const numTokens = this.numPages * config.pageSize;
this.pageTable[config.maxRunningReq] = new Array(this.maxSeqLen).fill(numTokens);
```

#### 6. SchedulerIOMixin 类

对齐 §9.6 L2193-2241。

```typescript
class SchedulerIOMixin {
  protected readonly config: SimulatorConfig;
  protected readonly _incomingQueue: SchedulerMsg[];
  protected readonly _outgoingQueue: SimRespMsg[];

  constructor(config: SimulatorConfig) {
    this.config = config;
    this._incomingQueue = [];
    this._outgoingQueue = [];
  }

  /** 接收消息；offline 模式下返回空数组 */
  receiveMsg(): SchedulerMsg[] {
    if (this.config.offlineMode) return [];
    const msgs = [...this._incomingQueue];
    this._incomingQueue.length = 0;
    return msgs;
  }

  /** 发送结果；offline 模式下为 noop */
  sendResult(reply: SimRespMsg[]): void {
    if (this.config.offlineMode) return;
    this._outgoingQueue.push(...reply);
  }

  /** 同步所有 rank；tp_size=1 时为 noop */
  syncAllRanks(): void {
    // 仿真中单 rank 为 noop
  }

  /** 推进一步消息处理（非 offline 模式下由外部驱动） */
  step(): void {
    // offline 模式下消息通过 runTick 参数传入，不需要主动拉取
  }
}
```

#### 7. SimScheduler 类

对齐 §9.11 L2888-3058。

```typescript
class SimScheduler extends SchedulerIOMixin {
  readonly engine: MockEngine;
  readonly tableManager: TableManager;
  readonly cacheManager: CacheManager;
  readonly decodeManager: DecodeManager;
  readonly prefillManager: PrefillManager;
  finishedReqs: Set<Req>;
  readonly eosTokenId: number;
  readonly tokenPool: number[][];
  readonly prefillBudget: number;
  readonly overlapEnabled: boolean;
  lastBatch: Batch | null;

  constructor(config: SimulatorConfig) {
    super(config);
    this.engine = new MockEngine(config);
    this.tableManager = new TableManager(config.maxRunningReq, this.engine.pageTable);
    this.cacheManager = new CacheManager(
      this.engine.numPages, config.pageSize, this.engine.pageTable, config.cacheType
    );
    this.decodeManager = new DecodeManager(config.pageSize);
    this.prefillManager = new PrefillManager(
      this.cacheManager, this.tableManager, this.decodeManager
    );
    this.finishedReqs = new Set();
    this.eosTokenId = config.eosTokenId;
    this.tokenPool = this.tableManager.tokenPool;
    this.prefillBudget = config.maxExtendTokens;
    this.overlapEnabled = config.enableOverlap;
    this.lastBatch = null;
  }

  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // S3: 仅实现 normal_tick；overlap 为 S5 范围
    // 当 overlapEnabled=true 时降级为 normal_tick，确保 end-to-end 可跑通
    return this._normalTick(incoming);
  }
}
```

**_normalTick 流程**（对齐 §9.11 L2916-2924）：
```
1. for msg in incoming: _processOneMsg(msg)
2. forwardInput = _scheduleNextBatch()
3. ongoingData = null
4. if forwardInput !== null: ongoingData = [forwardInput, _forward(forwardInput)]
5. return _processLastData(ongoingData)
```

**_processOneMsg(msg: SchedulerMsg)**（对齐 §9.11 L3027-3052，完整四种消息类型）：
- `BatchSchedulerMsg` → 递归展开，对每个子消息调用 `_processOneMsg`
- `ExitMsg` → 抛出 `Error("ExitSignal")`（仿真中终止调度循环）
- `UserMsg`（即 `SimRequestMsg` tag=`req_in`）→
  1. 计算 `maxOutputLen = engine.maxSeqLen - inputLen`
  2. 若 `maxOutputLen <= 0` → 跳过
  3. 若 `samplingParams.maxNewTokens > maxOutputLen` → 截断
  4. 若截断后 `maxNewTokens <= 0` → 跳过
  5. 调用 `prefillManager.addOneReq(msg)`
- `AbortMsg`（即 abort 语义）→
  1. 调用 `prefillManager.abortReq(uid)` 尝试从 pending 移除
  2. 若返回 ChunkedReq → 需释放资源
  3. 否则调用 `decodeManager.abortReq(uid)` 尝试从 running 移除
  4. 若找到 req → 调用 `_freeReqResources(req)`

**_scheduleNextBatch()**（对齐 §9.11 L2984-2992）：
- batch = prefillManager.scheduleNextBatch(prefillBudget) || decodeManager.scheduleNextBatch()
- 若 batch !== null → `_prepareBatch(batch)`，保存 `this.lastBatch = batch`
- 否则 → null

**_prepareBatch(batch: Batch)**（对齐 §9.11 L2994-3010）：
1. `engine.graphRunner.padBatch(batch)`
2. 对 batch 中每个 req 调用 `cacheManager.allocatePaged(req)`
3. 计算 positions（`_makePositions`）
4. 计算 input_mapping（`_makeInputTuple`）
5. 计算 write_mapping（`_makeWriteTuple`）
6. 计算 `batch.outLoc`
7. `engine.mockAttnBackend.prepareMetadata(batch)`
8. 构造 `ForwardInput` 返回

**_forward(forwardInput: ForwardInput)**（对齐 §9.11 L3012-3025）：
1. 从 tokenPool 读取 input_ids 填入 `batch.inputIds`
2. 调用 `engine.forward_batch(batch, sampleArgs)`
3. 将 next_tokens 写入 tokenPool（跳过 ChunkedReq 位置：`write_p < 0` 时跳过，对齐 §9.11 L3021-3023）
4. 调用 `decodeManager.filterReqs(batch.reqs 的数组形式)`
5. 返回 ForwardOutput

**_processLastData(ongoingData)**（对齐 §9.11 L2950-2982）：
1. `ongoingData === null` → return `[]`
2. 取 `batch = ongoingData[0].batch`，`forwardOutput = ongoingData[1]`
3. **调用 `forwardOutput.copyDoneEvent.synchronize()`**（对齐 §9.11 L2956）
4. 在 `cacheManager.beginLazyFree()` / `endLazyFree()` 上下文内遍历 batch.reqs：
   - ChunkedReq → 跳过（对齐 §9.11 L2962-2963）
   - `nextToken = forwardOutput.nextTokensCpu[i]`
   - `req.appendHost(nextToken)`
   - 判断 finished：`!req.canDecode` 或（`!samplingParams.ignoreEos && nextToken === eosTokenId`）
   - 构造 `SimRespMsg`
   - finished 且 `req not in this.finishedReqs` → `decodeManager.removeReq(req)` + `_freeReqResources(req)` + 加入 `newFinishedReqs`
   - 非 finished 且 batch 为 prefill（`batch.extendInputTokens > 0`）→ `cacheManager.cacheReq(req, false)`
5. **`this.finishedReqs = newFinishedReqs`**（对齐 §9.11 L2978）
6. 返回 reply

**_freeReqResources(req: Req)**（对齐 §9.11 L3054-3057）：
- `tableManager.free(req.tableIdx)`
- `cacheManager.freeCache(req)`

#### 8. 消息类型扩展

对齐 §9.11 L3027-3052，新增消息联合类型：

```typescript
/** 调度器消息联合类型（对齐 §9.11 四种消息） */
export type SchedulerMsg =
  | BatchSchedulerMsg
  | ExitMsg
  | UserMsg
  | AbortMsg;

/** 批量消息容器 */
export interface BatchSchedulerMsg {
  tag: "batch";
  data: SchedulerMsg[];
}

/** 退出信号 */
export interface ExitMsg {
  tag: "exit";
}

/** 用户请求消息（映射到 SimRequestMsg） */
export interface UserMsg {
  tag: "req_in";
  uid: number;
  inputIds: number[];
  samplingParams: SamplingParamsClass | null;
  outputLen: number;
}

/** 中止请求消息 */
export interface AbortMsg {
  tag: "abort";
  uid: number;
}
```

#### 9. Batch 扩展属性

新增属性（均在 Batch 构造中初始化为默认值）：
- `paddedReqs: (Req | null)[]` — CUDA Graph padding 后的请求列表（含 dummyReq）
- `inputIds: number[]` — 由 scheduler 填充
- `positions: number[]` — 由 scheduler 填充
- `outLoc: number[]` — KV cache 写入位置
- `attnMetadata: unknown` — 由 attention backend 填充

#### 10. ForwardOutput 扩展

对齐 §9.11 L3690-3694，扩展 `ForwardOutput` 接口：

```typescript
export interface ForwardOutput {
  logits: number[][] | null;        // 模型输出 logits（2D: [batchSize, vocabSize]）
  nextTokensGpu: number[] | null;   // GPU 端采样结果
  nextTokensCpu: number[] | null;   // CPU 端采样结果副本
  copyDoneEvent: MockEvent;         // 拷贝完成事件（对齐 §9.11 L3693）
  isIntermediate: boolean;          // true=中间 PP stage，不采样
  prefillBatchTime?: number;        // prefill 时间模拟 ticks
  decodeBatchTime?: number;         // decode 时间模拟 ticks
  isChunkPrefill?: boolean;         // batch 是否包含 ChunkedReq
  isGraphCapture?: boolean;         // 是否使用 CUDA Graph
  isPpLast?: boolean;               // 是否为 PP 最后 stage
  sampledIds?: number[] | null;     // 兼容 P4: 采样结果
}
```

#### 11. ForwardInput 接口

```typescript
export interface ForwardInput {
  batch: Batch;
  sampleArgs: BatchSamplingArgs;
  inputTuple: [number[], number[]];  // [tableIdx[], position[]]
  writeTuple: [number[], number[]];  // [tableIdx[], position[]]
}
```

#### 12. GraphRunner.padBatch 新增方法

对齐 §9.11 L3609-3614，使用 dummyReq 填充。

```typescript
padBatch(batch: Batch): void {
  let paddedSize: number;
  if (this.canUseCudaGraph(batch)) {
    // 从 graphBsList 中找第一个 >= batch.size 的值
    paddedSize = this.cudaGraphBs?.find(bs => bs >= batch.reqs.size) ?? batch.reqs.size;
  } else {
    paddedSize = batch.reqs.size;
  }
  const dummyCount = paddedSize - batch.reqs.size;
  // 使用 dummyReq 填充，对齐 §9.11 L3614
  batch.paddedReqs = [...batch.reqs.values(), ...Array(dummyCount).fill(/* dummyReq 由外部注入 */)];
}
```

注意：`dummyReq` 由 `MockEngine` 创建，通过 `GraphRunner` 构造参数注入（对齐 §9.11 L3674 `SimGraphRunner(config, model_config, dummy_req)`）。

### 接口变更

1. **`core/index.ts`**：新增 `BatchSamplingArgs` class、`MockEvent` class；`ForwardOutput` 接口扩展为包含 `nextTokensGpu`/`nextTokensCpu`/`copyDoneEvent` 及可选时间/标识字段；`Batch` class 新增 5 个属性
2. **`engine/index.ts`**：新增 `MockSampler` class、`MockAttnBackend` class；`MockEngine` 新增属性和方法；`GraphRunner` 新增 `padBatch` 方法并接受 `dummyReq` 参数
3. **`scheduler/index.ts`**：新增 `SchedulerIOMixin` class、`SimScheduler` class
4. **`types.ts`**：新增 `ForwardInput` interface、`SchedulerMsg`/`BatchSchedulerMsg`/`ExitMsg`/`UserMsg`/`AbortMsg` 消息类型；`SimScheduler` 从 interface 改为指向 class
5. **`index.ts`**：新增所有新 class/interface 的 re-export

### 数据结构改动

1. **新增 `MockEvent`** — 仿真事件同步桩
2. **新增 `BatchSamplingArgs`** — 采样参数批处理封装
3. **新增 `ForwardInput`** — forward 输入元数据
4. **扩展 `ForwardOutput`** — 新增 `nextTokensGpu`/`nextTokensCpu`/`copyDoneEvent` 及时间/标识字段
5. **扩展 `Batch`** — 新增 paddedReqs/inputIds/positions/outLoc/attnMetadata
6. **新增消息类型** — `SchedulerMsg`/`BatchSchedulerMsg`/`ExitMsg`/`UserMsg`/`AbortMsg`
7. **新增 `SimScheduler` 内部状态** — finishedReqs/lastBatch/tokenPool/dummyReq 等

### 修改点清单

1. **修改 `server/src/sglang/core/index.ts`**：新增 `BatchSamplingArgs` class；新增 `MockEvent` class；`ForwardOutput` 接口新增字段；`Batch` class 新增 5 个属性
2. **修改 `server/src/sglang/engine/index.ts`**：新增 `MockSampler` class（含 prepare/sample/apply_* 方法）；新增 `MockAttnBackend` class；`MockEngine` 新增 mockSampler/mockAttnBackend/dummyReq/pageTable/numPages/maxSeqLen 属性；`MockEngine` 新增 `forward_batch` 方法；`MockEngine` 新增 `_computePrefillTime`/`_computeDecodeTime` 方法；`GraphRunner` 构造函数新增 `dummyReq` 参数；`GraphRunner` 新增 `padBatch` 方法
3. **修改 `server/src/sglang/scheduler/index.ts`**：新增 `SchedulerIOMixin` class；新增 `SimScheduler` class
4. **修改 `server/src/sglang/types.ts`**：新增 `ForwardInput` interface；新增 `SchedulerMsg`/`BatchSchedulerMsg`/`ExitMsg`/`UserMsg`/`AbortMsg` 消息类型；更新 `SimScheduler` 引用
5. **修改 `server/src/sglang/index.ts`**：新增所有新增导出
6. **新建 `server/src/test/sglang-s3.test.ts`**：S3 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | MockEvent 构造与 synchronize | 创建 MockEvent，调用 record/synchronize 不抛异常 |
| T2 | MockSampler 构造 | vocabSize/mode/fixedToken 正确初始化 |
| T3 | MockSampler.prepare - greedy batch | 所有 req.isGreedy=true → temperatures=null |
| T4 | MockSampler.prepare - mixed batch | 存在非 greedy req → temperatures 数组非 null |
| T5 | MockSampler.sample - greedy 模式 | argmax 返回最大 logit 位置 |
| T6 | MockSampler.sample - random 模式 | 返回 [0, vocabSize) 范围随机整数 |
| T7 | MockSampler.sample - fixed 模式 | 返回固定 token |
| T8 | MockSampler.apply_temperature | logits /= temperature 正确执行 |
| T9 | MockSampler.apply_top_p_top_k | top-k 过滤 + top-p 核化正确执行 |
| T10 | MockSampler.apply_logits_penalty | 重复惩罚正确应用 |
| T11 | MockAttnBackend.prepare_metadata | 调用不抛异常，设置 attnMetadata |
| T12 | MockAttnBackend.simulate_kv_recycle | 返回 0 |
| T13 | MockEngine.forward_batch - prefill batch | 返回 ForwardOutput 含 prefillBatchTime > 0 |
| T14 | MockEngine.forward_batch - decode batch | 返回 ForwardOutput 含 decodeBatchTime > 0 |
| T15 | MockEngine.forward_batch - ChunkedReq 跳过 | ChunkedReq 的 completeOne 不被调用 |
| T16 | MockEngine.forward_batch - CUDA Graph | isGraphCapture 标识正确 |
| T17 | MockEngine.forward_batch - copyDoneEvent | copyDoneEvent 存在且 synchronize 不抛异常 |
| T18 | MockEngine.forward_batch - isChunkPrefill | batch 含 ChunkedReq 时 isChunkPrefill=true |
| T19 | SchedulerIOMixin - offline 模式 | receiveMsg 返回 []，sendResult 为 noop |
| T20 | SchedulerIOMixin - online 模式 | 内部队列可收发 |
| T21 | SimScheduler 构造 | 所有子组件正确初始化，dummyReq 存在 |
| T22 | SimScheduler._normalTick - 空 tick | 无 incoming、无可调度请求 → 返回 [] |
| T23 | SimScheduler end-to-end - 短 prompt | 一个短 prompt 经 prefill tick → decode tick → 完成 |
| T24 | SimScheduler._processOneMsg - req_in | 请求正确加入 prefillManager |
| T25 | SimScheduler._processOneMsg - maxTokens 调整 | inputLen 接近 maxSeqLen 时 maxNewTokens 被截断 |
| T26 | SimScheduler._processOneMsg - ExitMsg | 抛出 Error("ExitSignal") |
| T27 | SimScheduler._processOneMsg - BatchMsg | 递归展开子消息 |
| T28 | SimScheduler._processOneMsg - AbortMsg | 请求从 prefill/decode 中移除并释放资源 |
| T29 | SimScheduler._scheduleNextBatch - prefill 优先 | pending 和 decode 都有请求时优先返回 prefill batch |
| T30 | SimScheduler._scheduleNextBatch - 仅 decode | 无 pending 请求时返回 decode batch |
| T31 | SimScheduler._processLastData - copyDoneEvent | 调用 synchronize 后再处理结果 |
| T32 | SimScheduler._processLastData - prefill 完成 | 非 finished 的 prefill 请求调用 cacheReq(finished=false) |
| T33 | SimScheduler._processLastData - 请求完成 | finished 时调用 removeReq + _freeReqResources |
| T34 | SimScheduler._processLastData - EOS 终止 | nextToken === eosTokenId 时标记 finished |
| T35 | SimScheduler._processLastData - ChunkedReq 跳过 | ChunkedReq 不生成 resp_token |
| T36 | SimScheduler._processLastData - finishedReqs 更新 | newFinishedReqs 正确收集，finishedReqs 赋值更新 |
| T37 | SimScheduler end-to-end - 完整流程 | 单请求经历 prefill + 多个 decode tick 完成 |
| T38 | SimScheduler._freeReqResources | tableIdx 被 free，freeCache 被调用 |
| T39 | GraphRunner.padBatch - 使用 dummyReq | paddedReqs 中填充 dummyReq 而非 null |
| T40 | MockEngine dummyReq 初始化 | dummyReq.tableIdx = maxRunningReq，pageTable 最后一行正确填充 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | 空 incoming + 空 pending + 空 decode | normalTick 返回 [] |
| B2 | maxNewTokens 被截断为 0 | 请求被跳过 |
| B3 | 单 token 输入请求 | 正常 prefill + decode |
| B4 | ChunkedReq 在 _processLastData 中 | 跳过，不生成响应 |
| B5 | decode batch 空 | _scheduleNextBatch 返回 null，跳过 forward |
| B6 | prefill batch 含混合 ChunkedReq 和 Req | ChunkedReq 跳过采样，Req 正常采样 |
| B7 | greedy 采样 + temperature=0 | isGreedy=true，argmax 采样 |
| B8 | offline 模式 + 正常 runTick 调用 | 消息通过参数传入，结果通过返回值传出 |
| B9 | ExitMsg 在 BatchMsg 内 | 递归展开后抛出 Error("ExitSignal") |
| B10 | AbortMsg 目标请求不存在 | prefillManager 和 decodeManager 均返回 null，不释放资源 |
| B11 | copyDoneEvent.synchronize 多次调用 | 不抛异常（幂等） |

## 风险与注意事项

- **兼容性影响**：`ForwardOutput` 接口新增字段，`logits` 类型从 `number[] | null` 变为 `number[][] | null`（2D），可能影响已有代码。需确保 `ForwardOutput` 新字段均为可选或向后兼容。`Batch` class 新增属性均有默认值，不影响现有实例化。`SimScheduler` 从 interface 升级为 class，`types.ts` 中旧 interface 改为 type alias 指向新 class。
- **性能影响**：`SimScheduler` 构造开销可忽略。`_normalTick` 为纯同步逻辑，无异步瓶颈。
- **回滚方案**：所有改动在 `issue-17` 分支，合并前可安全回滚。
- **与现有 MockEngine 方法的关系**：`forward_batch`（snake_case）为 S3 新增方法，与已有的 `forwardBatchPP`/`forwardBatchSeq`/`forwardBatch`（camelCase）并存，不修改已有方法，确保 P3a/P4/P5 测试不受影响。
- **MockSampler vs Sampler**：保留原 `Sampler` class 不做修改，`MockEngine` 新增 `mockSampler` 属性，`forward_batch` 内部使用 `mockSampler`。已有的 `sampler` 属性和 `forwardBatchPP`/`forwardBatchSeq`/`forwardBatch` 方法继续使用原 `Sampler`。
- **SchedulerIOMixin 的 TS 实现**：采用 class 继承（单继承），`SimScheduler extends SchedulerIOMixin`。
- **lazy_free_region**：`_processLastData` 中需在遍历前调用 `cacheManager.beginLazyFree()`，遍历后调用 `endLazyFree()`。
- **overlap 模式降级**：S3 仅实现 `normal_tick`。当 `overlapEnabled=true` 时降级调用 `_normalTick`，确保 end-to-end 可跑通。
- **copyDoneEvent 机制**：仿真中 `MockEvent.synchronize()` 为 noop 立即返回，真实 SGLang 中等待 GPU→CPU 拷贝完成。`_processLastData` 必须在 `copyDoneEvent.synchronize()` 之后才能读取 `nextTokensCpu`（对齐 §9.11 L2955-2956）。
- **_processOneMsg 完整消息类型**：必须覆盖 BatchSchedulerMsg/ExitMsg/UserMsg/AbortMsg 四种类型（对齐 §9.11 L3027-3052），不能简化。
- **isChunkPrefill 判断**：使用 `batch.reqs 中存在 instanceof ChunkedReq` 判断（对齐 §9.11 L3686），不使用 `batch.hasIdleReqs`。
- **dummyReq**：MockEngine 构造时创建 dummyReq 用于 CUDA Graph padding（对齐 §9.11 L3662-3674），`tableIdx = maxRunningReq`，`pageTable` 最后一行填充 `numTokens`。`GraphRunner.padBatch` 使用 `dummyReq` 填充而非 `null`。
- **finishedReqs 更新**：每轮 `_processLastData` 维护 `newFinishedReqs` 集合，遍历结束后赋值 `this.finishedReqs = newFinishedReqs`（对齐 §9.11 L2959-2978），防止 overlap 模式重复释放。S3 仅 normal_tick 下此集合每轮更新。
