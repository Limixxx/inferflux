---
title: "Issue #17 解决方案"
issue_number: 17
issue_type: Feature
created: 2026-08-31
updated: 2026-08-31
status: draft
review_round: 1
---

# Issue #17 解决方案

## 需求分析

- **问题描述**：Issue #17 要求实现 SGLang 仿真器 S3 阶段的核心调度循环组件：`MockEngine.forward_batch`（含 prefill/decode 时间模拟、chunked prefill 标识、CUDA Graph 标识、PP last 标识）、`MockSampler`（含 `prepare`/`sample` 方法及采样参数处理）、`MockAttnBackend`（含 `prepare_metadata`/`simulate_kv_recycle` stub）、`SimScheduler.normal_tick`（含完整的调度循环：消息处理 → prefill/decode 调度 → forward → 结果处理）、`SchedulerIOMixin`（含 `step()`/`_process_messages`/`_send_resp` 及消息队列管理）。Phase 2 目标：单实例可跑通一个短 prompt end-to-end。

- **能力目标**：
  1. **MockEngine.forward_batch**（§4.3/§9.11）：返回 `ForwardOutput`，包含 `prefillBatchTime`/`decodeBatchTime` 时间模拟、`isChunkPrefill` 标识、`isGraphCapture` 标识、`isPpLast` 标识；对 ChunkedReq 跳过 `completeOne()`；调用 MockSampler 采样
  2. **MockSampler**（§4.3/§9.11）：`prepare(batch)` 生成 `BatchSamplingArgs`；`sample(logits, args)` 按 greedy/random/fixed 模式采样；实现 `apply_temperature`/`apply_top_p_top_k`/`apply_logits_penalty` 采样管线
  3. **MockAttnBackend**（§4.3）：`prepare_metadata(batch)` 为桩实现；`simulate_kv_recycle` 为 stub 返回 0
  4. **SimScheduler.normal_tick**（§9.2/§9.10）：串行执行消息处理 → prefill 调度 → decode 调度 → forward → 结果处理；`_scheduleNextBatch` prefill 优先、否则 decode
  5. **SchedulerIOMixin**（§4.3/§9.6）：`step()` 推进 req_in/resp_token；`self.messages = deque[msg]`；`_process_messages`/`_send_resp`；支持 offline 模式
  6. Phase 2 验收：单实例可跑通一个短 prompt end-to-end；TS strict zero-any

- **影响范围**：仅修改 `server/src/sglang/` 目录下的 `engine/index.ts`（重构 MockEngine.forward_batch + 新增 MockSampler/MockAttnBackend）、`scheduler/index.ts`（新增 SimScheduler + SchedulerIOMixin）、`types.ts`（新增消息与接口类型）、`core/index.ts`（扩展 ForwardOutput/BatchSamplingArgs/Batch）、`index.ts`（新增 re-export），以及 `server/src/test/` 目录（新增 `sglang-s3.test.ts`）。不修改已有测试代码。

- **依赖 Issue**：
  - #16 S2: PrefillManager + PrefillAdder + DecodeManager（已完成）
  - #15 K4: RadixPrefixCache（已完成）

- **阻塞 Issue**：
  - S4: SimGraphRunner（bs 分桶 CUDA 图）
  - S5: Overlap Scheduling（last_data 延迟空 tick 刷新）

## 改造方案

### 总体思路

按照 §9.11 完整实现代码集，将 S3 阶段的五个组件分为三个层次实现：

1. **底层组件**（`engine/index.ts`）：新增 `MockSampler`（替换现有 Sampler 的功能扩展）、`MockAttnBackend`（新增）；`MockEngine` 新增 `forward_batch` 方法
2. **中层调度器**（`scheduler/index.ts`）：新增 `SimScheduler` 类（含 `normal_tick` 完整循环）和 `SchedulerIOMixin` 类
3. **类型层**（`types.ts`/`core/index.ts`）：扩展消息类型、`ForwardOutput`、`BatchSamplingArgs`、`Batch` 等

核心设计决策：

1. **MockSampler 与现有 Sampler 并存**：当前 `Sampler` 是 P3a/P4 的简化桩，仅支持 `sample(logits, batchSize)`。S3 新增 `MockSampler` class，保留原 `Sampler` 不做改动（P3a/P4/P5 测试依赖），`MockEngine` 内部新增 `mockSampler` 属性，`forward_batch` 使用 `mockSampler`。
2. **MockAttnBackend 独立 class**：`prepare_metadata(batch)` 为空操作桩，`simulate_kv_recycle` 返回 0 ticks。
3. **SimScheduler 继承 SchedulerIOMixin**：`SchedulerIOMixin` 负责消息 I/O，`SimScheduler` 继承并实现调度逻辑。`offline_mode=true` 时 receive/send 为 noop。
4. **normal_tick 循环严格对齐 §9.11**：`_processOneMsg` → `_scheduleNextBatch` → `_forward` → `_processLastData`。
5. **ForwardOutput 扩展**：新增 `prefillBatchTime`/`decodeBatchTime`/`isChunkPrefill`/`isGraphCapture`/`isPpLast` 可选字段，不影响已有代码。
6. **Batch 扩展**：新增 `paddedReqs`/`inputIds`/`positions`/`outLoc`/`attnMetadata` 属性，供 `_prepareBatch` 和 `forward_batch` 使用。
7. **GraphRunner.padBatch**：新增方法，对齐 §9.11 `SimGraphRunner.pad_batch`。

### 详细设计

#### 1. BatchSamplingArgs 类

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

#### 2. MockSampler 类

核心职责：模拟采样器，支持 greedy/random/fixed 三种模式，实现采样参数处理管线。

**构造函数**：`vocabSize: number`, `mode: "random" | "greedy" | "fixed"`, `fixedToken: number`

**prepare(batch: Batch): BatchSamplingArgs**：
- 遍历 batch 中所有 req 的 `samplingParams`
- 若所有 req 均为 greedy → `temperatures: null`
- 否则 → `temperatures/topK/topP` 数组

**sample(logits: number[][], args: BatchSamplingArgs): number[]**：
- greedy/fixed/random 三种模式

**采样参数处理管线**（方法级，供 sample 内部调用）：
- `apply_temperature(logits, temperature)` — logits /= temperature
- `apply_top_p_top_k(logits, top_p, top_k)` — top-k 过滤 + top-p 核化
- `apply_logits_penalty(logits, token_ids, penalty)` — 重复惩罚
- `apply_logits_prob(logits)` — stub
- `apply_logits_bias(logits, bias)` — stub

#### 3. MockAttnBackend 类

- `prepare_metadata(batch: Batch): void` — 空操作，可选设置 `batch.attnMetadata = {}`
- `simulate_kv_recycle(): number` — 返回 0（stub）

#### 4. MockEngine.forward_batch 新增方法

```typescript
forward_batch(batch: Batch): ForwardOutput {
  // 1. CUDA Graph 判断
  const isGraphCapture = this.graphRunner.canUseCudaGraph(batch);

  // 2. mock forward
  let logits: number[];
  if (isGraphCapture) {
    logits = this.graphRunner.replay(batch);
  } else {
    logits = this._mockModelForward(batch);
  }

  // 3. complete_one（跳过 ChunkedReq）
  for (const req of batch.reqs.values()) {
    if (!(req instanceof ChunkedReq)) {
      req.completeOne();
    }
  }

  // 4. 采样
  const sampleArgs = this.mockSampler.prepare(batch);
  const nextTokens = this.mockSampler.sample(logits, sampleArgs);

  // 5. 时间模型
  const isPrefill = batch.extendInputTokens > 0;
  const isChunkPrefill = batch.hasIdleReqs;
  const prefillBatchTime = isPrefill ? this._computePrefillTime(batch) : 0;
  const decodeBatchTime = !isPrefill ? this._computeDecodeTime(batch) : 0;

  return {
    logits, sampledIds: nextTokens, isIntermediate: false,
    prefillBatchTime, decodeBatchTime, isChunkPrefill, isGraphCapture, isPpLast: true,
  };
}
```

**时间模型**（简化版，对齐 §4.3）：
- `_computePrefillTime(batch)` — 基于 `batch.extendInputTokens` 和 `config.eagerForwardCostTicks` 计算
- `_computeDecodeTime(batch)` — 基于 `batch.numDecodeTokens` 和 `config.graphReplayCostTicks`/`config.eagerForwardCostTicks` 计算

**MockEngine 新增属性**：`mockSampler: MockSampler`, `mockAttnBackend: MockAttnBackend`, `pageTable: number[][]`, `numPages: number`, `maxSeqLen: number`

#### 5. SchedulerIOMixin 类

```typescript
class SchedulerIOMixin {
  protected readonly config: SimulatorConfig;
  protected readonly _incomingQueue: SimRequestMsg[];
  protected readonly _outgoingQueue: SimRespMsg[];

  constructor(config: SimulatorConfig) {
    this.config = config;
    this._incomingQueue = [];
    this._outgoingQueue = [];
  }

  receiveMsg(): SimRequestMsg[] {
    if (this.config.offlineMode) return [];
    const msgs = [...this._incomingQueue];
    this._incomingQueue.length = 0;
    return msgs;
  }

  sendResult(reply: SimRespMsg[]): void {
    if (this.config.offlineMode) return;
    this._outgoingQueue.push(...reply);
  }

  step(): void {
    // 推进一步消息处理（非 offline 模式下由外部驱动）
  }
}
```

#### 6. SimScheduler 类

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
    if (this.overlapEnabled) {
      // 降级为 normal_tick，确保 end-to-end 可跑通
    }
    return this._normalTick(incoming);
  }
}
```

**_normalTick 流程**（对齐 §9.11）：
```
1. for msg in incoming: _processOneMsg(msg)
2. forwardInput = _scheduleNextBatch()
3. ongoingData = null
4. if forwardInput !== null: ongoingData = [forwardInput, _forward(forwardInput)]
5. return _processLastData(ongoingData)
```

**_processOneMsg(msg: SimRequestMsg)**：
- 调整 maxNewTokens = min(maxNewTokens, engine.maxSeqLen - inputLen)
- 若调整后 ≤ 0 → 跳过
- 调用 prefillManager.addOneReq(msg)

**_scheduleNextBatch()**：
- batch = prefillManager.scheduleNextBatch(prefillBudget) || decodeManager.scheduleNextBatch()
- 若 batch !== null → _prepareBatch(batch)
- 否则 → null

**_prepareBatch(batch: Batch)**：
1. engine.graphRunner.padBatch(batch)
2. 对 batch 中每个 req 调用 cacheManager.allocatePaged(req)
3. 计算 positions、input_ids、out_loc
4. engine.mockAttnBackend.prepareMetadata(batch)
5. 构造 ForwardInput 返回

**_forward(forwardInput: ForwardInput)**：
1. 从 tokenPool 读取 input_ids 填入 batch
2. 调用 engine.forward_batch(batch)
3. 将 next_tokens 写入 tokenPool（跳过 ChunkedReq）
4. 调用 decodeManager.filterReqs(batch.reqs)
5. 返回 ForwardOutput

**_processLastData(ongoingData)**：
1. null → return []
2. 在 beginLazyFree/endLazyFree 上下文内遍历 batch.reqs：
   - ChunkedReq 跳过
   - req.appendHost(nextToken)
   - 判断 finished：!canDecode || nextToken === eosTokenId
   - 构造 SimRespMsg
   - finished → decodeManager.removeReq + _freeReqResources
   - prefill 非 finished → cacheManager.cacheReq(req, false)
3. 返回 reply

**_freeReqResources(req: Req)**：
- tableManager.free(req.tableIdx)
- cacheManager.freeCache(req)

#### 7. Batch 扩展属性

新增属性（均在 Batch 构造中初始化为默认值）：
- `paddedReqs: Req[]` — CUDA Graph padding 后的请求列表
- `inputIds: number[]` — 由 scheduler 填充
- `positions: number[]` — 由 scheduler 填充
- `outLoc: number[]` — KV cache 写入位置
- `attnMetadata: unknown` — 由 attention backend 填充

#### 8. GraphRunner.padBatch 新增方法

```typescript
padBatch(batch: Batch): void {
  let paddedSize: number;
  if (this.canUseCudaGraph(batch)) {
    paddedSize = this.cudaGraphBs?.find(bs => bs >= batch.reqs.size) ?? batch.reqs.size;
  } else {
    paddedSize = batch.reqs.size;
  }
  // 使用 dummy req 填充至 paddedSize
  const dummyCount = paddedSize - batch.reqs.size;
  batch.paddedReqs = [...batch.reqs.values(), ...Array(dummyCount).fill(null)];
}
```

### 接口变更

1. **`core/index.ts`**：新增 `BatchSamplingArgs` class；`ForwardOutput` 新增 5 个可选字段
2. **`engine/index.ts`**：新增 `MockSampler` class、`MockAttnBackend` class；`MockEngine` 新增属性和方法
3. **`scheduler/index.ts`**：新增 `SchedulerIOMixin` class、`SimScheduler` class
4. **`types.ts`**：新增 `ForwardInput` interface；`SimScheduler` 从 interface 改为指向 class
5. **`index.ts`**：新增所有新 class/interface 的 re-export

### 数据结构改动

1. **新增 `BatchSamplingArgs`** — 采样参数批处理封装
2. **新增 `ForwardInput`** — forward 输入元数据
3. **扩展 `ForwardOutput`** — 新增时间模拟和标识字段
4. **扩展 `Batch`** — 新增 paddedReqs/inputIds/positions/outLoc/attnMetadata
5. **新增 `SimScheduler` 内部状态** — finishedReqs/lastBatch/tokenPool 等

### 修改点清单

1. **修改 `server/src/sglang/core/index.ts`**：新增 `BatchSamplingArgs` class；`ForwardOutput` 新增 5 个可选字段；`Batch` class 新增 5 个属性
2. **修改 `server/src/sglang/engine/index.ts`**：新增 `MockSampler` class（含 prepare/sample/apply_* 方法）；新增 `MockAttnBackend` class；`MockEngine` 新增 mockSampler/mockAttnBackend/pageTable/numPages/maxSeqLen 属性；`MockEngine` 新增 `forward_batch` 方法；`GraphRunner` 新增 `padBatch` 方法
3. **修改 `server/src/sglang/scheduler/index.ts`**：新增 `SchedulerIOMixin` class；新增 `SimScheduler` class
4. **修改 `server/src/sglang/types.ts`**：新增 `ForwardInput` interface；更新 `SimScheduler` 引用
5. **修改 `server/src/sglang/index.ts`**：新增所有新增导出
6. **新建 `server/src/test/sglang-s3.test.ts`**：S3 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | MockSampler 构造 | vocabSize/mode/fixedToken 正确初始化 |
| T2 | MockSampler.prepare - greedy batch | 所有 req.isGreedy=true → temperatures=null |
| T3 | MockSampler.prepare - mixed batch | 存在非 greedy req → temperatures 数组非 null |
| T4 | MockSampler.sample - greedy 模式 | argmax 返回最大 logit 位置 |
| T5 | MockSampler.sample - random 模式 | 返回 [0, vocabSize) 范围随机整数 |
| T6 | MockSampler.sample - fixed 模式 | 返回固定 token |
| T7 | MockSampler.apply_temperature | logits /= temperature 正确执行 |
| T8 | MockSampler.apply_top_p_top_k | top-k 过滤 + top-p 核化正确执行 |
| T9 | MockSampler.apply_logits_penalty | 重复惩罚正确应用 |
| T10 | MockAttnBackend.prepare_metadata | 调用不抛异常 |
| T11 | MockAttnBackend.simulate_kv_recycle | 返回 0 |
| T12 | MockEngine.forward_batch - prefill batch | 返回 ForwardOutput 含 prefillBatchTime > 0 |
| T13 | MockEngine.forward_batch - decode batch | 返回 ForwardOutput 含 decodeBatchTime > 0 |
| T14 | MockEngine.forward_batch - ChunkedReq 跳过 | ChunkedReq 的 completeOne 不被调用 |
| T15 | MockEngine.forward_batch - CUDA Graph | isGraphCapture 标识正确 |
| T16 | SchedulerIOMixin - offline 模式 | receiveMsg 返回 []，sendResult 为 noop |
| T17 | SchedulerIOMixin - online 模式 | 内部队列可收发 |
| T18 | SimScheduler 构造 | 所有子组件正确初始化 |
| T19 | SimScheduler._normalTick - 空 tick | 无 incoming、无可调度请求 → 返回 [] |
| T20 | SimScheduler end-to-end - 短 prompt | 一个短 prompt 经 prefill tick → decode tick → 完成 |
| T21 | SimScheduler._processOneMsg - req_in | 请求正确加入 prefillManager |
| T22 | SimScheduler._processOneMsg - maxTokens 调整 | inputLen 接近 maxSeqLen 时 maxNewTokens 被截断 |
| T23 | SimScheduler._scheduleNextBatch - prefill 优先 | pending 和 decode 都有请求时优先返回 prefill batch |
| T24 | SimScheduler._scheduleNextBatch - 仅 decode | 无 pending 请求时返回 decode batch |
| T25 | SimScheduler._processLastData - prefill 完成 | 非 finished 的 prefill 请求调用 cacheReq(finished=false) |
| T26 | SimScheduler._processLastData - 请求完成 | finished 时调用 removeReq + _freeReqResources |
| T27 | SimScheduler._processLastData - EOS 终止 | nextToken === eosTokenId 时标记 finished |
| T28 | SimScheduler._processLastData - ChunkedReq 跳过 | ChunkedReq 不生成 resp_token |
| T29 | SimScheduler end-to-end - 完整流程 | 单请求经历 prefill + 多个 decode tick 完成 |
| T30 | SimScheduler._freeReqResources | tableIdx 被 free，freeCache 被调用 |

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

## 风险与注意事项

- **兼容性影响**：`ForwardOutput` 接口新增 5 个可选字段，不影响现有代码。`Batch` class 新增属性均有默认值，不影响现有实例化。`SimScheduler` 从 interface 升级为 class，`types.ts` 中旧 interface 改为 type alias 指向新 class。
- **性能影响**：`SimScheduler` 构造开销可忽略。`_normalTick` 为纯同步逻辑，无异步瓶颈。
- **回滚方案**：所有改动在 `issue-17` 分支，合并前可安全回滚。
- **与现有 MockEngine 方法的关系**：`forward_batch` 为 S3 新增方法，不修改已有的 `forwardBatchPP`/`forwardBatchSeq`/`forwardBatch`，确保 P3a/P4/P5 测试不受影响。
- **MockSampler vs Sampler**：保留原 `Sampler` class 不做修改，`MockEngine` 新增 `mockSampler` 属性，`forward_batch` 内部使用 `mockSampler`。
- **SchedulerIOMixin 的 TS 实现**：采用 class 继承（单继承），`SimScheduler extends SchedulerIOMixin`。
- **lazy_free_region**：`_processLastData` 中需在遍历前调用 `cacheManager.beginLazyFree()`，遍历后调用 `endLazyFree()`。
- **overlap 模式降级**：S3 仅实现 `normal_tick`。当 `overlapEnabled=true` 时降级调用 `_normalTick`，确保 end-to-end 可跑通。
- **GraphRunner.padBatch**：S3 需在 `GraphRunner` 中新增 `padBatch` 方法，且 `Batch` class 需新增 `paddedReqs` 属性。
- **Batch 新增属性**：`paddedReqs`/`inputIds`/`positions`/`outLoc`/`attnMetadata` 在 S1 的 `Batch` class 中尚未包含，S3 中需扩展。
