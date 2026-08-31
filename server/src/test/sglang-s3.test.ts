import assert from "assert";
import {
  MockEvent,
  BatchSamplingArgs,
  SamplingParams,
  Req,
  Batch,
} from "../sglang/core";
import {
  MockSampler,
  MockAttnBackend,
  MockEngine,
} from "../sglang/engine";
import {
  SchedulerIOMixin,
  SimScheduler,
} from "../sglang/scheduler";
import {
  CacheManager,
  BaseCacheHandle,
} from "../sglang/cache";
import {
  ChunkedReq,
} from "../sglang/entities";
import type {
  SimulatorConfig,
  SimRequestMsg,
  SimRespMsg,
  SchedulerMsg,
  BatchSchedulerMsg,
  ExitMsg,
  UserMsg,
  AbortMsg,
  ForwardInput,
} from "../sglang/types";

/**
 * Issue #17 验收测试 — S3: MockEngine.forward_batch + SimScheduler.normal_tick + SchedulerIOMixin
 *
 * Run with:  npx ts-node src/test/sglang-s3.test.ts
 */

let passed = 0, failed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try { fn(); passed++; console.log("  \u2713 " + name); }
  catch (e) {
    failed++;
    const msg = e instanceof Error ? e.message : String(e);
    failures.push(name + ": " + msg);
    console.log("  \u2717 " + name + " — " + msg);
  }
}

// ===== 辅助函数 =====

/** 创建测试用 SimulatorConfig（最小配置，naive cache） */
function makeConfig(overrides?: Partial<SimulatorConfig>): SimulatorConfig {
  return {
    modelConfig: {
      numLayers: 2,
      hiddenSize: 256,
      numKvHeads: 4,
      headDim: 64,
      vocabSize: 128,
      isMoe: false,
      numExperts: 0,
      moeIntermediateSize: 0,
      moeTopK: 1,
      intermediateSize: 0,
      numAttentionHeads: 4,
      rmsNormEps: 1e-6,
      ropeTheta: 10000.0,
      maxPositionEmbeddings: 2048,
    },
    maxRunningReq: 8,
    maxSeqLen: 256,
    maxExtendTokens: 256,
    cacheType: "naive",
    pageSize: 1,
    numPages: 512,
    totalGpuMemory: 80 * 1024 ** 3,
    memoryRatio: 0.88,
    dtypeSize: 2,
    enableCudaGraph: false,
    cudaGraphBs: null,
    cudaGraphMaxBs: null,
    graphReplayCostTicks: 1,
    eagerForwardCostTicks: 10,
    enableOverlap: false,
    cpuScheduleCostTicks: 1,
    cpuProcessResultCostTicks: 1,
    tpSize: 1,
    allReduceCostPerByteTicks: 0.001,
    allReduceLatencyTicks: 2,
    tpCpuGroupType: "gloo",
    tpGpuGroupType: "nccl",
    dpSize: 1,
    dpLoadBalanceStrategy: "round_robin",
    enableDpAttention: false,
    dpAttentionAllGatherCostPerByteTicks: 0.0015,
    epSize: 1,
    allToAllCostPerByteTicks: 0.002,
    allToAllLatencyTicks: 3,
    moeRoutingMode: "mock",
    enableEplb: false,
    cpSize: 1,
    cpAllGatherCostPerByteTicks: 0.001,
    ppSize: 1,
    ppNumMicroBatches: 1,
    ppSendRecvCostPerByteTicks: 0.0005,
    ppPipelineSchedule: "1f1b",
    ppInterleavedNumChunks: 2,
    commBandwidthBytesPerTick: 1_000_000,
    commOverlapWithCompute: true,
    networkBandwidthGBps: 100,
    networkLatencyUs: 5,
    tpEfficiency: 0.95,
    epEfficiency: 0.90,
    cpEfficiency: 0.90,
    offlineMode: true,
    eosTokenId: 2,
    mockSampleMode: "greedy",
    fixedOutputToken: 0,
    maxTicks: null,
    logLevel: "INFO",
    enableMetrics: true,
    ...overrides,
  };
}

// ===== T1: MockEvent 构造与 synchronize =====
test("T1: MockEvent 构造与 synchronize", () => {
  const evt = new MockEvent();
  evt.record();
  evt.synchronize();
  // 多次调用不抛异常
  evt.synchronize();
  evt.synchronize();
});

// ===== T2: MockSampler 构造 =====
test("T2: MockSampler 构造", () => {
  const s = new MockSampler(128, "greedy", 0);
  assert.strictEqual(s.vocabSize, 128);
  assert.strictEqual(s.mode, "greedy");
  assert.strictEqual(s.fixedToken, 0);
});

// ===== T3: MockSampler.prepare - greedy batch =====
test("T3: MockSampler.prepare - greedy batch", () => {
  const s = new MockSampler(128, "greedy", 0);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ temperature: 0 }) });
  batch.reqs.set(1, req);
  const args = s.prepare(batch);
  assert.strictEqual(args.isGreedy, true);
  assert.strictEqual(args.temperatures, null);
});

// ===== T4: MockSampler.prepare - mixed batch =====
test("T4: MockSampler.prepare - mixed batch", () => {
  const s = new MockSampler(128, "greedy", 0);
  const batch = new Batch();
  const req1 = new Req({ rid: 1, inputIds: [1, 2], samplingParams: new SamplingParams({ temperature: 0 }) });
  const req2 = new Req({ rid: 2, inputIds: [3, 4], samplingParams: new SamplingParams({ temperature: 1.0 }) });
  batch.reqs.set(1, req1);
  batch.reqs.set(2, req2);
  const args = s.prepare(batch);
  assert.strictEqual(args.isGreedy, false);
  assert.strictEqual(args.temperatures!.length, 2);
  assert.strictEqual(args.temperatures![0], 0);
  assert.strictEqual(args.temperatures![1], 1.0);
});

// ===== T5: MockSampler.sample - greedy 模式 =====
test("T5: MockSampler.sample - greedy 模式", () => {
  const s = new MockSampler(128, "greedy", 0);
  const logits = [[0, 1, 2, 3]];
  const args = new BatchSamplingArgs({ temperatures: null });
  const tokens = s.sample(logits, args);
  assert.strictEqual(tokens.length, 1);
  assert.strictEqual(tokens[0], 3); // argmax
});

// ===== T6: MockSampler.sample - random 模式 =====
test("T6: MockSampler.sample - random 模式", () => {
  const s = new MockSampler(128, "random", 0);
  const logits = [[0, 0, 0, 0]];
  const args = new BatchSamplingArgs({ temperatures: [1.0] });
  const tokens = s.sample(logits, args);
  assert.strictEqual(tokens.length, 1);
  assert.ok(tokens[0] >= 0 && tokens[0] < 128);
});

// ===== T7: MockSampler.sample - fixed 模式 =====
test("T7: MockSampler.sample - fixed 模式", () => {
  const s = new MockSampler(128, "fixed", 42);
  const logits = [[0, 0, 0, 0]];
  const args = new BatchSamplingArgs({ temperatures: [1.0] });
  const tokens = s.sample(logits, args);
  assert.strictEqual(tokens.length, 1);
  assert.strictEqual(tokens[0], 42);
});

// ===== T8: MockSampler.apply_temperature =====
test("T8: MockSampler.apply_temperature", () => {
  const s = new MockSampler(128, "greedy", 0);
  const logits = [1, 2, 3];
  const result = s.apply_temperature(logits, 2.0);
  assert.strictEqual(result[0], 0.5);
  assert.strictEqual(result[1], 1.0);
  assert.strictEqual(result[2], 1.5);
});

// ===== T9: MockSampler.apply_top_p_top_k =====
test("T9: MockSampler.apply_top_p_top_k", () => {
  const s = new MockSampler(128, "greedy", 0);
  const logits = [1.0, 2.0, 3.0, 0.5];
  const result = s.apply_top_p_top_k(logits, 1.0, 2);
  assert.strictEqual(result.length, 4);
  // top-k=2 保留最大 2 个
  assert.ok(result[1] !== -Infinity);
  assert.ok(result[2] !== -Infinity);
});

// ===== T10: MockSampler.apply_logits_penalty =====
test("T10: MockSampler.apply_logits_penalty", () => {
  const s = new MockSampler(128, "greedy", 0);
  const logits = [1.0, 2.0, 3.0];
  const result = s.apply_logits_penalty(logits, [1], 2.0);
  assert.strictEqual(result[1], 4.0); // 2.0 * 2.0
});

// ===== T11: MockAttnBackend.prepare_metadata =====
test("T11: MockAttnBackend.prepare_metadata", () => {
  const backend = new MockAttnBackend();
  const batch = new Batch();
  backend.prepareMetadata(batch);
  assert.deepStrictEqual(batch.attnMetadata, {});
});

// ===== T12: MockAttnBackend.simulate_kv_recycle =====
test("T12: MockAttnBackend.simulate_kv_recycle", () => {
  const backend = new MockAttnBackend();
  assert.strictEqual(backend.simulate_kv_recycle(), 0);
});

// ===== T13: MockEngine.forward_batch - prefill batch =====
test("T13: MockEngine.forward_batch - prefill batch", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  (req as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, req);
  batch.extendInputTokens = 3;
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.ok(output.prefillBatchTime! > 0);
});

// ===== T14: MockEngine.forward_batch - decode batch =====
test("T14: MockEngine.forward_batch - decode batch", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  req.deviceLen = 3;
  (req as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, req);
  batch.numDecodeTokens = 1;
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.ok(output.decodeBatchTime! > 0);
});

// ===== T15: MockEngine.forward_batch - ChunkedReq 跳过 =====
test("T15: MockEngine.forward_batch - ChunkedReq 跳过 completeOne", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  const batch = new Batch();
  const creq = new ChunkedReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  creq.deviceLen = 2;
  (creq as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, creq);
  batch.extendInputTokens = 2;
  const sampleArgs = engine.mockSampler.prepare(batch);
  const deviceLenBefore = creq.deviceLen;
  engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(creq.deviceLen, deviceLenBefore); // completeOne 未被调用
});

// ===== T16: MockEngine.forward_batch - CUDA Graph =====
test("T16: MockEngine.forward_batch - CUDA Graph isGraphCapture", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4] });
  const engine = new MockEngine(config);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  (req as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, req);
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(output.isGraphCapture, true);
});

// ===== T17: MockEngine.forward_batch - copyDoneEvent =====
test("T17: MockEngine.forward_batch - copyDoneEvent", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  (req as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, req);
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.ok(output.copyDoneEvent !== undefined);
  output.copyDoneEvent.synchronize(); // 不抛异常
});

// ===== T18: MockEngine.forward_batch - isChunkPrefill =====
test("T18: MockEngine.forward_batch - isChunkPrefill", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  const batch = new Batch();
  const creq = new ChunkedReq({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  (creq as unknown as { tableIdx: number }).tableIdx = 0;
  batch.reqs.set(1, creq);
  batch.extendInputTokens = 2;
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(output.isChunkPrefill, true);
});

// ===== T19: SchedulerIOMixin - offline 模式 =====
test("T19: SchedulerIOMixin - offline 模式", () => {
  const config = makeConfig({ offlineMode: true });
  const mixin = new SchedulerIOMixin(config);
  assert.deepStrictEqual(mixin.receiveMsg(), []);
  mixin.sendResult([]); // 不抛异常
});

// ===== T20: SchedulerIOMixin - online 模式 =====
test("T20: SchedulerIOMixin - online 模式", () => {
  const config = makeConfig({ offlineMode: false });
  const mixin = new SchedulerIOMixin(config);
  // online 模式下 receiveMsg 返回内部队列
  assert.deepStrictEqual(mixin.receiveMsg(), []);
  mixin.step(); // 不抛异常
});

// ===== T21: SimScheduler 构造 =====
test("T21: SimScheduler 构造", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  assert.ok(scheduler.engine instanceof MockEngine);
  assert.ok(scheduler.tableManager !== undefined);
  assert.ok(scheduler.cacheManager instanceof CacheManager);
  assert.ok(scheduler.decodeManager !== undefined);
  assert.ok(scheduler.prefillManager !== undefined);
  assert.ok(scheduler.engine.dummyReq !== undefined);
  assert.strictEqual(scheduler.engine.dummyReq.rid, -1);
});

// ===== T22: SimScheduler._normalTick - 空 tick =====
test("T22: SimScheduler._normalTick - 空 tick", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const result = scheduler.runTick([]);
  assert.deepStrictEqual(result, []);
});

// ===== T23: SimScheduler end-to-end - 短 prompt =====
test("T23: SimScheduler end-to-end - 短 prompt", () => {
  const config = makeConfig({ mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 2 },
  ];
  // 第一个 tick: prefill
  const resp1 = scheduler.runTick(msgs);
  assert.ok(resp1.length >= 1, "prefill tick should produce responses");
});

// ===== T24: SimScheduler._processOneMsg - req_in =====
test("T24: SimScheduler._processOneMsg - req_in", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  // 直接调用 _processOneMsg 以隔离测试消息处理逻辑
  const userMsg: UserMsg = { tag: "req_in", uid: 42, inputIds: [1, 2, 3], samplingParams: null, outputLen: 5 };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(userMsg);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 1);
  assert.strictEqual(scheduler.prefillManager.pendingList[0].rid, 42);
});

// ===== T25: SimScheduler._processOneMsg - maxTokens 调整 =====
test("T25: SimScheduler._processOneMsg - maxTokens 调整", () => {
  const config = makeConfig({ maxSeqLen: 5 });
  const scheduler = new SimScheduler(config);
  // inputLen=4, maxSeqLen=5, maxOutputLen=1
  const sp = new SamplingParams({ maxNewTokens: 10 });
  const userMsg: UserMsg = { tag: "req_in", uid: 1, inputIds: [1, 2, 3, 4], samplingParams: sp, outputLen: 10 };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(userMsg);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 1);
  assert.strictEqual(scheduler.prefillManager.pendingList[0].samplingParams.maxNewTokens, 1);
});

// ===== T26: SimScheduler._processOneMsg - ExitMsg =====
test("T26: SimScheduler._processOneMsg - ExitMsg", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const exitMsg: ExitMsg = { tag: "exit" };
  assert.throws(() => {
    (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(exitMsg);
  }, /ExitSignal/);
});

// ===== T27: SimScheduler._processOneMsg - BatchMsg =====
test("T27: SimScheduler._processOneMsg - BatchMsg", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const batchMsg: BatchSchedulerMsg = {
    tag: "batch",
    data: [
      { tag: "req_in", uid: 1, inputIds: [1], samplingParams: null, outputLen: 1 },
      { tag: "req_in", uid: 2, inputIds: [2], samplingParams: null, outputLen: 1 },
    ],
  };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(batchMsg);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 2);
});

// ===== T28: SimScheduler._processOneMsg - AbortMsg =====
test("T28: SimScheduler._processOneMsg - AbortMsg", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  // 先通过 _processOneMsg 添加请求
  const userMsg: UserMsg = { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 5 };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(userMsg);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 1);
  // abort 该请求
  const abortMsg: AbortMsg = { tag: "abort", uid: 1 };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(abortMsg);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 0);
});

// ===== T29: SimScheduler._scheduleNextBatch - prefill 优先 =====
test("T29: SimScheduler._scheduleNextBatch - prefill 优先", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  // 先通过 _processOneMsg 添加请求到 prefillManager（不触发调度）
  const userMsg: UserMsg = { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: null, outputLen: 2 };
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(userMsg);
  // 此时 prefillManager 有请求，_scheduleNextBatch 应返回 prefill batch
  const result = (scheduler as unknown as { _scheduleNextBatch: () => ForwardInput | null })._scheduleNextBatch();
  assert.ok(result !== null);
  assert.ok(result.batch.extendInputTokens > 0, "should be a prefill batch");
});

// ===== T30: SimScheduler._scheduleNextBatch - 仅 decode =====
test("T30: SimScheduler._scheduleNextBatch - 仅 decode", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  // 手动向 decodeManager 添加一个请求
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  (req as unknown as { tableIdx: number }).tableIdx = 0;
  scheduler.decodeManager.addReq(req);
  const result = (scheduler as unknown as { _scheduleNextBatch: () => ForwardInput | null })._scheduleNextBatch();
  assert.ok(result !== null);
  assert.strictEqual(result.batch.numDecodeTokens, 1, "should be a decode batch");
});

// ===== T31: SimScheduler._processLastData - copyDoneEvent =====
test("T31: SimScheduler._processLastData - copyDoneEvent", () => {
  const config = makeConfig({ mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 1 },
  ];
  // 运行一个 tick，copyDoneEvent 应被 synchronize
  const resp = scheduler.runTick(msgs);
  // 如果没抛异常，说明 copyDoneEvent.synchronize() 成功执行
  assert.ok(true);
});

// ===== T32: SimScheduler._processLastData - prefill 完成 =====
test("T32: SimScheduler._processLastData - prefill 完成", () => {
  const config = makeConfig({ mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 3 });
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: sp, outputLen: 3 },
  ];
  // Prefill tick
  const resp = scheduler.runTick(msgs);
  // prefill 完成后应有 resp_token（未 finished）
  assert.ok(resp.length >= 1);
  assert.strictEqual(resp[0].finished, false, "prefill tick should produce resp_token");
});

// ===== T33: SimScheduler._processLastData - 请求完成 =====
test("T33: SimScheduler._processLastData - 请求完成", () => {
  const config = makeConfig({ mockSampleMode: "fixed", fixedOutputToken: 2 });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 1 });
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: sp, outputLen: 1 },
  ];
  // Prefill tick（maxNewTokens=1, fixedOutputToken=2=eosTokenId）
  const resp = scheduler.runTick(msgs);
  // 应产出完成的响应
  assert.ok(resp.length >= 1);
  assert.strictEqual(resp[0].finished, true, "request should be finished after producing 1 token");
});

// ===== T34: SimScheduler._processLastData - EOS 终止 =====
test("T34: SimScheduler._processLastData - EOS 终止", () => {
  const config = makeConfig({ mockSampleMode: "fixed", fixedOutputToken: 2, eosTokenId: 2 });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 10, skipSpecialTokens: false });
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: sp, outputLen: 10 },
  ];
  const resp = scheduler.runTick(msgs);
  // 应看到 resp_done（因为 fixedOutputToken=2=eosTokenId）
  const doneResp = resp.find(r => r.finished);
  assert.ok(doneResp, "should have a finished response due to EOS");
});

// ===== T35: SimScheduler._processLastData - ChunkedReq 跳过 =====
test("T35: SimScheduler._processLastData - ChunkedReq 跳过", () => {
  const config = makeConfig({ maxExtendTokens: 1, mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  // 输入长度=3, maxExtendTokens=1 → chunked prefill
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 1 },
  ];
  const resp = scheduler.runTick(msgs);
  // ChunkedReq 不应产出 resp_token
  const tokenResp = resp.find(r => r.uid === 1 && r.tag === "resp_token");
  assert.strictEqual(tokenResp, undefined, "ChunkedReq should not produce resp_token");
});

// ===== T36: SimScheduler._processLastData - finishedReqs 更新 =====
test("T36: SimScheduler._processLastData - finishedReqs 更新", () => {
  const config = makeConfig({ mockSampleMode: "fixed", fixedOutputToken: 2, eosTokenId: 2 });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 1, skipSpecialTokens: false });
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: sp, outputLen: 1 },
  ];
  scheduler.runTick(msgs);
  assert.strictEqual(scheduler.finishedReqs.size, 1, "finishedReqs should contain the completed request");
});

// ===== T37: SimScheduler end-to-end - 完整流程 =====
test("T37: SimScheduler end-to-end - 完整流程", () => {
  const config = makeConfig({ mockSampleMode: "fixed", fixedOutputToken: 5 });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 3 });
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: sp, outputLen: 3 },
  ];

  const allResp: SimRespMsg[] = [];

  // Prefill tick
  const resp1 = scheduler.runTick(msgs);
  allResp.push(...resp1);

  // Decode ticks until done
  for (let i = 0; i < 10; i++) {
    const resp = scheduler.runTick([]);
    allResp.push(...resp);
    if (resp.some(r => r.finished)) break;
  }

  const doneResp = allResp.find(r => r.finished);
  assert.ok(doneResp, "should have a finished response");
  assert.strictEqual(doneResp!.uid, 1);
});

// ===== T38: SimScheduler._freeReqResources =====
test("T38: SimScheduler._freeReqResources", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  const tableIdx = scheduler.tableManager.allocate();
  (req as unknown as { tableIdx: number }).tableIdx = tableIdx;
  (req as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle = null;

  const availableBefore = scheduler.tableManager.availableSize;
  (scheduler as unknown as { _freeReqResources: (r: Req) => void })._freeReqResources(req);
  const availableAfter = scheduler.tableManager.availableSize;
  assert.strictEqual(availableAfter, availableBefore + 1, "tableIdx should be freed");
});

// ===== T39: GraphRunner.padBatch - 使用 dummyReq =====
test("T39: GraphRunner.padBatch - 使用 dummyReq", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [2, 4] });
  const engine = new MockEngine(config);
  const batch = new Batch();
  const req = new Req({ rid: 1, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
  batch.reqs.set(1, req);
  engine.graphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 2, "should pad to cudaGraphBs size");
  assert.strictEqual(batch.paddedReqs[1], engine.dummyReq, "padding should use dummyReq");
});

// ===== T40: MockEngine dummyReq 初始化 =====
test("T40: MockEngine dummyReq 初始化", () => {
  const config = makeConfig({ maxRunningReq: 8 });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.dummyReq.rid, -1);
  assert.strictEqual(engine.dummyReq.deviceLen, 1);
  assert.strictEqual(engine.dummyReq.maxDeviceLen, 1);
  // pageTable 最后一行应填充 numTokens
  const numTokens = engine.numPages * config.pageSize;
  assert.strictEqual(engine.pageTable[config.maxRunningReq][0], numTokens);
});

// ===== 边界条件测试 =====

// ===== B1: 空 incoming + 空 pending + 空 decode =====
test("B1: 空 incoming + 空 pending + 空 decode", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const result = scheduler.runTick([]);
  assert.deepStrictEqual(result, []);
});

// ===== B2: maxNewTokens 被截断为 0 =====
test("B2: maxNewTokens 被截断为 0", () => {
  const config = makeConfig({ maxSeqLen: 3 });
  const scheduler = new SimScheduler(config);
  const sp = new SamplingParams({ maxNewTokens: 5 });
  // inputLen=3, maxSeqLen=3, maxOutputLen=0
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: sp, outputLen: 5 },
  ];
  scheduler.runTick(msgs);
  assert.strictEqual(scheduler.prefillManager.pendingList.length, 0, "request should be skipped");
});

// ===== B3: 单 token 输入请求 =====
test("B3: 单 token 输入请求", () => {
  const config = makeConfig({ mockSampleMode: "fixed", fixedOutputToken: 5 });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1], samplingParams: null, outputLen: 2 },
  ];
  const resp = scheduler.runTick(msgs);
  assert.ok(resp.length >= 1, "should produce response for single token input");
});

// ===== B4: ChunkedReq 在 _processLastData 中 =====
test("B4: ChunkedReq 在 _processLastData 中", () => {
  const config = makeConfig({ maxExtendTokens: 1, mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 1 },
  ];
  const resp = scheduler.runTick(msgs);
  // ChunkedReq 不应产出任何响应
  assert.ok(!resp.some(r => r.uid === 1), "ChunkedReq should not produce response");
});

// ===== B5: decode batch 空 =====
test("B5: decode batch 空", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  // 无任何请求时 runTick 应返回空
  const result = scheduler.runTick([]);
  assert.deepStrictEqual(result, []);
});

// ===== B6: prefill batch 含混合 ChunkedReq 和 Req =====
test("B6: prefill batch 含混合 ChunkedReq 和 Req", () => {
  const config = makeConfig({ maxExtendTokens: 2, mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  // 一个短请求（不 chunked），一个长请求（chunked）
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1], samplingParams: null, outputLen: 1 },
    { tag: "req_in", uid: 2, inputIds: [1, 2, 3, 4, 5], samplingParams: null, outputLen: 1 },
  ];
  const resp = scheduler.runTick(msgs);
  // 短请求应产出 resp_token
  assert.ok(resp.some(r => r.uid === 1), "short request should produce response");
});

// ===== B7: greedy 采样 + temperature=0 =====
test("B7: greedy 采样 + temperature=0", () => {
  const s = new MockSampler(128, "greedy", 0);
  const logits = [[0, 5, 2, 8, 1]];
  const args = new BatchSamplingArgs({ temperatures: null });
  const tokens = s.sample(logits, args);
  assert.strictEqual(tokens[0], 3, "greedy should pick argmax (index 3, value 8)");
});

// ===== B8: offline 模式 + 正常 runTick 调用 =====
test("B8: offline 模式 + 正常 runTick 调用", () => {
  const config = makeConfig({ offlineMode: true, mockSampleMode: "fixed", fixedOutputToken: 5 });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2], samplingParams: null, outputLen: 1 },
  ];
  const resp = scheduler.runTick(msgs);
  assert.ok(resp.length >= 1, "offline mode runTick should produce responses");
});

// ===== B9: ExitMsg 在 BatchMsg 内 =====
test("B9: ExitMsg 在 BatchMsg 内", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const batchMsg: BatchSchedulerMsg = {
    tag: "batch",
    data: [
      { tag: "exit" },
    ],
  };
  assert.throws(() => {
    (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(batchMsg);
  }, /ExitSignal/);
});

// ===== B10: AbortMsg 目标请求不存在 =====
test("B10: AbortMsg 目标请求不存在", () => {
  const config = makeConfig();
  const scheduler = new SimScheduler(config);
  const abortMsg: AbortMsg = { tag: "abort", uid: 9999 };
  // 不应抛异常
  (scheduler as unknown as { _processOneMsg: (m: SchedulerMsg) => void })._processOneMsg(abortMsg);
  assert.ok(true, "aborting non-existent request should not throw");
});

// ===== B11: copyDoneEvent.synchronize 多次调用 =====
test("B11: copyDoneEvent.synchronize 多次调用", () => {
  const evt = new MockEvent();
  evt.record();
  evt.synchronize();
  evt.synchronize();
  evt.synchronize();
  assert.ok(true, "multiple synchronize calls should not throw");
});

// ===== 结果汇总 =====
console.log(`\n=== S3 Test Results: ${passed} passed, ${failed} failed ===`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) {
    console.log("  - " + f);
  }
  process.exit(1);
}
