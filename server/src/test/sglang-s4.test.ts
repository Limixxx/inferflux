import assert from "assert";
import {
  SamplingParams,
  Req,
  Batch,
} from "../sglang/core";
import {
  SimGraphRunner,
  MockEngine,
  MockSampler,
} from "../sglang/engine";
import {
  SimScheduler,
} from "../sglang/scheduler";
import {
  CacheManager,
  estimateGraphBuffer,
} from "../sglang/cache";
import type {
  SimulatorConfig,
  ModelConfig,
  SimRequestMsg,
} from "../sglang/types";

/**
 * Issue #18 验收测试 — S4: SimGraphRunner — CUDA Graph bs 分桶 pad_batch graph_replay_cost
 *
 * Run with:  npx ts-node src/test/sglang-s4.test.ts
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

function makeModelConfig(overrides?: Partial<ModelConfig>): ModelConfig {
  return {
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
    ...overrides,
  };
}

function makeConfig(overrides?: Partial<SimulatorConfig>): SimulatorConfig {
  return {
    modelConfig: makeModelConfig(),
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

/** 创建 decode batch（numDecodeTokens>0, extendInputTokens=0） */
function makeDecodeBatch(bs: number): Batch {
  const batch = new Batch();
  for (let i = 1; i <= bs; i++) {
    const req = new Req({ rid: i, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
    (req as unknown as { tableIdx: number }).tableIdx = i - 1;
    batch.reqs.set(i, req);
  }
  batch.numDecodeTokens = bs;
  batch.extendInputTokens = 0;
  return batch;
}

/** 创建 prefill batch（extendInputTokens>0） */
function makePrefillBatch(bs: number, tokensPerReq: number = 3): Batch {
  const batch = new Batch();
  for (let i = 1; i <= bs; i++) {
    const req = new Req({ rid: i, inputIds: [1, 2, 3], samplingParams: new SamplingParams({ maxNewTokens: 5 }) });
    (req as unknown as { tableIdx: number }).tableIdx = i - 1;
    batch.reqs.set(i, req);
  }
  batch.extendInputTokens = bs * tokensPerReq;
  batch.numDecodeTokens = 0;
  return batch;
}

// ===== T1: SimGraphRunner 构造 =====
test("T1: SimGraphRunner 构造", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const runner = engine.simGraphRunner;
  assert.deepStrictEqual(runner.graphBsList, [1, 2, 4, 8]);
  assert.strictEqual(runner.maxGraphBs, 8);
  assert.strictEqual(runner.vocabSize, 128);
  assert.strictEqual(runner.isValid, true);
  // 验证 enableCudaGraph 通过 canUseCudaGraph 间接生效
  const batch = makeDecodeBatch(2);
  assert.strictEqual(runner.canUseCudaGraph(batch), true, "enableCudaGraph=true should allow CUDA Graph");
});

// ===== T2: determineCudaGraphBs - 用户指定 =====
test("T2: determineCudaGraphBs - 用户指定", () => {
  const config = makeConfig({ cudaGraphBs: [2, 4, 16] });
  const result = SimGraphRunner.determineCudaGraphBs(config);
  assert.deepStrictEqual(result, [2, 4, 16]);
});

// ===== T3: determineCudaGraphBs - 自动计算 =====
test("T3: determineCudaGraphBs - 自动计算", () => {
  const config = makeConfig({ cudaGraphBs: null, cudaGraphMaxBs: 24 });
  const result = SimGraphRunner.determineCudaGraphBs(config);
  assert.deepStrictEqual(result, [1, 2, 4, 8, 16, 24]);
});

// ===== T4: determineCudaGraphBs - 大显存自动推断 =====
test("T4: determineCudaGraphBs - 大显存自动推断", () => {
  // cudaGraphMaxBs=null 时，totalGpuMemory > 80GiB → 自动推断 maxBs=256
  const config = makeConfig({ cudaGraphBs: null, cudaGraphMaxBs: null, totalGpuMemory: 81 * 1024 ** 3 });
  const result = SimGraphRunner.determineCudaGraphBs(config);
  assert.ok(result.includes(256), "totalGpuMemory > 80GiB should auto-infer maxBs=256");
  assert.ok(!result.includes(264), "should not exceed maxBs=256");
});

// ===== T5: determineCudaGraphBs - 小显存自动推断 =====
test("T5: determineCudaGraphBs - 小显存自动推断", () => {
  // cudaGraphMaxBs=null 时，totalGpuMemory <= 80GiB → 自动推断 maxBs=160
  const config = makeConfig({ cudaGraphBs: null, cudaGraphMaxBs: null, totalGpuMemory: 64 * 1024 ** 3 });
  const result = SimGraphRunner.determineCudaGraphBs(config);
  assert.ok(result.includes(160), "totalGpuMemory <= 80GiB should auto-infer maxBs=160");
  assert.ok(!result.includes(168), "should not exceed maxBs=160");
});

// ===== T6: determineCudaGraphBs - 禁用 =====
test("T6: determineCudaGraphBs - 禁用", () => {
  const config = makeConfig({ cudaGraphBs: null, cudaGraphMaxBs: 0 });
  const result = SimGraphRunner.determineCudaGraphBs(config);
  assert.deepStrictEqual(result, []);
});

// ===== T7: canUseCudaGraph - 禁用 =====
test("T7: canUseCudaGraph - 禁用", () => {
  const config = makeConfig({ enableCudaGraph: false, cudaGraphBs: [1, 2, 4] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(1);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
});

// ===== T8: canUseCudaGraph - decode batch =====
test("T8: canUseCudaGraph - decode batch", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(2);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), true);
});

// ===== T9: canUseCudaGraph - prefill batch =====
test("T9: canUseCudaGraph - prefill batch", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makePrefillBatch(2);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
});

// ===== T10: canUseCudaGraph - bs 超限 =====
test("T10: canUseCudaGraph - bs 超限", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(5);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
});

// ===== T11: canUseCudaGraph - invalidate 后 =====
test("T11: canUseCudaGraph - invalidate 后", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(2);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), true);
  engine.simGraphRunner.invalidate();
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
});

// ===== T12: padBatch - decode batch pad 到分桶 =====
test("T12: padBatch - decode batch pad 到分桶", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(3);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 4, "bs=3 should pad to 4");
});

// ===== T13: padBatch - prefill batch 不 pad =====
test("T13: padBatch - prefill batch 不 pad", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makePrefillBatch(3);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 3, "prefill batch should not pad");
});

// ===== T14: padBatchToBs - 显式指定目标 =====
test("T14: padBatchToBs - 显式指定目标", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(5);
  engine.simGraphRunner.padBatchToBs(batch, 8);
  assert.strictEqual(batch.paddedReqs.length, 8, "should pad to targetBs=8");
});

// ===== T15: padBatchToBs - 使用 dummyReq =====
test("T15: padBatchToBs - 使用 dummyReq", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(2);
  engine.simGraphRunner.padBatchToBs(batch, 4);
  assert.strictEqual(batch.paddedReqs.length, 4);
  assert.strictEqual(batch.paddedReqs[2], engine.dummyReq, "padding should use dummyReq");
  assert.strictEqual(batch.paddedReqs[3], engine.dummyReq, "padding should use dummyReq");
});

// ===== T16: padBatchToBs - 不干扰 KV 分配计数 =====
test("T16: padBatchToBs - 不干扰 KV 分配计数", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8], cacheType: "naive" });
  const scheduler = new SimScheduler(config);
  const freeSlotsBefore = scheduler.cacheManager.freeSlots;
  const batch = makeDecodeBatch(2);
  engine: {
    const engine = scheduler.engine;
    engine.simGraphRunner.padBatchToBs(batch, 4);
  }
  const freeSlotsAfter = scheduler.cacheManager.freeSlots;
  assert.strictEqual(freeSlotsAfter, freeSlotsBefore, "padBatchToBs should not affect cacheManager freeSlots");
});

// ===== T17: graphReplayCostTicks - 基本值 =====
test("T17: graphReplayCostTicks - 基本值", () => {
  const config = makeConfig({ graphReplayCostTicks: 100, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const cost = engine.simGraphRunner.graphReplayCostTicks(1);
  // ≈ 100 × (1 + 0.05 × 1/128) = 100 × 1.000390625 = 100.039... → ceil = 101
  assert.ok(cost >= 100, "graphReplayCostTicks(bs=1) should be >= base cost");
});

// ===== T18: graphReplayCostTicks - 随 bs 增长 =====
test("T18: graphReplayCostTicks - 随 bs 增长", () => {
  const config = makeConfig({ graphReplayCostTicks: 100, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const cost128 = engine.simGraphRunner.graphReplayCostTicks(128);
  // 100 × (1 + 0.05 × 128/128) = 100 × 1.05 = 105
  assert.strictEqual(cost128, 105, "graphReplayCostTicks(bs=128) should be 105");
});

// ===== T19: graphReplayCostTicks - 大 bs =====
test("T19: graphReplayCostTicks - 大 bs", () => {
  const config = makeConfig({ graphReplayCostTicks: 100, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8, 256] });
  const engine = new MockEngine(config);
  const cost256 = engine.simGraphRunner.graphReplayCostTicks(256);
  // 100 × (1 + 0.05 × 256/128) = 100 × 1.1 = 110
  assert.strictEqual(cost256, 110, "graphReplayCostTicks(bs=256) should be 110");
});

// ===== T20: eagerForwardCostTicks - prefill =====
test("T20: eagerForwardCostTicks - prefill", () => {
  const config = makeConfig({ eagerForwardCostTicks: 10, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const cost = engine.simGraphRunner.eagerForwardCostTicks(4, 100);
  // prefill: 10 × 100 = 1000
  assert.strictEqual(cost, 1000, "eagerForwardCostTokens(bs=4, tokensPerSeq=100) should be 1000");
});

// ===== T21: eagerForwardCostTicks - decode =====
test("T21: eagerForwardCostTicks - decode", () => {
  const config = makeConfig({ eagerForwardCostTicks: 10, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const cost = engine.simGraphRunner.eagerForwardCostTicks(4, 1);
  // decode: ceil(10 × (1 + 0.1 × (4-1)/128)) = ceil(10 × 1.00234375) = ceil(10.0234375) = 11
  assert.ok(cost >= 10, "eagerForwardCostTicks decode should be >= base cost");
});

// ===== T22: estimateGraphBuffer - 空 graphBsList =====
test("T22: estimateGraphBuffer - 空 graphBsList", () => {
  const config = makeConfig({ enableCudaGraph: false, cudaGraphBs: null, cudaGraphMaxBs: 0 });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.simGraphRunner.estimateGraphBuffer(), 0, "empty graphBsList should return 0");
});

// ===== T23: estimateGraphBuffer - 正常计算 =====
test("T23: estimateGraphBuffer - 正常计算", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  // maxBs=8, hiddenSize=256, numLayers=2 → 8 × 256 × 2 × 4 = 16384
  const expected = 8 * 256 * 2 * 4;
  assert.strictEqual(engine.simGraphRunner.estimateGraphBuffer(), expected);
});

// ===== T24: estimateGraphBuffer - 与 budget.ts 一致 =====
test("T24: estimateGraphBuffer - 与 budget.ts 一致", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8, 16] });
  const engine = new MockEngine(config);
  const runnerResult = engine.simGraphRunner.estimateGraphBuffer();
  const budgetResult = estimateGraphBuffer(config.cudaGraphBs, config.modelConfig);
  assert.strictEqual(runnerResult, budgetResult, "SimGraphRunner.estimateGraphBuffer should match estimateGraphBuffer from budget.ts");
});

// ===== T25: invalidate - 标记失效 =====
test("T25: invalidate - 标记失效", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.simGraphRunner.isValid, true);
  engine.simGraphRunner.invalidate();
  assert.strictEqual(engine.simGraphRunner.isValid, false);
  const batch = makeDecodeBatch(2);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
});

// ===== T26: invalidate - destroyCudaGraphs 恢复 =====
test("T26: invalidate - destroyCudaGraphs 恢复", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  engine.simGraphRunner.invalidate();
  assert.strictEqual(engine.simGraphRunner.isValid, false);
  engine.simGraphRunner.destroyCudaGraphs();
  assert.strictEqual(engine.simGraphRunner.isValid, true);
  const batch = makeDecodeBatch(2);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), true);
});

// ===== T27: replay - 返回正确行数 =====
test("T27: replay - 返回正确行数", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(3);
  const logits = engine.simGraphRunner.replay(batch);
  assert.strictEqual(logits.length, 3, "replay should return batch.reqs.size rows (not padded)");
});

// ===== T28: replay - 返回正确列数 =====
test("T28: replay - 返回正确列数", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(2);
  const logits = engine.simGraphRunner.replay(batch);
  assert.strictEqual(logits[0].length, 128, "replay should return vocabSize columns");
});

// ===== T29: MockEngine.simGraphRunner 属性 =====
test("T29: MockEngine.simGraphRunner 属性", () => {
  const config = makeConfig();
  const engine = new MockEngine(config);
  assert.ok(engine.simGraphRunner instanceof SimGraphRunner);
  assert.ok(engine.simGraphRunner !== undefined);
});

// ===== T30: MockEngine.forward_batch - graph replay 时间 =====
test("T30: MockEngine.forward_batch - graph replay 时间", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8], graphReplayCostTicks: 100 });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(4);
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(output.isGraphCapture, true);
  // graphReplayCostTicks(4) = ceil(100 × (1 + 0.05 × 4/128)) = ceil(100 × 1.0015625) = ceil(100.15625) = 101
  assert.strictEqual(output.decodeBatchTime, 101, "decodeBatchTime should use graphReplayCostTicks formula");
});

// ===== T31: MockEngine.forward_batch - eager 时间 =====
test("T31: MockEngine.forward_batch - eager 时间", () => {
  const config = makeConfig({ enableCudaGraph: false, eagerForwardCostTicks: 10 });
  const engine = new MockEngine(config);
  const batch = makePrefillBatch(4, 3);
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(output.isGraphCapture, false);
  // eagerForwardCostTicks(4, 3) = 10 × 3 = 30 (prefill path)
  assert.strictEqual(output.prefillBatchTime, 30, "prefillBatchTime should use eagerForwardCostTicks prefill formula");
});

// ===== T32: MockEngine.forward_batch - isGraphCapture 标识 =====
test("T32: MockEngine.forward_batch - isGraphCapture 标识", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const decodeBatch = makeDecodeBatch(2);
  const decodeArgs = engine.mockSampler.prepare(decodeBatch);
  const decodeOutput = engine.forward_batch(decodeBatch, decodeArgs);
  assert.strictEqual(decodeOutput.isGraphCapture, true, "decode batch should use SimGraphRunner for isGraphCapture");

  const prefillBatch = makePrefillBatch(2);
  const prefillArgs = engine.mockSampler.prepare(prefillBatch);
  const prefillOutput = engine.forward_batch(prefillBatch, prefillArgs);
  assert.strictEqual(prefillOutput.isGraphCapture, false, "prefill batch should have isGraphCapture=false");
});

// ===== T33: SimScheduler._prepareBatch 使用 simGraphRunner =====
test("T33: SimScheduler._prepareBatch 使用 simGraphRunner", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8], mockSampleMode: "greedy" });
  const scheduler = new SimScheduler(config);
  const msgs: SimRequestMsg[] = [
    { tag: "req_in", uid: 1, inputIds: [1, 2, 3], samplingParams: null, outputLen: 2 },
  ];
  // Run one tick to trigger _prepareBatch
  const resp = scheduler.runTick(msgs);
  assert.ok(resp.length >= 1, "scheduler should produce responses");
});

// ===== T34: 分桶边界 bs=31→32 =====
test("T34: 分桶边界 bs=31→32", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8, 16, 24, 32], maxRunningReq: 64, numPages: 1024 });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(31);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), true, "bs=31 should be <= maxGraphBs=32");
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 32, "bs=31 should pad to 32");
});

// ===== T35: eager 与 graph 切换一致 =====
test("T35: eager 与 graph 切换一致", () => {
  const configOn = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const configOff = makeConfig({ enableCudaGraph: false });
  const engineOn = new MockEngine(configOn);
  const engineOff = new MockEngine(configOff);

  const batch = makeDecodeBatch(2);
  // Both should handle the batch — isGraphCapture differs but both produce valid output
  const argsOn = engineOn.mockSampler.prepare(batch);
  const argsOff = engineOff.mockSampler.prepare(batch);
  const outputOn = engineOn.forward_batch(batch, argsOn);
  // Reset batch for second forward
  const batch2 = makeDecodeBatch(2);
  const argsOff2 = engineOff.mockSampler.prepare(batch2);
  const outputOff = engineOff.forward_batch(batch2, argsOff2);

  assert.strictEqual(outputOn.isGraphCapture, true);
  assert.strictEqual(outputOff.isGraphCapture, false);
  // Both produce valid decodeBatchTime
  assert.ok(outputOn.decodeBatchTime! > 0);
  assert.ok(outputOff.decodeBatchTime! > 0);
});

// ===== T36: destroyCudaGraphs 为 noop =====
test("T36: destroyCudaGraphs 为 noop", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  // Should not throw
  engine.simGraphRunner.destroyCudaGraphs();
  assert.strictEqual(engine.simGraphRunner.isValid, true);
});

// ===== 边界条件测试 =====

// ===== B1: cudaGraphBs 为空列表 =====
test("B1: cudaGraphBs 为空列表", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [] });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.simGraphRunner.maxGraphBs, 0, "empty list → maxGraphBs=0");
  const batch = makeDecodeBatch(1);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false, "empty list → canUseCudaGraph=false");
});

// ===== B2: bs=0 的空 batch =====
test("B2: bs=0 的空 batch", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = new Batch();
  batch.numDecodeTokens = 0;
  batch.extendInputTokens = 0;
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 0, "empty batch should not pad");
});

// ===== B3: bs 恰好等于分桶值 =====
test("B3: bs 恰好等于分桶值", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(4);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 4, "bs=4 should not add dummy when bucket=4");
});

// ===== B4: bs=1 的 decode batch =====
test("B4: bs=1 的 decode batch", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(1);
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), true);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 1, "bs=1 should pad to bucket 1");
});

// ===== B5: chunked prefill batch =====
test("B5: chunked prefill batch", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makePrefillBatch(2);
  batch.extendInputTokens = 6;
  batch.numDecodeTokens = 0;
  assert.strictEqual(engine.simGraphRunner.canUseCudaGraph(batch), false, "chunked prefill should not use CUDA Graph");
});

// ===== B6: graphReplayCostTicks=0 =====
test("B6: graphReplayCostTicks=0", () => {
  const config = makeConfig({ graphReplayCostTicks: 0, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.simGraphRunner.graphReplayCostTicks(4), 0);
});

// ===== B7: eagerForwardCostTicks=0 =====
test("B7: eagerForwardCostTicks=0", () => {
  const config = makeConfig({ eagerForwardCostTicks: 0, enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.simGraphRunner.eagerForwardCostTicks(4, 1), 0);
  assert.strictEqual(engine.simGraphRunner.eagerForwardCostTicks(4, 100), 0);
});

// ===== B8: 多次 padBatch 调用 =====
test("B8: 多次 padBatch 调用", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  const batch = makeDecodeBatch(3);
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 4, "first pad: 3→4");
  engine.simGraphRunner.padBatch(batch);
  assert.strictEqual(batch.paddedReqs.length, 4, "second pad: should recalculate to 4 again, not accumulate");
});

// ===== B9: 连续多次 invalidate =====
test("B9: 连续多次 invalidate", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8] });
  const engine = new MockEngine(config);
  engine.simGraphRunner.invalidate();
  engine.simGraphRunner.invalidate();
  engine.simGraphRunner.invalidate();
  assert.strictEqual(engine.simGraphRunner.isValid, false, "multiple invalidate should keep isValid=false");
});

// ===== B10: invalidate 后 forward_batch 走 eager =====
test("B10: invalidate 后 forward_batch 走 eager", () => {
  const config = makeConfig({ enableCudaGraph: true, cudaGraphBs: [1, 2, 4, 8], graphReplayCostTicks: 100, eagerForwardCostTicks: 10 });
  const engine = new MockEngine(config);
  engine.simGraphRunner.invalidate();
  const batch = makeDecodeBatch(2);
  const sampleArgs = engine.mockSampler.prepare(batch);
  const output = engine.forward_batch(batch, sampleArgs);
  assert.strictEqual(output.isGraphCapture, false, "after invalidate, should not use CUDA Graph");
  // eagerForwardCostTicks(2, 1) = ceil(10 × (1 + 0.1 × 1/128)) = ceil(10.0078125) = 11
  assert.strictEqual(output.decodeBatchTime, 11, "after invalidate, decode should use eagerForwardCostTicks");
});

// ===== 结果汇总 =====
console.log(`\n=== S4 Test Results: ${passed} passed, ${failed} failed ===`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) {
    console.log("  - " + f);
  }
  process.exit(1);
}
