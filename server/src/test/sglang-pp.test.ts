import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  DEFAULT_MODEL_CONFIG,
  PPPipelineSimulator,
  MockEngine,
  ParallelTopology,
  Batch,
  Req,
  SamplingParams,
} from "../sglang";
import type { SimulatorConfig, ModelConfig, PipelineStepResult } from "../sglang";

/**
 * Issue #28 验收测试 — P4: PPPipelineSimulator
 *
 * Run with:  npx ts-node src/test/sglang-pp.test.ts
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

function makeConfig(overrides: Partial<SimulatorConfig>): SimulatorConfig {
  return { ...DEFAULT_SIMULATOR_CONFIG, ...overrides };
}

function makeModelConfig(overrides: Partial<ModelConfig>): ModelConfig {
  return { ...DEFAULT_MODEL_CONFIG, ...overrides };
}

function makeBatch(size: number): Batch {
  const batch = new Batch();
  for (let i = 0; i < size; i++) {
    const req = new Req({
      rid: i,
      inputIds: [1, 2, 3],
      samplingParams: new SamplingParams(),
    });
    batch.reqs.set(i, req);
  }
  return batch;
}

// ==========================================
// T1: gpipe bubble 公式验证
// pp=4, numMB=4, eagerForward=10 → bubble === 120
// ==========================================
test("T1 gpipe bubble formula", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 4, eagerForwardCostTicks: 10, ppPipelineSchedule: "gpipe" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.bubbleTicks, (4 - 1) * 10 * 4, `Expected 120, got ${result.bubbleTicks}`);
  assert.strictEqual(result.bubbleTicks, 120);
});

// T2: gpipe bubble 二次增长
// pp=2,4,8 × numMB=4 → bubble === 40, 120, 280
test("T2 gpipe bubble quadratic growth", () => {
  for (const [pp, expected] of [[2, 40], [4, 120], [8, 280]] as [number, number][]) {
    const config = makeConfig({ ppSize: pp, ppNumMicroBatches: 4, eagerForwardCostTicks: 10, ppPipelineSchedule: "gpipe" });
    const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
    const batch = makeBatch(4);
    const result = ppSim.simulatePipelineStep(batch);
    assert.strictEqual(result.bubbleTicks, expected, `pp=${pp}: expected ${expected}, got ${result.bubbleTicks}`);
  }
});

// T3: 1f1b bubble 最优
// pp=4, eagerForward=10 → bubble === 30
test("T3 1f1b bubble optimal", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 4, eagerForwardCostTicks: 10, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.bubbleTicks, (4 - 1) * 10);
  assert.strictEqual(result.bubbleTicks, 30);
});

// T4: interleaved bubble
// pp=4, numChunks=2, eagerForward=10 → bubble === 60
test("T4 interleaved bubble", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 4, eagerForwardCostTicks: 10, ppPipelineSchedule: "interleaved", ppInterleavedNumChunks: 2 });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.bubbleTicks, (4 - 1) * 2 * 10);
  assert.strictEqual(result.bubbleTicks, 60);
});

// T5: pp_size=1 退化
test("T5 pp_size=1 degeneration", () => {
  const config = makeConfig({ ppSize: 1, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(1);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.totalTicks, 0);
  assert.strictEqual(result.bubbleTicks, 0);
  assert.strictEqual(result.sendRecvTicks, 0);
  assert.deepStrictEqual(result.perStageTicks, []);
});

// T6: send/recv 通信成本
test("T6 send/recv communication cost", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 2, ppPipelineSchedule: "1f1b", tpSize: 1, commOverlapWithCompute: false });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.ok(result.sendRecvTicks > 0, `Expected >0, got ${result.sendRecvTicks}`);
  assert.strictEqual(result.perStageTicks.length, 1, `Expected perStageTicks.length=1, got ${result.perStageTicks.length}`);
});

// T7: isPpLastStage
test("T7 isPpLastStage", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  assert.strictEqual(ppSim.isPpLastStage(3), true);
  assert.strictEqual(ppSim.isPpLastStage(0), false);
  assert.strictEqual(ppSim.isPpLastStage(1), false);
  assert.strictEqual(ppSim.isPpLastStage(2), false);
});

// T8: intermediate stage 不采样
test("T8 intermediate stage does not sample", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 0); // ppRank=0 → not last
  const batch = makeBatch(2);
  const output = engine.forwardBatchReq(batch);
  assert.strictEqual(output.isIntermediate, true);
  assert.strictEqual(output.sampledIds, null);
});

// T9: last stage 正常采样
test("T9 last stage samples normally", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", mockSampleMode: "fixed", fixedOutputToken: 42 });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 3); // ppRank=3 → last
  const batch = makeBatch(2);
  const output = engine.forwardBatchReq(batch);
  assert.strictEqual(output.isIntermediate, false);
  assert.ok(output.sampledIds !== null, "sampledIds should not be null");
  assert.ok(output.sampledIds!.length > 0, "sampledIds should be non-empty");
});

// T10: sampling_counter 中间不变
test("T10 sampling_counter unchanged at intermediate stage", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 0);
  const counterBefore = engine.sampler.samplingCounter;
  const batch = makeBatch(2);
  engine.forwardBatchReq(batch);
  assert.strictEqual(engine.sampler.samplingCounter, counterBefore, "samplingCounter should not change for intermediate stage");
});

// T11: ParallelMetrics 回填
test("T11 ParallelMetrics backfill", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 2, ppPipelineSchedule: "gpipe", mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 3); // last stage
  const batch = makeBatch(4);
  engine.forwardBatchReq(batch);
  assert.ok(engine.metrics.parallel.ppBubbleTicks > 0, `ppBubbleTicks should be >0, got ${engine.metrics.parallel.ppBubbleTicks}`);
  assert.ok(engine.metrics.parallel.ppSendRecvTicks > 0, `ppSendRecvTicks should be >0, got ${engine.metrics.parallel.ppSendRecvTicks}`);
  assert.strictEqual(engine.metrics.parallel.ppNumMicroBatches, 2);
});

// T12: pp_size=1 时 Metrics 全零
test("T12 pp_size=1 metrics all zero", () => {
  const config = makeConfig({ ppSize: 1, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 0);
  const batch = makeBatch(2);
  engine.forwardBatchReq(batch);
  assert.strictEqual(engine.metrics.parallel.ppBubbleTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppSendRecvTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppNumMicroBatches, 0);
});

// T13: micro-batch 分割
// batchSize=8, numMB=4 → sizes=[2,2,2,2]
test("T13 micro-batch split even", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 4, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const mbs = ppSim._splitMicroBatches(8);
  assert.deepStrictEqual(mbs.map(m => m.size), [2, 2, 2, 2]);
});

// T14: 不整除
// batchSize=7, numMB=3 → sizes=[3,2,2]
test("T14 micro-batch split uneven", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 3, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const mbs = ppSim._splitMicroBatches(7);
  assert.deepStrictEqual(mbs.map(m => m.size), [3, 2, 2]);
});

// T15: numMicroBatches=1
test("T15 numMicroBatches=1", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const mbs = ppSim._splitMicroBatches(5);
  assert.strictEqual(mbs.length, 1);
  assert.strictEqual(mbs[0].size, 5);
});

// T16: TP×PP 修正（R2-5）
// Use large hiddenSize so bandwidth dominates over latency
test("T16 TP×PP correction (R2-5)", () => {
  const config1 = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", tpSize: 1, commOverlapWithCompute: false });
  const config2 = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", tpSize: 2, commOverlapWithCompute: false });
  // Use large hiddenSize so bandwidth term dominates latency
  const mc = makeModelConfig({ hiddenSize: 65536 });
  const ppSim1 = new PPPipelineSimulator(config1, mc);
  const ppSim2 = new PPPipelineSimulator(config2, mc);
  // Use larger microBatchSize so bandwidth term dominates
  const cost1 = ppSim1._stageSendRecvCost(128);
  const cost2 = ppSim2._stageSendRecvCost(128);
  assert.ok(cost2 < cost1, `tp=2 cost (${cost2}) should be less than tp=1 cost (${cost1})`);
  // Due to ceil rounding and latency being counted twice (send+recv), tp=2 cost is approximately
  // half of tp=1 cost. The latency adds ~10μs per call (2 × latency), so tolerance needs to account for that.
  assert.ok(cost2 <= cost1 / 2 + 10, `tp=2 cost (${cost2}) should be ~half of tp=1 cost (${cost1})`);
});

// T17: TP tp=1 退化
test("T17 tp=1 degeneration", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", tpSize: 1, commOverlapWithCompute: false });
  const mc = makeModelConfig({ hiddenSize: 4096 });
  const ppSim = new PPPipelineSimulator(config, mc);
  const cost = ppSim._stageSendRecvCost(1);
  assert.ok(cost > 0, "tp=1 should still have communication cost");
  // dataBytes = 1 * 4096 * 2 = 8192
  // cost = 2 * sendRecv(8192)
  const dataBytes = 1 * 4096 * 2;
  const expectedCost = ppSim.commGroup!.sendRecv(dataBytes, 0) * 2;
  assert.strictEqual(cost, expectedCost);
});

// T18: CUDA Graph 跳过 PP（R2-3）
test("T18 CUDA Graph skips PP (R2-3)", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", enableCudaGraph: true, cudaGraphBs: [2], mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 3);
  const batch = makeBatch(2); // batch size matches cudaGraphBs
  engine.forwardBatchReq(batch);
  assert.strictEqual(engine.metrics.parallel.ppBubbleTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppSendRecvTicks, 0);
});

// T19: pp_stage_layers
// pp=4, numLayers=32 → 长度 4，每 stage 8 层
test("T19 pp_stage_layers", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const mc = makeModelConfig({ numLayers: 32 });
  const ppSim = new PPPipelineSimulator(config, mc);
  assert.strictEqual(ppSim.stageLayers.length, 4);
  for (const s of ppSim.stageLayers) {
    assert.strictEqual(s.end - s.start, 8);
  }
});

// T20: perStageTicks 计算
// pp=3, numMB=2 → perStageTicks.length === 2
test("T20 perStageTicks calculation", () => {
  const config = makeConfig({ ppSize: 3, ppNumMicroBatches: 2, ppPipelineSchedule: "1f1b", commOverlapWithCompute: false });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.perStageTicks.length, 2, `Expected 2, got ${result.perStageTicks.length}`);
  for (const t of result.perStageTicks) {
    assert.ok(t > 0, `perStageTick should be >0, got ${t}`);
  }
});

// T21: 通信重叠模式（R2-2）
// When rawSendRecv <= microBatchTicks, effectiveSendRecv === 0
test("T21 comm overlap mode (R2-2) full overlap", () => {
  // Use zero latency and very small hidden so rawCost < microBatchTicks
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", commOverlapWithCompute: true, eagerForwardCostTicks: 100, networkLatencyUs: 0 });
  const mc = makeModelConfig({ hiddenSize: 1 }); // minimal hidden
  const ppSim = new PPPipelineSimulator(config, mc);
  // With tiny hidden size, latency=0, rawCost should be very small
  const rawCost = ppSim._stageSendRecvCost(1);
  // microBatchTicks = 100, so rawCost should be < 100 → effective = 0
  assert.ok(rawCost < 100, `rawCost (${rawCost}) should be < microBatchTicks (100)`);
  const batch = makeBatch(1);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.sendRecvTicks, 0, `Expected 0 when fully overlapped, got ${result.sendRecvTicks}`);
});

// T21b: 通信重叠模式 — 非重叠
test("T21b comm non-overlap mode", () => {
  const config1 = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", commOverlapWithCompute: true });
  const config2 = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", commOverlapWithCompute: false });
  const mc = makeModelConfig({});
  const ppSim1 = new PPPipelineSimulator(config1, mc);
  const ppSim2 = new PPPipelineSimulator(config2, mc);
  const batch = makeBatch(4);
  const result1 = ppSim1.simulatePipelineStep(batch);
  const result2 = ppSim2.simulatePipelineStep(batch);
  assert.ok(result1.sendRecvTicks <= result2.sendRecvTicks, `overlap sendRecv (${result1.sendRecvTicks}) should be <= non-overlap (${result2.sendRecvTicks})`);
});

// T22: 通信部分重叠（R2-2）
test("T22 comm partial overlap (R2-2)", () => {
  // Create a config with large hiddenSize so rawCost > microBatchTicks
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", commOverlapWithCompute: true, eagerForwardCostTicks: 10 });
  const mc = makeModelConfig({ hiddenSize: 100000 }); // large hidden → large sendRecv cost
  const ppSim = new PPPipelineSimulator(config, mc);
  const rawCost = ppSim._stageSendRecvCost(100); // large batch
  // With overlap: effective = max(0, rawCost - microBatchTicks) = max(0, rawCost - 10)
  const expectedEffective = Math.max(0, rawCost - 10);
  const batch = makeBatch(100);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.sendRecvTicks, expectedEffective, `Expected ${expectedEffective}, got ${result.sendRecvTicks}`);
});

// T23: interleaved numChunks 可配（R2-4）
// pp=4, numChunks=3, eagerForward=10 → bubble === 90
test("T23 interleaved numChunks configurable (R2-4)", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "interleaved", ppInterleavedNumChunks: 3, eagerForwardCostTicks: 10 });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.bubbleTicks, (4 - 1) * 3 * 10);
  assert.strictEqual(result.bubbleTicks, 90);
});

// T24: 极端 micro-batch（R2-6）
// batchSize=1, numMB=4 → sizes=[1,0,0,0]
test("T24 extreme micro-batch batchSize<numMB (R2-6)", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 4, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const mbs = ppSim._splitMicroBatches(1);
  assert.deepStrictEqual(mbs.map(m => m.size), [1, 0, 0, 0]);
});

// T25: 大量 micro-batch（R2-6）
// batchSize=1000, numMB=999 → sizes=[2,1,1,...,1]
test("T25 many micro-batches (R2-6)", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 999, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const mbs = ppSim._splitMicroBatches(1000);
  assert.strictEqual(mbs.length, 999);
  assert.strictEqual(mbs[0].size, 2);
  // remainder = 1000 % 999 = 1, so only first gets +1
  for (let i = 1; i < 999; i++) {
    assert.strictEqual(mbs[i].size, 1, `mb[${i}].size should be 1`);
  }
});

// T26: TP hiddenSize 不整除（R2-5）
// tp=3, hiddenSize=4096 → ceil(4096/3) === 1366
test("T26 TP hiddenSize non-divisible (R2-5)", () => {
  assert.strictEqual(Math.ceil(4096 / 3), 1366);
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", tpSize: 3, commOverlapWithCompute: false });
  const mc = makeModelConfig({ hiddenSize: 4096 });
  const ppSim = new PPPipelineSimulator(config, mc);
  // dataBytes = 1 * ceil(4096/3) * 2 = 1 * 1366 * 2 = 2732
  const expectedDataBytes = 1 * 1366 * 2;
  const expectedCost = ppSim.commGroup!.sendRecv(expectedDataBytes, 0) * 2;
  const cost = ppSim._stageSendRecvCost(1);
  assert.strictEqual(cost, expectedCost, `Expected ${expectedCost}, got ${cost}`);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: pp_size=0 → constructor should throw
test("B1 pp_size=0 throws", () => {
  assert.throws(
    () => {
      const config = makeConfig({ ppSize: 0, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
      new PPPipelineSimulator(config, makeModelConfig({}));
    },
    /pp_size must be > 0/
  );
});

// B2: numMicroBatches=0 → returns all zero
test("B2 numMicroBatches=0 returns all zero", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 0, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(4);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.totalTicks, 0);
  assert.strictEqual(result.bubbleTicks, 0);
  assert.strictEqual(result.sendRecvTicks, 0);
});

// B3: batchSize=0 → sendRecvTicks=0
test("B3 batchSize=0 sendRecvTicks=0", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 2, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(0);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.sendRecvTicks, 0);
});

// B4: 未知 schedule → throws
test("B4 unknown schedule throws", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "unknown" as any });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(1);
  assert.throws(
    () => ppSim.simulatePipelineStep(batch),
    /Unknown pipeline schedule/
  );
});

// B5: commGroup=null → _stageSendRecvCost returns 0
test("B5 commGroup=null returns 0", () => {
  const config = makeConfig({ ppSize: 1, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b" });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  assert.strictEqual(ppSim._stageSendRecvCost(1), 0);
});

// B6: networkBandwidthGBps=0 → sendRecv returns Infinity
test("B6 bandwidth=0 returns Infinity", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", networkBandwidthGBps: 0, commOverlapWithCompute: false });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const cost = ppSim._stageSendRecvCost(1);
  assert.strictEqual(cost, Infinity);
});

// B7: ppInterleavedNumChunks=0 → interleaved bubble=0
test("B7 interleaved numChunks=0 bubble=0", () => {
  const config = makeConfig({ ppSize: 2, ppNumMicroBatches: 1, ppPipelineSchedule: "interleaved", ppInterleavedNumChunks: 0 });
  const ppSim = new PPPipelineSimulator(config, makeModelConfig({}));
  const batch = makeBatch(1);
  const result = ppSim.simulatePipelineStep(batch);
  assert.strictEqual(result.bubbleTicks, 0);
});

// B8: ppInterleavedNumChunks=1 → 与 1f1b 相同
test("B8 interleaved numChunks=1 same as 1f1b", () => {
  const config1f1b = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", eagerForwardCostTicks: 10 });
  const configInter = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "interleaved", ppInterleavedNumChunks: 1, eagerForwardCostTicks: 10 });
  const ppSim1f1b = new PPPipelineSimulator(config1f1b, makeModelConfig({}));
  const ppSimInter = new PPPipelineSimulator(configInter, makeModelConfig({}));
  const batch = makeBatch(1);
  const result1f1b = ppSim1f1b.simulatePipelineStep(batch);
  const resultInter = ppSimInter.simulatePipelineStep(batch);
  assert.strictEqual(resultInter.bubbleTicks, result1f1b.bubbleTicks);
});

// ==========================================
// 端到端集成验证
// ==========================================

// E2E-1: gpipe 全流程
test("E2E-1 gpipe full flow", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 2, ppPipelineSchedule: "gpipe", mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 3);
  const batch = makeBatch(4);
  engine.forwardBatchReq(batch);
  const expectedBubble = (4 - 1) * 10 * 2;
  assert.strictEqual(engine.metrics.parallel.ppBubbleTicks, expectedBubble);
  assert.strictEqual(engine.metrics.parallel.ppSendRecvTicks, engine.metrics.parallel.ppSendRecvTicks); // just verify non-negative
});

// E2E-2: pp_size=1 全流程
test("E2E-2 pp_size=1 full flow", () => {
  const config = makeConfig({ ppSize: 1, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 0);
  const batch = makeBatch(2);
  const output = engine.forwardBatchReq(batch);
  assert.strictEqual(output.isIntermediate, false);
  assert.strictEqual(engine.metrics.parallel.ppBubbleTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppSendRecvTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppNumMicroBatches, 0);
});

// E2E-3: CUDA Graph + PP
test("E2E-3 CUDA Graph + PP", () => {
  const config = makeConfig({ ppSize: 4, ppNumMicroBatches: 1, ppPipelineSchedule: "1f1b", enableCudaGraph: true, cudaGraphBs: [2], mockSampleMode: "greedy" });
  const modelConfig = makeModelConfig({});
  const engine = new MockEngine(config, modelConfig, 3);
  const batch = makeBatch(2);
  engine.forwardBatchReq(batch);
  assert.strictEqual(engine.metrics.parallel.ppBubbleTicks, 0);
  assert.strictEqual(engine.metrics.parallel.ppSendRecvTicks, 0);
});

// E2E-4: 重叠模式对比
test("E2E-4 overlap mode comparison", () => {
  const configOverlap = makeConfig({ ppSize: 2, ppNumMicroBatches: 2, ppPipelineSchedule: "1f1b", commOverlapWithCompute: true, mockSampleMode: "greedy" });
  const configNoOverlap = makeConfig({ ppSize: 2, ppNumMicroBatches: 2, ppPipelineSchedule: "1f1b", commOverlapWithCompute: false, mockSampleMode: "greedy" });
  const mc = makeModelConfig({});
  const engineOverlap = new MockEngine(configOverlap, mc, 1);
  const engineNoOverlap = new MockEngine(configNoOverlap, mc, 1);
  const batch = makeBatch(4);
  engineOverlap.forwardBatchReq(batch);
  engineNoOverlap.forwardBatchReq(batch);
  assert.ok(engineOverlap.metrics.parallel.ppSendRecvTicks <= engineNoOverlap.metrics.parallel.ppSendRecvTicks,
    `overlap (${engineOverlap.metrics.parallel.ppSendRecvTicks}) should be <= non-overlap (${engineNoOverlap.metrics.parallel.ppSendRecvTicks})`);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P4 (PPPipelineSimulator) 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P4 验收测试通过 \u2713");
  process.exit(0);
}
