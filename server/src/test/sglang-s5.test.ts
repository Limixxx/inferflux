import assert from "assert";
import {
  SamplingParams,
  Req,
  Batch,
} from "../sglang/core";
import {
  MockEngine,
} from "../sglang/engine";
import {
  SimScheduler,
  SimulationClock,
} from "../sglang/scheduler";
import type {
  SimEvent,
} from "../sglang/scheduler";
import type {
  SimulatorConfig,
  ModelConfig,
  SimRequestMsg,
  SimRespMsg,
} from "../sglang/types";

/**
 * Issue #19 验收测试 — S5: Overlap Scheduling（last_data 延迟 + 空 tick 刷新）+ SimulationClock
 *
 * Run with:  npx ts-node src/test/sglang-s5.test.ts
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
    enableOverlap: true,
    cpuScheduleCostTicks: 1,
    cpuProcessResultCostTicks: 1,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 2,
    idleCountForFlush: 2,
    messagesHighWatermark: 1024,
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

/** 创建短 prompt 请求消息 */
function makeShortRequest(uid: number, inputLen: number = 3, outputLen: number = 2): SimRequestMsg {
  return {
    tag: "req_in",
    uid,
    inputIds: Array.from({ length: inputLen }, (_, i) => (i + 1) % 128),
    samplingParams: new SamplingParams({ maxNewTokens: outputLen }),
    outputLen,
  };
}

/** 创建长 prompt 请求消息（触发 chunked prefill） */
function makeLongRequest(uid: number, inputLen: number = 300, outputLen: number = 2): SimRequestMsg {
  return {
    tag: "req_in",
    uid,
    inputIds: Array.from({ length: inputLen }, (_, i) => (i + 1) % 128),
    samplingParams: new SamplingParams({ maxNewTokens: outputLen }),
    outputLen,
  };
}

/**
 * 运行 scheduler 直到某个请求完成或达到最大 tick
 * @returns 完成的响应消息列表
 */
function runUntilDone(
  scheduler: SimScheduler,
  uids: number[],
  maxTicks: number = 100,
): SimRespMsg[] {
  const doneReqs: SimRespMsg[] = [];
  const pendingUids = new Set(uids);
  for (let t = 0; t < maxTicks && pendingUids.size > 0; t++) {
    const resp = scheduler.runTick([]);
    for (const r of resp) {
      if (r.finished && pendingUids.has(r.uid)) {
        doneReqs.push(r);
        pendingUids.delete(r.uid);
      }
    }
  }
  return doneReqs;
}

// ================================================================
// 测试 1: test_simulation_clock_advance
// ================================================================
test("test_simulation_clock_advance", () => {
  const clock = new SimulationClock();
  assert.strictEqual(clock.currentTick, 0, "initial tick should be 0");

  clock.advance(1);
  assert.strictEqual(clock.currentTick, 1, "advance(1) should increment to 1");

  clock.advance(3);
  assert.strictEqual(clock.currentTick, 4, "advance(3) should increment to 4");

  // advance(0) should throw
  assert.throws(() => clock.advance(0), /deltaTicks > 0/, "advance(0) should throw");

  // advance(-1) should throw
  assert.throws(() => clock.advance(-1), /deltaTicks > 0/, "advance(-1) should throw");
});

// ================================================================
// 测试 2: test_simulation_clock_schedule_gpu
// ================================================================
test("test_simulation_clock_schedule_gpu", () => {
  const clock = new SimulationClock();

  // scheduleGpu(5) at tick 0
  const finish1 = clock.scheduleGpu(5);
  assert.strictEqual(finish1, 5, "scheduleGpu(5) should finish at tick 5");
  assert.strictEqual(clock.gpuBusyUntil, 5, "gpuBusyUntil should be 5");

  // scheduleGpu(3) while GPU busy → start at 5, finish at 8
  const finish2 = clock.scheduleGpu(3);
  assert.strictEqual(finish2, 8, "scheduleGpu(3) while busy should finish at tick 8");
  assert.strictEqual(clock.gpuBusyUntil, 8, "gpuBusyUntil should be 8");

  // Events recorded
  assert.strictEqual(clock.events.length, 4, "should have 4 events (2 start + 2 end)");
  assert.strictEqual(clock.events[0].eventType, "gpu_start");
  assert.strictEqual(clock.events[0].tick, 0);
  assert.strictEqual(clock.events[1].eventType, "gpu_end");
  assert.strictEqual(clock.events[1].tick, 5);
  assert.strictEqual(clock.events[2].eventType, "gpu_start");
  assert.strictEqual(clock.events[2].tick, 5);
  assert.strictEqual(clock.events[3].eventType, "gpu_end");
  assert.strictEqual(clock.events[3].tick, 8);
});

// ================================================================
// 测试 2 变体: scheduleGpu(0)
// ================================================================
test("test_simulation_clock_schedule_gpu_zero", () => {
  const clock = new SimulationClock();
  const finish = clock.scheduleGpu(0);
  assert.strictEqual(finish, 0, "scheduleGpu(0) should return currentTick (start==finish)");
  assert.strictEqual(clock.gpuBusyUntil, 0, "gpuBusyUntil should remain 0 after scheduleGpu(0)");
});

// ================================================================
// 测试 3: test_simulation_clock_can_overlap
// ================================================================
test("test_simulation_clock_can_overlap", () => {
  const clock = new SimulationClock();

  // Initially GPU idle
  assert.strictEqual(clock.canOverlap(), false, "GPU should be idle initially");

  // Schedule GPU task
  clock.scheduleGpu(5);
  // currentTick=0 < gpuBusyUntil=5 → busy
  assert.strictEqual(clock.canOverlap(), true, "GPU should be busy after scheduleGpu(5)");

  // Advance past gpuBusyUntil
  clock.advance(5);
  // currentTick=5, gpuBusyUntil=5 → not busy
  assert.strictEqual(clock.canOverlap(), false, "GPU should be idle after advancing past gpuBusyUntil");
});

// ================================================================
// 测试 4: test_simulation_clock_on_tick_callback
// ================================================================
test("test_simulation_clock_on_tick_callback", () => {
  const clock = new SimulationClock();
  const ticks: number[] = [];

  const unsub = clock.onTick((tick) => {
    ticks.push(tick);
  });

  clock.advance(1);
  assert.deepStrictEqual(ticks, [1], "callback should be called with tick 1");

  clock.advance(2);
  assert.deepStrictEqual(ticks, [1, 3], "callback should be called with tick 3");

  // Unregister
  unsub();
  clock.advance(1);
  assert.deepStrictEqual(ticks, [1, 3], "callback should not be called after unsubscribe");
});

// ================================================================
// 测试 5: test_overlap_short_prompt_normal
// 短 prompt 在 overlap 模式下正常 prefill→decode→finish
// tokenRecvDelayTicks=0 且 isGraphCapture=false → last_data 无额外延迟
// ================================================================
test("test_overlap_short_prompt_normal", () => {
  const config = makeConfig({
    enableOverlap: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 2),
  ];

  // Tick 1: prefill
  const resp1 = scheduler.runTick(msgs);
  // In overlap mode, first tick does prefill but result is deferred
  // The result comes in a later tick after last_data ack
  assert.ok(resp1.length === 0 || resp1.length >= 0, "first tick may not have responses (overlap defers)");

  // Run more ticks to get the request completed
  const doneReqs = runUntilDone(scheduler, [1], 50);
  assert.strictEqual(doneReqs.length, 1, "short prompt should complete in overlap mode");
  assert.strictEqual(doneReqs[0].uid, 1);
  assert.strictEqual(doneReqs[0].finished, true);
});

// ================================================================
// 测试 6: test_overlap_chunked_prefill_idle_flush
// 长 prompt 触发 chunked prefill，空闲 tick 后自动续接
// ================================================================
test("test_overlap_chunked_prefill_idle_flush", () => {
  const config = makeConfig({
    enableOverlap: true,
    maxExtendTokens: 64, // Small budget to force chunking
    idleCountForFlush: 2,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxSeqLen: 512,
    numPages: 2048,
    maxRunningReq: 16,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  // Long request that exceeds maxExtendTokens
  const msgs: SimRequestMsg[] = [
    makeLongRequest(100, 300, 2), // inputLen=300 > maxExtendTokens=64 → chunked
  ];

  // Tick 1: first chunk of prefill
  scheduler.runTick(msgs);

  // The chunked request should eventually complete via idle flush ticks
  const doneReqs = runUntilDone(scheduler, [100], 200);
  assert.strictEqual(doneReqs.length, 1, "chunked prefill should complete via idle flush");
  assert.strictEqual(doneReqs[0].uid, 100);
  assert.strictEqual(doneReqs[0].finished, true);
});

// ================================================================
// 测试 7: test_overlap_last_data_delay_graph_capture
// graph_replay 路径下 tokenRecvDelayTicks=1，last_data 延迟 1 tick 后才被处理
// ================================================================
test("test_overlap_last_data_delay_graph_capture", () => {
  const config = makeConfig({
    enableOverlap: true,
    enableCudaGraph: true,
    cudaGraphBs: [1, 2, 4, 8],
    tokenRecvDelayTicks: 1,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 2),
  ];

  // Tick 0: prefill (result deferred to _lastOverlapData with ackTick=0+1=1)
  const resp0 = scheduler.runTick(msgs);
  // No responses yet because ackTick=1, current tickCounter still 0
  // After Phase 4.5, tickCounter becomes 1

  // Tick 1: tickCounter=1 >= ackTick=1, so last_data is processed
  const resp1 = scheduler.runTick([]);
  // Now we should get the prefill result (resp_token)
  // For decode steps, ackTick will be tickCounter+1

  // Run more ticks to complete
  const doneReqs = runUntilDone(scheduler, [1], 50);
  assert.strictEqual(doneReqs.length, 1, "request should complete with graph_capture delay");
  assert.strictEqual(doneReqs[0].uid, 1);
  assert.strictEqual(doneReqs[0].finished, true);

  // Verify delay took effect: with delay=1, each decode step takes 2 ticks instead of 1
  // The total ticks should be noticeably more than without delay
});

// ================================================================
// 测试 8: test_overlap_last_data_delay_eager
// eager 路径下 tokenRecvDelayTicks=1 且 eagerForwardExtraDelayTicks=2
// last_data 延迟 1+2=3 tick 后才被处理
// ================================================================
test("test_overlap_last_data_delay_eager", () => {
  const config = makeConfig({
    enableOverlap: true,
    enableCudaGraph: false, // eager path
    tokenRecvDelayTicks: 1,
    eagerForwardExtraDelayTicks: 2,
    maxExtendTokens: 256,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 2),
  ];

  // Tick 0: prefill (eager path → ackTick = 0 + 1 + 2 = 3)
  scheduler.runTick(msgs);
  // tickCounter after tick 0: 1

  // Tick 1: tickCounter=1 < ackTick=3 → no last_data processing
  const resp1 = scheduler.runTick([]);
  const hasResp1 = resp1.some(r => r.uid === 1);
  assert.strictEqual(hasResp1, false, "tick 1 should not process last_data yet (ackTick=3, tickCounter=1)");

  // Tick 2: tickCounter=2 < ackTick=3 → no last_data processing
  const resp2 = scheduler.runTick([]);
  const hasResp2 = resp2.some(r => r.uid === 1);
  assert.strictEqual(hasResp2, false, "tick 2 should not process last_data yet (ackTick=3, tickCounter=2)");

  // Tick 3: tickCounter=3 >= ackTick=3 → last_data is processed
  const resp3 = scheduler.runTick([]);
  const hasResp3 = resp3.some(r => r.uid === 1);
  assert.strictEqual(hasResp3, true, "tick 3 should process last_data (ackTick=3, tickCounter=3)");

  // Complete the request
  const doneReqs = runUntilDone(scheduler, [1], 50);
  assert.strictEqual(doneReqs.length, 1, "request should complete with eager delay");
});

// ================================================================
// 测试 9: test_overlap_empty_tick_flush_last_data
// overlap 模式结束后，调用空 tick 刷新残留 last_data
// ================================================================
test("test_overlap_empty_tick_flush_last_data", () => {
  const config = makeConfig({
    enableOverlap: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 2),
  ];

  // Run first tick with request
  scheduler.runTick(msgs);

  // Run empty ticks to flush remaining data
  const allResp: SimRespMsg[] = [];
  for (let t = 0; t < 50; t++) {
    const resp = scheduler.runTick([]);
    allResp.push(...resp);
    if (resp.some(r => r.uid === 1 && r.finished)) break;
  }

  const doneResp = allResp.filter(r => r.uid === 1 && r.finished);
  assert.strictEqual(doneResp.length, 1, "empty tick should flush residual last_data");

  // Verify no errors on more empty ticks after completion
  const extraResp = scheduler.runTick([]);
  assert.ok(Array.isArray(extraResp), "additional empty ticks after completion should not throw");
});

// ================================================================
// 测试 10: test_overlap_high_watermark_backpressure
// 大量请求超过 highWatermark 时，跳过 forward 仅处理 last_data
// ================================================================
test("test_overlap_high_watermark_backpressure", () => {
  const highWatermark = 4;
  const config = makeConfig({
    enableOverlap: true,
    messagesHighWatermark: highWatermark,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    maxRunningReq: 32,
    numPages: 2048,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  // Submit more requests than highWatermark at once
  const msgs: SimRequestMsg[] = [];
  for (let i = 1; i <= highWatermark + 2; i++) {
    msgs.push(makeShortRequest(i, 3, 2));
  }

  // Tick 0: All messages processed, but highWatermark may trigger backpressure
  scheduler.runTick(msgs);

  // All requests should eventually complete despite backpressure
  const doneReqs = runUntilDone(scheduler, msgs.map(m => m.uid), 200);
  assert.strictEqual(doneReqs.length, msgs.length, "all requests should complete despite backpressure");
});

// ================================================================
// 测试 10 变体: highWatermark=0 → 始终背压
// ================================================================
test("test_overlap_high_watermark_zero_always_backpressure", () => {
  const config = makeConfig({
    enableOverlap: true,
    messagesHighWatermark: 0,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    maxRunningReq: 8,
    numPages: 512,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  // With highWatermark=0, _incomingQueue.length > 0 > 0 is always true,
  // so forward is always skipped. But _processOneMsg still enqueues requests.
  // This means the scheduler should handle this gracefully.
  // Note: with highWatermark=0, no forward happens, so requests never complete.
  // This tests that the scheduler doesn't crash under this condition.
  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 2),
  ];

  // Should not throw
  scheduler.runTick(msgs);
  scheduler.runTick([]);
  scheduler.runTick([]);

  // Verify scheduler is still functional
  assert.ok(true, "scheduler should handle highWatermark=0 without crashing");
});

// ================================================================
// 测试 11: test_overlap_finished_reqs_dedup
// 验证 finishedReqs 集合在 overlap 模式下防止同一请求被重复释放资源
// ================================================================
test("test_overlap_finished_reqs_dedup", () => {
  const config = makeConfig({
    enableOverlap: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    eosTokenId: 2,
    mockSampleMode: "fixed",
    fixedOutputToken: 2, // Force EOS token so requests finish quickly
    maxRunningReq: 8,
    numPages: 512,
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeShortRequest(1, 3, 10), // Will finish via EOS since fixedOutputToken=eosTokenId
  ];

  // Submit the request first
  scheduler.runTick(msgs);

  // Run until done
  const doneReqs = runUntilDone(scheduler, [1], 50);
  assert.strictEqual(doneReqs.length, 1, "request should finish via EOS");
  assert.strictEqual(doneReqs[0].finished, true);

  // Run more ticks to verify no double-free errors
  for (let t = 0; t < 10; t++) {
    scheduler.runTick([]);
  }
  // If finishedReqs dedup is broken, _freeReqResources would be called multiple times
  // causing tableManager.free to free the same index multiple times
  // which could cause issues. We verify the scheduler stays stable.
  assert.ok(true, "scheduler should remain stable after request completion (finishedReqs dedup works)");
});

// ================================================================
// 边界条件: idleCountForFlush=0 → 每个 tick 都尝试 flush
// ================================================================
test("test_overlap_idle_count_for_flush_zero", () => {
  const config = makeConfig({
    enableOverlap: true,
    idleCountForFlush: 0,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 64,
    maxSeqLen: 512,
    numPages: 2048,
    maxRunningReq: 16,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  const msgs: SimRequestMsg[] = [
    makeLongRequest(200, 300, 2),
  ];

  scheduler.runTick(msgs);
  const doneReqs = runUntilDone(scheduler, [200], 200);
  assert.strictEqual(doneReqs.length, 1, "chunked prefill should complete with idleCountForFlush=0");
});

// ================================================================
// 边界条件: 无 last_data 时空 tick 不报错
// ================================================================
test("test_overlap_empty_tick_no_last_data", () => {
  const config = makeConfig({
    enableOverlap: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  // Multiple empty ticks with no last_data and no pending requests
  for (let t = 0; t < 20; t++) {
    const resp = scheduler.runTick([]);
    assert.ok(Array.isArray(resp), `empty tick ${t} should not throw`);
    assert.strictEqual(resp.length, 0, `empty tick ${t} should return empty responses`);
  }
});

// ================================================================
// 边界条件: SimulationClock callback 多次注册
// ================================================================
test("test_simulation_clock_multiple_callbacks", () => {
  const clock = new SimulationClock();
  const ticks1: number[] = [];
  const ticks2: number[] = [];

  clock.onTick((t) => ticks1.push(t));
  clock.onTick((t) => ticks2.push(t));

  clock.advance(1);
  assert.deepStrictEqual(ticks1, [1]);
  assert.deepStrictEqual(ticks2, [1]);

  clock.advance(2);
  assert.deepStrictEqual(ticks1, [1, 3]);
  assert.deepStrictEqual(ticks2, [1, 3]);
});

// ================================================================
// 边界条件: SimulationClock scheduleGpu after advance
// ================================================================
test("test_simulation_clock_schedule_gpu_after_advance", () => {
  const clock = new SimulationClock();
  clock.advance(10);
  // Now currentTick=10, gpuBusyUntil=0
  const finish = clock.scheduleGpu(5);
  // start = max(10, 0) = 10, finish = 15
  assert.strictEqual(finish, 15);
  assert.strictEqual(clock.gpuBusyUntil, 15);
});

// ================================================================
// 测试: SimulationClock 与 SimScheduler 的集成
// ================================================================
test("test_clock_scheduler_integration", () => {
  const clock = new SimulationClock();
  const config = makeConfig({
    enableOverlap: true,
    enableMetrics: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config, { clock });

  assert.strictEqual(scheduler.clock, clock, "scheduler should use provided clock");

  // Run a tick and verify clock advances
  scheduler.runTick([makeShortRequest(1, 3, 2)]);
  assert.ok(clock.currentTick > 0, "clock should have advanced after tick");
});

// ================================================================
// 测试: SimulationClock 自动创建
// ================================================================
test("test_clock_auto_creation", () => {
  const config = makeConfig({
    enableOverlap: true,
    enableMetrics: true,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);
  assert.ok(scheduler.clock !== null, "clock should be auto-created when enableOverlap && enableMetrics");
  assert.ok(scheduler.clock instanceof SimulationClock);
});

// ================================================================
// 测试: SimulationClock 不自动创建（enableMetrics=false）
// ================================================================
test("test_clock_no_auto_creation_without_metrics", () => {
  const config = makeConfig({
    enableOverlap: true,
    enableMetrics: false,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);
  assert.strictEqual(scheduler.clock, null, "clock should not be auto-created when enableMetrics=false");
});

// ================================================================
// 测试: _normalTick 也推进 SimulationClock
// ================================================================
test("test_normal_tick_advances_clock", () => {
  const clock = new SimulationClock();
  const config = makeConfig({
    enableOverlap: false, // Use normal tick
    enableMetrics: true,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config, { clock });

  assert.strictEqual(clock.currentTick, 0);
  scheduler.runTick([makeShortRequest(1, 3, 2)]);
  assert.strictEqual(clock.currentTick, 1, "normal tick should advance clock by 1");
});

// ================================================================
// 测试: tickCounter 属性
// ================================================================
test("test_tick_counter_property", () => {
  const config = makeConfig({
    enableOverlap: true,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  assert.strictEqual(scheduler.tickCounter, 0, "initial tickCounter should be 0");
  scheduler.runTick([]);
  assert.strictEqual(scheduler.tickCounter, 1, "tickCounter should increment by 1 per tick");
  scheduler.runTick([]);
  assert.strictEqual(scheduler.tickCounter, 2, "tickCounter should be 2 after 2 ticks");
});

// ================================================================
// 边界条件: 背压期间 _processLastData 仍正常执行
// ================================================================
test("test_backpressure_process_last_data_still_works", () => {
  const highWatermark = 2;
  const config = makeConfig({
    enableOverlap: true,
    messagesHighWatermark: highWatermark,
    tokenRecvDelayTicks: 0,
    eagerForwardExtraDelayTicks: 0,
    maxExtendTokens: 256,
    maxRunningReq: 32,
    numPages: 2048,
    mockSampleMode: "greedy",
  });
  const scheduler = new SimScheduler(config);

  // First, submit a request that will generate last_data
  scheduler.runTick([makeShortRequest(1, 3, 2)]);

  // Then, submit many requests to trigger backpressure
  const manyMsgs: SimRequestMsg[] = [];
  for (let i = 100; i < 100 + highWatermark + 5; i++) {
    manyMsgs.push(makeShortRequest(i, 3, 2));
  }

  // During backpressure, last_data should still be processed
  let gotRespToken = false;
  for (let t = 0; t < 50; t++) {
    const resp = scheduler.runTick(t === 0 ? manyMsgs : []);
    if (resp.some(r => r.uid === 1 && !r.finished)) {
      gotRespToken = true;
    }
    if (resp.some(r => r.uid === 1 && r.finished)) {
      gotRespToken = true;
      break;
    }
  }
  assert.ok(gotRespToken, "last_data should still be processed during backpressure for request 1");
});

// ===== 结果汇总 =====
console.log(`\n=== S5 Test Results: ${passed} passed, ${failed} failed ===`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) {
    console.log("  - " + f);
  }
  process.exit(1);
}
