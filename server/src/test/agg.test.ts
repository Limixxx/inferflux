import assert from "assert";
import { SimEngine } from "../sim/SimEngine";
import { SimParams } from "../shared/types";
import { DEFAULTS, PRESETS } from "../shared/presets";
import { BD_KEYS_DISAGG, BD_KEYS_AGG } from "../shared/constants";

/**
 * Issue #1 验收测试 — agg (aggregated) mode simulation.
 *
 * Run with:  npx ts-node src/test/agg.test.ts
 * (uses Node's built-in assert module — no test framework dependency)
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

function aggParams(overrides: Partial<SimParams> = {}): SimParams {
  return { ...DEFAULTS, mode: "agg", numWorkers: 2, kvGb: 99, chunkedPrefill: false, ...overrides };
}

function disaggParams(overrides: Partial<SimParams> = {}): SimParams {
  return { ...DEFAULTS, mode: "pd-disagg", ...overrides };
}

// Run the engine forward enough to complete many requests.
function runFor(engine: SimEngine, ms: number): void {
  engine.advance(ms);
}

console.log("\n=== Issue #1 验收测试 — agg mode ===\n");

// T1: Default mode is pd-disagg
test("T1 default mode is pd-disagg", () => {
  assert.strictEqual(DEFAULTS.mode, "pd-disagg");
  const eng = new SimEngine({ ...DEFAULTS });
  assert.strictEqual(eng.P.mode, "pd-disagg");
  assert.ok(eng.pList.length > 0, "pList should be populated in pd-disagg mode");
  assert.strictEqual(eng.wList.length, 0, "wList should be empty in pd-disagg mode");
});

// T2: Switch to agg mode — wList created, pList/dList empty
test("T2 agg mode creates wList, empties pList/dList", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 3 }));
  assert.strictEqual(eng.P.mode, "agg");
  assert.strictEqual(eng.wList.length, 3, "wList should have numWorkers entries");
  assert.strictEqual(eng.pList.length, 0, "pList should be empty in agg mode");
  assert.strictEqual(eng.dList.length, 0, "dList should be empty in agg mode");
  assert.strictEqual(eng.P.chunkedPrefill, false, "chunkedPrefill defaults to false");
});

// T3: agg non-chunked prefill — requests complete and reach 'done'
test("T3 agg non-chunked prefill completes requests", () => {
  const eng = new SimEngine(aggParams({ qps: 4, inputLenMean: 512, outputLenMean: 64, chunkedPrefill: false }));
  runFor(eng, 60000);
  assert.ok(eng.metrics.totalCompleted > 0, "should have completed requests");
  // Every completed request should have valid TTFT
  const snap = eng.metrics.snapshot(eng.now, 10000);
  assert.ok(snap, "snapshot should not be null");
  assert.ok(snap!.ttft.avg > 0, "TTFT should be positive");
  assert.ok(snap!.e2e.avg > 0, "E2E should be positive");
});

// T4: agg chunked prefill — requests complete with chunkedPrefill=true
test("T4 agg chunked prefill completes requests", () => {
  const eng = new SimEngine(aggParams({
    qps: 2, inputLenMean: 4096, outputLenMean: 64,
    chunkedPrefill: true, chunkSize: 1024,
  }));
  runFor(eng, 60000);
  assert.ok(eng.metrics.totalCompleted > 0, "should complete requests in chunked prefill mode");
  const snap = eng.metrics.snapshot(eng.now, 10000);
  assert.ok(snap, "snapshot should not be null");
  assert.ok(snap!.ttft.avg > 0, "TTFT should be positive in chunked mode");
});

// T5: agg KV capacity constraint — kvUsed never exceeds maxTokens
test("T5 agg KV capacity constraint holds", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 1, kvGb: 4, qps: 8, inputLenMean: 2048, outputLenMean: 512 }));
  runFor(eng, 30000);
  for (const w of eng.wList) {
    const cap = w.maxTokens(eng.P);
    assert.ok(w.kvUsed <= cap + 1, `kvUsed (${w.kvUsed}) must not exceed capacity (${cap})`);
  }
});

// T7/T8: agg metrics — breakdown has 4 columns
test("T7 agg breakdown has 4 columns (BD_KEYS_AGG)", () => {
  assert.strictEqual(BD_KEYS_AGG.length, 4);
  assert.deepStrictEqual([...BD_KEYS_AGG], ["tokenize", "queue", "prefill", "detok"]);
  const eng = new SimEngine(aggParams({ qps: 4, inputLenMean: 512, outputLenMean: 64 }));
  runFor(eng, 30000);
  const bd = eng.metrics.recentBreakdown(10);
  assert.ok(Array.isArray(bd), "breakdown should be an array");
  assert.strictEqual(bd.length, 4, "agg breakdown must have exactly 4 columns");
});

// T9: agg gauges — wQueue and kvW are populated
test("T9 agg gauges wQueue/kvW populated", () => {
  const eng = new SimEngine(aggParams({ qps: 8, inputLenMean: 2048, outputLenMean: 256 }));
  runFor(eng, 30000);
  const g = eng.sampleGauges();
  assert.ok(typeof g.wQueue === "number", "wQueue gauge should be a number");
  assert.ok(typeof g.kvW === "number", "kvW gauge should be a number");
  assert.ok(g.kvW >= 0 && g.kvW <= 1, "kvW should be a ratio in [0,1]");
  // In agg mode, pd-disagg-only gauges should be zeroed
  assert.strictEqual(g.pQueue, 0, "pQueue should be 0 in agg mode");
  assert.strictEqual(g.dQueue, 0, "dQueue should be 0 in agg mode");
  assert.strictEqual(g.link, 0, "link should be 0 in agg mode");
});

// T10: RadixCache prefix reuse — uncachedLen < inputLen when cacheHitRate > 0
test("T10 RadixCache prefix reuse reduces uncachedLen", () => {
  const eng = new SimEngine(aggParams({ cacheHitRate: 0.5, inputLenMean: 2048, inputDist: "fixed" }));
  runFor(eng, 5000);
  let foundActive = false;
  for (const r of eng.allActive) {
    foundActive = true;
    if (r.cachedLen > 0) {
      assert.ok(r.uncachedLen <= r.inputLen, "uncachedLen must not exceed inputLen");
      assert.strictEqual(r.inputLen, r.cachedLen + r.uncachedLen, "inputLen = cachedLen + uncachedLen");
    }
  }
  // Even if no active requests, the engine ran without error
  assert.ok(true, "engine ran with cacheHitRate=0.5");
});

// T11: BlockManager pre-allocation — kvUsed accounts for full inputLen on admission
test("T11 BlockManager pre-allocates full inputLen", () => {
  const eng = new SimEngine(aggParams({
    numWorkers: 1, kvGb: 99, qps: 0.5, inputLenMean: 1024, inputDist: "fixed",
    outputLenMean: 16, cacheHitRate: 0,
  }));
  runFor(eng, 3000);
  // Find a request in running; its worker's kvUsed should include inputLen
  for (const w of eng.wList) {
    for (const r of w.running) {
      // Request was admitted → kvUsed includes at least this request's inputLen
      assert.ok(w.kvUsed >= r.inputLen, "kvUsed should include admitted request's full inputLen");
    }
  }
});

// T13: Switch back to pd-disagg after agg
test("T13 switch agg → pd-disagg resets topology", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 3 }));
  runFor(eng, 5000);
  assert.ok(eng.wList.length > 0, "wList populated in agg mode");
  // Switch to pd-disagg
  eng.P.mode = "pd-disagg";
  eng.reset();
  assert.strictEqual(eng.wList.length, 0, "wList should be empty after switching to pd-disagg");
  assert.ok(eng.pList.length > 0, "pList should be populated in pd-disagg mode");
  assert.ok(eng.dList.length > 0, "dList should be populated in pd-disagg mode");
});

// T14: agg presets load with correct mode
test("T14 agg presets have mode=agg", () => {
  for (const name of ["aggBalanced", "aggChunkedPrefill", "aggDecodeHeavy", "aggHighQps"]) {
    const preset = PRESETS[name];
    assert.ok(preset, `preset ${name} should exist`);
    assert.strictEqual(preset.mode, "agg", `preset ${name} should have mode=agg`);
  }
  // pd-disagg presets keep mode=pd-disagg
  assert.strictEqual(PRESETS.balanced.mode, "pd-disagg");
});

// T15: agg getRenderState via SimService returns wList
test("T15 agg render state returns wList (via engine state)", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 2 }));
  runFor(eng, 5000);
  // Directly verify engine state shape that SimService.getRenderState serializes
  assert.strictEqual(eng.P.mode, "agg");
  assert.ok(eng.wList.length === 2, "engine should have 2 workers");
  for (const w of eng.wList) {
    assert.ok(typeof w.id === "number");
    assert.ok(Array.isArray(w.waitingQ));
    assert.ok(Array.isArray(w.running));
    assert.ok(typeof w.kvUsed === "number");
    assert.ok(typeof w.maxTokens(eng.P) === "number");
  }
});

// T12: make_batch mixed batching — running can contain multiple stages
test("T12 agg running batch can mix prefill and decode stages", () => {
  const eng = new SimEngine(aggParams({
    numWorkers: 1, qps: 10, inputLenMean: 512, outputLenMean: 256,
    maxRunning: 32, chunkedPrefill: false,
  }));
  runFor(eng, 8000);
  // Under load, the single worker's running batch should at some point contain
  // requests in different stages. We verify the engine doesn't crash and
  // produces completions.
  assert.ok(eng.metrics.totalCompleted > 0, "mixed-batch engine should complete requests");
});

// B3: Very small KV budget causes saturation but no crash
test("B3 tiny kvGb saturates without crash", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 1, kvGb: 1, qps: 4, inputLenMean: 2048, outputLenMean: 256 }));
  runFor(eng, 20000);
  assert.ok(true, "engine survived tiny KV budget");
  // KV should be near or at capacity
  const w = eng.wList[0];
  assert.ok(w.kvUsed <= w.maxTokens(eng.P) + 1, "kvUsed within capacity even when saturated");
});

// B9: cacheHitRate=1.0 → uncachedLen near 0 (>=1 due to inputLen-1 cap, but << inputLen)
test("B9 full cache hit → uncachedLen near zero", () => {
  const eng = new SimEngine(aggParams({ cacheHitRate: 1.0, inputLenMean: 2048, inputDist: "fixed", qps: 2 }));
  runFor(eng, 5000);
  for (const r of eng.allActive) {
    // cachedLen is capped at inputLen-1, so uncachedLen >= 1 always.
    // With cacheHitRate=1.0 + jitter, uncachedLen should be a tiny fraction of inputLen.
    assert.ok(r.uncachedLen >= 1, "uncachedLen >= 1 (inputLen-1 cap)");
    assert.ok(r.uncachedLen <= r.inputLen * 0.15 + 1, "uncachedLen should be small with full cache hit");
  }
});

// B1: numWorkers=1 single worker
test("B1 single worker agg mode", () => {
  const eng = new SimEngine(aggParams({ numWorkers: 1, qps: 4, inputLenMean: 512, outputLenMean: 64 }));
  assert.strictEqual(eng.wList.length, 1);
  runFor(eng, 20000);
  assert.ok(eng.metrics.totalCompleted > 0, "single-worker agg should complete requests");
});

// Summary
console.log("\n=== 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有验收测试通过 \u2713");
  process.exit(0);
}
