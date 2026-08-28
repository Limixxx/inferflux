import assert from "assert";
import {
  DPAttentionSimulator,
} from "../sglang";
import type { DPAttentionSimulatorOpts } from "../sglang";

/**
 * Issue #25 验收测试 — P2b: DPAttentionSimulator
 *
 * Run with:  npx ts-node src/test/sglang-p2b.test.ts
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

function makeOpts(overrides: Partial<DPAttentionSimulatorOpts>): DPAttentionSimulatorOpts {
  return {
    dpSize: 2,
    hiddenSize: 4096,
    dtypeSize: 2,
    useMla: true,
    enableDpAttention: true,
    networkBandwidthGBps: 100,
    networkLatencyUs: 5,
    ...overrides,
  };
}

// ==========================================
// T1: 启用条件 — useMla=true && enableDpAttention=true && dpSize=2
// ==========================================
test("T1 enabled when useMla=true && enableDpAttention=true && dpSize=2", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  assert.strictEqual(sim.enabled, true);
  assert.ok(sim.commGroup !== null, "commGroup should not be null");
});

// ==========================================
// T2: 不启用 — useMla=false
// ==========================================
test("T2 disabled when useMla=false", () => {
  const sim = new DPAttentionSimulator(makeOpts({ useMla: false }));
  assert.strictEqual(sim.enabled, false);
  assert.strictEqual(sim.commGroup, null);
  const result = sim.simulateMlpForward([1, 1]);
  assert.deepStrictEqual(result, { commTicks: 0, allGatherBytes: 0 });
});

// ==========================================
// T3: 不启用 — enableDpAttention=false
// ==========================================
test("T3 disabled when enableDpAttention=false", () => {
  const sim = new DPAttentionSimulator(makeOpts({ enableDpAttention: false }));
  assert.strictEqual(sim.enabled, false);
  assert.strictEqual(sim.commGroup, null);
});

// ==========================================
// T4: 不启用 — dpSize=1
// ==========================================
test("T4 disabled when dpSize=1", () => {
  const sim = new DPAttentionSimulator(makeOpts({ dpSize: 1 }));
  assert.strictEqual(sim.enabled, false);
  assert.strictEqual(sim.commGroup, null);
});

// ==========================================
// T5: simulateMlpForward dpSize=2 batch=2 分块 1+1
// ==========================================
test("T5 simulateMlpForward dpSize=2 batch=2 split 1+1", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  const result = sim.simulateMlpForward([1, 1]);
  assert.ok(result.commTicks > 0, `commTicks should be >0, got ${result.commTicks}`);
  // allGatherBytes = (1 * hiddenSize * dtypeSize) + (1 * hiddenSize * dtypeSize) = 2 * 4096 * 2 = 16384
  assert.strictEqual(result.allGatherBytes, 2 * 4096 * 2);
});

// ==========================================
// T6: all_gather_ticks 随 dpSize 线性增长
// ==========================================
test("T6 all_gather_ticks increase with dpSize", () => {
  // Use larger batch sizes so bandwidth cost dominates over latency
  const sim2 = new DPAttentionSimulator(makeOpts({ dpSize: 2, networkLatencyUs: 1 }));
  const sim4 = new DPAttentionSimulator(makeOpts({ dpSize: 4, networkLatencyUs: 1 }));
  const result2 = sim2.simulateMlpForward([64, 64]);
  const result4 = sim4.simulateMlpForward([64, 64, 64, 64]);
  assert.ok(result4.commTicks > result2.commTicks,
    `dpSize=4 commTicks (${result4.commTicks}) should be > dpSize=2 commTicks (${result2.commTicks})`);
});

// ==========================================
// T7: simulateMlpForward 未启用返回 0
// ==========================================
test("T7 simulateMlpForward returns 0 when disabled (mla=false)", () => {
  const sim = new DPAttentionSimulator(makeOpts({ useMla: false }));
  const result = sim.simulateMlpForward([1, 1]);
  assert.deepStrictEqual(result, { commTicks: 0, allGatherBytes: 0 });
});

// ==========================================
// T8: totalAllGatherBytesPerStep 启用时返回正值
// ==========================================
test("T8 totalAllGatherBytesPerStep returns positive when enabled", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  const batch = 4;
  const result = sim.totalAllGatherBytesPerStep(batch);
  // result = batch × hiddenSize × dtypeSize = 4 * 4096 * 2 = 32768
  assert.strictEqual(result, batch * 4096 * 2);
});

// ==========================================
// T9: totalAllGatherBytesPerStep 未启用返回 0
// ==========================================
test("T9 totalAllGatherBytesPerStep returns 0 when disabled", () => {
  const sim = new DPAttentionSimulator(makeOpts({ useMla: false }));
  assert.strictEqual(sim.totalAllGatherBytesPerStep(4), 0);
});

// ==========================================
// T10: commGroup 类型为 dp_attn
// ==========================================
test("T10 commGroup.groupType === dp_attn", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  assert.ok(sim.commGroup !== null);
  assert.strictEqual(sim.commGroup!.groupType, "dp_attn");
});

// ==========================================
// T11: SimCommGroup size — dpSize=2 时 commGroup.size=2
// ==========================================
test("T11 commGroup.size equals dpSize", () => {
  const sim = new DPAttentionSimulator(makeOpts({ dpSize: 2 }));
  assert.ok(sim.commGroup !== null);
  assert.strictEqual(sim.commGroup!.size, 2);
  // dpSize=1 时 commGroup 不创建
  const sim1 = new DPAttentionSimulator(makeOpts({ dpSize: 1 }));
  assert.strictEqual(sim1.commGroup, null);
});

// ==========================================
// T12: allGatherBytes 计算正确
// ==========================================
test("T12 allGatherBytes calculation correct", () => {
  const hiddenSize = 5120;
  const dtypeSize = 2;
  const sim = new DPAttentionSimulator(makeOpts({ hiddenSize, dtypeSize }));
  const result = sim.simulateMlpForward([1, 1]);
  // allGatherBytes = (1 * 5120 * 2) + (1 * 5120 * 2) = 20480
  assert.strictEqual(result.allGatherBytes, 2 * hiddenSize * dtypeSize);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: localBatchSizes 为空数组
test("B1 simulateMlpForward with empty localBatchSizes", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  const result = sim.simulateMlpForward([]);
  assert.strictEqual(result.allGatherBytes, 0);
  // SimCommGroup.allGather([]) → size=2, totalBytes=0 → latency only
  assert.ok(result.commTicks > 0, `commTicks should include latency, got ${result.commTicks}`);
});

// B2: localBatchSizes 全为 0
test("B2 simulateMlpForward with all-zero localBatchSizes", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  const result = sim.simulateMlpForward([0, 0]);
  assert.strictEqual(result.allGatherBytes, 0);
  // SimCommGroup.allGather([0,0]) → totalBytes=0, size=2 → latency only
  assert.ok(result.commTicks > 0, `commTicks should include latency, got ${result.commTicks}`);
});

// B3: batch=0 传入 totalAllGatherBytesPerStep
test("B3 totalAllGatherBytesPerStep with batch=0", () => {
  const sim = new DPAttentionSimulator(makeOpts({}));
  assert.strictEqual(sim.totalAllGatherBytesPerStep(0), 0);
});

// B5: useMla=undefined (falsy)
test("B5 disabled when useMla is undefined (falsy)", () => {
  const sim = new DPAttentionSimulator(makeOpts({ useMla: undefined as any }));
  assert.strictEqual(sim.enabled, false);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P2b 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P2b 验收测试通过 \u2713");
  process.exit(0);
}
