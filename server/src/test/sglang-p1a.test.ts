import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  DEFAULT_MODEL_CONFIG,
  TPSimulator,
  TPCommInfraSimulator,
  ParallelMetrics,
  SimCommGroupImpl,
} from "../sglang";
import type { SimulatorConfig, ModelConfig } from "../sglang";
import { SimCommGroup } from "../sglang/parallel/comm_group";

/**
 * Issue #22 验收测试 — P1a: TPSimulator + TPCommInfraSimulator
 *
 * Run with:  npx ts-node src/test/sglang-p1a.test.ts
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

function makeConfig(overrides: Partial<SimulatorConfig> = {}): SimulatorConfig {
  return { ...DEFAULT_SIMULATOR_CONFIG, ...overrides };
}

function makeModelConfig(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return { ...DEFAULT_MODEL_CONFIG, ...overrides };
}

// ==========================================
// TPSimulator 测试
// ==========================================

// T1: tpSize=1 全 noop
test("T1 tpSize=1 all noop — allReduceAfterAttn/Mlp return 0, localNumHeads=orig", () => {
  const config = makeConfig({ tpSize: 1 });
  const model = makeModelConfig({ numAttentionHeads: 32, numKvHeads: 8, intermediateSize: 11008 });
  const tp = new TPSimulator(config, model);
  assert.strictEqual(tp.allReduceAfterAttn(1), 0);
  assert.strictEqual(tp.allReduceAfterMlp(1), 0);
  assert.strictEqual(tp.localNumHeads, 32);
  assert.strictEqual(tp.localNumKvHeads, 8);
  assert.strictEqual(tp.localIntermediate, 11008);
});

// T2: tpSize=2 内存修正
test("T2 tpSize=2 memory correction", () => {
  const config = makeConfig({ tpSize: 2 });
  const model = makeModelConfig({ numAttentionHeads: 32, numKvHeads: 8, intermediateSize: 11008 });
  const tp = new TPSimulator(config, model);
  assert.strictEqual(tp.localNumHeads, 16);
  assert.strictEqual(tp.localNumKvHeads, 4);
  assert.strictEqual(tp.localIntermediate, 5504);
});

// T3: allReduceAfterAttn 正值
test("T3 allReduceAfterAttn returns positive when tpSize=2", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const batchSize = 4;
  const result = tp.allReduceAfterAttn(batchSize);
  assert.ok(result > 0, `Expected >0, got ${result}`);
  // Verify formula: dataBytes = 4 * 4096 * 2 = 32768
  // SimCommGroup.allReduce(32768) with size=2, bw=100GB/s, latency=5us
  const expected = new SimCommGroup({
    groupType: "tp", size: 2, networkBandwidthGBps: 100, latencyUs: 5, efficiency: 1.0,
  }).allReduce(32768);
  assert.strictEqual(result, expected);
});

// T4: allReduceAfterMlp 正值（相同数据量，返回相同值）
test("T4 allReduceAfterMlp returns same value as allReduceAfterAttn for same batch", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp1 = new TPSimulator(config, model);
  const tp2 = new TPSimulator(config, model);
  const batchSize = 4;
  const attnResult = tp1.allReduceAfterAttn(batchSize);
  const mlpResult = tp2.allReduceAfterMlp(batchSize);
  assert.strictEqual(attnResult, mlpResult);
});

// T5: totalCommTicksPerStep 累加
test("T5 totalCommTicksPerStep accumulates", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const r1 = tp.allReduceAfterAttn(2);
  const r2 = tp.allReduceAfterMlp(2);
  const r3 = tp.allReduceAfterAttn(3);
  assert.strictEqual(tp.totalCommTicksPerStep(), r1 + r2 + r3);
});

// T6: resetStepComm 清零
test("T6 resetStepComm clears total", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  tp.allReduceAfterAttn(2);
  tp.allReduceAfterMlp(2);
  tp.resetStepComm();
  assert.strictEqual(tp.totalCommTicksPerStep(), 0);
});

// T7: divEven GQA kv_heads 复制 — numKvHeads=2, tpSize=4
test("T7 divEven GQA kv_heads replication — numKvHeads=2, tpSize=4", () => {
  const config = makeConfig({ tpSize: 4 });
  const model = makeModelConfig({ numKvHeads: 2, numAttentionHeads: 32, intermediateSize: 11008 });
  const tp = new TPSimulator(config, model);
  // divEven(2, 4, true) = [1, 1, 0, 0], first element = 1
  assert.strictEqual(tp.localNumKvHeads, 1);
});

// T8: SimCommGroup 效率因子
test("T8 TPSimulator uses config.tpEfficiency", () => {
  const config1 = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const config05 = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 0.5 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp1 = new TPSimulator(config1, model);
  const tp05 = new TPSimulator(config05, model);
  const r1 = tp1.allReduceAfterAttn(4);
  const r05 = tp05.allReduceAfterAttn(4);
  assert.ok(r05 > r1, `efficiency=0.5 result (${r05}) should be > efficiency=1.0 result (${r1})`);
});

// ==========================================
// TPCommInfraSimulator 测试
// ==========================================

// T9: tpSize=1 全 noop
test("T9 tpSize=1 TPCommInfraSimulator all noop", () => {
  const config = makeConfig({ tpSize: 1 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  assert.strictEqual(comm.zmqBroadcast(1024), 0);
  assert.strictEqual(comm.cpuBarrier(), 0);
  assert.strictEqual(comm.gpuAllReduce(1024), 0);
  assert.strictEqual(comm.broadcastAll([[1, 2, 3]]), 0);
});

// T10: zmqBroadcast 正值
test("T10 zmqBroadcast returns positive when tpSize=2", () => {
  const config = makeConfig({ tpSize: 2, commBandwidthBytesPerTick: 1000 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  const msgSize = 5000;
  const result = comm.zmqBroadcast(msgSize);
  assert.strictEqual(result, Math.ceil(5000 / 1000));
  assert.ok(result > 0);
});

// T11: cpuBarrier 固定 1 tick
test("T11 cpuBarrier always returns 1 when tpSize=2", () => {
  const config = makeConfig({ tpSize: 2 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  assert.strictEqual(comm.cpuBarrier(), 1);
  assert.strictEqual(comm.cpuBarrier(), 1);
});

// T12: gpuAllReduce 委托
test("T12 gpuAllReduce delegates to SimCommGroup.allReduce", () => {
  const config = makeConfig({ tpSize: 4, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 0.95 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  const expected = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5, efficiency: 0.95,
  }).allReduce(1024);
  assert.strictEqual(comm.gpuAllReduce(1024), expected);
});

// T13: broadcastAll 批量
test("T13 broadcastAll aggregates bytes correctly", () => {
  const config = makeConfig({ tpSize: 2, commBandwidthBytesPerTick: 1000 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  // 3 lists with 2, 3, 1 tokens respectively = 6 tokens × 4 bytes = 24 bytes
  const result = comm.broadcastAll([[1, 2], [3, 4, 5], [6]]);
  assert.strictEqual(result, Math.ceil(24 / 1000));
});

// T14: zmqBroadcastTicks 累加
test("T14 zmqBroadcastTicks accumulates", () => {
  const config = makeConfig({ tpSize: 2, commBandwidthBytesPerTick: 1000 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  const r1 = comm.zmqBroadcast(1000);
  const r2 = comm.zmqBroadcast(2000);
  assert.strictEqual(comm.zmqBroadcastTicks, r1 + r2);
});

// T15: barrierTicks 累加
test("T15 barrierTicks accumulates", () => {
  const config = makeConfig({ tpSize: 2 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  comm.cpuBarrier();
  comm.cpuBarrier();
  comm.cpuBarrier();
  assert.strictEqual(comm.barrierTicks, 3);
});

// T16: cpuGroupType/gpuGroupType 读取
test("T16 cpuGroupType/gpuGroupType read from config", () => {
  const config = makeConfig({ tpSize: 2, tpCpuGroupType: "gloo", tpGpuGroupType: "nccl" });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  assert.strictEqual(comm.cpuGroupType, "gloo");
  assert.strictEqual(comm.gpuGroupType, "nccl");
});

// ==========================================
// 组合测试
// ==========================================

// T17: TPSimulator + TPCommInfraSimulator 独立
test("T17 TPSimulator + TPCommInfraSimulator independent instances", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const comm = new TPCommInfraSimulator(config, model);
  // TPSimulator operation does not affect TPCommInfraSimulator
  tp.allReduceAfterAttn(4);
  assert.strictEqual(comm.zmqBroadcastTicks, 0);
  assert.strictEqual(comm.barrierTicks, 0);
  // And vice versa
  comm.cpuBarrier();
  assert.strictEqual(tp.totalCommTicksPerStep(), tp.allReduceAfterAttn(4) + 0);
});

// T18: tpSize=1 退化单实例
test("T18 tpSize=1 degeneration — all return 0", () => {
  const config = makeConfig({ tpSize: 1 });
  const model = makeModelConfig();
  const tp = new TPSimulator(config, model);
  const comm = new TPCommInfraSimulator(config, model);
  assert.strictEqual(tp.allReduceAfterAttn(4), 0);
  assert.strictEqual(tp.allReduceAfterMlp(4), 0);
  assert.strictEqual(tp.totalCommTicksPerStep(), 0);
  assert.strictEqual(comm.zmqBroadcast(1024), 0);
  assert.strictEqual(comm.cpuBarrier(), 0);
  assert.strictEqual(comm.gpuAllReduce(1024), 0);
  assert.strictEqual(comm.broadcastAll([[1, 2, 3]]), 0);
});

// T19: 与 ParallelMetrics 字段对应
test("T19 TPSimulator comm ticks can be written to ParallelMetrics.tpCommTicks", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096, numLayers: 32 });
  const tp = new TPSimulator(config, model);
  const metrics = new ParallelMetrics();
  // Simulate one step: 32 layers × (attn + mlp) all_reduce
  let totalComm = 0;
  let allReduceCount = 0;
  for (let i = 0; i < model.numLayers; i++) {
    totalComm += tp.allReduceAfterAttn(4);
    totalComm += tp.allReduceAfterMlp(4);
    allReduceCount += 2;
  }
  metrics.tpCommTicks = totalComm;
  metrics.tpAllReduceCount = allReduceCount;
  assert.strictEqual(metrics.tpCommTicks, tp.totalCommTicksPerStep());
  assert.strictEqual(metrics.tpAllReduceCount, 64);
  assert.ok(metrics.tpCommTicks > 0);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: batchSize=0 → allReduceAfterAttn(0) 数据量为 0
test("B1 allReduceAfterAttn(0) — dataBytes=0, SimCommGroup.allReduce(0) returns latency", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const result = tp.allReduceAfterAttn(0);
  // dataBytes = 0 * 4096 * 2 = 0
  // SimCommGroup.allReduce(0): raw = 2*0*1/2/100000 + 5 = 5, ceil(5/1.0) = 5
  assert.strictEqual(result, 5);
});

// B2: msgSize=0 → zmqBroadcast(0) 返回 0
test("B2 zmqBroadcast(0) returns 0", () => {
  const config = makeConfig({ tpSize: 2, commBandwidthBytesPerTick: 1000 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  assert.strictEqual(comm.zmqBroadcast(0), 0);
});

// B3: tpSize 极大 → allReduce 成本随 size 增大趋近于 2×bytes/bw + latency
test("B3 large tpSize — allReduce cost approaches 2×bytes/bw + latency", () => {
  const config = makeConfig({ tpSize: 256, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const result = tp.allReduceAfterAttn(1);
  // dataBytes = 1 * 4096 * 2 = 8192
  // raw = 2 * 8192 * 255 / 256 / 100000 + 5 ≈ 0.1632 + 5 = 5.1632
  // ceil(5.1632) = 6
  // As size→∞: 2*8192/100000 + 5 = 0.16384 + 5 = 5.16384
  assert.ok(result > 5, `Large tpSize should give result > 5, got ${result}`);
  assert.ok(result <= 7, `Large tpSize should give result <= 7, got ${result}`);
});

// B4: numKvHeads=0 → divEven(0, tpSize) 返回全 0
test("B4 numKvHeads=0 — localNumKvHeads=0", () => {
  const config = makeConfig({ tpSize: 2 });
  const model = makeModelConfig({ numKvHeads: 0, numAttentionHeads: 32, intermediateSize: 11008 });
  const tp = new TPSimulator(config, model);
  assert.strictEqual(tp.localNumKvHeads, 0);
});

// B5: commBandwidthBytesPerTick=0 → zmqBroadcast 中 Math.ceil(msgSize/max(1,0)) = msgSize
test("B5 commBandwidthBytesPerTick=0 — zmqBroadcast returns msgSize (1 tick per byte)", () => {
  const config = makeConfig({ tpSize: 2, commBandwidthBytesPerTick: 0 });
  const model = makeModelConfig();
  const comm = new TPCommInfraSimulator(config, model);
  const msgSize = 500;
  const result = comm.zmqBroadcast(msgSize);
  // Math.ceil(500 / Math.max(1, 0)) = Math.ceil(500 / 1) = 500
  assert.strictEqual(result, 500);
});

// B6: tpEfficiency=1.0 → 结果与无效率因子一致
test("B6 tpEfficiency=1.0 — result same as no efficiency factor", () => {
  const config = makeConfig({ tpSize: 2, networkBandwidthGBps: 100, networkLatencyUs: 5, tpEfficiency: 1.0 });
  const model = makeModelConfig({ hiddenSize: 4096 });
  const tp = new TPSimulator(config, model);
  const expected = new SimCommGroup({
    groupType: "tp", size: 2, networkBandwidthGBps: 100, latencyUs: 5,
  }).allReduce(4 * 4096 * 2);
  assert.strictEqual(tp.allReduceAfterAttn(4), expected);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P1a 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P1a 验收测试通过 \u2713");
  process.exit(0);
}
