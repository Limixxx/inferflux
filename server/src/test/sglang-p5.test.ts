import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  CPSimulator,
  CPAttnResult,
  SimCommGroupImpl,
  ParallelMetrics,
  SimulationMetrics,
} from "../sglang";
import type { SimulatorConfig, ModelConfig } from "../sglang";
import { MockEngine } from "../sglang/engine";
import { SimCommGroup } from "../sglang/parallel/comm_group";
import { divCeil } from "../sglang/core";

/**
 * Issue #29 验收测试 — P5: CPSimulator Context Parallel KV all-gather
 *
 * Run with:  npx ts-node src/test/sglang-p5.test.ts
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

/** Helper: 创建 CPSimulator 配置 */
function cpConfig(cpSize: number, extra?: Partial<SimulatorConfig>): SimulatorConfig {
  return makeConfig({ cpSize, tpSize: cpSize, ...extra });
}

// ==========================================
// CPSimulator 单元测试
// ==========================================

// T1: cp_size=1 时 simulateAttnForward 返回零
test("T1 cp_size=1 simulateAttnForward returns zero", () => {
  const config = cpConfig(1);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(1024);
  assert.strictEqual(result.commTicks, 0);
  assert.strictEqual(result.allGatherBytes, 0);
  assert.strictEqual(result.seqLenPerRank, 1024);
});

// T2: cp_size=4 时 comm_ticks 大于零
test("T2 cp_size=4 comm_ticks > 0", () => {
  const seqLen = 1024;
  const config4 = cpConfig(4);
  const seqLenPerRank = divCeil(seqLen, 4); // 256
  const g4 = new SimCommGroup({
    groupType: "cp", size: 4,
    networkBandwidthGBps: config4.networkBandwidthGBps,
    latencyUs: config4.networkLatencyUs,
    efficiency: config4.cpEfficiency,
  });
  // 单层 kv_bytes_per_rank = seqLenPerRank * numKvHeads * headDim * dtypeSize * 2
  const kvBytesPerRank = seqLenPerRank * config4.modelConfig.numKvHeads *
    config4.modelConfig.headDim * config4.dtypeSize * 2;
  const ticks4 = g4.allGather([kvBytesPerRank]);
  assert.ok(ticks4 > 0, `cp_size=4 comm_ticks should be >0, got ${ticks4}`);

  // 验证 CPSimulator 返回同样的结果
  const sim4 = new CPSimulator(config4, config4.modelConfig);
  const result4 = sim4.simulateAttnForward(seqLen);
  assert.strictEqual(result4.commTicks, ticks4);
});

// T3: seq_len 不能整除 cp_size 时 seq_per_rank 分布正确
test("T3 divCeil seq_per_rank when seq_len not divisible", () => {
  assert.strictEqual(divCeil(10, 4), 3);
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(10);
  assert.strictEqual(result.seqLenPerRank, 3);
});

// T4: cp_size=4 时 allGatherBytes 计算正确（单层，使用 seqLenPerRank）
test("T4 allGatherBytes formula correct (single layer, per-rank)", () => {
  const seqLen = 2048;
  const cpSize = 4;
  const config = cpConfig(cpSize);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(seqLen);
  const seqLenPerRank = divCeil(seqLen, cpSize); // 512
  const expected = seqLenPerRank * config.modelConfig.numKvHeads *
    config.modelConfig.headDim * config.dtypeSize * 2;
  assert.strictEqual(result.allGatherBytes, expected);
});

// T5: cp_size=4 时 cpAllGatherCount 每层递增
test("T5 cpAllGatherCount increments per layer", () => {
  const config = cpConfig(4);
  const engine = new MockEngine(config);
  engine.forwardBatch(1024);
  assert.strictEqual(engine.metrics.parallel.cpAllGatherCount, config.modelConfig.numLayers);
});

// T6: totalCommTicks 累加正确
test("T6 totalCommTicks accumulates correctly", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const r1 = sim.simulateAttnForward(512);
  const r2 = sim.simulateAttnForward(1024);
  const r3 = sim.simulateAttnForward(2048);
  assert.strictEqual(sim.totalCommTicks, r1.commTicks + r2.commTicks + r3.commTicks);
});

// T7: cp_size=1 时 CPSimulator commGroup 为 null
test("T7 cp_size=1 commGroup is null", () => {
  const config = cpConfig(1);
  const sim = new CPSimulator(config, config.modelConfig);
  assert.strictEqual(sim.commGroup, null);
  assert.strictEqual(sim.cpSize, 1);
});

// ==========================================
// 精确性断言 — 修复驳回问题4
// ==========================================

// P1: 总通信量精确性 — cp_size=4 forwardBatch 总 cpCommTicks 等于 numLayers × 单层 commTicks
test("P1 precise total cpCommTicks = numLayers * singleLayerCommTicks", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const singleLayerResult = sim.simulateAttnForward(1024);
  const singleLayerTicks = singleLayerResult.commTicks;

  const engine = new MockEngine(config);
  engine.forwardBatch(1024);
  const expectedTotal = singleLayerTicks * config.modelConfig.numLayers;
  assert.strictEqual(engine.metrics.parallel.cpCommTicks, expectedTotal,
    `total cpCommTicks (${engine.metrics.parallel.cpCommTicks}) should equal ` +
    `singleLayerTicks (${singleLayerTicks}) * numLayers (${config.modelConfig.numLayers}) = ${expectedTotal}`);
});

// P2: 精确的 kvBytes_per_rank 计算 — 验证公式中不含 numLayers
test("P2 kvBytesPerRank does NOT include numLayers factor", () => {
  const seqLen = 1024;
  const cpSize = 4;
  const config = cpConfig(cpSize);
  const seqLenPerRank = divCeil(seqLen, cpSize);
  // 单层每 rank 的 KV bytes
  const expectedPerRank = seqLenPerRank * config.modelConfig.numKvHeads *
    config.modelConfig.headDim * config.dtypeSize * 2;
  // 不含 numLayers 的版本
  const withLayers = expectedPerRank * config.modelConfig.numLayers;
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(seqLen);
  // allGatherBytes 应等于不含 numLayers 的版本
  assert.strictEqual(result.allGatherBytes, expectedPerRank);
  // allGatherBytes 不应等于含 numLayers 的版本
  assert.notStrictEqual(result.allGatherBytes, withLayers);
});

// P3: allGather 接收的 bytes 基于 seqLenPerRank 而非完整 seqLen
test("P3 allGather bytes based on seqLenPerRank not full seqLen", () => {
  const seqLen = 1024;
  const cpSize = 4;
  const config = cpConfig(cpSize);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(seqLen);
  const seqLenPerRank = divCeil(seqLen, cpSize);
  // 用 seqLenPerRank 计算的 kvBytes
  const kvPerRank = seqLenPerRank * config.modelConfig.numKvHeads *
    config.modelConfig.headDim * config.dtypeSize * 2;
  // 用完整 seqLen 计算的 kvBytes（错误方式）
  const kvFull = seqLen * config.modelConfig.numKvHeads *
    config.modelConfig.headDim * config.dtypeSize * 2;
  assert.strictEqual(result.allGatherBytes, kvPerRank);
  assert.notStrictEqual(result.allGatherBytes, kvFull);
});

// ==========================================
// 集成测试
// ==========================================

// T8: cp_size=4 forward_batch 后 cpCommTicks > 0
test("T8 cp_size=4 forwardBatch cpCommTicks > 0", () => {
  const config = cpConfig(4);
  const engine = new MockEngine(config);
  engine.forwardBatch(1024);
  assert.ok(engine.metrics.parallel.cpCommTicks > 0,
    `cpCommTicks should be >0, got ${engine.metrics.parallel.cpCommTicks}`);
});

// T9: cp_size=1 forward_batch 后 cpCommTicks = 0
test("T9 cp_size=1 forwardBatch cpCommTicks = 0", () => {
  const config = cpConfig(1);
  const engine = new MockEngine(config);
  engine.forwardBatch(1024);
  assert.strictEqual(engine.metrics.parallel.cpCommTicks, 0);
  assert.strictEqual(engine.metrics.parallel.cpAllGatherCount, 0);
  assert.strictEqual(engine.metrics.parallel.cpSeqLenPerRank, 0);
});

// T10: cp_size=4 时 cpSeqLenPerRank 正确
test("T10 cp_size=4 seqLen=1024 cpSeqLenPerRank=256", () => {
  const config = cpConfig(4);
  const engine = new MockEngine(config);
  engine.forwardBatch(1024);
  assert.strictEqual(engine.metrics.parallel.cpSeqLenPerRank, 256);
});

// T11: CP + TP 组合：commTicksTotal 包含 cp
test("T11 CP+TP combined commTicksTotal includes cp", () => {
  const m = new ParallelMetrics();
  m.tpCommTicks = 100;
  m.cpCommTicks = 50;
  m.dpAttnCommTicks = 0;
  m.epCommTicks = 0;
  m.ppSendRecvTicks = 0;
  assert.strictEqual(m.commTicksTotal, 150);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: seq_len=0
test("B1 seqLen=0 kv_bytes=0, commTicks latency-based", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(0);
  assert.strictEqual(result.allGatherBytes, 0);
  assert.ok(result.commTicks > 0, `commTicks should include latency, got ${result.commTicks}`);
});

// B2: seq_len < cp_size
test("B2 seqLen=2 cpSize=4 seqLenPerRank=1", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(2);
  assert.strictEqual(result.seqLenPerRank, 1);
});

// B3: cp_size=tp_size
test("B3 cp_size=tp_size=4 CPSimulator works", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(1024);
  assert.ok(result.commTicks > 0);
  assert.strictEqual(result.seqLenPerRank, 256);
});

// B4: num_layers=1 — 验证单层场景精确性
test("B4 numLayers=1 precise total commTicks", () => {
  const config1 = cpConfig(4, {
    modelConfig: { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, numLayers: 1 },
  });
  const sim1 = new CPSimulator(config1, config1.modelConfig);
  const singleResult = sim1.simulateAttnForward(1024);

  const engine = new MockEngine(config1);
  engine.forwardBatch(1024);

  // num_layers=1 时总 cpCommTicks 应等于单层 commTicks
  assert.strictEqual(engine.metrics.parallel.cpCommTicks, singleResult.commTicks);
  assert.strictEqual(engine.metrics.parallel.cpAllGatherCount, 1);
});

// B4b: numLayers=1 vs numLayers=32 allGatherBytes 相同（单层不含 numLayers 因子）
test("B4b numLayers=1 vs numLayers=32 allGatherBytes identical per-layer", () => {
  const config32 = cpConfig(4);
  const config1 = cpConfig(4, {
    modelConfig: { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, numLayers: 1 },
  });
  const sim32 = new CPSimulator(config32, config32.modelConfig);
  const sim1 = new CPSimulator(config1, config1.modelConfig);
  const r32 = sim32.simulateAttnForward(1024);
  const r1 = sim1.simulateAttnForward(1024);
  // 单层 allGatherBytes 应相同（不含 numLayers 因子）
  assert.strictEqual(r1.allGatherBytes, r32.allGatherBytes,
    "Per-layer allGatherBytes should be the same regardless of numLayers");
  // 但总 cpCommTicks 应为 numLayers 倍
  const engine32 = new MockEngine(config32);
  engine32.forwardBatch(1024);
  const engine1 = new MockEngine(config1);
  engine1.forwardBatch(1024);
  assert.strictEqual(engine32.metrics.parallel.cpCommTicks,
    engine1.metrics.parallel.cpCommTicks * 32);
});

// B5: cpEfficiency=1.0 vs 0.90
test("B5 cpEfficiency=1.0 vs 0.90", () => {
  const config090 = cpConfig(4);
  const config100 = cpConfig(4, { cpEfficiency: 1.0 });
  const sim090 = new CPSimulator(config090, config090.modelConfig);
  const sim100 = new CPSimulator(config100, config100.modelConfig);
  const r090 = sim090.simulateAttnForward(1024);
  const r100 = sim100.simulateAttnForward(1024);
  assert.ok(r100.commTicks <= r090.commTicks,
    `efficiency=1.0 (${r100.commTicks}) should be <= efficiency=0.9 (${r090.commTicks})`);
});

// B6: 极大 seq_len（128K）
test("B6 large seqLen=131072 significant comm cost", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(131072);
  assert.ok(result.allGatherBytes > 0);
  assert.ok(result.commTicks > 0);
  assert.strictEqual(result.seqLenPerRank, 32768);
});

// ==========================================
// 导出验证
// ==========================================

test("T_export CPSimulator exported from sglang index", () => {
  assert.ok(CPSimulator !== undefined);
  assert.ok(typeof CPSimulator === "function");
});

test("T_export CPAttnResult interface fields exist", () => {
  const config = cpConfig(4);
  const sim = new CPSimulator(config, config.modelConfig);
  const result = sim.simulateAttnForward(1024);
  assert.ok("commTicks" in result);
  assert.ok("allGatherBytes" in result);
  assert.ok("seqLenPerRank" in result);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P5 (Issue #29) 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P5 验收测试通过 \u2713");
  process.exit(0);
}
