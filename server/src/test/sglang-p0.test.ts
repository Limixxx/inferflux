import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  SimCommGroupImpl,
  MockTPGroup,
  ParallelTopology,
  ParallelMetrics,
  SimulationMetrics,
} from "../sglang";
import type { SimulatorConfig, CommGroupType } from "../sglang";
import { SimCommGroup } from "../sglang/parallel/comm_group";

/**
 * Issue #21 验收测试 — P0: SimCommGroup/ParallelTopology/ParallelMetrics
 *
 * Run with:  npx ts-node src/test/sglang-p0.test.ts
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

// ==========================================
// SimCommGroup 测试
// ==========================================

// T1: allReduce size=1 返回 0
test("T1 allReduce size=1 returns 0", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 1, networkBandwidthGBps: 100, latencyUs: 5,
  });
  assert.strictEqual(g.allReduce(1024), 0);
});

// T2: allReduce size>1 返回正数
test("T2 allReduce size>1 returns positive", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  const result = g.allReduce(1024);
  assert.ok(result > 0, `Expected >0, got ${result}`);
  // Verify formula: ceil((2 * 1024 * 3 / 4 / 100000 + 5) / 1.0)
  const bwBytesPerUs = 100 * 1000; // 100000
  const expected = Math.ceil((2 * 1024 * 3 / 4 / bwBytesPerUs + 5) / 1.0);
  assert.strictEqual(result, expected);
});

// T3: allGather size=1 返回 0
test("T3 allGather size=1 returns 0", () => {
  const g = new SimCommGroup({
    groupType: "ep", size: 1, networkBandwidthGBps: 100, latencyUs: 5,
  });
  assert.strictEqual(g.allGather([1024, 2048]), 0);
});

// T4: allGather size>1 返回正数
test("T4 allGather size>1 returns positive", () => {
  const g = new SimCommGroup({
    groupType: "ep", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  const result = g.allGather([1024, 2048]);
  assert.ok(result > 0);
  const bwBytesPerUs = 100 * 1000;
  const totalBytes = 1024 + 2048;
  const expected = Math.ceil((totalBytes * 3 / bwBytesPerUs + 5) / 1.0);
  assert.strictEqual(result, expected);
});

// T5: allToAll size=1 返回 0
test("T5 allToAll size=1 returns 0", () => {
  const g = new SimCommGroup({
    groupType: "ep", size: 1, networkBandwidthGBps: 100, latencyUs: 5,
  });
  assert.strictEqual(g.allToAll([1024], [2048]), 0);
});

// T6: allToAll size>1 返回正数
test("T6 allToAll size>1 returns positive with latency*size", () => {
  const g = new SimCommGroup({
    groupType: "ep", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  const result = g.allToAll([1024, 2048], [512, 256]);
  assert.ok(result > 0);
  const bwBytesPerUs = 100 * 1000;
  const totalBytes = (1024 + 2048) + (512 + 256);
  const expected = Math.ceil((totalBytes * 4 / bwBytesPerUs + 5 * 4) / 1.0);
  assert.strictEqual(result, expected);
});

// T7: sendRecv 正常计算（不受 size=1 影响）
test("T7 sendRecv always computes", () => {
  const g1 = new SimCommGroup({
    groupType: "pp", size: 1, networkBandwidthGBps: 100, latencyUs: 5,
  });
  const result1 = g1.sendRecv(1024, 0);
  assert.ok(result1 > 0, `size=1 sendRecv should be >0, got ${result1}`);
  const bwBytesPerUs = 100 * 1000;
  const expected = Math.ceil((1024 / bwBytesPerUs + 5) / 1.0);
  assert.strictEqual(result1, expected);
});

// T8: barrier 为 noop
test("T8 barrier is noop", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  // Should not throw
  g.barrier();
});

// T9: efficiency 缩放
test("T9 efficiency scales result", () => {
  const g1 = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5, efficiency: 1.0,
  });
  const g05 = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5, efficiency: 0.5,
  });
  const r1 = g1.allReduce(1024);
  const r05 = g05.allReduce(1024);
  assert.ok(r05 > r1, `efficiency=0.5 result (${r05}) should be > efficiency=1.0 result (${r1})`);
  // With efficiency=0.5, result should be approximately double
  assert.ok(r05 >= r1 * 2 - 2, `r05=${r05} should be ~2x r1=${r1}`);
});

// T10: CommGroupType 全类型
test("T10 all CommGroupType values work", () => {
  const types: CommGroupType[] = ["tp", "ep", "pp", "cp", "dp_attn"];
  for (const t of types) {
    const g = new SimCommGroup({
      groupType: t, size: 2, networkBandwidthGBps: 100, latencyUs: 5,
    });
    assert.strictEqual(g.groupType, t);
    assert.strictEqual(g.size, 2);
  }
});

// ==========================================
// ParallelTopology 测试
// ==========================================

// T11: worldSize 计算正确
test("T11 worldSize = tp × dp × pp", () => {
  const t = new ParallelTopology({ tpSize: 2, dpSize: 4, ppSize: 3 });
  assert.strictEqual(t.worldSize, 24);
  assert.strictEqual(t.worldSize, t.tpSize * t.dpSize * t.ppSize);
});

// T12: rankToCoord/coordToRank 互逆
test("T12 rankToCoord and coordToRank are inverse", () => {
  const t = new ParallelTopology({ tpSize: 2, dpSize: 3, ppSize: 2 });
  for (let rank = 0; rank < t.worldSize; rank++) {
    const [tp, dp, pp] = t.rankToCoord(rank);
    assert.strictEqual(t.coordToRank(tp, dp, pp), rank);
  }
});

// T13: computeMoeRanks tp=8 dp=2 ep=2
test("T13 computeMoeRanks tp=8 dp=2 ep=2", () => {
  const t = new ParallelTopology({ tpSize: 8, dpSize: 2, epSize: 2 });
  // moe_dp_size = 2, moe_tp_size = max(1, 8/2/2) = 2
  // For tpRank=0: moe_dp_rank = 0 // (8//2) = 0 // 4 = 0
  //   inner_rank = 0 % 4 = 0
  //   moe_ep_rank = 0 // 2 = 0
  //   moe_tp_rank = 0 % 2 = 0
  const [dp0, ep0, tp0] = t.computeMoeRanks(0);
  assert.strictEqual(dp0, 0);
  assert.strictEqual(ep0, 0);
  assert.strictEqual(tp0, 0);

  // For tpRank=5: moe_dp_rank = 5 // 4 = 1
  //   inner_rank = 5 % 4 = 1
  //   moe_ep_rank = 1 // 2 = 0
  //   moe_tp_rank = 1 % 2 = 1
  const [dp5, ep5, tp5] = t.computeMoeRanks(5);
  assert.strictEqual(dp5, 1);
  assert.strictEqual(ep5, 0);
  assert.strictEqual(tp5, 1);

  // For tpRank=7: moe_dp_rank = 7 // 4 = 1
  //   inner_rank = 7 % 4 = 3
  //   moe_ep_rank = 3 // 2 = 1
  //   moe_tp_rank = 3 % 2 = 1
  const [dp7, ep7, tp7] = t.computeMoeRanks(7);
  assert.strictEqual(dp7, 1);
  assert.strictEqual(ep7, 1);
  assert.strictEqual(tp7, 1);
});

// T14: computeAttnRanks tp=8 cp=2
test("T14 computeAttnRanks tp=8 cp=2", () => {
  const t = new ParallelTopology({ tpSize: 8, cpSize: 2 });
  // attn_dp_size = 1 (enableDpAttention=false)
  // attn_tp_size = max(1, 8/1/2) = 4
  // For tpRank=0: attn_cp_rank = (0 // 4) % 2 = 0, attn_tp_rank = 0 % 4 = 0
  const [cp0, tp0] = t.computeAttnRanks(0);
  assert.strictEqual(cp0, 0);
  assert.strictEqual(tp0, 0);

  // For tpRank=5: attn_cp_rank = (5 // 4) % 2 = 1, attn_tp_rank = 5 % 4 = 1
  const [cp5, tp5] = t.computeAttnRanks(5);
  assert.strictEqual(cp5, 1);
  assert.strictEqual(tp5, 1);
});

// T15: ppStageLayers 32层 pp=4
test("T15 ppStageLayers 32 layers pp=4", () => {
  const t = new ParallelTopology({ ppSize: 4 });
  const stages = t.ppStageLayers(32);
  assert.strictEqual(stages.length, 4);
  for (const s of stages) {
    assert.strictEqual(s.end - s.start, 8);
  }
  assert.strictEqual(stages[0].start, 0);
  assert.strictEqual(stages[3].end, 32);
});

// T16: ppStageLayers 33层 pp=4
test("T16 ppStageLayers 33 layers pp=4", () => {
  const t = new ParallelTopology({ ppSize: 4 });
  const stages = t.ppStageLayers(33);
  assert.strictEqual(stages.length, 4);
  // 33 / 4 = 8 remainder 1, so first stage has 9 layers, rest 8
  assert.strictEqual(stages[0].end - stages[0].start, 9);
  for (let i = 1; i < 4; i++) {
    assert.strictEqual(stages[i].end - stages[i].start, 8);
  }
  assert.strictEqual(stages[3].end, 33);
});

// T17: cp_size 整除 tp_size 约束
test("T17 cp_size must divide tp_size", () => {
  assert.throws(
    () => new ParallelTopology({ tpSize: 8, cpSize: 3 }),
    /cp_size.*must divide tp_size/
  );
});

// T18: ep_size 整除 tp_size/cp_size 约束
test("T18 ep_size must divide tp_size/cp_size", () => {
  assert.throws(
    () => new ParallelTopology({ tpSize: 8, cpSize: 2, epSize: 3 }),
    /ep_size.*must divide/
  );
});

// ==========================================
// ParallelMetrics 测试
// ==========================================

// T19: 默认值全为 0/空
test("T19 ParallelMetrics defaults are 0/empty", () => {
  const m = new ParallelMetrics();
  assert.strictEqual(m.tpCommTicks, 0);
  assert.strictEqual(m.tpAllReduceCount, 0);
  assert.strictEqual(m.tpWeightBytes, 0);
  assert.deepStrictEqual(m.dpRankLoad, []);
  assert.deepStrictEqual(m.dpAllocatePagesPerRank, []);
  assert.strictEqual(m.dpAttnCommTicks, 0);
  assert.strictEqual(m.epCommTicks, 0);
  assert.strictEqual(m.epAllToAllCount, 0);
  assert.strictEqual(m.epCrossRankTokens, 0);
  assert.deepStrictEqual(m.epExpertLoad, []);
  assert.strictEqual(m.epRebalanceCostTicks, 0);
  assert.strictEqual(m.ppBubbleTicks, 0);
  assert.strictEqual(m.ppNumMicroBatches, 0);
  assert.strictEqual(m.ppSendRecvTicks, 0);
  assert.strictEqual(m.cpCommTicks, 0);
  assert.strictEqual(m.cpAllGatherCount, 0);
  assert.strictEqual(m.cpSeqLenPerRank, 0);
  assert.strictEqual(m.worldSize, 1);
  assert.strictEqual(m.tpSize, 1);
  assert.strictEqual(m.dpSize, 1);
  assert.strictEqual(m.epSize, 1);
  assert.strictEqual(m.ppSize, 1);
  assert.strictEqual(m.cpSize, 1);
});

// T20: commTicksTotal 计算
test("T20 commTicksTotal sums tp+dp+ep+pp+cp", () => {
  const m = new ParallelMetrics();
  m.tpCommTicks = 10;
  m.dpAttnCommTicks = 20;
  m.epCommTicks = 30;
  m.ppSendRecvTicks = 40;
  m.cpCommTicks = 50;
  assert.strictEqual(m.commTicksTotal, 150);
});

// T21: reset 清零
test("T21 reset clears all fields", () => {
  const m = new ParallelMetrics();
  m.tpCommTicks = 10;
  m.dpRankLoad = [1, 2];
  m.epExpertLoad = [3, 4];
  m.worldSize = 8;
  m.reset();
  assert.strictEqual(m.tpCommTicks, 0);
  assert.deepStrictEqual(m.dpRankLoad, []);
  assert.deepStrictEqual(m.epExpertLoad, []);
  assert.strictEqual(m.worldSize, 1);
});

// T22: summary 包含全部 22 字段
test("T22 summary contains all 22 fields", () => {
  const m = new ParallelMetrics();
  const s = m.summary();
  const keys = Object.keys(s);
  assert.strictEqual(keys.length, 23, `Expected 23 keys, got ${keys.length}: ${keys.join(",")}`);
  const expectedKeys = [
    "tpCommTicks", "tpAllReduceCount", "tpWeightBytes",
    "dpRankLoad", "dpAllocatePagesPerRank", "dpAttnCommTicks",
    "epCommTicks", "epAllToAllCount", "epCrossRankTokens", "epExpertLoad", "epRebalanceCostTicks",
    "ppBubbleTicks", "ppNumMicroBatches", "ppSendRecvTicks",
    "cpCommTicks", "cpAllGatherCount", "cpSeqLenPerRank",
    "worldSize", "tpSize", "dpSize", "epSize", "ppSize", "cpSize",
  ];
  for (const k of expectedKeys) {
    assert.ok(k in s, `Missing key: ${k}`);
  }
});

// ==========================================
// MockTPGroup 测试
// ==========================================

// T23: MockTPGroup(1) allReduce=0
test("T23 MockTPGroup(1) allReduce=0", () => {
  const m = new MockTPGroup(1, DEFAULT_SIMULATOR_CONFIG);
  assert.strictEqual(m.allReduce(1024), 0);
});

// T24: MockTPGroup(2) allReduce>0
test("T24 MockTPGroup(2) allReduce>0", () => {
  const m = new MockTPGroup(2, DEFAULT_SIMULATOR_CONFIG);
  const result = m.allReduce(1024);
  assert.ok(result > 0, `Expected >0, got ${result}`);
});

// T25: mockAllReduceSum 兼容
test("T25 mockAllReduceSum compatible with allReduce", () => {
  const m = new MockTPGroup(4, DEFAULT_SIMULATOR_CONFIG);
  assert.strictEqual(m.mockAllReduceSum(2048), m.allReduce(2048));
});

// ==========================================
// SimulatorConfig 测试
// ==========================================

// T26: DEFAULT 含新增字段
test("T26 DEFAULT_SIMULATOR_CONFIG has new P0 fields", () => {
  assert.strictEqual(DEFAULT_SIMULATOR_CONFIG.networkBandwidthGBps, 100);
  assert.strictEqual(DEFAULT_SIMULATOR_CONFIG.networkLatencyUs, 5);
  assert.strictEqual(DEFAULT_SIMULATOR_CONFIG.tpEfficiency, 0.95);
  assert.strictEqual(DEFAULT_SIMULATOR_CONFIG.epEfficiency, 0.90);
  assert.strictEqual(DEFAULT_SIMULATOR_CONFIG.cpEfficiency, 0.90);
});

// T27: 新字段类型正确
test("T27 new P0 fields are numbers", () => {
  assert.strictEqual(typeof DEFAULT_SIMULATOR_CONFIG.networkBandwidthGBps, "number");
  assert.strictEqual(typeof DEFAULT_SIMULATOR_CONFIG.networkLatencyUs, "number");
  assert.strictEqual(typeof DEFAULT_SIMULATOR_CONFIG.tpEfficiency, "number");
  assert.strictEqual(typeof DEFAULT_SIMULATOR_CONFIG.epEfficiency, "number");
  assert.strictEqual(typeof DEFAULT_SIMULATOR_CONFIG.cpEfficiency, "number");
});

// ==========================================
// 集成测试
// ==========================================

// T28: SimulationMetrics.parallel 存在
test("T28 SimulationMetrics.parallel exists", () => {
  const sm = new SimulationMetrics();
  assert.ok(sm.parallel instanceof ParallelMetrics);
  assert.strictEqual(sm.parallel.tpCommTicks, 0);
  sm.parallel.tpCommTicks = 42;
  sm.reset();
  assert.strictEqual(sm.parallel.tpCommTicks, 0);
});

// T29: 全并行 size=1 退化为单实例
test("T29 size=1 full degeneration to single instance", () => {
  const topo = new ParallelTopology({ tpSize: 1, dpSize: 1, ppSize: 1 });
  assert.strictEqual(topo.worldSize, 1);
  const [tp, dp, pp] = topo.rankToCoord(0);
  assert.strictEqual(tp, 0);
  assert.strictEqual(dp, 0);
  assert.strictEqual(pp, 0);

  const types: CommGroupType[] = ["tp", "ep", "pp", "cp", "dp_attn"];
  for (const t of types) {
    const g = new SimCommGroup({
      groupType: t, size: 1, networkBandwidthGBps: 100, latencyUs: 5,
    });
    assert.strictEqual(g.allReduce(1024), 0, `allReduce for ${t} should be 0`);
    assert.strictEqual(g.allGather([1024]), 0, `allGather for ${t} should be 0`);
    assert.strictEqual(g.allToAll([1024], [1024]), 0, `allToAll for ${t} should be 0`);
  }
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: SimCommGroup bytes=0
test("B1 allReduce bytes=0 returns latency only", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  const result = g.allReduce(0);
  // 2*0*(3)/4 / bw + 5 = 5, ceil(5/1.0) = 5
  assert.strictEqual(result, 5);
});

// B2: SimCommGroup bandwidth 极大 → ticks 趋近于 latency（ceil 取整可能+1）
test("B2 bandwidth very large → ticks approach latency", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 1e9, latencyUs: 5,
  });
  const result = g.allReduce(1024);
  // bw_bytes_per_us = 1e9 * 1000 = 1e12
  // 2*1024*3/4 / 1e12 + 5 = 0.000001536 + 5 = 5.000001536
  // ceil(5.000001536) = 6
  assert.strictEqual(result, 6);
});

// B3: SimCommGroup bandwidth=0 → Infinity
test("B3 bandwidth=0 returns Infinity", () => {
  const g = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 0, latencyUs: 5,
  });
  const result = g.allReduce(1024);
  assert.strictEqual(result, Infinity);
});

// B4: ParallelTopology world_size=1
test("B4 world_size=1 rankToCoord(0)=(0,0,0)", () => {
  const t = new ParallelTopology();
  const [tp, dp, pp] = t.rankToCoord(0);
  assert.strictEqual(tp, 0);
  assert.strictEqual(dp, 0);
  assert.strictEqual(pp, 0);
});

// B5: ParallelTopology ppStageLayers pp=1
test("B5 ppStageLayers pp=1 covers all layers", () => {
  const t = new ParallelTopology({ ppSize: 1 });
  const stages = t.ppStageLayers(32);
  assert.strictEqual(stages.length, 1);
  assert.strictEqual(stages[0].start, 0);
  assert.strictEqual(stages[0].end, 32);
});

// B6: ParallelMetrics commTicksTotal 各项为 0 → 总和为 0
test("B6 commTicksTotal all zero → total is 0", () => {
  const m = new ParallelMetrics();
  assert.strictEqual(m.commTicksTotal, 0);
});

// B7: MockTPGroup tp_size=1 mockAllReduceSum(0)
test("B7 MockTPGroup(1) mockAllReduceSum(0) returns 0", () => {
  const m = new MockTPGroup(1, DEFAULT_SIMULATOR_CONFIG);
  assert.strictEqual(m.mockAllReduceSum(0), 0);
});

// B8: efficiency=1.0 → 与无效率因子一致
test("B8 efficiency=1.0 same as no efficiency factor", () => {
  const gExplicit = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5, efficiency: 1.0,
  });
  const gDefault = new SimCommGroup({
    groupType: "tp", size: 4, networkBandwidthGBps: 100, latencyUs: 5,
  });
  assert.strictEqual(gExplicit.allReduce(1024), gDefault.allReduce(1024));
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P0 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P0 验收测试通过 \u2713");
  process.exit(0);
}
