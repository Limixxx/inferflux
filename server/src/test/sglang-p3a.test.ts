import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  SimCommGroupImpl,
  ParallelTopology,
  ParallelMetrics,
  SimMoeBackend,
  MockEngine,
} from "../sglang";
import type { SimulatorConfig, ModelConfig } from "../sglang";
import { SimCommGroup } from "../sglang/parallel/comm_group";

/**
 * Issue #26 验收测试 — P3a: SimMoeBackend（EP 路由 + all-to-all 正反）3 种 moe_routing_mode
 *
 * Run with:  npx ts-node src/test/sglang-p3a.test.ts
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

function makeMoeModelConfig(overrides: Partial<ModelConfig> = {}): ModelConfig {
  return {
    numLayers: 8,
    hiddenSize: 4096,
    numKvHeads: 8,
    headDim: 128,
    vocabSize: 128256,
    isMoe: true,
    numExperts: 8,
    moeIntermediateSize: 14336,
    moeTopK: 2,
    intermediateSize: 14336,
    numAttentionHeads: 32,
    rmsNormEps: 1e-6,
    ropeTheta: 10000.0,
    maxPositionEmbeddings: 8192,
    ...overrides,
  };
}

function makeMoeBackend(
  modelOverrides: Partial<ModelConfig> = {},
  configOverrides: Partial<SimulatorConfig> = {},
  seed?: number
): SimMoeBackend {
  const modelConfig = makeMoeModelConfig(modelOverrides);
  const epSize = configOverrides.epSize ?? 2;
  // ParallelTopology 约束: ep_size must divide tp_size/cp_size
  // 当 epSize > 1 时确保 tpSize >= epSize
  const minTpSize = epSize > 1 ? epSize : (configOverrides.tpSize ?? 1);
  const config = makeConfig({
    modelConfig,
    epSize,
    tpSize: minTpSize,
    moeRoutingMode: configOverrides.moeRoutingMode ?? "mock",
    ...configOverrides,
  });

  const topology = new ParallelTopology({
    tpSize: config.tpSize,
    dpSize: config.dpSize,
    epSize: config.epSize,
    ppSize: config.ppSize,
    cpSize: config.cpSize,
  });

  const epCommGroup = new SimCommGroup({
    groupType: "ep",
    size: config.epSize,
    networkBandwidthGBps: config.networkBandwidthGBps,
    latencyUs: config.networkLatencyUs,
    efficiency: config.epEfficiency,
  });

  const metrics = new ParallelMetrics();

  return new SimMoeBackend({
    modelConfig,
    topology,
    config,
    epCommGroup,
    metrics,
    seed,
  });
}

// ==========================================
// T1: 构造 — 专家均匀分片
// ==========================================
test("T1 构造 — expertsPerRank numExperts=8 epSize=2", () => {
  const moe = makeMoeBackend();
  assert.deepStrictEqual(moe.expertsPerRank, [4, 4]);
});

// T2: _expertToRank 正确映射
test("T2 _expertToRank numExperts=8 epSize=2", () => {
  const moe = makeMoeBackend();
  assert.strictEqual(moe._expertToRank(0), 0);
  assert.strictEqual(moe._expertToRank(3), 0);
  assert.strictEqual(moe._expertToRank(4), 1);
  assert.strictEqual(moe._expertToRank(7), 1);
});

// T2a: _expertToRank 与 topology 一致
test("T2a _expertToRank 与 topology.computeMoeRanks 一致", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  // 构造时自检已验证，此处再做显式断言
  const topo = moe.topology;
  const expertsPerRank = moe.expertsPerRank;
  for (let expertId = 0; expertId < moe.numExperts; expertId++) {
    let foundEpRank = -1;
    for (let tpRank = 0; tpRank < topo.tpSize; tpRank++) {
      const [, epRank] = topo.computeMoeRanks(tpRank);
      let eOffset = 0;
      for (let r = 0; r < epRank; r++) {
        eOffset += expertsPerRank[r];
      }
      if (expertId >= eOffset && expertId < eOffset + expertsPerRank[epRank]) {
        foundEpRank = epRank;
        break;
      }
    }
    if (foundEpRank !== -1) {
      assert.strictEqual(moe._expertToRank(expertId), foundEpRank,
        `expert ${expertId}: expected rank ${foundEpRank}, got ${moe._expertToRank(expertId)}`);
    }
  }
});

// T3: _expertToRank 非均分
test("T3 _expertToRank 非均分 numExperts=7 epSize=2", () => {
  const moe = makeMoeBackend({ numExperts: 7 }, { epSize: 2, tpSize: 2 });
  assert.deepStrictEqual(moe.expertsPerRank, [4, 3]);
  assert.strictEqual(moe._expertToRank(5), 1);
});

// T4: hash 路由可复现
test("T4 hash 路由可复现", () => {
  const moe = makeMoeBackend({}, { moeRoutingMode: "hash", tpSize: 2 });
  const tokenIds = [1, 2, 3, 4, 5];
  const r1 = moe._routeTokens(tokenIds, 0);
  const r2 = moe._routeTokens(tokenIds, 0);
  assert.deepStrictEqual(r1.rankDistribution, r2.rankDistribution);
});

// T4a: hash 路由 layer 区分
test("T4a hash 路由不同 layerIdx 结果不同", () => {
  const moe = makeMoeBackend({}, { moeRoutingMode: "hash", tpSize: 2 });
  const tokenIds = [1, 2, 3, 4, 5];
  const r1 = moe._routeTokens(tokenIds, 0);
  const r2 = moe._routeTokens(tokenIds, 1);
  // 不同 layer 的 rankDistribution 应不同
  let diff = false;
  for (const [rank, count] of r1.rankDistribution) {
    if (count !== r2.rankDistribution.get(rank)) { diff = true; break; }
  }
  assert.ok(diff, "hash routing should differ across layers");
});

// T5: hash 路由分布合理
test("T5 hash 路由分布合理 — 各 rank 负载方差 < 均值的 20%", () => {
  const moe = makeMoeBackend(
    { numExperts: 8, moeTopK: 2 },
    { moeRoutingMode: "hash", epSize: 2, tpSize: 2 }
  );
  const tokenIds = Array.from({ length: 10000 }, (_, i) => i);
  const r = moe._routeTokens(tokenIds, 0);
  const loads: number[] = [];
  for (const [, count] of r.rankDistribution) { loads.push(count); }
  const mean = loads.reduce((s, v) => s + v, 0) / loads.length;
  const variance = loads.reduce((s, v) => s + (v - mean) ** 2, 0) / loads.length;
  const stdDev = Math.sqrt(variance);
  assert.ok(stdDev < mean * 0.2,
    `stdDev=${stdDev.toFixed(2)} should be < 20% of mean=${mean.toFixed(2)}`);
});

// T6: mock 路由平衡方差低
test("T6 mock 路由平衡方差低", () => {
  const moe = makeMoeBackend(
    { numExperts: 8, moeTopK: 2 },
    { epSize: 2, tpSize: 2 }
  );
  const tokenIds = Array.from({ length: 1000 }, (_, i) => i);
  const r = moe._routeTokens(tokenIds, 0);
  const loads: number[] = [];
  for (const [, count] of r.rankDistribution) { loads.push(count); }
  const mean = loads.reduce((s, v) => s + v, 0) / loads.length;
  const variance = loads.reduce((s, v) => s + (v - mean) ** 2, 0) / loads.length;
  assert.ok(variance / (mean * mean) < 0.05,
    `variance/mean^2 should be < 5%`);
});

// T7: simulated 路由可复现
test("T7 simulated 路由可复现", () => {
  const moe = makeMoeBackend({}, { moeRoutingMode: "simulated", tpSize: 2 }, 42);
  const tokenIds = [1, 2, 3, 4, 5];
  const r1 = moe._routeTokens(tokenIds, 0);
  const r2 = moe._routeTokens(tokenIds, 0);
  assert.deepStrictEqual(r1.rankDistribution, r2.rankDistribution);
});

// T7a: simulated 不同 seed 不同结果
test("T7a simulated 不同 seed 不同结果", () => {
  const moe1 = makeMoeBackend({}, { moeRoutingMode: "simulated", tpSize: 2 }, 42);
  const moe2 = makeMoeBackend({}, { moeRoutingMode: "simulated", tpSize: 2 }, 99);
  const tokenIds = [1, 2, 3, 4, 5];
  const r1 = moe1._routeTokens(tokenIds, 0);
  const r2 = moe2._routeTokens(tokenIds, 0);
  let diff = false;
  for (const [rank, count] of r1.rankDistribution) {
    if (count !== r2.rankDistribution.get(rank)) { diff = true; break; }
  }
  assert.ok(diff, "different seeds should produce different routing");
});

// T8: simulated 路由分布非退化
test("T8 simulated 路由分布非退化 — 各 rank 有非零 token", () => {
  const moe = makeMoeBackend(
    { numExperts: 8, moeTopK: 2 },
    { moeRoutingMode: "simulated", epSize: 2, tpSize: 2 },
    0
  );
  const tokenIds = Array.from({ length: 1000 }, (_, i) => i);
  const r = moe._routeTokens(tokenIds, 0);
  for (const [rank, count] of r.rankDistribution) {
    assert.ok(count > 0, `rank ${rank} should have non-zero tokens, got ${count}`);
  }
});

// T9: all-to-all 正反字节数守恒
test("T9 all-to-all 正反字节数守恒", () => {
  const moe = makeMoeBackend(
    { numExperts: 8, moeTopK: 2 },
    { epSize: 2, tpSize: 2 }
  );
  const tokenIds = Array.from({ length: 100 }, (_, i) => i);
  const route = moe._routeTokens(tokenIds, 0);
  const bytesPerToken = moe.hiddenSize * moe.dtypeSize;
  const sendSizes: number[] = [];
  const recvSizes: number[] = [];
  for (let r = 0; r < moe.epSize; r++) {
    const tokenCount = route.rankDistribution.get(r) ?? 0;
    sendSizes.push(tokenCount * bytesPerToken);
    recvSizes.push(tokenCount * bytesPerToken);
  }
  const totalSend = sendSizes.reduce((s, v) => s + v, 0);
  const totalRecv = recvSizes.reduce((s, v) => s + v, 0);
  assert.strictEqual(totalSend, totalRecv, "send and recv total bytes should be equal");
  // 反向：reverseSendSizes = recvSizes, reverseRecvSizes = sendSizes
  const revTotalSend = recvSizes.reduce((s, v) => s + v, 0);
  const revTotalRecv = sendSizes.reduce((s, v) => s + v, 0);
  assert.strictEqual(revTotalSend, revTotalRecv);
});

// T9a: comm_ticks 与公式一致
test("T9a comm_ticks 与公式一致", () => {
  const config = makeConfig({
    modelConfig: makeMoeModelConfig({ numExperts: 8, moeTopK: 2 }),
    epSize: 2,
    tpSize: 2,
    networkBandwidthGBps: 100,
    networkLatencyUs: 5,
    epEfficiency: 0.9,
    moeRoutingMode: "mock",
  });

  const topology = new ParallelTopology({ tpSize: 2, epSize: 2 });
  const epCommGroup = new SimCommGroup({
    groupType: "ep", size: 2,
    networkBandwidthGBps: 100, latencyUs: 5, efficiency: 0.9,
  });
  const metrics = new ParallelMetrics();
  const moe = new SimMoeBackend({
    modelConfig: config.modelConfig,
    topology,
    config,
    epCommGroup,
    metrics,
  });

  const tokenIds = Array.from({ length: 100 }, (_, i) => i);
  const result = moe.forward(tokenIds, 0);

  // 手动计算
  const route = moe._routeTokens(tokenIds, 0);
  const bytesPerToken = moe.hiddenSize * moe.dtypeSize;
  const sendSizes: number[] = [];
  const recvSizes: number[] = [];
  for (let r = 0; r < moe.epSize; r++) {
    const tc = route.rankDistribution.get(r) ?? 0;
    sendSizes.push(tc * bytesPerToken);
    recvSizes.push(tc * bytesPerToken);
  }
  const totalBytes = sendSizes.reduce((s, v) => s + v, 0) + recvSizes.reduce((s, v) => s + v, 0);
  const bwBytesPerUs = 100 * 1000;
  const rawCost = (totalBytes * 2 / bwBytesPerUs + 5 * 2);  // size=2
  const expectedPerA2A = Math.ceil(rawCost / 0.9);
  const expectedTotal = expectedPerA2A * 2;  // fwd + rev
  assert.strictEqual(result.commTicks, expectedTotal,
    `commTicks=${result.commTicks} should equal expected=${expectedTotal}`);
});

// T10: crossRankTokens 非负
test("T10 crossRankTokens 非负", () => {
  for (const mode of ["mock", "hash", "simulated"] as const) {
    const moe = makeMoeBackend({}, { moeRoutingMode: mode, tpSize: 2 });
    const tokenIds = [1, 2, 3, 4, 5];
    const r = moe._routeTokens(tokenIds, 0);
    assert.ok(r.crossRankTokens >= 0, `mode=${mode}: crossRankTokens should be >= 0`);
  }
});

// T11: epSize=1 退化 — commTicks=0
test("T11 epSize=1 退化 — commTicks=0", () => {
  const moe = makeMoeBackend({}, { epSize: 1, tpSize: 1 });
  const result = moe.forward([1, 2, 3], 0);
  assert.strictEqual(result.commTicks, 0);
});

// T12: epSize=1 退化 — crossRankTokens=0
test("T12 epSize=1 退化 — crossRankTokens=0", () => {
  const moe = makeMoeBackend({}, { epSize: 1, tpSize: 1 });
  const result = moe.forward([1, 2, 3], 0);
  assert.strictEqual(result.crossRankTokens, 0);
});

// T13: epSize>1 — forward 返回 commTicks>0
test("T13 epSize>1 — forward 返回 commTicks>0", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const result = moe.forward(Array.from({ length: 100 }, (_, i) => i), 0);
  assert.ok(result.commTicks > 0, `commTicks should be > 0, got ${result.commTicks}`);
});

// T14: 指标写入 epCommTicks
test("T14 指标 epCommTicks 累加", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const tokenIds = Array.from({ length: 50 }, (_, i) => i);
  const r1 = moe.forward(tokenIds, 0);
  const r2 = moe.forward(tokenIds, 1);
  assert.strictEqual(moe.metrics.epCommTicks, r1.commTicks + r2.commTicks);
});

// T15: 指标写入 epAllToAllCount
test("T15 指标 epAllToAllCount 每次 forward 增加 2", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const tokenIds = [1, 2, 3];
  moe.forward(tokenIds, 0);
  assert.strictEqual(moe.metrics.epAllToAllCount, 2);
  moe.forward(tokenIds, 1);
  assert.strictEqual(moe.metrics.epAllToAllCount, 4);
});

// T16: 指标写入 epCrossRankTokens
test("T16 指标 epCrossRankTokens 累加", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const tokenIds = [1, 2, 3];
  const r1 = moe.forward(tokenIds, 0);
  const r2 = moe.forward(tokenIds, 1);
  assert.strictEqual(moe.metrics.epCrossRankTokens, r1.crossRankTokens + r2.crossRankTokens);
});

// T17: 指标写入 epExpertLoad
test("T17 指标 epExpertLoad — length=numExperts, 总和=batchSize×moeTopK", () => {
  const moe = makeMoeBackend({ numExperts: 8, moeTopK: 2 }, { epSize: 2, tpSize: 2 });
  const tokenIds = Array.from({ length: 100 }, (_, i) => i);
  moe.forward(tokenIds, 0);
  assert.strictEqual(moe.metrics.epExpertLoad.length, 8);
  const total = moe.metrics.epExpertLoad.reduce((s, v) => s + v, 0);
  assert.strictEqual(total, 100 * 2, `total expert load should be 200, got ${total}`);
});

// T17a: 指标 epExpertLoad 多次 forward 累加
test("T17a 指标 epExpertLoad 多次 forward 累加", () => {
  const moe = makeMoeBackend({ numExperts: 8, moeTopK: 2 }, { epSize: 2, tpSize: 2 });
  const tokenIds = [1, 2, 3, 4, 5];
  const r1 = moe._routeTokens(tokenIds, 0);
  moe.forward(tokenIds, 0);
  const counts1 = [...moe.metrics.epExpertLoad];

  const r2 = moe._routeTokens(tokenIds, 1);
  moe.forward(tokenIds, 1);

  for (let e = 0; e < moe.numExperts; e++) {
    assert.strictEqual(moe.metrics.epExpertLoad[e], counts1[e] + r2.expertCounts[e],
      `expert ${e}: expected ${counts1[e] + r2.expertCounts[e]}, got ${moe.metrics.epExpertLoad[e]}`);
  }
});

// T18: isMoe=false — 不创建实例
test("T18 isMoe=false — MockEngine 不创建 moeBackend", () => {
  const config = makeConfig({
    modelConfig: {
      ...DEFAULT_SIMULATOR_CONFIG.modelConfig,
      isMoe: false,
    },
  });
  const engine = new MockEngine(config);
  assert.strictEqual(engine.moeBackend, undefined);
});

// T19: 多层 forward 指标累加
test("T19 多层 forward 指标累加 — 3 层 MoE", () => {
  const moe = makeMoeBackend({ numExperts: 8, moeTopK: 2, numLayers: 3 }, { epSize: 2, tpSize: 2 });
  const tokenIds = Array.from({ length: 50 }, (_, i) => i);
  let totalComm = 0;
  for (let layer = 0; layer < 3; layer++) {
    totalComm += moe.forward(tokenIds, layer).commTicks;
  }
  assert.strictEqual(moe.metrics.epCommTicks, totalComm);
  assert.strictEqual(moe.metrics.epAllToAllCount, 6);  // 3 layers × 2
});

// T20: 多 batch forward 指标累加
test("T20 多 batch forward 指标累加", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const batch1 = Array.from({ length: 30 }, (_, i) => i);
  const batch2 = Array.from({ length: 50 }, (_, i) => i);
  const r1 = moe.forward(batch1, 0);
  const r2 = moe.forward(batch2, 0);
  assert.strictEqual(moe.metrics.epCommTicks, r1.commTicks + r2.commTicks);
  assert.strictEqual(moe.metrics.epAllToAllCount, 4);
  assert.strictEqual(moe.metrics.epCrossRankTokens, r1.crossRankTokens + r2.crossRankTokens);
});

// T21: hash 模式 seed=0 正常运行
test("T21 hash 模式 seed=0 正常运行", () => {
  const moe = makeMoeBackend({}, { moeRoutingMode: "hash", tpSize: 2 }, 0);
  const tokenIds = [1, 2, 3, 4, 5];
  const r = moe._routeTokens(tokenIds, 0);
  assert.ok(r.rankDistribution.size > 0);
  assert.ok(r.crossRankTokens >= 0);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: batchSize=0
test("B1 batchSize=0 — rankDistribution 值全为 0, crossRankTokens=0", () => {
  // epSize=1 确保 commTicks=0（allToAll noop 退化）
  const moe = makeMoeBackend({}, { epSize: 1, tpSize: 1 });
  const r = moe.forward([], 0);
  assert.strictEqual(r.commTicks, 0);
  assert.strictEqual(r.crossRankTokens, 0);
  // rankDistribution 已初始化但值全为 0
  for (const [, count] of r.rankDistribution) {
    assert.strictEqual(count, 0);
  }
});

// B2: numExperts=1, moeTopK=1
test("B2 numExperts=1, moeTopK=1 — 所有 token 路由到唯一专家", () => {
  const moe = makeMoeBackend(
    { numExperts: 1, moeTopK: 1 },
    { epSize: 1, tpSize: 1 }
  );
  const tokenIds = [1, 2, 3];
  const r = moe.forward(tokenIds, 0);
  assert.strictEqual(r.rankDistribution.get(0), 3);
  assert.strictEqual(r.crossRankTokens, 0);
});

// B3: moeTopK=numExperts
test("B3 moeTopK=numExperts — 每个 token 选择所有专家", () => {
  const moe = makeMoeBackend(
    { numExperts: 8, moeTopK: 8 },
    { epSize: 2, tpSize: 2 }
  );
  const tokenIds = [1, 2];
  const r = moe.forward(tokenIds, 0);
  const total = Array.from(r.rankDistribution.values()).reduce((s, v) => s + v, 0);
  assert.strictEqual(total, 2 * 8, `total tokens should be 16, got ${total}`);
});

// B4: epSize > numExperts → divEven 抛出
test("B4 epSize > numExperts → divEven throws", () => {
  assert.throws(
    () => makeMoeBackend({ numExperts: 2 }, { epSize: 4, tpSize: 4 }),
    /divEven/
  );
});

// B5: numExperts % epSize !== 0
test("B5 numExperts=7 epSize=2 非整除分配正确", () => {
  const moe = makeMoeBackend({ numExperts: 7 }, { epSize: 2, tpSize: 2 });
  assert.deepStrictEqual(moe.expertsPerRank, [4, 3]);
  assert.strictEqual(moe.expertToRankMap[0], 0);
  assert.strictEqual(moe.expertToRankMap[3], 0);
  assert.strictEqual(moe.expertToRankMap[4], 1);
  assert.strictEqual(moe.expertToRankMap[6], 1);
});

// B6: seed=0 simulated 路由正常
test("B6 seed=0 simulated 路由正常", () => {
  const moe = makeMoeBackend(
    {},
    { moeRoutingMode: "simulated", tpSize: 2 },
    0
  );
  const tokenIds = [1, 2, 3, 4, 5];
  const r = moe._routeTokens(tokenIds, 0);
  assert.ok(r.crossRankTokens >= 0);
  assert.ok(r.rankDistribution.size > 0);
});

// B7: 单 token batch
test("B7 单 token batch — 路由正常，不除零", () => {
  const moe = makeMoeBackend({}, { epSize: 2, tpSize: 2 });
  const r = moe.forward([42], 0);
  assert.ok(r.commTicks >= 0);
  assert.ok(r.crossRankTokens >= 0);
  const total = Array.from(r.rankDistribution.values()).reduce((s, v) => s + v, 0);
  assert.strictEqual(total, moe.moeTopK, `total should be moeTopK=${moe.moeTopK}`);
});

// B8: epSize=1, numExperts=1, moeTopK=1 极端退化
test("B8 epSize=1, numExperts=1, moeTopK=1 极端退化", () => {
  const moe = makeMoeBackend(
    { numExperts: 1, moeTopK: 1 },
    { epSize: 1, tpSize: 1 }
  );
  const r = moe.forward([1, 2, 3], 0);
  assert.strictEqual(r.commTicks, 0);
  assert.strictEqual(r.crossRankTokens, 0);
  assert.strictEqual(r.rankDistribution.get(0), 3);
});

// ==========================================
// MockEngine 集成测试
// ==========================================

test("MockEngine isMoe=true 创建 moeBackend", () => {
  const config = makeConfig({
    modelConfig: makeMoeModelConfig(),
    epSize: 2,
    tpSize: 2,
  });
  const engine = new MockEngine(config);
  assert.ok(engine.moeBackend instanceof SimMoeBackend);
});

test("MockEngine forwardBatch MoE 层返回 commTicks", () => {
  const config = makeConfig({
    modelConfig: makeMoeModelConfig({ numExperts: 8, moeTopK: 2, numLayers: 4 }),
    epSize: 2,
    tpSize: 2,
  });
  const engine = new MockEngine(config);
  const tokenIds = Array.from({ length: 100 }, (_, i) => i);
  const ticks = engine.forwardBatch(tokenIds, 0);
  assert.ok(ticks > 0, `MoE layer should return commTicks > 0, got ${ticks}`);
});

test("MockEngine forwardBatch 非 MoE 层返回 0", () => {
  const config = makeConfig({
    modelConfig: { ...DEFAULT_SIMULATOR_CONFIG.modelConfig, isMoe: false },
  });
  const engine = new MockEngine(config);
  const ticks = engine.forwardBatch([1, 2, 3], 0);
  assert.strictEqual(ticks, 0);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P3a 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P3a 验收测试通过 \u2713");
  process.exit(0);
}
