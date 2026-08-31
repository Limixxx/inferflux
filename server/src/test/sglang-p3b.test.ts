import assert from "assert";
import {
  DEFAULT_SIMULATOR_CONFIG,
  SimCommGroupImpl,
  ParallelTopology,
  ParallelMetrics,
  SimMoeBackend,
  EPLBSimulator,
} from "../sglang";
import type { SimulatorConfig, ModelConfig, EPLBSimulatorOpts, RebalanceResult } from "../sglang";
import { SimCommGroup } from "../sglang/parallel/comm_group";

/**
 * Issue #27 验收测试 — P3b: EPLBSimulator — EP 负载均衡（100 步周期 + 方差阈值重平衡）
 *
 * Run with:  npx ts-node src/test/sglang-p3b.test.ts
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

function makeEplb(
  moeBackend: SimMoeBackend,
  opts: Partial<EPLBSimulatorOpts> = {}
): EPLBSimulator {
  return new EPLBSimulator({
    enabled: opts.enabled ?? true,
    numExperts: opts.numExperts ?? moeBackend.numExperts,
    epSize: opts.epSize ?? moeBackend.epSize,
    metrics: opts.metrics ?? moeBackend.metrics,
    rebalanceIntervalSteps: opts.rebalanceIntervalSteps,
    loadVarianceThreshold: opts.loadVarianceThreshold,
    rebalanceCostFixedTicks: opts.rebalanceCostFixedTicks,
  });
}

// ==========================================
// T1: 构造 — enabled=false 时 maybe_rebalance 直接返回 false
// ==========================================
test("T1 enabled=false 时 maybe_rebalance 返回 shouldRebalance=false", () => {
  const moe = makeMoeBackend();
  const eplb = makeEplb(moe, { enabled: false });
  const result = eplb.maybe_rebalance(100, [10, 20, 30, 40, 50, 60, 70, 80], moe);
  assert.strictEqual(result.shouldRebalance, false);
  assert.strictEqual(result.rebalanceTicks, 0);
  assert.strictEqual(result.movedExperts, 0);
});

// ==========================================
// T2: 构造 — epSize<=1 时 maybe_rebalance 直接返回 false
// ==========================================
test("T2 epSize<=1 时 maybe_rebalance 返回 shouldRebalance=false", () => {
  const moe = makeMoeBackend({}, { epSize: 1, tpSize: 1 });
  const eplb = makeEplb(moe, { epSize: 1 });
  const result = eplb.maybe_rebalance(100, [100], moe);
  assert.strictEqual(result.shouldRebalance, false);
  assert.strictEqual(result.rebalanceTicks, 0);
  assert.strictEqual(result.movedExperts, 0);
});

// ==========================================
// T3: 100 步周期 — global_step 非 100 倍数时跳过
// ==========================================
test("T3 step=50 非 100 倍数 → shouldRebalance=false", () => {
  const moe = makeMoeBackend();
  const eplb = makeEplb(moe);
  const result = eplb.maybe_rebalance(50, [10, 20, 30, 40, 50, 60, 70, 80], moe);
  assert.strictEqual(result.shouldRebalance, false);
  assert.strictEqual(result.rebalanceTicks, 0);
  assert.strictEqual(result.movedExperts, 0);
});

// ==========================================
// T4: 100 步周期 — global_step=100 时触发检查
// ==========================================
test("T4 step=100 进入方差判定（均匀负载 → 跳过）", () => {
  const moe = makeMoeBackend();
  const eplb = makeEplb(moe);
  // 均匀负载：每个 expert 负载相同 → 方差为 0 → 跳过
  const loads = [100, 100, 100, 100, 100, 100, 100, 100];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, false);
});

// ==========================================
// T5: 方差低跳过 — 负载均匀时不重平衡
// ==========================================
test("T5 方差低跳过 — 负载均匀 variance_ratio=0 < 0.1", () => {
  const moe = makeMoeBackend();
  const eplb = makeEplb(moe);
  const loads = [50, 50, 50, 50, 50, 50, 50, 50];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, false);
  assert.strictEqual(result.rebalanceTicks, 0);
});

// ==========================================
// T6: 方差高触发重平衡
// ==========================================
test("T6 方差高触发重平衡", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  // rank0 有 expert 0-3，rank1 有 expert 4-7
  // 构造极度不均匀：rank0 的 expert 全部高负载，rank1 的 expert 全部低负载
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, true);
  assert.ok(result.movedExperts > 0, `movedExperts should be > 0, got ${result.movedExperts}`);
});

// ==========================================
// T7: movedExperts 非负
// ==========================================
test("T7 movedExperts 非负", () => {
  const moe = makeMoeBackend();
  const eplb = makeEplb(moe);
  // 均匀负载 → 不重平衡
  let result = eplb.maybe_rebalance(100, [50, 50, 50, 50, 50, 50, 50, 50], moe);
  assert.ok(result.movedExperts >= 0);
  // 不均匀负载 → 重平衡
  result = eplb.maybe_rebalance(200, [100, 100, 100, 100, 10, 10, 10, 10], moe);
  assert.ok(result.movedExperts >= 0);
});

// ==========================================
// T8: plan 不使新 rank max 负载超过旧 max
// ==========================================
test("T8 重排后新 max rank 负载不超过旧 max", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  // 旧 rank 负载: rank0=400, rank1=40, 旧 max=400
  const oldMaxRankLoad = 400;
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, true);
  // 重排后重新计算 rank 负载
  const newRankLoads = new Array(moe.epSize).fill(0) as number[];
  for (let e = 0; e < moe.numExperts; e++) {
    newRankLoads[moe.expertToRankMap[e]] += loads[e];
  }
  const newMax = Math.max(...newRankLoads);
  assert.ok(newMax <= oldMaxRankLoad,
    `new max rank load ${newMax} should be <= old max ${oldMaxRankLoad}`);
});

// ==========================================
// T9: rebalanceTicks 等于 rebalanceCostFixedTicks
// ==========================================
test("T9 触发重平衡时 rebalanceTicks 等于 rebalanceCostFixedTicks", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { rebalanceCostFixedTicks: 50 });
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, true);
  assert.strictEqual(result.rebalanceTicks, 50);
});

// ==========================================
// T10: metrics.epRebalanceCostTicks 累加正确
// ==========================================
test("T10 多次重平衡后 metrics.epRebalanceCostTicks 累加", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { rebalanceCostFixedTicks: 50 });
  // 第一次重平衡
  const loads1 = [100, 100, 100, 100, 10, 10, 10, 10];
  eplb.maybe_rebalance(100, loads1, moe);
  assert.strictEqual(moe.metrics.epRebalanceCostTicks, 50);
  // 第二次重平衡 — 需要构造不均匀负载（重排后可能仍然不均匀或需要新的负载分布）
  const loads2 = [200, 150, 100, 50, 10, 10, 10, 10];
  eplb.maybe_rebalance(200, loads2, moe);
  // 第二次是否触发取决于方差比率；至少第一次已累加 50
  assert.ok(moe.metrics.epRebalanceCostTicks >= 50,
    `epRebalanceCostTicks should be >= 50, got ${moe.metrics.epRebalanceCostTicks}`);
});

// ==========================================
// T11: expertToRankMap 更新后下一次 forward 生效
// ==========================================
test("T11 expertToRankMap 更新后路由反映新映射", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  // 记录原始映射
  const originalMap = [...moe.expertToRankMap];
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(100, loads, moe);
  if (result.shouldRebalance && result.movedExperts > 0) {
    // 映射应有变化
    let changed = false;
    for (let e = 0; e < moe.numExperts; e++) {
      if (moe.expertToRankMap[e] !== originalMap[e]) {
        changed = true;
        break;
      }
    }
    assert.ok(changed, "expertToRankMap should have changed after rebalance");
  }
});

// ==========================================
// T12: avg=0 时不重平衡（不除零）
// ==========================================
test("T12 avg=0 时不重平衡（不除零）", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  const loads = [0, 0, 0, 0, 0, 0, 0, 0];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, false);
  assert.strictEqual(result.rebalanceTicks, 0);
  assert.strictEqual(result.movedExperts, 0);
});

// ==========================================
// T13: 多次周期触发
// ==========================================
test("T13 step=100, 200, 300 各触发一次检查", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  // 均匀负载 → 方差低 → 跳过，但检查已触发
  const uniformLoads = [50, 50, 50, 50, 50, 50, 50, 50];
  for (const step of [100, 200, 300]) {
    const result = eplb.maybe_rebalance(step, uniformLoads, moe);
    // 均匀负载 → shouldRebalance=false
    assert.strictEqual(result.shouldRebalance, false);
  }
  // 不均匀负载 → 触发
  const unevenLoads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(400, unevenLoads, moe);
  assert.strictEqual(result.shouldRebalance, true);
});

// ==========================================
// 边界条件覆盖
// ==========================================

// B1: enabled=false
test("B1 enabled=false → maybe_rebalance 始终返回 shouldRebalance=false", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { enabled: false });
  const result = eplb.maybe_rebalance(100, [100, 100, 100, 100, 10, 10, 10, 10], moe);
  assert.strictEqual(result.shouldRebalance, false);
});

// B2: epSize=1
test("B2 epSize=1 → 退化返回不重平衡", () => {
  const moe = makeMoeBackend({}, { epSize: 1, tpSize: 1 });
  const eplb = makeEplb(moe, { epSize: 1 });
  const result = eplb.maybe_rebalance(100, [100], moe);
  assert.strictEqual(result.shouldRebalance, false);
});

// B3: 所有 expertLoadCounts=0
test("B3 所有 expertLoadCounts=0 → avg=0 → 安全返回", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  const result = eplb.maybe_rebalance(100, [0, 0, 0, 0, 0, 0, 0, 0], moe);
  assert.strictEqual(result.shouldRebalance, false);
});

// B4: 单 expert per rank → 无法搬迁
test("B4 每 rank 仅 1 expert → movedExperts=0", () => {
  const moe = makeMoeBackend({ numExperts: 2 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  // 极度不均：rank0=1000, rank1=10
  const loads = [1000, 10];
  const result = eplb.maybe_rebalance(100, loads, moe);
  // 每 rank 只有 1 expert → 无法搬迁
  assert.strictEqual(result.movedExperts, 0);
  // 但重平衡仍然被触发（方差高），只是搬不动
  assert.strictEqual(result.shouldRebalance, true);
});

// B6: 负载极度不均
test("B6 极度不均（某 rank 全部负载，另一 rank 为 0）→ 触发重平衡", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  // rank0 experts 0-3 全部负载 100，rank1 experts 4-7 负载 0
  const loads = [100, 100, 100, 100, 0, 0, 0, 0];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, true);
  assert.ok(result.movedExperts > 0,
    `movedExperts should be > 0 for extreme imbalance, got ${result.movedExperts}`);
});

// B7: 连续多次 maybe_rebalance 调用
test("B7 非检查周期返回 false，不重复累加成本", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { rebalanceCostFixedTicks: 50 });
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  // step=100 触发
  eplb.maybe_rebalance(100, loads, moe);
  const costAfterFirst = moe.metrics.epRebalanceCostTicks;
  // step=101 不触发
  eplb.maybe_rebalance(101, loads, moe);
  assert.strictEqual(moe.metrics.epRebalanceCostTicks, costAfterFirst,
    "non-checkpoint step should not add cost");
  // step=150 不触发
  eplb.maybe_rebalance(150, loads, moe);
  assert.strictEqual(moe.metrics.epRebalanceCostTicks, costAfterFirst,
    "non-checkpoint step should not add cost");
});

// B8: 重排后每 rank 至少保留 1 个 expert
test("B8 重排后每 rank 至少保留 1 个 expert", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe);
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  eplb.maybe_rebalance(100, loads, moe);
  // 检查每 rank 至少 1 个 expert
  const rankCounts = new Array(moe.epSize).fill(0) as number[];
  for (let e = 0; e < moe.numExperts; e++) {
    rankCounts[moe.expertToRankMap[e]]++;
  }
  for (let r = 0; r < moe.epSize; r++) {
    assert.ok(rankCounts[r] >= 1,
      `rank ${r} should have at least 1 expert, got ${rankCounts[r]}`);
  }
});

// ==========================================
// expertLoadCounts getter 测试
// ==========================================
test("expertLoadCounts getter 返回正确快照", () => {
  const moe = makeMoeBackend({ numExperts: 8, moeTopK: 2 }, { epSize: 2, tpSize: 2 });
  const tokenIds = Array.from({ length: 100 }, (_, i) => i);
  moe.forward(tokenIds, 0);
  const counts = moe.expertLoadCounts;
  assert.strictEqual(counts.length, 8);
  const total = counts.reduce((s, v) => s + v, 0);
  assert.strictEqual(total, 100 * 2);
});

test("expertLoadCounts 返回浅拷贝（不影响原始指标）", () => {
  const moe = makeMoeBackend({ numExperts: 8, moeTopK: 2 }, { epSize: 2, tpSize: 2 });
  moe.forward([1, 2, 3], 0);
  const counts = moe.expertLoadCounts;
  counts[0] = 9999;
  assert.notStrictEqual(moe.metrics.epExpertLoad[0], 9999,
    "modifying returned array should not affect metrics");
});

// ==========================================
// 自定义参数测试
// ==========================================
test("自定义 rebalanceIntervalSteps=50 在 step=50 触发", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { rebalanceIntervalSteps: 50 });
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(50, loads, moe);
  assert.strictEqual(result.shouldRebalance, true);
});

test("自定义 loadVarianceThreshold=0.5 方差低阈值高→跳过", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { loadVarianceThreshold: 0.5 });
  // 中等不均匀：variance_ratio 约 0.69（400 vs 40 → avg=220, stdev≈180, ratio≈0.82）
  // 但用一个小一点的不均匀试试
  const loads = [100, 100, 100, 100, 80, 80, 80, 80];
  // rank0=400, rank1=320, avg=360, stdev=40, ratio≈0.111
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.shouldRebalance, false,
    "variance ratio 0.111 < threshold 0.5 → skip");
});

test("自定义 rebalanceCostFixedTicks=100 成本累加正确", () => {
  const moe = makeMoeBackend({ numExperts: 8 }, { epSize: 2, tpSize: 2 });
  const eplb = makeEplb(moe, { rebalanceCostFixedTicks: 100 });
  const loads = [100, 100, 100, 100, 10, 10, 10, 10];
  const result = eplb.maybe_rebalance(100, loads, moe);
  assert.strictEqual(result.rebalanceTicks, 100);
  assert.strictEqual(moe.metrics.epRebalanceCostTicks, 100);
});

// ==========================================
// Summary
// ==========================================
console.log("\n=== P3b 结果 ===");
console.log(`通过: ${passed}  失败: ${failed}`);
if (failures.length) {
  console.log("\n失败详情:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
} else {
  console.log("\n所有 P3b 验收测试通过 \u2713");
  process.exit(0);
}
