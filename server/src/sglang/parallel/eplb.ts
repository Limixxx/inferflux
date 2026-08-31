// parallel/eplb.ts — P3b: EPLBSimulator — EP 负载均衡（100 步周期 + 方差阈值重平衡）

import type { ParallelMetrics } from "./metrics";
import type { SimMoeBackend } from "./moe";

// ===== 构造选项 =====

export interface EPLBSimulatorOpts {
  enabled: boolean;
  numExperts: number;
  epSize: number;
  metrics: ParallelMetrics;
  rebalanceIntervalSteps?: number;
  loadVarianceThreshold?: number;
  rebalanceCostFixedTicks?: number;
}

// ===== maybe_rebalance 返回值 =====

export interface RebalanceResult {
  shouldRebalance: boolean;
  rebalanceTicks: number;
  movedExperts: number;
}

// ===== EPLBSimulator =====

export class EPLBSimulator {
  readonly enabled: boolean;
  readonly numExperts: number;
  readonly epSize: number;
  readonly metrics: ParallelMetrics;
  readonly rebalanceIntervalSteps: number;
  readonly loadVarianceThreshold: number;
  readonly rebalanceCostFixedTicks: number;

  constructor(opts: EPLBSimulatorOpts) {
    this.enabled = opts.enabled;
    this.numExperts = opts.numExperts;
    this.epSize = opts.epSize;
    this.metrics = opts.metrics;
    this.rebalanceIntervalSteps = opts.rebalanceIntervalSteps ?? 100;
    this.loadVarianceThreshold = opts.loadVarianceThreshold ?? 0.1;
    this.rebalanceCostFixedTicks = opts.rebalanceCostFixedTicks ?? 50;
  }

  /**
   * 检查并执行重平衡
   * @param globalStep 当前全局步数
   * @param expertLoadCounts 各 expert 累计负载（来自 moeBackend.expertLoadCounts）
   * @param moeBackend MoE 后端实例（用于更新 expertToRankMap）
   */
  maybe_rebalance(
    globalStep: number,
    expertLoadCounts: number[],
    moeBackend: SimMoeBackend,
  ): RebalanceResult {
    // 1. 禁用或单 rank → 直接返回
    if (!this.enabled || this.epSize <= 1) {
      return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 };
    }

    // 2. 非检查周期 → 跳过
    if (globalStep % this.rebalanceIntervalSteps !== 0) {
      return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 };
    }

    // 3. 计算 rank 负载
    const rankLoads = new Array(this.epSize).fill(0) as number[];
    const expertToRankMap = moeBackend.expertToRankMap;
    for (let e = 0; e < this.numExperts; e++) {
      rankLoads[expertToRankMap[e]] += expertLoadCounts[e];
    }

    // 4. 计算均值
    const avg = rankLoads.reduce((s, v) => s + v, 0) / this.epSize;

    // 5. avg=0 → 安全返回不重平衡
    if (avg === 0) {
      return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 };
    }

    // 6. 计算标准差和方差比率
    const variance = rankLoads.reduce((s, v) => s + (v - avg) ** 2, 0) / this.epSize;
    const stdev = Math.sqrt(variance);
    const varianceRatio = stdev / avg;

    // 7. 方差低于阈值 → 跳过
    if (varianceRatio < this.loadVarianceThreshold) {
      return { shouldRebalance: false, rebalanceTicks: 0, movedExperts: 0 };
    }

    // 8. 贪心重排
    const movedExperts = this._greedyRebalance(expertLoadCounts, rankLoads, avg, moeBackend);

    // 9. 累加重平衡成本
    this.metrics.epRebalanceCostTicks += this.rebalanceCostFixedTicks;

    // 10. 返回结果
    return {
      shouldRebalance: true,
      rebalanceTicks: this.rebalanceCostFixedTicks,
      movedExperts,
    };
  }

  /**
   * 贪心重排策略：从过载 rank 搬走负载最小的 expert 到欠载 rank
   */
  private _greedyRebalance(
    expertLoadCounts: number[],
    initialRankLoads: number[],
    avg: number,
    moeBackend: SimMoeBackend,
  ): number {
    const expertToRankMap = moeBackend.expertToRankMap;
    const targetLoadPerRank = avg * 1.02;
    const oldMaxRankLoad = Math.max(...initialRankLoads);

    // 按 rank 分组 expert
    const expertsByRank: Array<Array<{ expertId: number; load: number }>> = [];
    for (let r = 0; r < this.epSize; r++) {
      expertsByRank.push([]);
    }
    for (let e = 0; e < this.numExperts; e++) {
      const rank = expertToRankMap[e];
      expertsByRank[rank].push({ expertId: e, load: expertLoadCounts[e] });
    }

    // 当前 rank 负载（可变）
    const currentRankLoads = [...initialRankLoads];
    let movedExperts = 0;

    // 贪心循环
    for (;;) {
      // 找过载最多的 rank
      let maxSurplusRank = -1;
      let maxSurplus = -Infinity;
      for (let r = 0; r < this.epSize; r++) {
        const surplus = currentRankLoads[r] - targetLoadPerRank;
        if (surplus > maxSurplus) {
          maxSurplus = surplus;
          maxSurplusRank = r;
        }
      }

      // 所有 rank 不再过载 → 终止
      if (maxSurplus <= 0) break;

      // 确保源 rank 有多个 expert 可以搬走
      if (expertsByRank[maxSurplusRank].length <= 1) break;

      // 找欠载最多的 rank
      let maxDeficitRank = -1;
      let maxDeficit = -Infinity;
      for (let r = 0; r < this.epSize; r++) {
        const deficit = targetLoadPerRank - currentRankLoads[r];
        if (deficit > maxDeficit) {
          maxDeficit = deficit;
          maxDeficitRank = r;
        }
      }

      if (maxDeficit <= 0) break;

      // 从过载 rank 的 expert 中选择负载最小的（搬迁代价最小）
      // 尝试找到满足约束的 expert
      const sourceExperts = expertsByRank[maxSurplusRank];
      // 按负载升序排列
      sourceExperts.sort((a, b) => a.load - b.load);

      let moved = false;
      for (let i = 0; i < sourceExperts.length; i++) {
        const candidate = sourceExperts[i];
        // 约束检查：搬迁后目标 rank 新负载 ≤ 旧 max rank 负载
        const newTargetLoad = currentRankLoads[maxDeficitRank] + candidate.load;
        if (newTargetLoad <= oldMaxRankLoad) {
          // 执行搬迁
          expertToRankMap[candidate.expertId] = maxDeficitRank;
          currentRankLoads[maxSurplusRank] -= candidate.load;
          currentRankLoads[maxDeficitRank] += candidate.load;

          // 从源 rank 移除，加入目标 rank
          sourceExperts.splice(i, 1);
          expertsByRank[maxDeficitRank].push(candidate);

          movedExperts++;
          moved = true;
          break;
        }
      }

      // 没有可搬迁的 expert → 终止
      if (!moved) break;
    }

    return movedExperts;
  }
}
