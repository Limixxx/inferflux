// parallel/moe.ts — P3a: SimMoeBackend（EP 路由 + all-to-all 正反）3 种 moe_routing_mode

import { divEven } from "../core";
import type { ModelConfig, SimulatorConfig } from "../types";
import { SimCommGroup } from "./comm_group";
import { ParallelMetrics } from "./metrics";
import { ParallelTopology } from "./topology";

// ===== 路由结果 =====

export interface MoeRouteResult {
  /** 每个 EP rank 处理的 token 数（含本地和远程） */
  rankDistribution: Map<number, number>;
  /** 非本地 expert 的 token 数（需要跨 rank 通信） */
  crossRankTokens: number;
  /** 本次路由中每个专家被选中的次数（用于 epExpertLoad 更新） */
  expertCounts: number[];
}

// ===== Forward 结果 =====

export interface MoeForwardResult {
  /** all-to-all 正向 + 反向总通信 ticks */
  commTicks: number;
  /** 跨 rank 的 token 数 */
  crossRankTokens: number;
  /** 各 rank 处理的 token 数分布 */
  rankDistribution: Map<number, number>;
}

// ===== 构造选项 =====

export interface SimMoeBackendOpts {
  modelConfig: ModelConfig;
  topology: ParallelTopology;
  config: SimulatorConfig;
  epCommGroup: SimCommGroup;  // group_type="ep" 的通信组
  metrics: ParallelMetrics;    // 指标写入目标
  dtypeSize?: number;          // 默认 2 (float16)
  seed?: number;               // 用于 simulated 路由的种子（默认 0）
}

// ===== SimMoeBackend =====

/**
 * MoE 仿真后端（§10.4）
 *
 * 三种路由模式：mock / hash / simulated
 * EP 路由 → 正向 all-to-all → mock forward → 反向 all-to-all
 * 指标写入 ParallelMetrics 已有的 EP 字段
 */
export class SimMoeBackend {
  readonly modelConfig: ModelConfig;
  readonly epSize: number;
  readonly routingMode: "mock" | "hash" | "simulated";
  readonly moeTopK: number;
  readonly numExperts: number;
  readonly hiddenSize: number;
  readonly topology: ParallelTopology;
  readonly commGroup: SimCommGroup;
  readonly metrics: ParallelMetrics;
  readonly dtypeSize: number;
  readonly seed: number;

  readonly expertsPerRank: number[];
  readonly expertToRankMap: number[];

  callCount: number = 0;
  totalTokens: number = 0;
  commTicksTotal: number = 0;

  constructor(opts: SimMoeBackendOpts) {
    this.modelConfig = opts.modelConfig;
    this.topology = opts.topology;
    this.commGroup = opts.epCommGroup;
    this.metrics = opts.metrics;
    this.dtypeSize = opts.dtypeSize ?? 2;
    this.seed = opts.seed ?? 0;

    this.epSize = opts.config.epSize;
    this.routingMode = opts.config.moeRoutingMode;
    this.moeTopK = opts.modelConfig.moeTopK;
    this.numExperts = opts.modelConfig.numExperts;
    this.hiddenSize = opts.modelConfig.hiddenSize;

    this.expertsPerRank = divEven(this.numExperts, this.epSize);

    this.expertToRankMap = new Array(this.numExperts);
    let offset = 0;
    for (let rank = 0; rank < this.epSize; rank++) {
      for (let i = 0; i < this.expertsPerRank[rank]; i++) {
        this.expertToRankMap[offset + i] = rank;
      }
      offset += this.expertsPerRank[rank];
    }

    // 构造时自检：验证 expertToRankMap 与 topology.computeMoeRanks 一致
    for (let expertId = 0; expertId < this.numExperts; expertId++) {
      let foundEpRank = -1;
      for (let tpRank = 0; tpRank < this.topology.tpSize; tpRank++) {
        const [, epRank] = this.topology.computeMoeRanks(tpRank);
        let eOffset = 0;
        for (let r = 0; r < epRank; r++) {
          eOffset += this.expertsPerRank[r];
        }
        if (expertId >= eOffset && expertId < eOffset + this.expertsPerRank[epRank]) {
          foundEpRank = epRank;
          break;
        }
      }
      if (foundEpRank !== -1 && foundEpRank !== this.expertToRankMap[expertId]) {
        throw new Error(
          `SimMoeBackend: expertToRankMap[${expertId}]=${this.expertToRankMap[expertId]} ` +
          `differs from topology.computeMoeRanks epRank=${foundEpRank}`
        );
      }
    }
  }

  /** splitmix32 — 32 位确定性整数哈希，良好雪崩效应 */
  private _splitmix32(x: number): number {
    x = (x + 0x9e3779b9) | 0;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
    return (x ^ (x >>> 16)) >>> 0;
  }

  /** mulberry32 — 确定性种子化伪随机数生成器，返回 [0, 1) */
  private _mulberry32(seed: number): number {
    let t = (seed + 0x6d2b79f5) | 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** O(1) 查表获取 expert 所在的 EP rank */
  _expertToRank(expertId: number): number {
    return this.expertToRankMap[expertId];
  }

  /** 获取当前各 expert 的累计负载快照（来自 metrics.epExpertLoad） */
  get expertLoadCounts(): number[] {
    return [...this.metrics.epExpertLoad];
  }

  /**
   * 对一批 token 执行路由决策
   */
  _routeTokens(tokenIds: number[], layerIdx: number): MoeRouteResult {
    const rankDistribution = new Map<number, number>();
    const expertCounts = new Array(this.numExperts).fill(0);
    let crossRankTokens = 0;

    for (let r = 0; r < this.epSize; r++) {
      rankDistribution.set(r, 0);
    }

    for (let tokenIdx = 0; tokenIdx < tokenIds.length; tokenIdx++) {
      const tokenId = tokenIds[tokenIdx];
      const selectedExperts = this._selectExperts(tokenId, tokenIdx, layerIdx);

      for (const expertId of selectedExperts) {
        const epRank = this.expertToRankMap[expertId];
        rankDistribution.set(epRank, (rankDistribution.get(epRank) ?? 0) + 1);
        expertCounts[expertId]++;
        if (epRank !== 0) {
          crossRankTokens++;
        }
      }
    }

    return { rankDistribution, crossRankTokens, expertCounts };
  }

  private _selectExperts(tokenId: number, tokenIdx: number, layerIdx: number): number[] {
    switch (this.routingMode) {
      case "mock":
        return this._selectMockExperts(tokenIdx);
      case "hash":
        return this._selectHashExperts(tokenId, layerIdx);
      case "simulated":
        return this._selectSimulatedExperts(tokenIdx, layerIdx);
    }
  }

  /** mock 模式：均匀轮转选 top_k 个专家 */
  private _selectMockExperts(tokenIdx: number): number[] {
    const experts: number[] = [];
    for (let k = 0; k < this.moeTopK; k++) {
      experts.push((tokenIdx * this.moeTopK + k) % this.numExperts);
    }
    return experts;
  }

  /** hash 模式：splitmix32 确定性哈希选 top_k 个专家 */
  private _selectHashExperts(tokenId: number, layerIdx: number): number[] {
    const layerMix = Math.imul(layerIdx, 0x9e3779b9);
    const scores: Array<{ expertId: number; hashScore: number }> = [];
    for (let e = 0; e < this.numExperts; e++) {
      const hashScore = this._splitmix32(tokenId ^ layerMix ^ e);
      scores.push({ expertId: e, hashScore });
    }
    scores.sort((a, b) => a.hashScore - b.hashScore);
    return scores.slice(0, this.moeTopK).map((s) => s.expertId);
  }

  /** simulated 模式：mulberry32 种子化伪随机选 top_k 个专家 */
  private _selectSimulatedExperts(tokenIdx: number, layerIdx: number): number[] {
    const scores: Array<{ expertId: number; score: number }> = [];
    for (let e = 0; e < this.numExperts; e++) {
      const prngSeed = this.seed + tokenIdx * this.numExperts + layerIdx * this.numExperts + e;
      const score = this._mulberry32(prngSeed);
      scores.push({ expertId: e, score });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, this.moeTopK).map((s) => s.expertId);
  }

  /**
   * MoE forward：路由 → 正向 all-to-all → mock forward → 反向 all-to-all
   */
  forward(tokenIds: number[], layerIdx: number): MoeForwardResult {
    this.callCount++;
    this.totalTokens += tokenIds.length;

    // 1. 路由决策
    const routeResult = this._routeTokens(tokenIds, layerIdx);

    // 2. 更新指标 — epExpertLoad
    if (this.metrics.epExpertLoad.length < this.numExperts) {
      const oldLen = this.metrics.epExpertLoad.length;
      this.metrics.epExpertLoad.length = this.numExperts;
      for (let i = oldLen; i < this.numExperts; i++) {
        this.metrics.epExpertLoad[i] = 0;
      }
    }
    for (let e = 0; e < this.numExperts; e++) {
      this.metrics.epExpertLoad[e] += routeResult.expertCounts[e];
    }

    // 3. 构造 all-to-all sizes
    const bytesPerToken = this.hiddenSize * this.dtypeSize;
    const sendSizes: number[] = new Array(this.epSize).fill(0);
    const recvSizes: number[] = new Array(this.epSize).fill(0);
    for (let r = 0; r < this.epSize; r++) {
      const tokenCount = routeResult.rankDistribution.get(r) ?? 0;
      sendSizes[r] = tokenCount * bytesPerToken;
    }
    for (let r = 0; r < this.epSize; r++) {
      recvSizes[r] = sendSizes[r];
    }

    // 4. 正向 all-to-all
    const fwdTicks = this.commGroup.allToAll(sendSizes, recvSizes);

    // 5. Mock forward：0 cost

    // 6. 反向 all-to-all
    const revTicks = this.commGroup.allToAll(recvSizes, sendSizes);

    // 7. 更新指标
    const totalCommTicks = fwdTicks + revTicks;
    this.metrics.epCommTicks += totalCommTicks;
    this.metrics.epAllToAllCount += 2;
    this.metrics.epCrossRankTokens += routeResult.crossRankTokens;

    // 8. 内部统计
    this.commTicksTotal += totalCommTicks;

    // 9. 返回
    return {
      commTicks: totalCommTicks,
      crossRankTokens: routeResult.crossRankTokens,
      rankDistribution: routeResult.rankDistribution,
    };
  }
}
