// parallel/topology.ts — P0: ParallelTopology 并行拓扑映射

// ===== 构造选项 =====

export interface ParallelTopologyOpts {
  tpSize?: number;
  dpSize?: number;
  epSize?: number;
  ppSize?: number;
  cpSize?: number;
  enableDpAttention?: boolean;
}

// ===== ParallelTopology =====

/**
 * 并行拓扑配置（§4.2）
 *
 * 描述 TP×DP×PP 进程网格，支持 rank↔coord 双向映射、
 * MoE/Attention 层级 rank 推导、PP stage 层分割。
 *
 * CP 与 EP 是 TP group 内部重编号：
 * - cp_size 整除 tp_size
 * - ep_size 整除 tp_size / cp_size
 */
export class ParallelTopology {
  readonly tpSize: number;
  readonly dpSize: number;
  readonly epSize: number;
  readonly ppSize: number;
  readonly cpSize: number;
  readonly enableDpAttention: boolean;

  constructor(opts?: ParallelTopologyOpts) {
    this.tpSize = opts?.tpSize ?? 1;
    this.dpSize = opts?.dpSize ?? 1;
    this.epSize = opts?.epSize ?? 1;
    this.ppSize = opts?.ppSize ?? 1;
    this.cpSize = opts?.cpSize ?? 1;
    this.enableDpAttention = opts?.enableDpAttention ?? false;

    // 约束验证
    if (this.cpSize > 1 && this.tpSize % this.cpSize !== 0) {
      throw new Error(
        `cp_size (${this.cpSize}) must divide tp_size (${this.tpSize})`
      );
    }
    const tpPerCp = this.tpSize / this.cpSize;
    if (this.epSize > 1 && tpPerCp % this.epSize !== 0) {
      throw new Error(
        `ep_size (${this.epSize}) must divide tp_size/cp_size (${tpPerCp})`
      );
    }
  }

  /** 总进程数 = tp × dp × pp */
  get worldSize(): number {
    return this.tpSize * this.dpSize * this.ppSize;
  }

  /** DP 组数量 = tp × pp */
  get numDpGroups(): number {
    return this.tpSize * this.ppSize;
  }

  /** PP stage 数量 */
  get numPpStages(): number {
    return this.ppSize;
  }

  /**
   * rank → (tp_idx, dp_idx, pp_idx)
   *
   * tp 在最内层，dp 居中，pp 最外层：
   * rank = pp_idx × (dp_size × tp_size) + dp_idx × tp_size + tp_idx
   */
  rankToCoord(rank: number): [number, number, number] {
    const tpIdx = rank % this.tpSize;
    const dpIdx = Math.floor(rank / this.tpSize) % this.dpSize;
    const ppIdx = Math.floor(rank / (this.tpSize * this.dpSize));
    return [tpIdx, dpIdx, ppIdx];
  }

  /**
   * (tp_idx, dp_idx, pp_idx) → rank
   */
  coordToRank(tpIdx: number, dpIdx: number, ppIdx: number): number {
    return ppIdx * (this.dpSize * this.tpSize) + dpIdx * this.tpSize + tpIdx;
  }

  /**
   * MoE 层级 rank 推导（对应 SGLang _compute_parallelism_ranks）
   *
   * 从 tpRank 推导出 (moe_dp_rank, moe_ep_rank, moe_tp_rank)
   *
   * moe_dp_size = dp_size
   * moe_tp_size = max(1, tp_size / dp_size / ep_size)
   * moe_dp_rank = tp_rank // (tp_size // moe_dp_size)
   * moe_ep_rank = (tp_rank % (tp_size // moe_dp_size)) // moe_tp_size
   * moe_tp_rank = tp_rank % moe_tp_size
   */
  computeMoeRanks(tpRank: number): [number, number, number] {
    const moeDpSize = this.dpSize;
    const moeTpSize = Math.max(1, Math.floor(this.tpSize / this.dpSize / this.epSize));
    const moeDpRank = Math.floor(tpRank / Math.floor(this.tpSize / moeDpSize));
    const innerRank = tpRank % Math.floor(this.tpSize / moeDpSize);
    const moeEpRank = Math.floor(innerRank / moeTpSize);
    const moeTpRank = innerRank % moeTpSize;
    return [moeDpRank, moeEpRank, moeTpRank];
  }

  /**
   * Attention 层级 rank 推导
   *
   * 从 tpRank 推导出 (attn_cp_rank, attn_tp_rank)
   *
   * attn_dp_size = dp_size (若 enableDpAttention) 或 1
   * attn_tp_size = max(1, tp_size / attn_dp_size / cp_size)
   * attn_cp_rank = (tp_rank // attn_tp_size) % cp_size
   * attn_tp_rank = tp_rank % attn_tp_size
   */
  computeAttnRanks(tpRank: number): [number, number] {
    const attnDpSize = this.enableDpAttention ? this.dpSize : 1;
    const attnTpSize = Math.max(1, Math.floor(this.tpSize / attnDpSize / this.cpSize));
    const attnCpRank = Math.floor(tpRank / attnTpSize) % this.cpSize;
    const attnTpRank = tpRank % attnTpSize;
    return [attnCpRank, attnTpRank];
  }

  /**
   * PP stage 层分割
   *
   * num_layers 按 pp_size 均分，余数分配到前几个 stage。
   * 返回每个 stage 的层范围 [start, end) 列表
   */
  ppStageLayers(numLayers: number): Array<{ start: number; end: number }> {
    const stages: Array<{ start: number; end: number }> = [];
    const base = Math.floor(numLayers / this.ppSize);
    const remainder = numLayers % this.ppSize;
    let offset = 0;
    for (let i = 0; i < this.ppSize; i++) {
      const count = base + (i < remainder ? 1 : 0);
      stages.push({ start: offset, end: offset + count });
      offset += count;
    }
    return stages;
  }
}
