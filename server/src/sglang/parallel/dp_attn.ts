// parallel/dp_attn.ts — P2b: DPAttentionSimulator — MLA 专用（all-gather → forward → slice）

import { SimCommGroup } from "./comm_group";

// ===== 构造选项 =====

export interface DPAttentionSimulatorOpts {
  dpSize: number;
  hiddenSize: number;
  dtypeSize: number;
  useMla: boolean;
  enableDpAttention: boolean;
  networkBandwidthGBps: number;
  networkLatencyUs: number;
}

// ===== DPAttentionSimulator =====

/**
 * DP Attention 仿真器（§10.3 DP Attention，MLA 专用）
 *
 * 仅当 useMla && enableDpAttention && dpSize > 1 时启用。
 * Attention 层不通信（MLA 规则：KV cache 每 rank 自己维护）。
 * MLP 层执行 all-gather → forward → slice，仅 all-gather 产生通信成本。
 */
export class DPAttentionSimulator {
  readonly dpSize: number;
  readonly hiddenSize: number;
  readonly dtypeSize: number;
  readonly useMla: boolean;
  readonly enableDpAttention: boolean;

  /** all-gather 通信组（仅当启用时非 null） */
  readonly commGroup: SimCommGroup | null;

  /** 是否启用（useMla && enableDpAttention && dpSize > 1） */
  readonly enabled: boolean;

  constructor(opts: DPAttentionSimulatorOpts) {
    this.dpSize = opts.dpSize;
    this.hiddenSize = opts.hiddenSize;
    this.dtypeSize = opts.dtypeSize;
    this.useMla = opts.useMla;
    this.enableDpAttention = opts.enableDpAttention;

    this.enabled = !!(opts.useMla && opts.enableDpAttention && opts.dpSize > 1);

    this.commGroup = this.enabled
      ? new SimCommGroup({
          groupType: "dp_attn",
          size: opts.dpSize,
          networkBandwidthGBps: opts.networkBandwidthGBps,
          latencyUs: opts.networkLatencyUs,
          efficiency: 1.0,
        })
      : null;
  }

  /**
   * 仿真 MLP 层的 all-gather → forward → slice 通信成本。
   * localBatchSizes: 各 DP rank 的本地 batch_size 列表。
   * 返回 { commTicks, allGatherBytes }。
   */
  simulateMlpForward(localBatchSizes: number[]): {
    commTicks: number;
    allGatherBytes: number;
  } {
    if (!this.enabled || this.commGroup === null) {
      return { commTicks: 0, allGatherBytes: 0 };
    }

    const gatherSizes = localBatchSizes.map(sz => sz * this.hiddenSize * this.dtypeSize);
    const allGatherBytes = gatherSizes.reduce((s, v) => s + v, 0);

    const commTicks = this.commGroup.allGather(gatherSizes);

    return { commTicks, allGatherBytes };
  }

  /**
   * 纯预览函数：计算给定 batch 下每步 all-gather 数据量（字节数）。
   * batch: 总 batch 大小。
   * 返回每步 all-gather 总字节数。
   */
  totalAllGatherBytesPerStep(batch: number): number {
    if (!this.enabled) {
      return 0;
    }
    const localBatchSize = batch / this.dpSize;
    const perRankBytes = localBatchSize * this.hiddenSize * this.dtypeSize;
    return perRankBytes * this.dpSize;
  }
}
