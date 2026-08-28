// parallel/cp_simulator.ts — P5: CPSimulator Context Parallel KV all-gather (§10.8)

import { SimCommGroup } from "./comm_group";
import { divCeil } from "../core";

// ===== CP Attention 结果 =====

export interface CPAttnResult {
  commTicks: number;
  allGatherBytes: number;
  seqLenPerRank: number;
}

// ===== CPSimulator =====

/**
 * Context Parallel 仿真器（§10.8）
 *
 * 仅当 cp_size > 1 时启用；否则 simulateAttnForward 返回零结果。
 * 每层 attention 结束后执行 KV all-gather，计算通信成本。
 * MLP 层不通信。
 */
export class CPSimulator {
  readonly cpSize: number;
  readonly commGroup: SimCommGroup | null;
  totalCommTicks: number = 0;

  private readonly numKvHeads: number;
  private readonly headDim: number;
  private readonly dtypeSize: number;
  private readonly numLayers: number;

  constructor(config: { cpSize: number; networkBandwidthGBps: number; networkLatencyUs: number; cpEfficiency: number; dtypeSize: number }, modelConfig: { numKvHeads: number; headDim: number; numLayers: number }) {
    this.cpSize = config.cpSize;
    this.numKvHeads = modelConfig.numKvHeads;
    this.headDim = modelConfig.headDim;
    this.dtypeSize = config.dtypeSize;
    this.numLayers = modelConfig.numLayers;

    if (this.cpSize > 1) {
      this.commGroup = new SimCommGroup({
        groupType: "cp",
        size: this.cpSize,
        networkBandwidthGBps: config.networkBandwidthGBps,
        latencyUs: config.networkLatencyUs,
        efficiency: config.cpEfficiency,
      });
    } else {
      this.commGroup = null;
    }
  }

  /**
   * 仿真 CP attention 的 KV all-gather 通信成本
   *
   * @param seqLen 完整序列长度
   * @returns CPAttnResult 包含通信 ticks、all-gather 字节数、每 rank 序列长度
   */
  simulateAttnForward(seqLen: number): CPAttnResult {
    if (this.cpSize <= 1 || this.commGroup === null) {
      return { commTicks: 0, allGatherBytes: 0, seqLenPerRank: seqLen };
    }

    const seqLenPerRank = divCeil(seqLen, this.cpSize);

    // KV all-gather 数据量: seqLen × num_kv_heads × head_dim × dtype_size × num_layers × 2
    // ×2 因为 K 和 V 两份
    const kvBytes = seqLen * this.numKvHeads * this.headDim * this.dtypeSize * this.numLayers * 2;

    // 调用 SimCommGroup("cp").allGather 计算通信 ticks
    const commTicks = this.commGroup.allGather([kvBytes]);

    this.totalCommTicks += commTicks;

    return { commTicks, allGatherBytes: kvBytes, seqLenPerRank };
  }
}
