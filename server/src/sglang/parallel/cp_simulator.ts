// parallel/cp_simulator.ts — P5: CPSimulator Context Parallel KV all-gather (§10.8)

import type { SimulatorConfig, ModelConfig } from "../types";
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
 *
 * simulateAttnForward 计算单层 attention 后的 KV all-gather 通信成本，
 * 在 MockEngine.forwardBatch 的层循环中逐层调用并累加。
 */
export class CPSimulator {
  readonly cpSize: number;
  readonly commGroup: SimCommGroup | null;
  totalCommTicks: number = 0;

  private readonly numKvHeads: number;
  private readonly headDim: number;
  private readonly dtypeSize: number;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig) {
    this.cpSize = config.cpSize;
    this.numKvHeads = modelConfig.numKvHeads;
    this.headDim = modelConfig.headDim;
    this.dtypeSize = config.dtypeSize;

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
   * 仿真单层 CP attention 的 KV all-gather 通信成本
   *
   * 每次 call 代表一个 layer 的 attention 结束后的 KV all-gather。
   * 在 MockEngine.forwardBatch 的层循环中逐层调用。
   *
   * @param seqLen 完整序列长度
   * @returns CPAttnResult 包含通信 ticks、all-gather 字节数（每 rank）、每 rank 序列长度
   */
  simulateAttnForward(seqLen: number): CPAttnResult {
    if (this.cpSize <= 1 || this.commGroup === null) {
      return { commTicks: 0, allGatherBytes: 0, seqLenPerRank: seqLen };
    }

    const seqLenPerRank = divCeil(seqLen, this.cpSize);

    // 单层 KV all-gather 数据量（每 rank 持有 1/cp_size 的 KV）
    // kv_bytes_per_rank = seq_len_per_rank × num_kv_heads × head_dim × dtype_size × 2
    // ×2 因为 K 和 V 两份
    const kvBytesPerRank = seqLenPerRank * this.numKvHeads * this.headDim * this.dtypeSize * 2;

    // allGather 接收每 rank 的字节数数组
    const commTicks = this.commGroup.allGather([kvBytesPerRank]);

    this.totalCommTicks += commTicks;

    return { commTicks, allGatherBytes: kvBytesPerRank, seqLenPerRank };
  }
}
