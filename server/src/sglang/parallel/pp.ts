// parallel/pp.ts — P4: PPPipelineSimulator 流水并行仿真器（§10.5）

import { SimCommGroup } from "./comm_group";
import { ParallelTopology } from "./topology";
import type { SimulatorConfig, ModelConfig } from "../types";
import type { Batch } from "../core";

// ===== 结果接口 =====

/** 流水线步骤仿真结果 */
export interface PipelineStepResult {
  totalTicks: number;       // bubbleTicks + sendRecvTicks
  bubbleTicks: number;      // 流水线气泡 ticks（μs）
  sendRecvTicks: number;    // stage 间 send/recv 总成本（μs）
  perStageTicks: number[];  // 每个 stage 边界的通信 ticks（长度 = pp_size - 1）
}

// ===== PPPipelineSimulator =====

/**
 * 流水并行仿真器（§10.5）
 *
 * 纯算术仿真三种流水线调度（gpipe / 1f1b / interleaved）的
 * bubble 和 stage 间通信成本。pp_size=1 时全部退化为 noop。
 *
 * Tick 定义对齐 §4.2 行1059：
 * - microBatchTicks = config.eagerForwardCostTicks（默认 10μs）
 * - SimCommGroup.sendRecv() 返回值单位为 μs
 * - 仿真内部约定 1 tick = 1 μs
 */
export class PPPipelineSimulator {
  readonly ppSize: number;
  readonly schedule: "1f1b" | "gpipe" | "interleaved";
  readonly numMicroBatches: number;
  readonly tpSize: number;
  readonly commGroup: SimCommGroup | null;
  readonly stageLayers: Array<{ start: number; end: number }>;
  readonly numChunks: number;                  // interleaved 专用，默认 2（R2-4）
  readonly microBatchTicks: number;            // = config.eagerForwardCostTicks（R2-1）
  readonly commOverlapWithCompute: boolean;     // 通信是否可与计算重叠（R2-2）
  readonly hiddenSize: number;
  readonly dtypeSize: number;

  bubbleTicks: number;
  commTicksTotal: number;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig) {
    if (config.ppSize <= 0) {
      throw new Error(`pp_size must be > 0, got ${config.ppSize}`);
    }
    this.ppSize = config.ppSize;
    this.schedule = config.ppPipelineSchedule;
    this.numMicroBatches = config.ppNumMicroBatches;
    this.tpSize = config.tpSize;
    this.numChunks = config.ppInterleavedNumChunks ?? 2; // R2-4
    this.microBatchTicks = config.eagerForwardCostTicks;  // R2-1
    this.commOverlapWithCompute = config.commOverlapWithCompute; // R2-2
    this.hiddenSize = modelConfig.hiddenSize;
    this.dtypeSize = config.dtypeSize;

    // 构造器内自动推导 stageLayers
    const topology = new ParallelTopology({ ppSize: this.ppSize });
    this.stageLayers = topology.ppStageLayers(modelConfig.numLayers);

    // 通信组：pp_size > 1 时创建，否则为 null
    this.commGroup = this.ppSize > 1
      ? new SimCommGroup({
          groupType: "pp",
          size: this.ppSize,
          networkBandwidthGBps: config.networkBandwidthGBps,
          latencyUs: config.networkLatencyUs,
        })
      : null;

    this.bubbleTicks = 0;
    this.commTicksTotal = 0;
  }

  /** 仿真整个 pipeline forward，返回 PipelineStepResult */
  simulatePipelineStep(batch: Batch): PipelineStepResult {
    // pp_size=1 退化：全部为零
    if (this.ppSize <= 1) {
      return { totalTicks: 0, bubbleTicks: 0, sendRecvTicks: 0, perStageTicks: [] };
    }

    // numMicroBatches=0 时返回全零
    if (this.numMicroBatches <= 0) {
      return { totalTicks: 0, bubbleTicks: 0, sendRecvTicks: 0, perStageTicks: new Array(this.ppSize - 1).fill(0) };
    }

    // 1. 计算 bubble ticks（严格遵循 Issue 规格）
    let bubbleTicks: number;
    switch (this.schedule) {
      case "gpipe":
        // gpipe: bubble = (pp_size - 1) × microBatchTicks × numMicroBatches
        bubbleTicks = (this.ppSize - 1) * this.microBatchTicks * this.numMicroBatches;
        break;
      case "1f1b":
        // 1f1b: bubble = (pp_size - 1) × microBatchTicks（最优）
        bubbleTicks = (this.ppSize - 1) * this.microBatchTicks;
        break;
      case "interleaved":
        // interleaved: bubble = (pp_size - 1) × numChunks × microBatchTicks
        bubbleTicks = (this.ppSize - 1) * this.numChunks * this.microBatchTicks;
        break;
      default:
        throw new Error(`Unknown pipeline schedule: ${this.schedule}`);
    }

    // 2. 计算 micro-batch 分割
    const batchSize = this._computeBatchSize(batch);
    const microBatches = this._splitMicroBatches(batchSize);

    // 3. 计算 per-stage 通信成本
    const perStageTicks: number[] = new Array(this.ppSize - 1).fill(0);
    for (const mb of microBatches) {
      const rawCost = this._stageSendRecvCost(mb.size);
      const effectiveCost = this._effectiveSendRecv(rawCost);
      for (let i = 0; i < this.ppSize - 1; i++) {
        perStageTicks[i] += effectiveCost;
      }
    }

    const sendRecvTicks = perStageTicks.reduce((a, b) => a + b, 0);
    const totalTicks = bubbleTicks + sendRecvTicks;

    // 累积状态
    this.bubbleTicks += bubbleTicks;
    this.commTicksTotal += sendRecvTicks;

    return { totalTicks, bubbleTicks, sendRecvTicks, perStageTicks };
  }

  /**
   * 将 batch 按 numMicroBatches 分割。
   * 不整除时采用 ceil 分配，不 padding。
   * batchSize < numMicroBatches 时：前 batchSize 个各 1，剩余为 0。
   */
  _splitMicroBatches(batchSize: number): Array<{ size: number }> {
    if (this.numMicroBatches <= 0) return [];
    if (batchSize <= 0) return Array.from({ length: this.numMicroBatches }, () => ({ size: 0 }));

    const base = Math.floor(batchSize / this.numMicroBatches);
    const remainder = batchSize % this.numMicroBatches;

    return Array.from({ length: this.numMicroBatches }, (_, i) => ({
      size: base + (i < remainder ? 1 : 0),
    }));
  }

  /**
   * 计算单个 stage 间 send/recv 通信成本（原始值）。
   * R2-5: dataBytes = microBatchSize × ceil(hiddenSize / tpSize) × dtypeSize
   * 参考 §10.5.2 行4423，增加 TP 分割修正。
   */
  _stageSendRecvCost(microBatchSize: number): number {
    if (!this.commGroup) return 0;
    if (microBatchSize <= 0) return 0;

    const dataBytes = microBatchSize * Math.ceil(this.hiddenSize / this.tpSize) * this.dtypeSize;
    // send + recv = 2 次 sendRecv
    const cost = this.commGroup.sendRecv(dataBytes, 0) * 2;
    return cost;
  }

  /**
   * 通信与计算重叠折算（R2-2）。
   * 根据 config.commOverlapWithCompute（§4.2 行1101，默认 true）：
   * - 重叠模式：effectiveSendRecv = max(0, rawSendRecv - microBatchTicks)
   * - 非重叠模式：effectiveSendRecv = rawSendRecv
   */
  private _effectiveSendRecv(rawCost: number): number {
    if (this.commOverlapWithCompute) {
      return Math.max(0, rawCost - this.microBatchTicks);
    }
    return rawCost;
  }

  /** 是否最后一个 PP stage */
  isPpLastStage(stageIdx: number): boolean {
    return stageIdx === this.ppSize - 1;
  }

  /** 从 Batch 计算 batchSize */
  private _computeBatchSize(batch: Batch): number {
    return batch.reqs.size;
  }
}
