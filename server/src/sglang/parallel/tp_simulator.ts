// parallel/tp_simulator.ts — P1a: TPSimulator TP 张量并行仿真器（§10.2）

import { SimCommGroup } from "./comm_group";
import type { SimulatorConfig, ModelConfig } from "../types";
import { divEven } from "../core";

/**
 * TP 张量并行仿真器（§10.2）
 *
 * 封装通信组与内存修正逻辑，在 forward 路径中注入 all-reduce 通信成本。
 * tpSize=1 时退化为 noop（所有方法返回 0）。
 */
export class TPSimulator {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly tpSize: number;
  readonly commGroup: SimCommGroup;

  // 内存修正：本地（单 rank）可见的参数量
  readonly localNumHeads: number;
  readonly localNumKvHeads: number;
  readonly localIntermediate: number;

  // 通信记录
  private _commTicksLog: number[] = [];

  constructor(config: SimulatorConfig, modelConfig: ModelConfig) {
    this.config = config;
    this.modelConfig = modelConfig;
    this.tpSize = config.tpSize;

    this.commGroup = new SimCommGroup({
      groupType: "tp",
      size: config.tpSize,
      networkBandwidthGBps: config.networkBandwidthGBps,
      latencyUs: config.networkLatencyUs,
      efficiency: config.tpEfficiency,
    });

    // 内存修正
    if (this.tpSize > 1) {
      this.localNumHeads = divEven(modelConfig.numAttentionHeads, this.tpSize, true)[0];
      this.localNumKvHeads = divEven(modelConfig.numKvHeads, this.tpSize, true)[0];
      this.localIntermediate = Math.floor(modelConfig.intermediateSize / this.tpSize);
    } else {
      this.localNumHeads = modelConfig.numAttentionHeads;
      this.localNumKvHeads = modelConfig.numKvHeads;
      this.localIntermediate = modelConfig.intermediateSize;
    }
  }

  /** attention 后 all-reduce，数据量 = batch × hidden × dtype，返回 comm_ticks */
  allReduceAfterAttn(batchSize: number): number {
    if (this.tpSize <= 1) return 0;
    const dataBytes = batchSize * this.modelConfig.hiddenSize * this.config.dtypeSize;
    const ticks = this.commGroup.allReduce(dataBytes);
    this._commTicksLog.push(ticks);
    return ticks;
  }

  /** MLP 后 all-reduce，数据量 = batch × hidden × dtype，返回 comm_ticks */
  allReduceAfterMlp(batchSize: number): number {
    if (this.tpSize <= 1) return 0;
    const dataBytes = batchSize * this.modelConfig.hiddenSize * this.config.dtypeSize;
    const ticks = this.commGroup.allReduce(dataBytes);
    this._commTicksLog.push(ticks);
    return ticks;
  }

  /** 返回当前 step 累计的 TP 通信 ticks */
  totalCommTicksPerStep(): number {
    return this._commTicksLog.reduce((sum, v) => sum + v, 0);
  }

  /** 重置当前 step 的通信记录 */
  resetStepComm(): void {
    this._commTicksLog = [];
  }
}
