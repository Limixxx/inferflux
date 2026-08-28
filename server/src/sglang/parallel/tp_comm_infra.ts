// parallel/tp_comm_infra.ts — P1a: TPCommInfraSimulator TP 通信基础设施仿真器（§10.6）

import { SimCommGroup } from "./comm_group";
import type { SimulatorConfig, ModelConfig } from "../types";

/**
 * TP 通信基础设施仿真器（§10.6）
 *
 * 仿真 ZMQ 广播 + gloo barrier + nccl all-reduce 的三层通信成本。
 * tpSize=1 时退化为 noop（所有方法返回 0）。
 */
export class TPCommInfraSimulator {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly tpSize: number;

  readonly commGroup: SimCommGroup;

  readonly cpuGroupType: string; // "gloo"
  readonly gpuGroupType: string; // "nccl" | "pynccl"

  zmqBroadcastTicks: number = 0;
  barrierTicks: number = 0;

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

    this.cpuGroupType = config.tpCpuGroupType;
    this.gpuGroupType = config.tpGpuGroupType;
  }

  /** ZMQ 广播成本（primary rank → 其他 TP rank） */
  zmqBroadcast(msgSize: number): number {
    if (this.tpSize <= 1) return 0;
    const cost = Math.ceil(msgSize / Math.max(1, this.config.commBandwidthBytesPerTick));
    this.zmqBroadcastTicks += cost;
    return cost;
  }

  /** gloo barrier 成本（固定延迟 1 tick） */
  cpuBarrier(): number {
    if (this.tpSize <= 1) return 0;
    const cost = 1;
    this.barrierTicks += cost;
    return cost;
  }

  /** nccl all-reduce 成本（委托 SimCommGroup） */
  gpuAllReduce(dataBytes: number): number {
    return this.commGroup.allReduce(dataBytes);
  }

  /** 批量 ZMQ 广播：汇总 token_ids_list 的字节数后调用 zmqBroadcast */
  broadcastAll(tokenIdsList: number[][]): number {
    if (this.tpSize <= 1) return 0;
    const totalBytes = tokenIdsList.reduce((sum, list) => sum + list.length * 4, 0);
    return this.zmqBroadcast(totalBytes);
  }
}
