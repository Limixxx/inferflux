// parallel/comm_group.ts — P0: SimCommGroup 统一通信成本模型 + MockTPGroup

import type { SimulatorConfig } from "../types";

// ===== 通信组类型 =====

/** 通信组类型标识 */
export type CommGroupType = "tp" | "ep" | "pp" | "cp" | "dp_attn";

// ===== 通信操作类型（内部使用） =====

type CommOpType = "all_reduce" | "all_gather" | "all_to_all" | "send_recv";

// ===== 构造选项 =====

export interface SimCommGroupOpts {
  groupType: CommGroupType;
  size: number;
  ranks?: number[];
  globalRanks?: number[];
  networkBandwidthGBps: number;
  latencyUs: number;
  efficiency?: number;
}

// ===== SimCommGroup =====

/**
 * 统一通信成本模型（§3.4.4）
 *
 * 纯算术通信成本计算：给定通信操作类型和数据量，返回 comm_ticks。
 * size=1 时 allReduce/allGather/allToAll 退化为 noop（返回 0）。
 * 所有公开方法返回整数 ticks（向上取整 ceil）。
 */
export class SimCommGroup {
  readonly groupType: CommGroupType;
  readonly size: number;
  readonly ranks: number[];
  readonly globalRanks: number[];
  readonly networkBandwidthGBps: number;
  readonly latencyUs: number;
  readonly efficiency: number;

  private readonly bandwidthBytesPerUs: number;

  constructor(opts: SimCommGroupOpts) {
    this.groupType = opts.groupType;
    this.size = opts.size;
    this.ranks = opts.ranks ?? Array.from({ length: opts.size }, (_, i) => i);
    this.globalRanks = opts.globalRanks ?? this.ranks;
    this.networkBandwidthGBps = opts.networkBandwidthGBps;
    this.latencyUs = opts.latencyUs;
    this.efficiency = opts.efficiency ?? 1.0;

    // bandwidth: GB/s → bytes/μs
    // 1 GB/s = 1e9 B / 1e6 μs = 1000 B/μs
    this.bandwidthBytesPerUs = opts.networkBandwidthGBps * 1000;
  }

  /**
   * 内部统一成本计算
   * 公式根据 opType 选择对应的通信模型
   */
  private _computeCost(opType: CommOpType, bytes: number): number {
    if (this.bandwidthBytesPerUs === 0) {
      return Infinity;
    }

    switch (opType) {
      case "all_reduce":
        // 2 × bytes × (size-1) / size / bandwidth + latency
        return (2 * bytes * (this.size - 1)) / this.size / this.bandwidthBytesPerUs + this.latencyUs;

      case "all_gather":
        // bytes × (size-1) / bandwidth + latency
        return (bytes * (this.size - 1)) / this.bandwidthBytesPerUs + this.latencyUs;

      case "all_to_all":
        // bytes × size / bandwidth + latency × size
        return (bytes * this.size) / this.bandwidthBytesPerUs + this.latencyUs * this.size;

      case "send_recv":
        // bytes / bandwidth + latency
        return bytes / this.bandwidthBytesPerUs + this.latencyUs;
    }
  }

  /** All-Reduce：tensor 全规约，返回 comm_ticks */
  allReduce(tensorBytes: number): number {
    if (this.size === 1) return 0;
    const raw = this._computeCost("all_reduce", tensorBytes);
    return Math.ceil(raw / this.efficiency);
  }

  /** All-Gather：多个 size 的数据收集，返回 comm_ticks */
  allGather(sizes: number[]): number {
    if (this.size === 1) return 0;
    const totalBytes = sizes.reduce((s, v) => s + v, 0);
    const raw = this._computeCost("all_gather", totalBytes);
    return Math.ceil(raw / this.efficiency);
  }

  /** All-to-All：MoE 专家路由通信，返回 comm_ticks */
  allToAll(sendSizes: number[], recvSizes: number[]): number {
    if (this.size === 1) return 0;
    const totalSend = sendSizes.reduce((s, v) => s + v, 0);
    const totalRecv = recvSizes.reduce((s, v) => s + v, 0);
    const totalBytes = totalSend + totalRecv;
    const raw = this._computeCost("all_to_all", totalBytes);
    return Math.ceil(raw / this.efficiency);
  }

  /** Send/Recv：点对点通信，返回 comm_ticks */
  sendRecv(bytes: number, _peer: number): number {
    const raw = this._computeCost("send_recv", bytes);
    return Math.ceil(raw / this.efficiency);
  }

  /** Barrier：同步屏障，noop，不产生成本 */
  barrier(): void {
    // noop
  }
}

// ===== MockTPGroup — 向后兼容薄包装 =====

/**
 * MockTPGroup 向后兼容包装（对应 Issue 中的薄包装需求）
 * 内部创建 SimCommGroup("tp")，旧的 mockAllReduceSum 返回 SimCommGroup.allReduce
 * tpSize=1 时 inner.size=1，allReduce 直接返回 0
 */
export class MockTPGroup {
  readonly groupType: CommGroupType = "tp";
  readonly size: number;
  private readonly inner: SimCommGroup;

  constructor(tpSize: number, config: SimulatorConfig) {
    this.size = tpSize;
    this.inner = new SimCommGroup({
      groupType: "tp",
      size: tpSize,
      networkBandwidthGBps: config.networkBandwidthGBps,
      latencyUs: config.networkLatencyUs,
      efficiency: config.tpEfficiency,
    });
  }

  allReduce(tensorBytes: number): number {
    return this.inner.allReduce(tensorBytes);
  }

  allGather(sizes: number[]): number {
    return this.inner.allGather(sizes);
  }

  allToAll(sendSizes: number[], recvSizes: number[]): number {
    return this.inner.allToAll(sendSizes, recvSizes);
  }

  sendRecv(bytes: number, peer: number): number {
    return this.inner.sendRecv(bytes, peer);
  }

  barrier(): void {
    this.inner.barrier();
  }

  /** 旧接口兼容 */
  mockAllReduceSum(dataBytes: number): number {
    return this.allReduce(dataBytes);
  }
}
