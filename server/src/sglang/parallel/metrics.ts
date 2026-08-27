// parallel/metrics.ts — P0: ParallelMetrics 并行指标子结构（§10.9）

/**
 * 并行仿真指标子结构（§10.9）
 *
 * 嵌入 SimulationMetrics，收集所有并行维度的通信与负载指标。
 * 18+ 字段 + 6 通用维度字段。
 */
export class ParallelMetrics {
  // ===== TP 指标 =====
  tpCommTicks: number = 0;
  tpAllReduceCount: number = 0;
  tpWeightBytes: number = 0;

  // ===== DP 指标 =====
  dpRankLoad: number[] = [];
  dpAllocatePagesPerRank: number[] = [];
  dpAttnCommTicks: number = 0;

  // ===== EP 指标 =====
  epCommTicks: number = 0;
  epAllToAllCount: number = 0;
  epCrossRankTokens: number = 0;
  epExpertLoad: number[] = [];
  epRebalanceCostTicks: number = 0;

  // ===== PP 指标 =====
  ppBubbleTicks: number = 0;
  ppNumMicroBatches: number = 0;
  ppSendRecvTicks: number = 0;

  // ===== CP 指标 =====
  cpCommTicks: number = 0;
  cpAllGatherCount: number = 0;
  cpSeqLenPerRank: number = 0;

  // ===== 通用维度 =====
  worldSize: number = 1;
  tpSize: number = 1;
  dpSize: number = 1;
  epSize: number = 1;
  ppSize: number = 1;
  cpSize: number = 1;

  /** 通信 ticks 总和（TP + DP-Attn + EP + PP + CP） */
  get commTicksTotal(): number {
    return this.tpCommTicks + this.dpAttnCommTicks + this.epCommTicks +
           this.ppSendRecvTicks + this.cpCommTicks;
  }

  /** 汇总输出 */
  summary(): Record<string, unknown> {
    return {
      tpCommTicks: this.tpCommTicks,
      tpAllReduceCount: this.tpAllReduceCount,
      tpWeightBytes: this.tpWeightBytes,
      dpRankLoad: this.dpRankLoad,
      dpAllocatePagesPerRank: this.dpAllocatePagesPerRank,
      dpAttnCommTicks: this.dpAttnCommTicks,
      epCommTicks: this.epCommTicks,
      epAllToAllCount: this.epAllToAllCount,
      epCrossRankTokens: this.epCrossRankTokens,
      epExpertLoad: this.epExpertLoad,
      epRebalanceCostTicks: this.epRebalanceCostTicks,
      ppBubbleTicks: this.ppBubbleTicks,
      ppNumMicroBatches: this.ppNumMicroBatches,
      ppSendRecvTicks: this.ppSendRecvTicks,
      cpCommTicks: this.cpCommTicks,
      cpAllGatherCount: this.cpAllGatherCount,
      cpSeqLenPerRank: this.cpSeqLenPerRank,
      worldSize: this.worldSize,
      tpSize: this.tpSize,
      dpSize: this.dpSize,
      epSize: this.epSize,
      ppSize: this.ppSize,
      cpSize: this.cpSize,
    };
  }

  /** 重置所有指标到默认值 */
  reset(): void {
    this.tpCommTicks = 0;
    this.tpAllReduceCount = 0;
    this.tpWeightBytes = 0;

    this.dpRankLoad = [];
    this.dpAllocatePagesPerRank = [];
    this.dpAttnCommTicks = 0;

    this.epCommTicks = 0;
    this.epAllToAllCount = 0;
    this.epCrossRankTokens = 0;
    this.epExpertLoad = [];
    this.epRebalanceCostTicks = 0;

    this.ppBubbleTicks = 0;
    this.ppNumMicroBatches = 0;
    this.ppSendRecvTicks = 0;

    this.cpCommTicks = 0;
    this.cpAllGatherCount = 0;
    this.cpSeqLenPerRank = 0;

    this.worldSize = 1;
    this.tpSize = 1;
    this.dpSize = 1;
    this.epSize = 1;
    this.ppSize = 1;
    this.cpSize = 1;
  }
}
