// engine — S1: MockEngine/GraphRunner/Sampler + P3a: MoE 集成

import type { SimulatorConfig, ModelConfig } from "../types";
import { ParallelTopology, SimCommGroup as SimCommGroupImpl, ParallelMetrics, SimMoeBackend } from "../parallel";

/**
 * MockEngine — 仿真引擎桩（P3a: 集成 MoE 路由）
 *
 * 当 modelConfig.isMoe=true 时创建 SimMoeBackend 实例，
 * 对 MoE 层条件调用 moeBackend.forward。
 */
export class MockEngine {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly topology: ParallelTopology;
  readonly metrics: ParallelMetrics;
  readonly moeBackend?: SimMoeBackend;

  /** MoE 层索引列表（在 isMoe=true 时由外部设定，默认全部 MoE 层） */
  moeLayers: number[];

  constructor(config: SimulatorConfig) {
    this.config = config;
    this.modelConfig = config.modelConfig;
    this.metrics = new ParallelMetrics();

    // 创建并行拓扑
    this.topology = new ParallelTopology({
      tpSize: config.tpSize,
      dpSize: config.dpSize,
      epSize: config.epSize,
      ppSize: config.ppSize,
      cpSize: config.cpSize,
      enableDpAttention: config.enableDpAttention,
    });

    // MoE 层索引：默认为所有层（isMoe 时）
    this.moeLayers = this.modelConfig.isMoe
      ? Array.from({ length: this.modelConfig.numLayers }, (_, i) => i)
      : [];

    // 条件创建 SimMoeBackend
    if (this.modelConfig.isMoe) {
      const epCommGroup = new SimCommGroupImpl({
        groupType: "ep",
        size: config.epSize,
        networkBandwidthGBps: config.networkBandwidthGBps,
        latencyUs: config.networkLatencyUs,
        efficiency: config.epEfficiency,
      });

      this.moeBackend = new SimMoeBackend({
        modelConfig: this.modelConfig,
        topology: this.topology,
        config: config,
        epCommGroup: epCommGroup,
        metrics: this.metrics,
        seed: config.moeRoutingSeed,
      });
    }
  }

  /**
   * forward_batch — 仿真一层的 forward
   * @param tokenIds 当前 batch 的 token ID 列表
   * @param layerIdx 层索引
   * @returns 该层消耗的 ticks（含通信 + 计算）
   */
  forwardBatch(tokenIds: number[], layerIdx: number): number {
    // MoE 层：替换普通 MLP 为 moeBackend.forward
    if (this.modelConfig.isMoe && this.moeLayers.includes(layerIdx) && this.moeBackend) {
      const result = this.moeBackend.forward(tokenIds, layerIdx);
      return result.commTicks;  // mock forward 计算 cost 为 0，仅返回通信 ticks
    }

    // 非 MoE 层：返回 0（仿真桩，后续 Issue 实现完整计算成本）
    return 0;
  }
}
