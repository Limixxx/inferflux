// engine — S1: MockEngine/GraphRunner/Sampler + P5: CPSimulator integration

import type { SimulatorConfig, ModelConfig } from "../types";
import { CPSimulator } from "../parallel/cp_simulator";
import { SimulationMetrics } from "../metrics";

/**
 * MockEngine — 仿真推理引擎（S1 桩 + P5 CPSimulator 集成）
 *
 * forwardBatch 逐层执行 attention + MLP，
 * 每层 attention 结束后若 cp_size > 1 则注入 KV all-gather 通信成本。
 */
export class MockEngine {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly metrics: SimulationMetrics;
  readonly cpSim: CPSimulator | null;

  constructor(config: SimulatorConfig) {
    this.config = config;
    this.modelConfig = config.modelConfig;
    this.metrics = new SimulationMetrics();

    // P5: 仅当 cp_size > 1 时创建 CPSimulator
    this.cpSim = config.cpSize > 1
      ? new CPSimulator(config, this.modelConfig)
      : null;
  }

  /**
   * 仿真一次 forward batch
   *
   * @param seqLen 序列长度
   * @param numLayers 覆盖模型层数（默认取 modelConfig.numLayers）
   * @returns 更新后的 metrics
   */
  forwardBatch(seqLen: number, numLayers?: number): SimulationMetrics {
    const layers = numLayers ?? this.modelConfig.numLayers;

    for (let layerIdx = 0; layerIdx < layers; layerIdx++) {
      // 1. Attention 计算（已由其他 Issue 实现，此处仅模拟）

      // 2. CP KV all-gather（仅 attn 层后）
      if (this.cpSim) {
        const cpResult = this.cpSim.simulateAttnForward(seqLen);
        this.metrics.parallel.cpCommTicks += cpResult.commTicks;
        this.metrics.parallel.cpAllGatherCount += 1;
        this.metrics.parallel.cpSeqLenPerRank = cpResult.seqLenPerRank;
      }

      // 3. MLP 计算（不触发 CP 通信）
    }

    // 设置并行维度信息
    this.metrics.parallel.cpSize = this.config.cpSize;
    this.metrics.parallel.tpSize = this.config.tpSize;

    return this.metrics;
  }
}
