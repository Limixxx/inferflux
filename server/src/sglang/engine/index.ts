// engine — S1: MockEngine/GraphRunner/Sampler + P4: PP + P5: CP + P3a: MoE 集成

import type { SimulatorConfig, ModelConfig } from "../types";
import type { Batch } from "../core";
import { ForwardOutput } from "../core";
import { PPPipelineSimulator } from "../parallel/pp";
import type { PipelineStepResult } from "../parallel/pp";
import { CPSimulator } from "../parallel/cp_simulator";
import { ParallelTopology, SimCommGroup as SimCommGroupImpl, ParallelMetrics } from "../parallel";
import { SimMoeBackend } from "../parallel/moe";
import { SimulationMetrics } from "../metrics";

// ===== GraphRunner =====

/** CUDA Graph 运行器桩 */
export class GraphRunner {
  private readonly enableCudaGraph: boolean;
  private readonly cudaGraphBs: number[] | null;
  private readonly cudaGraphMaxBs: number | null;

  constructor(config: SimulatorConfig) {
    this.enableCudaGraph = config.enableCudaGraph;
    this.cudaGraphBs = config.cudaGraphBs;
    this.cudaGraphMaxBs = config.cudaGraphMaxBs;
  }

  /** 判断 batch 是否可以使用 CUDA Graph replay */
  canUseCudaGraph(batch: Batch): boolean {
    if (!this.enableCudaGraph) return false;
    const bs = batch.reqs.size;
    if (this.cudaGraphBs !== null) {
      return this.cudaGraphBs.includes(bs);
    }
    if (this.cudaGraphMaxBs !== null) {
      return bs <= this.cudaGraphMaxBs;
    }
    return false;
  }

  /** CUDA Graph replay（桩实现） */
  replay(batch: Batch): number[] {
    return new Array(batch.reqs.size * 128).fill(0);
  }
}

// ===== Sampler =====

/** 采样器桩 */
export class Sampler {
  private readonly mode: "random" | "greedy" | "fixed";
  private readonly fixedToken: number;
  samplingCounter: number = 0;

  constructor(config: SimulatorConfig) {
    this.mode = config.mockSampleMode;
    this.fixedToken = config.fixedOutputToken;
  }

  /** 对 logits 进行采样，返回 token ID 列表 */
  sample(logits: number[], batchSize: number): number[] {
    this.samplingCounter += 1;
    switch (this.mode) {
      case "greedy":
        return new Array(batchSize).fill(0);
      case "fixed":
        return new Array(batchSize).fill(this.fixedToken);
      case "random":
      default:
        return new Array(batchSize).fill(0).map(() => Math.floor(Math.random() * 100));
    }
  }
}

// ===== MockEngine =====

/**
 * MockEngine — 仿真引擎（S1 + P4 PP + P5 CP + P3a MoE 集成）
 *
 * P4 集成点：
 * - forwardBatch 前先切 micro_batch；按 schedule 循环；
 *   last 走采样；中间 stage 返回 intermediate
 * - CUDA Graph replay 路径跳过 PP 通信仿真（§10.5.3 行4480-4482）
 *
 * P5 集成点：
 * - forwardBatch 逐层执行 attention + MLP，
 *   每层 attention 结束后若 cp_size > 1 则注入 KV all-gather 通信成本。
 *
 * P3a 集成点：
 * - 当 modelConfig.isMoe=true 时创建 SimMoeBackend 实例，
 *   对 MoE 层条件调用 moeBackend.forward。
 */
export class MockEngine {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly simMetrics: SimulationMetrics;
  readonly parallelMetrics: ParallelMetrics;
  readonly graphRunner: GraphRunner;
  readonly sampler: Sampler;
  readonly ppSim: PPPipelineSimulator;
  readonly ppRank: number;
  readonly isPpLast: boolean;
  readonly cpSim: CPSimulator | null;
  readonly topology: ParallelTopology;
  readonly moeBackend?: SimMoeBackend;

  /** MoE 层索引列表（在 isMoe=true 时由外部设定，默认全部 MoE 层） */
  moeLayers: number[];

  constructor(config: SimulatorConfig, modelConfig?: ModelConfig, ppRank: number = 0) {
    this.config = config;
    this.modelConfig = modelConfig ?? config.modelConfig;
    this.simMetrics = new SimulationMetrics();
    this.parallelMetrics = new ParallelMetrics();

    // P4: PP 集成
    this.ppRank = ppRank;
    this.isPpLast = (ppRank === config.ppSize - 1);
    this.ppSim = new PPPipelineSimulator(config, this.modelConfig);
    this.graphRunner = new GraphRunner(config);
    this.sampler = new Sampler(config);

    // P5: CP 集成
    this.cpSim = config.cpSize > 1
      ? new CPSimulator(config, this.modelConfig)
      : null;

    // P3a: 创建并行拓扑
    this.topology = new ParallelTopology({
      tpSize: config.tpSize,
      dpSize: config.dpSize,
      epSize: config.epSize,
      ppSize: config.ppSize,
      cpSize: config.cpSize,
      enableDpAttention: config.enableDpAttention,
    });

    // P3a: MoE 层索引
    this.moeLayers = this.modelConfig.isMoe
      ? Array.from({ length: this.modelConfig.numLayers }, (_, i) => i)
      : [];

    // P3a: 条件创建 SimMoeBackend
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
        metrics: this.parallelMetrics,
        seed: config.moeRoutingSeed,
      });
    }
  }

  /** 兼容属性：metrics 指向 simMetrics（保持 P3a 测试兼容） */
  get metrics(): SimulationMetrics {
    return this.simMetrics;
  }

  /** Mock 模型前向（桩实现，返回 logits 数组） */
  private _mockModelForward(batch: Batch): number[] {
    return new Array(batch.reqs.size * 128).fill(0);
  }

  /**
   * P4: 执行一个 batch 的 forward（Batch 版本）
   *
   * 行为合约（§10.5.3）：
   * - isIntermediate=true（中间 PP stage）：
   *   sampler 不调用，samplingCounter 不增加，sampledIds=null
   * - isIntermediate=false（最后 PP stage 或 pp_size=1）：
   *   sampler 调用，samplingCounter 增加，sampledIds 非空
   * - CUDA Graph replay 路径跳过 PP 通信仿真
   */
  forwardBatchPP(batch: Batch): ForwardOutput {
    let ppStepResult: PipelineStepResult | null = null;
    let logits: number[];

    if (this.graphRunner.canUseCudaGraph(batch)) {
      logits = this.graphRunner.replay(batch);
    } else {
      logits = this._mockModelForward(batch);
      ppStepResult = this.ppSim.simulatePipelineStep(batch);
    }

    if (ppStepResult && ppStepResult.totalTicks > 0) {
      this.simMetrics.parallel.ppSendRecvTicks += ppStepResult.sendRecvTicks;
      this.simMetrics.parallel.ppBubbleTicks += ppStepResult.bubbleTicks;
      this.simMetrics.parallel.ppNumMicroBatches += this.ppSim.numMicroBatches;
    }

    if (!this.isPpLast) {
      return { logits, sampledIds: null, isIntermediate: true };
    }

    const nextTokenIds = this.sampler.sample(logits, batch.reqs.size);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }

  /**
   * P5: 仿真一次 forward batch（逐层版本）
   *
   * @param seqLen 序列长度
   * @param numLayers 覆盖模型层数（默认取 modelConfig.numLayers）
   * @returns 更新后的 metrics
   */
  forwardBatchSeq(seqLen: number, numLayers?: number): SimulationMetrics {
    const layers = numLayers ?? this.modelConfig.numLayers;

    for (let layerIdx = 0; layerIdx < layers; layerIdx++) {
      // 1. Attention 计算（已由其他 Issue 实现，此处仅模拟）

      // 2. CP KV all-gather（仅 attn 层后）
      if (this.cpSim) {
        const cpResult = this.cpSim.simulateAttnForward(seqLen);
        this.simMetrics.parallel.cpCommTicks += cpResult.commTicks;
        this.simMetrics.parallel.cpAllGatherCount += 1;
        this.simMetrics.parallel.cpSeqLenPerRank = cpResult.seqLenPerRank;
      }

      // 3. MLP 计算（不触发 CP 通信）
    }

    this.simMetrics.parallel.cpSize = this.config.cpSize;
    this.simMetrics.parallel.tpSize = this.config.tpSize;

    return this.simMetrics;
  }

  /**
   * P3a: forward_batch — 仿真一层的 forward
   * @param tokenIds 当前 batch 的 token ID 列表
   * @param layerIdx 层索引
   * @returns 该层消耗的 ticks（含通信 + 计算）
   */
  forwardBatch(tokenIds: number[], layerIdx: number): number {
    // MoE 层：替换普通 MLP 为 moeBackend.forward
    if (this.modelConfig.isMoe && this.moeLayers.includes(layerIdx) && this.moeBackend) {
      const result = this.moeBackend.forward(tokenIds, layerIdx);
      return result.commTicks;
    }

    // 非 MoE 层：返回 0
    return 0;
  }
}
