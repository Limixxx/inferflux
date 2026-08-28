// engine — S1: MockEngine/GraphRunner/Sampler + P4: PP integration

import type { SimulatorConfig, ModelConfig } from "../types";
import type { Batch } from "../core";
import { ForwardOutput } from "../core";
import { PPPipelineSimulator } from "../parallel/pp";
import type { PipelineStepResult } from "../parallel/pp";
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
 * MockEngine — 仿真引擎（S1 + P4 集成）
 *
 * P4 集成点：
 * - forwardBatch 前先切 micro_batch；按 schedule 循环；
 *   last 走采样；中间 stage 返回 intermediate
 * - CUDA Graph replay 路径跳过 PP 通信仿真（§10.5.3 行4480-4482）
 */
export class MockEngine {
  readonly graphRunner: GraphRunner;
  readonly sampler: Sampler;
  readonly ppSim: PPPipelineSimulator;
  readonly metrics: SimulationMetrics;
  readonly ppRank: number;
  readonly isPpLast: boolean;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig, ppRank: number = 0) {
    this.ppRank = ppRank;
    this.isPpLast = (ppRank === config.ppSize - 1);
    this.ppSim = new PPPipelineSimulator(config, modelConfig);
    this.graphRunner = new GraphRunner(config);
    this.sampler = new Sampler(config);
    this.metrics = new SimulationMetrics();
  }

  /** Mock 模型前向（桩实现，返回 logits 数组） */
  private _mockModelForward(batch: Batch): number[] {
    return new Array(batch.reqs.size * 128).fill(0);
  }

  /**
   * 执行一个 batch 的 forward
   *
   * 行为合约（§10.5.3）：
   * - isIntermediate=true（中间 PP stage）：
   *   sampler 不调用，samplingCounter 不增加，sampledIds=null
   * - isIntermediate=false（最后 PP stage 或 pp_size=1）：
   *   sampler 调用，samplingCounter 增加，sampledIds 非空
   * - CUDA Graph replay 路径跳过 PP 通信仿真
   */
  forwardBatch(batch: Batch): ForwardOutput {
    // R2-3: CUDA Graph replay 路径跳过 PP 通信仿真
    // 引用 §10.5.3 行4480-4482
    let ppStepResult: PipelineStepResult | null = null;
    let logits: number[];

    if (this.graphRunner.canUseCudaGraph(batch)) {
      logits = this.graphRunner.replay(batch);
      // CUDA Graph 内 PP 通信成本为 0
    } else {
      logits = this._mockModelForward(batch);
      ppStepResult = this.ppSim.simulatePipelineStep(batch);
    }

    // PP 指标回填（R2-7: 精确映射）
    if (ppStepResult && ppStepResult.totalTicks > 0) {
      this.metrics.parallel.ppSendRecvTicks += ppStepResult.sendRecvTicks;
      this.metrics.parallel.ppBubbleTicks += ppStepResult.bubbleTicks;
      this.metrics.parallel.ppNumMicroBatches += this.ppSim.numMicroBatches;
    }

    // 严格遵循行为合约
    if (!this.isPpLast) {
      return { logits, sampledIds: null, isIntermediate: true };
    }

    const nextTokenIds = this.sampler.sample(logits, batch.reqs.size);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }
}
