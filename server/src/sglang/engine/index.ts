// engine — S1: MockEngine/GraphRunner/Sampler + P6: ParallelGroups 集成

import type { SimulatorConfig, ModelConfig } from "../types";
import type { ForwardOutput } from "../core";
import { Batch, Req, SamplingParams } from "../core";
import { ParallelTopology, ParallelMetrics } from "../parallel";
import type { ParallelGroups } from "../parallel/groups";
import { initParallelGroups } from "../parallel/groups";
import { calculateMemoryBudgetParallel } from "../parallel/budget";
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
 * MockEngine — 仿真引擎（S1 + P6 ParallelGroups 集成）
 *
 * P6 集成点：
 * - 构造器接收 optional ParallelGroups；未提供时内部调用 initParallelGroups 创建
 * - forwardBatch 实现完整层循环：
 *   层循环前：ZMQ 广播 token IDs
 *   每层：Attention + CP KV all-gather → TP all-reduce after attn → MLP/MoE → TP all-reduce after MLP（非 MoE）→ DP-Attn all-gather
 *   层循环后：CPU barrier → PP 通信仿真 → TP 通信指标汇总 → 采样
 * - EPLB maybe_rebalance 不在 forwardBatch 内调用（移至 scheduler tick 末尾）
 */
export class MockEngine {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly simMetrics: SimulationMetrics;
  readonly parallelMetrics: ParallelMetrics;
  readonly graphRunner: GraphRunner;
  readonly sampler: Sampler;
  readonly groups: ParallelGroups;
  readonly isPpLast: boolean;
  readonly ppRank: number;

  /** MoE 层索引列表（在 isMoe=true 时由外部设定，默认全部 MoE 层） */
  moeLayers: number[];

  constructor(
    config: SimulatorConfig,
    modelConfig?: ModelConfig,
    ppRank: number = 0,
    parallelGroups?: ParallelGroups,
  ) {
    this.config = config;
    this.modelConfig = modelConfig ?? config.modelConfig;
    this.simMetrics = new SimulationMetrics();
    this.parallelMetrics = new ParallelMetrics();

    this.ppRank = ppRank;
    this.isPpLast = (ppRank === config.ppSize - 1);
    this.graphRunner = new GraphRunner(config);
    this.sampler = new Sampler(config);

    // P6: 使用传入的 parallelGroups 或内部创建
    if (parallelGroups) {
      this.groups = parallelGroups;
    } else {
      const budget = calculateMemoryBudgetParallel(
        config,
        this.modelConfig,
        config.totalGpuMemory,
      );
      this.groups = initParallelGroups({
        config,
        modelConfig: this.modelConfig,
        numPages: budget.numPages,
        metrics: this.parallelMetrics,
      });
    }

    // MoE 层索引
    this.moeLayers = this.modelConfig.isMoe
      ? Array.from({ length: this.modelConfig.numLayers }, (_, i) => i)
      : [];
  }

  /** 兼容属性：metrics 指向 simMetrics */
  get metrics(): SimulationMetrics {
    return this.simMetrics;
  }

  /** 兼容属性：topology 指向 groups.topology */
  get topology(): ParallelTopology {
    return this.groups.topology;
  }

  /** 兼容属性：ppSim 指向 groups.ppSim */
  get ppSim() {
    return this.groups.ppSim;
  }

  /** 兼容属性：cpSim 指向 groups.cpSim */
  get cpSim() {
    return this.groups.cpSim;
  }

  /** 兼容属性：moeBackend 指向 groups.moeBackend（null → undefined 兼容旧断言） */
  get moeBackend() {
    return this.groups.moeBackend ?? undefined;
  }

  /** Mock 模型前向（桩实现，返回 logits 数组） */
  private _mockModelForward(batch: Batch): number[] {
    return new Array(batch.reqs.size * 128).fill(0);
  }

  /**
   * 便捷方法：仅传入 batch，自动提取第一个 req 的 tokenIds/seqLen
   * 用于 PP 等已有 batch 对象的场景（旧测试兼容）
   */
  forwardBatchReq(batch: Batch, localBatchSizes?: number[]): ForwardOutput {
    const firstReq = batch.reqs.values().next().value;
    const tokenIds = firstReq ? firstReq.inputIds : [];
    const seqLen = tokenIds.length;
    return this.forwardBatch(tokenIds, seqLen, batch, localBatchSizes);
  }

  /**
   * 便捷方法：用 seqLen 自动构造最小 batch 后调用 forwardBatch
   * 用于不需要精确 batch 内容的场景（如旧测试兼容）
   */
  forwardBatchSeqLen(seqLen: number): ForwardOutput {
    const batch = new Batch();
    const req = new Req({
      rid: 0,
      inputIds: new Array(seqLen).fill(0),
      samplingParams: new SamplingParams(),
    });
    batch.reqs.set(0, req);
    batch.readyIds.push(0);
    return this.forwardBatch(batch.reqs.get(0)!.inputIds, seqLen, batch);
  }

  /**
   * P6: 统一 forwardBatch — 完整层循环并行仿真
   *
   * @param tokenIds 当前 batch 的 token ID 列表
   * @param seqLen 序列长度
   * @param batch 批处理
   * @param localBatchSizes DP-Attn 各 rank 的本地 batch 大小（可选）
   * @returns ForwardOutput
   */
  forwardBatch(
    tokenIds: number[],
    seqLen: number,
    batch: Batch,
    localBatchSizes?: number[],
  ): ForwardOutput {
    // ── 层循环前：ZMQ 广播 token IDs ──
    let totalCommTicks = this.groups.tpComm.broadcastAll([tokenIds]);

    const isMoELayer = (idx: number): boolean =>
      this.modelConfig.isMoe && this.moeLayers.includes(idx);

    for (let layerIdx = 0; layerIdx < this.modelConfig.numLayers; layerIdx++) {
      // 步骤 1: Attention 计算 + CP KV all-gather
      if (this.groups.cpSim) {
        const cpResult = this.groups.cpSim.simulateAttnForward(seqLen);
        this.simMetrics.parallel.cpCommTicks += cpResult.commTicks;
        this.simMetrics.parallel.cpAllGatherCount += 1;
        this.simMetrics.parallel.cpSeqLenPerRank = cpResult.seqLenPerRank;
      }

      // 步骤 2: TP all-reduce after attention
      totalCommTicks += this.groups.tpSim.allReduceAfterAttn(batch.reqs.size);

      // 步骤 3: MLP / MoE
      if (isMoELayer(layerIdx) && this.groups.moeBackend) {
        const moeResult = this.groups.moeBackend.forward(tokenIds, layerIdx);
        totalCommTicks += moeResult.commTicks;
        // MoE 层不调用 tpSim.allReduceAfterMlp（EP all-to-all 替代 TP all-reduce）
      } else {
        // 步骤 4: TP all-reduce after MLP（非 MoE 层）
        totalCommTicks += this.groups.tpSim.allReduceAfterMlp(batch.reqs.size);
      }

      // 步骤 5: DP-Attn all-gather after MLP
      if (this.groups.dpAttnSim && localBatchSizes) {
        const dpResult = this.groups.dpAttnSim.simulateMlpForward(localBatchSizes);
        this.simMetrics.parallel.dpAttnCommTicks += dpResult.commTicks;
        totalCommTicks += dpResult.commTicks;
      }
    }

    // ── 层循环后：CPU barrier ──
    totalCommTicks += this.groups.tpComm.cpuBarrier();

    // 步骤 6: PP 通信仿真（CUDA Graph replay 时跳过 PP）
    const useCudaGraph = this.graphRunner.canUseCudaGraph(batch);
    if (this.groups.ppSim.ppSize > 1 && !useCudaGraph) {
      const ppResult = this.groups.ppSim.simulatePipelineStep(batch);
      this.simMetrics.parallel.ppSendRecvTicks += ppResult.sendRecvTicks;
      this.simMetrics.parallel.ppBubbleTicks += ppResult.bubbleTicks;
      this.simMetrics.parallel.ppNumMicroBatches += this.groups.ppSim.numMicroBatches;
    }

    // 步骤 7: TP 通信指标汇总
    this.simMetrics.parallel.tpCommTicks +=
      this.groups.tpSim.totalCommTicksPerStep() +
      this.groups.tpComm.zmqBroadcastTicks +
      this.groups.tpComm.barrierTicks;
    this.groups.tpSim.resetStepComm();

    // 更新通用维度
    this.simMetrics.parallel.worldSize = this.groups.topology.worldSize;
    this.simMetrics.parallel.tpSize = this.config.tpSize;
    this.simMetrics.parallel.dpSize = this.config.dpSize;
    this.simMetrics.parallel.epSize = this.config.epSize;
    this.simMetrics.parallel.ppSize = this.config.ppSize;
    this.simMetrics.parallel.cpSize = this.config.cpSize;

    // 步骤 8: 采样（仅最后 PP stage）
    const logits = this._mockModelForward(batch);
    if (!this.isPpLast) {
      return { logits, sampledIds: null, isIntermediate: true };
    }
    const nextTokenIds = this.sampler.sample(logits, batch.reqs.size);
    return { logits, sampledIds: nextTokenIds, isIntermediate: false };
  }
}
