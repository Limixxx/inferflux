// engine — S1: MockEngine/GraphRunner/Sampler + P4: PP + P5: CP + P3a: MoE + S3: MockSampler/MockAttnBackend

import type { SimulatorConfig, ModelConfig } from "../types";
import { Req, Batch, SamplingParams, BatchSamplingArgs, MockEvent } from "../core";
import type { ForwardOutput } from "../core";
import { ChunkedReq } from "../entities";
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
  private readonly dummyReq: Req | null;

  constructor(config: SimulatorConfig, dummyReq?: Req) {
    this.enableCudaGraph = config.enableCudaGraph;
    this.cudaGraphBs = config.cudaGraphBs;
    this.cudaGraphMaxBs = config.cudaGraphMaxBs;
    this.dummyReq = dummyReq ?? null;
  }

  /** 判断 batch 是否可以使用 CUDA Graph replay（含 padding 对齐） */
  canUseCudaGraph(batch: Batch): boolean {
    if (!this.enableCudaGraph) return false;
    const bs = batch.reqs.size;
    if (this.cudaGraphBs !== null) {
      return this.cudaGraphBs.some(cbs => cbs >= bs);
    }
    if (this.cudaGraphMaxBs !== null) {
      return bs <= this.cudaGraphMaxBs;
    }
    return false;
  }

  /** CUDA Graph replay（桩实现） */
  replay(batch: Batch): number[][] {
    const bs = batch.paddedReqs.length || batch.reqs.size;
    return Array.from({ length: bs }, () => new Array(128).fill(0));
  }

  /** S3: padBatch — 使用 dummyReq 填充（对齐 §9.11 L3609-3614） */
  padBatch(batch: Batch): void {
    let paddedSize: number;
    if (this.canUseCudaGraph(batch)) {
      paddedSize = this.cudaGraphBs?.find(bs => bs >= batch.reqs.size) ?? batch.reqs.size;
    } else {
      paddedSize = batch.reqs.size;
    }
    const dummyCount = paddedSize - batch.reqs.size;
    const reqsArray = [...batch.reqs.values()];
    batch.paddedReqs = [
      ...reqsArray,
      ...Array(dummyCount).fill(this.dummyReq),
    ];
  }
}

// ===== Sampler（P4 兼容保留） =====

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

// ===== MockSampler（S3 §9.11 L3558-3585） =====

/** 仿真采样器（对齐 §9.11 L3558-3585） */
export class MockSampler {
  readonly vocabSize: number;
  readonly mode: "random" | "greedy" | "fixed";
  readonly fixedToken: number;

  constructor(vocabSize: number, mode: "random" | "greedy" | "fixed", fixedToken: number = 0) {
    this.vocabSize = vocabSize;
    this.mode = mode;
    this.fixedToken = fixedToken;
  }

  /** 生成采样参数（对齐 §9.11 L3568-3577） */
  prepare(batch: Batch): BatchSamplingArgs {
    const reqsArray = [...batch.reqs.values()];
    const allGreedy = reqsArray.every(r => r.samplingParams.isGreedy);
    if (allGreedy) {
      return new BatchSamplingArgs({ temperatures: null });
    }
    const temperatures: number[] = [];
    const topK: number[] = [];
    const topP: number[] = [];
    for (const req of reqsArray) {
      const sp = req.samplingParams;
      temperatures.push(sp.temperature);
      topK.push(sp.topK);
      topP.push(sp.topP);
    }
    return new BatchSamplingArgs({ temperatures, topK, topP });
  }

  /** 采样（对齐 §9.11 L3579-3584） */
  sample(logits: number[][], args: BatchSamplingArgs): number[] {
    const batchSize = logits.length;
    switch (this.mode) {
      case "greedy":
        return logits.map(row => {
          let maxIdx = 0;
          let maxVal = row[0];
          for (let i = 1; i < row.length; i++) {
            if (row[i] > maxVal) { maxVal = row[i]; maxIdx = i; }
          }
          return maxIdx;
        });
      case "fixed":
        return new Array(batchSize).fill(this.fixedToken);
      case "random":
      default:
        return new Array(batchSize).fill(0).map(() => Math.floor(Math.random() * this.vocabSize));
    }
  }

  /** 采样参数处理管线方法 */
  apply_temperature(logits: number[], temperature: number): number[] {
    if (temperature <= 0) return logits;
    return logits.map(v => v / temperature);
  }

  apply_top_p_top_k(logits: number[], topP: number, topK: number): number[] {
    let result = [...logits];
    // top-k 过滤
    if (topK > 0 && topK < result.length) {
      const sorted = [...result].sort((a, b) => b - a);
      const threshold = sorted[topK - 1];
      result = result.map(v => v >= threshold ? v : -Infinity);
    }
    // top-p 核化
    if (topP < 1.0) {
      const exps = result.map(v => v === -Infinity ? 0 : Math.exp(v - Math.max(...result.filter(x => x !== -Infinity))));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / sum);
      let cumSum = 0;
      const sortedIndices = probs
        .map((p, i) => ({ p, i }))
        .sort((a, b) => b.p - a.p);
      const cutoffIndices = new Set<number>();
      for (const { p, i } of sortedIndices) {
        cumSum += p;
        cutoffIndices.add(i);
        if (cumSum >= topP) break;
      }
      result = result.map((v, i) => cutoffIndices.has(i) ? v : -Infinity);
    }
    return result;
  }

  apply_logits_penalty(logits: number[], tokenIds: number[], penalty: number): number[] {
    const result = [...logits];
    for (const tid of tokenIds) {
      if (tid >= 0 && tid < result.length) {
        if (penalty > 1.0) {
          result[tid] = result[tid] > 0 ? result[tid] * penalty : result[tid] / penalty;
        } else if (penalty < 1.0 && penalty > 0) {
          result[tid] = result[tid] > 0 ? result[tid] * penalty : result[tid] / penalty;
        }
      }
    }
    return result;
  }

  apply_logits_prob(_logits: number[]): number[] {
    return _logits;
  }

  apply_logits_bias(logits: number[], _bias: number[]): number[] {
    return logits;
  }
}

// ===== MockAttnBackend（S3 §9.11 L876-883） =====

/** 仿真 Attention Backend（桩实现） */
export class MockAttnBackend {
  /** 准备元数据（桩实现，设置 attnMetadata） */
  prepareMetadata(batch: Batch): void {
    batch.attnMetadata = {};
  }

  /** 模拟 KV 回收（stub，返回 0 ticks） */
  simulate_kv_recycle(): number {
    return 0;
  }
}

// ===== MockEngine =====

/**
 * MockEngine — 仿真引擎（S1 + P4 PP + P5 CP + P3a MoE + S3 集成）
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
 *
 * S3 集成点：
 * - 新增 forward_batch（snake_case）方法对齐 §9.11 L3676-3694
 * - 新增 mockSampler/mockAttnBackend/dummyReq 等属性
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

  // S3 新增属性
  readonly mockSampler: MockSampler;
  readonly mockAttnBackend: MockAttnBackend;
  readonly dummyReq: Req;
  readonly pageTable: number[][];
  readonly numPages: number;
  readonly maxSeqLen: number;

  constructor(config: SimulatorConfig, modelConfig?: ModelConfig, ppRank: number = 0) {
    this.config = config;
    this.modelConfig = modelConfig ?? config.modelConfig;
    this.simMetrics = new SimulationMetrics();
    this.parallelMetrics = new ParallelMetrics();

    // S3: 初始化 pageTable 和缓存相关
    this.maxSeqLen = config.maxSeqLen;
    this.numPages = config.numPages ?? 1024;
    this.pageTable = Array.from(
      { length: config.maxRunningReq + 1 },
      () => new Array(this.maxSeqLen).fill(0)
    );

    // S3: 创建 dummyReq（对齐 §9.11 L3662-3674）
    this.dummyReq = new Req({
      rid: -1,
      inputIds: [0],
      samplingParams: new SamplingParams({ maxNewTokens: 0 }),
    });
    this.dummyReq.deviceLen = 1;
    this.dummyReq.maxDeviceLen = 1;

    // S3: pageTable 最后一行预留给 dummyReq
    const numTokens = this.numPages * config.pageSize;
    this.pageTable[config.maxRunningReq] = new Array(this.maxSeqLen).fill(numTokens);

    // S3: 创建 GraphRunner（注入 dummyReq）
    this.graphRunner = new GraphRunner(config, this.dummyReq);

    // P4: PP 集成
    this.ppRank = ppRank;
    this.isPpLast = (ppRank === config.ppSize - 1);
    this.ppSim = new PPPipelineSimulator(config, this.modelConfig);
    this.sampler = new Sampler(config);

    // S3: 创建 MockSampler 和 MockAttnBackend
    this.mockSampler = new MockSampler(
      this.modelConfig.vocabSize,
      config.mockSampleMode,
      config.fixedOutputToken,
    );
    this.mockAttnBackend = new MockAttnBackend();

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

  /** Mock 模型前向（桩实现，返回 2D logits 数组） */
  private _mockModelForward(batch: Batch): number[][] {
    const bs = batch.paddedReqs.length || batch.reqs.size;
    return Array.from({ length: bs }, () => new Array(128).fill(0));
  }

  /**
   * S3: forward_batch — 对齐 §9.11 L3676-3694
   * 返回完整 ForwardOutput，含采样结果和时间/标识字段
   */
  forward_batch(batch: Batch, sampleArgs: BatchSamplingArgs): ForwardOutput {
    // 1. CUDA Graph 判断
    const isGraphCapture = this.graphRunner.canUseCudaGraph(batch);

    // 2. mock forward
    let logits: number[][];
    if (isGraphCapture) {
      logits = this.graphRunner.replay(batch);
    } else {
      logits = this._mockModelForward(batch);
    }

    // 3. complete_one（跳过 ChunkedReq），对齐 §9.11 L3684-3686
    for (const req of batch.reqs.values()) {
      if (!(req instanceof ChunkedReq)) {
        req.completeOne();
      }
    }

    // 4. 采样，对齐 §9.11 L3689
    const nextTokens = this.mockSampler.sample(logits, sampleArgs);

    // 5. 时间模型
    const isChunkPrefill = [...batch.reqs.values()].some(r => r instanceof ChunkedReq);
    const prefillBatchTime = batch.extendInputTokens > 0 ? this._computePrefillTime(batch) : 0;
    const decodeBatchTime = batch.numDecodeTokens > 0 ? this._computeDecodeTime(batch) : 0;

    // 6. 构造 ForwardOutput，对齐 §9.11 L3690-3694
    const copyDoneEvent = new MockEvent();
    copyDoneEvent.record();

    return {
      logits,
      nextTokensGpu: nextTokens,
      nextTokensCpu: [...nextTokens],
      copyDoneEvent,
      isIntermediate: false,
      prefillBatchTime,
      decodeBatchTime,
      isChunkPrefill,
      isGraphCapture,
      isPpLast: true,
      sampledIds: nextTokens,
    };
  }

  /** S3: 计算 prefill 时间 */
  private _computePrefillTime(batch: Batch): number {
    return batch.extendInputTokens * this.config.eagerForwardCostTicks;
  }

  /** S3: 计算 decode 时间 */
  private _computeDecodeTime(batch: Batch): number {
    if (this.graphRunner.canUseCudaGraph(batch)) {
      return batch.numDecodeTokens * this.config.graphReplayCostTicks;
    }
    return batch.numDecodeTokens * this.config.eagerForwardCostTicks;
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
  forwardBatchPP(batch: Batch): { logits: number[]; sampledIds: number[] | null; isIntermediate: boolean } {
    let ppStepResult: PipelineStepResult | null = null;
    let logits: number[];

    if (this.graphRunner.canUseCudaGraph(batch)) {
      logits = this.graphRunner.replay(batch).flat();
    } else {
      logits = this._mockModelForward(batch).flat();
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
