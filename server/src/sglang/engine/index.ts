// engine — S1: MockEngine/GraphRunner/Sampler + P6: ParallelGroups 集成

import type { SimulatorConfig, ModelConfig } from "../types";
import type { ForwardOutput } from "../core";
import { Batch, Req, SamplingParams } from "../core";
import { ParallelTopology, ParallelMetrics } from "../parallel";
import type { ParallelGroups } from "../parallel/groups";
import { initParallelGroups } from "../parallel/groups";
import { calculateMemoryBudgetParallel } from "../parallel/budget";
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

/** CUDA Graph 运行器（§3.3.1 / §9.11 SimGraphRunner） */
export class GraphRunner {
  private readonly enableCudaGraph: boolean;
  private readonly cudaGraphBs: number[] | null;
  private readonly cudaGraphMaxBs: number | null;
  readonly dummyReq: Req;
  private readonly dummyReq: Req | null;

  constructor(config: SimulatorConfig, dummyReq?: Req) {
    this.enableCudaGraph = config.enableCudaGraph;
    this.cudaGraphBs = config.cudaGraphBs;
    this.cudaGraphMaxBs = config.cudaGraphMaxBs;
    // dummyReq 用于 CUDA Graph padding（table_idx = max_running_req）
    if (dummyReq) {
      this.dummyReq = dummyReq;
    } else {
      this.dummyReq = new Req({
        rid: -1,
        inputIds: [0],
        samplingParams: new SamplingParams(),
      });
      this.dummyReq.deviceLen = 1;
      this.dummyReq.maxDeviceLen = 1;
    }
    this.dummyReq = dummyReq ?? null;
  }

  /** 判断 batch 是否可以使用 CUDA Graph replay（含 padding 对齐） */
  canUseCudaGraph(batch: Batch): boolean {
    if (!this.enableCudaGraph) return false;
    const bs = batch.reqs.size;
    if (this.cudaGraphBs !== null) {
      // 找到 >= bs 的最小 graph 尺寸（§9.11 使用 some 而非 includes）
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

  /** 将 batch padding 到 CUDA graph 尺寸（§9.11 pad_batch） */
  padBatch(batch: Batch): void {
    if (this.cudaGraphBs !== null && this.enableCudaGraph) {
      const bs = batch.reqs.size;
      const paddedSize = this.cudaGraphBs.find(cbs => cbs >= bs) ?? bs;
      // 在 batch 上记录 padding 信息（不影响 batch.reqs 本身）
      (batch as unknown as { paddedSize: number }).paddedSize = paddedSize;
    }
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
 * MockEngine — 仿真引擎（S1 + P6 ParallelGroups 集成）
 *
 * P6 集成点：
 * - 构造器接收 optional ParallelGroups；未提供时内部调用 initParallelGroups 创建
 * - forwardBatch 实现完整层循环：
 *   层循环前：ZMQ 广播 token IDs
 *   每层：Attention + CP KV all-gather → TP all-reduce after attn → MLP/MoE → TP all-reduce after MLP（非 MoE）→ DP-Attn all-gather
 *   层循环后：CPU barrier → PP 通信仿真 → TP 通信指标汇总 → 采样
 * - EPLB maybe_rebalance 不在 forwardBatch 内调用（移至 scheduler tick 末尾）
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
   * 便捷方法：仅传入 batch，自动提取第一个 req 的 tokenIds/seqLen
   * 用于 PP 等已有 batch 对象的场景（旧测试兼容）
   */
  forwardBatchReq(batch: Batch, localBatchSizes?: number[]): ForwardOutput {
    const firstReq = batch.reqs.values().next().value;
    const tokenIds = firstReq ? firstReq.inputIds : [];
    const seqLen = tokenIds.length;
    return this.forwardBatch(tokenIds, seqLen, batch, localBatchSizes);
  }
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
