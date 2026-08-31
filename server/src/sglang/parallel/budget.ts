// parallel/budget.ts — P1b: 并行组合内存预算 calculateMemoryBudgetParallel (§10.7.2)

import type { ModelConfig, SimulatorConfig } from "../types";
import { divEven } from "../core";
import { estimateGraphBuffer, type MemoryBudgetResult } from "../cache/budget";

/** 并行维度内存修正明细 */
export interface ParallelMemoryCorrections {
  /** TP 修正：权重除以 tp_size */
  tpWeightDivisor: number;
  /** DP 修正：标准 DP 时 kv_budget_per_rank = kv_budget / dp_size；DP Attention 时 kv_per_tok_bytes ×= dp_size */
  dpKvMultiplier: number;
  /** EP 修正：MoE FFN 权重除以 ep_size（附加专家矩阵开销已在权重估算中） */
  epWeightDivisor: number;
  /** PP 修正：权重按 stage 切分，除以 pp_size */
  ppWeightDivisor: number;
  /** CP 修正：kv_per_tok_bytes ×= cp_size */
  cpKvMultiplier: number;
}

/**
 * 估算 attention 权重占用的字节数
 * 公式：numLayers × (numAttentionHeads × headDim × 3 + hiddenSize) × dtypeSize
 * 其中 3 = Q/K/V 三个权重矩阵，+ hiddenSize 为 output projection
 */
function _estimateAttnWeightBytes(modelConfig: ModelConfig, dtypeSize: number): number {
  return modelConfig.numLayers *
    (modelConfig.numAttentionHeads * modelConfig.headDim * 3 + modelConfig.hiddenSize) *
    dtypeSize;
}

/**
 * 估算 MLP 权重占用的字节数
 * - 非 MoE：numLayers × hiddenSize × intermediateSize × 3 × dtypeSize
 *   （gate/up/down 三个矩阵）
 * - MoE：numLayers × numExperts × hiddenSize × moeIntermediateSize × 3 × dtypeSize
 *   仅计算 MoE 层的专家矩阵权重
 */
function _estimateMlpWeightBytes(modelConfig: ModelConfig, dtypeSize: number): number {
  if (modelConfig.isMoe && modelConfig.numExperts > 0) {
    return modelConfig.numLayers *
      modelConfig.numExperts *
      modelConfig.hiddenSize *
      modelConfig.moeIntermediateSize *
      3 *
      dtypeSize;
  }
  return modelConfig.numLayers *
    modelConfig.hiddenSize *
    modelConfig.intermediateSize *
    3 *
    dtypeSize;
}

/**
 * 组合并行维度的内存预算计算
 * 对应 §10.7.2 calculate_memory_budget_parallel
 *
 * @param config - SimulatorConfig（含并行配置、内存配置）
 * @param modelConfig - ModelConfig（含模型特征参数）
 * @param totalMemory - GPU 总显存（bytes）
 * @returns MemoryBudgetResult（含 parallelCorrections 修正明细）
 */
export function calculateMemoryBudgetParallel(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
  totalMemory: number,
): MemoryBudgetResult {
  const tpSize = config.tpSize;
  const dpSize = config.dpSize;
  const epSize = config.epSize;
  const ppSize = config.ppSize;
  const cpSize = config.cpSize;

  // 1. 基础权重估算
  let weightBytes: number;
  if (config.enableDpAttention) {
    // DP Attention 时：attention 权重复制不除 tp，MLP 权重按 TP 切分
    const attnWeight = _estimateAttnWeightBytes(modelConfig, config.dtypeSize);
    const mlpWeight = _estimateMlpWeightBytes(modelConfig, config.dtypeSize) / tpSize;
    weightBytes = attnWeight + mlpWeight;
  } else {
    // 非 DP Attention：全部权重按 TP 切分
    const totalWeight = _estimateAttnWeightBytes(modelConfig, config.dtypeSize) +
      _estimateMlpWeightBytes(modelConfig, config.dtypeSize);
    weightBytes = totalWeight / tpSize;
  }

  // 2. EP 修正：若 epSize > 1 且 isMoe，weight_bytes /= epSize
  let epWeightDivisor = 1;
  if (epSize > 1 && modelConfig.isMoe) {
    weightBytes = weightBytes / epSize;
    epWeightDivisor = epSize;
  }

  // 3. PP 修正：weight_bytes /= ppSize
  weightBytes = weightBytes / ppSize;

  // 4. graphBuffer
  const graphBuffer = estimateGraphBuffer(config.cudaGraphBs, modelConfig);

  // 5. available = floor(memoryRatio × totalMemory) - weight_bytes - graphBuffer
  const available = Math.floor(config.memoryRatio * totalMemory) - weightBytes - graphBuffer;

  // 6. DP 修正
  let kvBudgetPerRank: number;
  if (config.enableDpAttention) {
    // DP Attention：不除 dpSize，但后面 kv_per_tok_bytes 会乘 dpSize
    kvBudgetPerRank = available;
  } else {
    // 标准 DP：每 rank 独立 KV pool
    kvBudgetPerRank = available / dpSize;
  }

  // 7. KV per token per layer 计算
  // local_kv_heads = sum(divEven(numKvHeads, tpSize, true))
  const localKvHeads = divEven(modelConfig.numKvHeads, tpSize, true)
    .reduce((sum, v) => sum + v, 0);
  let kvPerTokPerLayer = 2 * localKvHeads * modelConfig.headDim * config.dtypeSize;

  // 8. CP 修正：kv_per_tok_per_layer *= cpSize
  kvPerTokPerLayer *= cpSize;

  // 9. DP Attention 修正：kv_per_tok_per_layer *= dpSize
  let dpKvMultiplier = 1;
  if (config.enableDpAttention) {
    kvPerTokPerLayer *= dpSize;
    dpKvMultiplier = dpSize;
  }

  // 10. bytes_per_token = kv_per_tok_per_layer × numLayers
  const bytesPerToken = kvPerTokPerLayer * modelConfig.numLayers;

  // 构建 parallelCorrections（在除零保护前构建，以便早返回时也可使用）
  const parallelCorrections: ParallelMemoryCorrections = {
    tpWeightDivisor: tpSize,
    dpKvMultiplier,
    epWeightDivisor,
    ppWeightDivisor: ppSize,
    cpKvMultiplier: cpSize,
  };

  // 除零保护：bytesPerToken 为 0 时（如 numLayers=0）返回 numPages=0
  if (bytesPerToken === 0) {
    console.warn(
      `[calculateMemoryBudgetParallel] bytesPerToken=0, ` +
      `kvPerTokPerLayer=${kvPerTokPerLayer}, numLayers=${modelConfig.numLayers}`
    );
    return {
      numPages: 0,
      modelMemory: weightBytes,
      graphBuffer,
      parallelCorrections,
    };
  }

  // 11. num_tokens = floor(kv_budget_per_rank / bytes_per_token)
  const numTokens = Math.floor(kvBudgetPerRank / bytesPerToken);

  // 12. num_pages = max(0, floor(num_tokens / pageSize))
  const numPages = Math.max(0, Math.floor(numTokens / config.pageSize));

  // OOM 预测警告
  if (numPages < 1) {
    console.warn(
      `[calculateMemoryBudgetParallel] OOM: numPages=0, available=${available}, ` +
      `weightBytes=${weightBytes}, graphBuffer=${graphBuffer}, bytesPerToken=${bytesPerToken}`
    );
  }

  return {
    numPages,
    modelMemory: weightBytes,
    graphBuffer,
    parallelCorrections,
  };
}
