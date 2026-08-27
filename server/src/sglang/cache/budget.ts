// budget — K5: 内存预算基础公式 (§3.3.9)

import type { ModelConfig, SimulatorConfig } from "../types";
import { divEven } from "../core";

/** calculateMemoryBudget 返回值结构 */
export interface MemoryBudgetResult {
  /** 可分配的 KV cache 页数（≥0，0 表示 OOM） */
  numPages: number;
  /** 模型权重占用的显存（bytes） */
  modelMemory: number;
  /** CUDA Graph buffer 占用的显存（bytes） */
  graphBuffer: number;
}

/**
 * 估算模型权重占用的显存（bytes）
 * 对应 §3.3.9 estimate_model_memory
 *
 * 粗略估算：每层参数量 ≈ hidden² × 12（QKV + FFN + embed）
 * 公式：numLayers × hiddenSize × hiddenSize × 12 × dtypeSize
 */
export function estimateModelMemory(
  modelConfig: ModelConfig,
  dtypeSize: number,
): number {
  return modelConfig.numLayers * modelConfig.hiddenSize * modelConfig.hiddenSize * 12 * dtypeSize;
}

/**
 * 估算 CUDA Graph buffer 占用的显存（bytes）
 * 对应 §3.3.9 estimate_graph_buffer
 *
 * 每层 buffer ≈ max_bs × hidden × 4（中间激活、logits 等）
 * 当 cudaGraphBs 为空或 null 时返回 0
 */
export function estimateGraphBuffer(
  cudaGraphBs: number[] | null,
  modelConfig: ModelConfig,
): number {
  if (!cudaGraphBs || cudaGraphBs.length === 0) {
    return 0;
  }
  const maxBs = Math.max(...cudaGraphBs);
  return maxBs * modelConfig.hiddenSize * modelConfig.numLayers * 4;
}

/**
 * 计算可分配的 KV cache 页数
 * 对应 §3.3.9 calculate_memory_budget / mem_fraction_static
 *
 * @param config - SimulatorConfig（含 memoryRatio, pageSize, dtypeSize, tpSize, cudaGraphBs）
 * @param modelConfig - ModelConfig（含 numLayers, hiddenSize, numKvHeads, headDim）
 * @param totalMemory - GPU 总显存（bytes）
 * @returns MemoryBudgetResult
 */
export function calculateMemoryBudget(
  config: SimulatorConfig,
  modelConfig: ModelConfig,
  totalMemory: number,
): MemoryBudgetResult {
  // 模型权重占用
  const modelMemory = estimateModelMemory(modelConfig, config.dtypeSize);
  // CUDA Graph buffer 占用
  const graphBuffer = estimateGraphBuffer(config.cudaGraphBs, modelConfig);

  // 剩余可用 = 比例预算 - 模型权重 - graph buffer
  const available = Math.floor(config.memoryRatio * totalMemory) - modelMemory - graphBuffer;

  // KV cache 每页大小
  // divEven 返回每 GPU 的 KV head 分布列表，sum 后得到总 KV head 数
  const kvHeadsPerGpu = divEven(modelConfig.numKvHeads, config.tpSize, true)
    .reduce((sum, v) => sum + v, 0);
  const cachePerPage =
    2 *                    // key + value
    modelConfig.headDim *
    kvHeadsPerGpu *
    config.pageSize *
    config.dtypeSize *     // float16=2, bfloat16=2, float8=1
    modelConfig.numLayers;

  // 除零保护：cachePerPage 为 0 时（如 numLayers=0）返回 numPages=0
  if (cachePerPage === 0) {
    console.warn(
      `[calculateMemoryBudget] cachePerPage=0, ` +
      `kvHeadsPerGpu=${kvHeadsPerGpu}, numLayers=${modelConfig.numLayers}`
    );
    return { numPages: 0, modelMemory, graphBuffer };
  }

  // 可分配的页数（OOM 保护：负数时返回 0，由调用方触发 OOM 处理）
  const numPages = Math.max(0, Math.floor(available / cachePerPage));

  // OOM 预测警告
  if (numPages < 1) {
    console.warn(
      `[calculateMemoryBudget] OOM: numPages=0, available=${available}, ` +
      `modelMemory=${modelMemory}, graphBuffer=${graphBuffer}, cachePerPage=${cachePerPage}`
    );
  }

  return { numPages, modelMemory, graphBuffer };
}
