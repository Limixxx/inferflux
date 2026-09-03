// metrics — S6: SimulationMetrics 完整指标体系（§4.3 / §4.5）

import { ParallelMetrics } from "../parallel";
import type { SimRespMsg, SimScheduler } from "../types";
import type { Batch, ForwardOutput } from "../core";

/**
 * 仿真指标集合（§4.3 / §4.5）
 *
 * P0: 嵌入 ParallelMetrics 子结构，收集并行通信与负载指标。
 * P6: toJSON() 方法暴露并行指标汇总。
 * S6: 补全 §4.5 定义的全部指标字段和记录方法。
 */
export class SimulationMetrics {
  // ===== 已有：并行指标子结构 =====
  readonly parallel: ParallelMetrics = new ParallelMetrics();

  // ===== 吞吐量指标 =====
  totalRequests: number = 0;
  completedRequests: number = 0;
  totalTokensGenerated: number = 0;
  totalTicks: number = 0;

  // ===== 延迟指标 =====
  requestLatencies: number[] = [];
  prefillLatencies: number[] = [];
  decodeLatencies: number[] = [];

  // ===== 调度指标 =====
  prefillBatches: number = 0;
  decodeBatches: number = 0;
  avgPrefillBatchSize: number = 0.0;
  avgDecodeBatchSize: number = 0.0;
  chunkedPrefillCount: number = 0;

  // ===== Cache 指标 =====
  cacheHitRate: number = 0.0;
  cacheEvictionCount: number = 0;
  avgCacheUtilization: number = 0.0;

  // ===== 内存指标（严格对齐 §4.5） =====
  peakMemoryUsage: number = 0;
  oomCount: number = 0;

  // ===== GPU 利用率 =====
  gpuBusyTicks: number = 0;
  gpuIdleTicks: number = 0;
  gpuUtilization: number = 0.0;

  // ===== CUDA Graph 指标 =====
  cudaGraphReplayCount: number = 0;
  eagerForwardCount: number = 0;

  // ===== 内部计数器 =====
  private _prefillBatchTotal: number = 0;
  private _decodeBatchTotal: number = 0;

  /**
   * 记录一个 tick 的回复消息（对应 §9.11 SimulationMetrics.record_reply）
   * 遍历 replies，对每个消息递增 totalTokensGenerated，
   * 对 finished === true 的消息递增 completedRequests
   */
  recordReply(replies: SimRespMsg[], tick: number): void {
    for (const reply of replies) {
      if (reply.nextToken !== null) {
        this.totalTokensGenerated += 1;
      }
      if (reply.finished) {
        this.completedRequests += 1;
      }
    }
  }

  /**
   * 记录一次 batch forward（对应 §9.11 SimulationMetrics.record_batch）
   * 区分 prefill/decode batch，使用增量平均公式更新 avgBatchSize
   */
  recordBatch(batch: Batch, gpuTicks: number): void {
    const batchSize = batch.reqs.size;

    if (batch.extendInputTokens > 0) {
      // Prefill batch
      this.prefillBatches += 1;
      this._prefillBatchTotal += batchSize;
      this.avgPrefillBatchSize = this._prefillBatchTotal / this.prefillBatches;

      // 检查是否包含 ChunkedReq
      for (const req of batch.reqs.values()) {
        if ((req as any).constructor.name === "ChunkedReq") {
          this.chunkedPrefillCount += 1;
          break;
        }
      }
    }

    if (batch.numDecodeTokens > 0) {
      // Decode batch
      this.decodeBatches += 1;
      this._decodeBatchTotal += batchSize;
      this.avgDecodeBatchSize = this._decodeBatchTotal / this.decodeBatches;
    }

    // CUDA Graph / Eager 计数
    if (gpuTicks > 0) {
      // 无法从 batch 单独判断 graph replay，由调用方通过 recordCudaGraph/replay 显式记录
    }
  }

  /**
   * 记录一个 tick 的 GPU 使用情况（对应 §9.11 SimulationMetrics.record_tick）
   */
  recordTick(tick: number, gpuBusy: number = 0): void {
    this.totalTicks = Math.max(this.totalTicks, tick + 1);
    this.gpuBusyTicks += gpuBusy;
    this.gpuIdleTicks = Math.max(0, this.totalTicks - this.gpuBusyTicks);
    this.gpuUtilization = this.totalTicks > 0
      ? this.gpuBusyTicks / this.totalTicks
      : 0.0;
  }

  /**
   * 记录单个请求完成的延迟数据（TTFT/TBT/E2E）— [扩展项]
   * TTFT = firstTokenTick - arrivalTick
   * TBT = (finishTick - firstTokenTick) / max(1, decodeSteps)
   * E2E = finishTick - arrivalTick
   */
  recordRequestLatency(
    uid: number,
    arrivalTick: number,
    firstTokenTick: number,
    finishTick: number,
    decodeSteps: number,
  ): void {
    const ttft = firstTokenTick - arrivalTick;
    const tbt = (finishTick - firstTokenTick) / Math.max(1, decodeSteps);
    const e2e = finishTick - arrivalTick;

    this.prefillLatencies.push(ttft);
    this.decodeLatencies.push(tbt);
    this.requestLatencies.push(e2e);
  }

  /**
   * 记录 cache 指标快照（由 CacheManager 回调）— [扩展项]
   * 同时更新 peakMemoryUsage
   */
  recordCacheSnapshot(
    hitRate: number,
    evictionCount: number,
    utilization: number,
  ): void {
    this.cacheHitRate = hitRate;
    this.cacheEvictionCount = evictionCount;
    this.avgCacheUtilization = utilization;
  }

  /** 记录 CUDA Graph replay */
  recordCudaGraphReplay(): void {
    this.cudaGraphReplayCount += 1;
  }

  /** 记录 Eager forward */
  recordEagerForward(): void {
    this.eagerForwardCount += 1;
  }

  /** 时钟 tick 回调（供 SimulationClock.onTick 注册） */
  tick(currentTicks: number): void {
    // 占位：供 SimulationClock.onTick 注册；未来可用于周期性聚合指标
  }

  /** 重置所有指标到默认值 */
  reset(): void {
    this.parallel.reset();

    this.totalRequests = 0;
    this.completedRequests = 0;
    this.totalTokensGenerated = 0;
    this.totalTicks = 0;

    this.requestLatencies = [];
    this.prefillLatencies = [];
    this.decodeLatencies = [];

    this.prefillBatches = 0;
    this.decodeBatches = 0;
    this.avgPrefillBatchSize = 0.0;
    this.avgDecodeBatchSize = 0.0;
    this.chunkedPrefillCount = 0;

    this.cacheHitRate = 0.0;
    this.cacheEvictionCount = 0;
    this.avgCacheUtilization = 0.0;

    this.peakMemoryUsage = 0;
    this.oomCount = 0;

    this.gpuBusyTicks = 0;
    this.gpuIdleTicks = 0;
    this.gpuUtilization = 0.0;

    this.cudaGraphReplayCount = 0;
    this.eagerForwardCount = 0;

    this._prefillBatchTotal = 0;
    this._decodeBatchTotal = 0;
  }

  /** 序列化为 JSON 可序列化对象（包含所有 §4.5 指标字段 + parallel） */
  toJSON(): Record<string, unknown> {
    return {
      // 吞吐量
      totalRequests: this.totalRequests,
      completedRequests: this.completedRequests,
      totalTokensGenerated: this.totalTokensGenerated,
      totalTicks: this.totalTicks,
      // 延迟
      requestLatencies: this.requestLatencies,
      prefillLatencies: this.prefillLatencies,
      decodeLatencies: this.decodeLatencies,
      // 调度
      prefillBatches: this.prefillBatches,
      decodeBatches: this.decodeBatches,
      avgPrefillBatchSize: this.avgPrefillBatchSize,
      avgDecodeBatchSize: this.avgDecodeBatchSize,
      chunkedPrefillCount: this.chunkedPrefillCount,
      // Cache
      cacheHitRate: this.cacheHitRate,
      cacheEvictionCount: this.cacheEvictionCount,
      avgCacheUtilization: this.avgCacheUtilization,
      // 内存
      peakMemoryUsage: this.peakMemoryUsage,
      oomCount: this.oomCount,
      // GPU
      gpuBusyTicks: this.gpuBusyTicks,
      gpuIdleTicks: this.gpuIdleTicks,
      gpuUtilization: this.gpuUtilization,
      // CUDA Graph
      cudaGraphReplayCount: this.cudaGraphReplayCount,
      eagerForwardCount: this.eagerForwardCount,
      // 并行
      parallel: this.parallel.summary(),
    };
  }
}
