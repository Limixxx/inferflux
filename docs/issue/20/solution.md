---
title: "Issue #20 解决方案"
issue_number: 20
issue_type: Feature
created: 2026-09-02
updated: 2026-09-03
status: revised
review_round: 2
---

# Issue #20 解决方案

## 需求分析

- **问题描述**：Issue #20 要求实现 S6 阶段的三大组件：`WorkloadGenerator`、`SimulationMetrics` 完整指标体系、`SGHttpApi`（仿真 HTTP API），以及顶层导出函数 `createSimulator(config): SgSimInstance`。

- **能力目标**：
  1. **WorkloadGenerator（§4.3 / §9.10）**：Poisson/固定速率请求生成；生成 `SimRequestMsg`（含 uid/inputIds/samplingParams/outputLen）；支持 trace 回放（从预定义序列重放请求）
  2. **SimulationMetrics 完整指标（§4.3 / §4.5）**：
     - 延迟指标：`request_latencies` / `prefill_latencies` / `decode_latencies`
     - 吞吐指标：`total_requests` / `completed_requests` / `total_tokens_generated`
     - 调度指标：`prefill_batches` / `decode_batches` / `avg_prefill_batch_size` / `avg_decode_batch_size` / `chunked_prefill_count`
     - Cache 指标：`cache_hit_rate` / `cache_eviction_count` / `avg_cache_utilization`
     - 内存指标：`peak_memory_usage` / `oom_count`
     - GPU 指标：`gpu_busy_ticks` / `gpu_idle_ticks` / `gpu_utilization`
     - CUDA Graph 指标：`cuda_graph_replay_count` / `eager_forward_count`
     - 方法：`record(stepResult)`、`tick(currentTicks)`、`toJSON()`
     - 已有 `parallel: ParallelMetrics` 子结构保持不变
  3. **SGHttpApi（§4.3 / §2.3）**：
     - `POST /v1/chat/completions`：OpenAI 兼容请求，转成 `SimRequestMsg` 发送到 SchedulerIOMixin
     - `GET /v1/internal/metrics`：返回 `SimulationMetrics.toJSON()` + Snapshot
     - `GET /v1/internal/state`：返回 Scheduler 状态快照
  4. **导出函数 `createSimulator(config): SgSimInstance`**：`start()` / `enqueue(req)` / `getMetrics()` / `shutdown()`

- **影响范围**：修改 `server/src/sglang/workload/index.ts`（WorkloadGenerator 实现）、`server/src/sglang/metrics/index.ts`（SimulationMetrics 完整指标升级）、`server/src/sglang/api/index.ts`（SGHttpApi 实现）、`server/src/http/HttpService.ts`（新增路由）、`server/src/sglang/Simulator.ts`（升级为 SgSimInstance + createSimulator）、`server/src/sglang/index.ts`（新增导出）、新增测试文件 `server/src/test/sglang-s6.test.ts`。不修改已有测试代码。

- **依赖 Issue**：
  - #19 S5: Overlap Scheduling + SimulationClock（已完成 — SimulationClock 提供 tick 回调用于 metrics.tick）
  - #12 K5: 内存预算（Metrics memory 字段依赖 CacheManager/CacheSizeInfo 数据）

- **阻塞 Issue**：
  - P0: SimCommGroup + ParallelTopology + ParallelMetrics（已完成 — ParallelMetrics 已嵌入 SimulationMetrics）

## 改造方案

### 总体思路

在现有骨架代码基础上实现 S6 三大组件。当前代码状态：
- `workload/index.ts`：仅一行注释桩
- `metrics/index.ts`：已有 `SimulationMetrics` 类，包含 `parallel: ParallelMetrics` 子结构、`reset()` 和 `toJSON()`，但缺少 §4.5 定义的完整指标字段和记录方法
- `api/index.ts`：仅一行注释桩
- `HttpService.ts`：已有 `/api/internal/metrics` 路由，但缺少 `/v1/chat/completions` 和 `/v1/internal/state` 路由
- `Simulator.ts`：已有 `Simulator` 类桩，需升级为 `SgSimInstance` 并添加 `createSimulator` 工厂函数

核心变更分四部分：

1. **WorkloadGenerator**：在 `workload/index.ts` 中实现请求生成器，支持 Poisson/固定/trace 三种到达分布和 uniform/normal 两种长度分布
2. **SimulationMetrics 升级**：在 `metrics/index.ts` 中补全 §4.5 定义的全部指标字段和记录方法（`recordReply`、`recordBatch`、`recordTick`），升级 `toJSON()` 输出全部指标
3. **SGHttpApi + HttpService 路由扩展**：在 `api/index.ts` 实现 OpenAI 兼容的 chat completions 端点和内部状态端点；在 `HttpService.ts` 新增路由
4. **SgSimInstance + createSimulator**：在 `Simulator.ts` 升级为完整实例类，在 `index.ts` 导出工厂函数

### 详细设计

#### 1. WorkloadGenerator（`server/src/sglang/workload/index.ts`）

**配置类型**：

```typescript
/** 工作负载生成器配置（对应 §4.4 WorkloadConfig / §9.10） */
export interface WorkloadConfig {
  /** 生成请求数量 */
  numRequests: number;
  /** 输入长度分布类型 */
  inputLenDistribution: "uniform" | "normal";
  /** 输入长度最小值 */
  inputLenMin: number;
  /** 输入长度最大值 */
  inputLenMax: number;
  /** 输入长度均值（normal 分布用） */
  inputLenMean?: number;
  /** 输入长度标准差（normal 分布用） */
  inputLenStd?: number;
  /** 输出长度分布类型 */
  outputLenDistribution: "uniform" | "normal";
  /** 输出长度最小值 */
  outputLenMin: number;
  /** 输出长度最大值 */
  outputLenMax: number;
  /** 输出长度均值（normal 分布用） */
  outputLenMean?: number;
  /** 输出长度标准差（normal 分布用） */
  outputLenStd?: number;
  /** 请求到达速率（每 tick 请求数） */
  arrivalRate: number;
  /** 到达分布类型 */
  arrivalDistribution: "poisson" | "uniform" | "trace";
  /** 共享前缀比例 */
  sharedPrefixRatio: number;
  /** 共享前缀长度 */
  sharedPrefixLen: number;
  /** Trace 数据（trace 模式用） */
  trace?: SimRequestMsg[];
}

export const DEFAULT_WORKLOAD_CONFIG: WorkloadConfig = {
  numRequests: 100,
  inputLenDistribution: "uniform",
  inputLenMin: 128,
  inputLenMax: 1024,
  outputLenDistribution: "uniform",
  outputLenMin: 100,
  outputLenMax: 1024,
  arrivalRate: 10.0,
  arrivalDistribution: "poisson",
  sharedPrefixRatio: 0.3,
  sharedPrefixLen: 100,
};
```

**WorkloadGenerator 类**：

```typescript
export class WorkloadGenerator {
  private _rng: () => number;  // 可注入随机数生成器

  constructor(rng?: () => number) {
    this._rng = rng ?? Math.random;
  }

  /** 生成模拟请求序列 */
  generate(config: WorkloadConfig): SimRequestMsg[] {
    if (config.arrivalDistribution === "trace" && config.trace) {
      return config.trace;
    }
    const requests: SimRequestMsg[] = [];
    for (let i = 0; i < config.numRequests; i++) {
      const inputLen = this._sampleLen(
        config.inputLenDistribution,
        config.inputLenMin, config.inputLenMax,
        config.inputLenMean ?? 0, config.inputLenStd ?? 0,
      );
      const outputLen = this._sampleLen(
        config.outputLenDistribution,
        config.outputLenMin, config.outputLenMax,
        config.outputLenMean ?? 0, config.outputLenStd ?? 0,
      );
      const arrivalTick = this._sampleArrival(config, i);
      const inputIds = this._generateTokens(
        inputLen, config.sharedPrefixRatio, config.sharedPrefixLen, i,
      );
      requests.push({
        tag: "req_in",
        uid: i,
        inputIds,
        samplingParams: { maxTokens: outputLen },
        outputLen,
      });
    }
    // 按 arrivalTick 排序
    requests.sort((a, b) => (a as any).arrivalTick - (b as any).arrivalTick);
    return requests;
  }

  private _sampleLen(distribution: string, minVal: number, maxVal: number,
                     mean: number, std: number): number { ... }
  private _sampleArrival(config: WorkloadConfig, index: number): number { ... }
  private _generateTokens(length: number, ratio: number, prefixLen: number,
                          uid: number): number[] { ... }
}
```

**设计要点**：
- `_sampleArrival` Poisson 模式使用 `index / arrivalRate`（对应 §9.10 伪代码），不依赖真实 Poisson 过程库，与报告伪代码保持一致
- `_generateTokens` 中 `sharedPrefixRatio > 0 && uid % 3 === 0` 的共享前缀策略与 §9.10 伪代码精确对齐
- 返回的 `SimRequestMsg[]` 附带 `arrivalTick` 元数据（通过扩展属性或独立的 `SimRequest` 类型），供 Simulator 的 tick 循环按时间注入请求

**▶ 回应评审偏离 #1：`samplingParams` 设为 null → 改为对齐报告规格**

上一轮方案将 `samplingParams` 设为 `null`，偏离了 §4.4 L1333 中 `sampling_params=SamplingParams(max_tokens=output_len)` 的明确规格。本轮修正为**严格对齐报告**：`generate()` 方法直接构造 `SamplingParams({ maxTokens: outputLen })`，与 §9.10 伪代码一致。

偏离的理由不再成立——原先认为"由 Simulator 层赋值默认 SamplingParams"更灵活，但这引入了不必要的间接层：
- 如果调用方需要覆盖 `samplingParams`，可以在获得 `SimRequestMsg` 后自行修改
- Simulator 的 `enqueue()` 方法中已有 `max_tokens` 裁剪逻辑（§9.11 L3041-3046），会确保 `max_tokens ≤ max_seq_len - input_len`

因此，`samplingParams` 在 `generate()` 中直接赋值，与报告保持一致，无需额外论证。

#### 2. SimulationMetrics 升级（`server/src/sglang/metrics/index.ts`）

**完整指标字段**（严格对齐 §4.5）：

```typescript
export class SimulationMetrics {
  // ===== 已有：并行指标子结构 =====
  readonly parallel: ParallelMetrics = new ParallelMetrics();

  // ===== 吞吐量指标 =====
  totalRequests: number = 0;
  completedRequests: number = 0;
  totalTokensGenerated: number = 0;
  totalTicks: number = 0;

  // ===== 延迟指标 =====
  requestLatencies: number[] = [];      // per-request ticks
  prefillLatencies: number[] = [];      // TTFT (ticks)
  decodeLatencies: number[] = [];       // TBT (ticks)

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

  // ===== 方法 =====

  /** 记录一个 tick 的回复消息（对应 §9.11 SimulationMetrics.record_reply） */
  recordReply(replies: SimRespMsg[], tick: number): void;

  /** 记录一次 batch forward（对应 §9.11 SimulationMetrics.record_batch） */
  recordBatch(batch: Batch, gpuTicks: number): void;

  /** 记录一个 tick 的 GPU 使用情况（对应 §9.11 SimulationMetrics.record_tick） */
  recordTick(tick: number, gpuBusy: number = 0): void;

  /** 记录单个请求完成的延迟数据（TTFT/TBT/E2E）— [扩展项] */
  recordRequestLatency(uid: number, arrivalTick: number, firstTokenTick: number,
                       finishTick: number, decodeSteps: number): void;

  /** 记录 cache 指标快照（由 CacheManager 回调）— [扩展项] */
  recordCacheSnapshot(hitRate: number, evictionCount: number,
                      utilization: number): void;

  /** 时钟 tick 回调（供 SimulationClock.onTick 注册） */
  tick(currentTicks: number): void;

  /** 重置所有指标到默认值 */
  reset(): void;

  /** 序列化为 JSON 可序列化对象 */
  toJSON(): Record<string, unknown>;
}
```

**▶ 回应评审偏离 #2：`pagesAllocated` / `pagesFree` 字段**

上一轮方案新增了 `pagesAllocated` / `pagesFree` 两个字段，超出 §4.5 规格。本轮**移除这两个字段**，严格对齐 §4.5 的内存指标（仅 `peak_memory_usage` + `oom_count`）。

内存页面分配信息已经通过 `CacheSizeInfo` 对象暴露在 `GET /v1/internal/state` 端点的调度器状态快照中，无需在 `SimulationMetrics` 中重复存储。如果未来需要页面级别的指标趋势，可通过 `recordCacheSnapshot` 扩展方法实现（当前标记为扩展项，不影响 §4.5 对齐）。

**▶ 回应评审偏离 #3：`recordRequestLatency` / `recordCacheSnapshot` / `recordMemorySnapshot` — 标注为扩展项**

- `recordRequestLatency`：§4.5 定义了 `request_latencies` / `prefill_latencies` / `decode_latencies` 字段但未定义填充方法。此方法填补了报告的空白，TTFT/TBT/E2E 计算逻辑合理（TTFT = firstTokenTick - arrivalTick, TBT = (finishTick - firstTokenTick) / max(1, decodeSteps), E2E = finishTick - arrivalTick）。标记为 **[扩展项]**，与 §4.5 字段定义互补。
- `recordCacheSnapshot`：§4.5 定义了 Cache 指标字段但未定义数据来源。Cache 指标需要从 CacheManager 定期回调获取，此方法定义了集成接口。标记为 **[扩展项]**。
- `recordMemorySnapshot`：上一轮方案中的此方法已无需保留（`pagesAllocated`/`pagesFree` 字段已移除）。`peakMemoryUsage` 的更新改为在 `recordCacheSnapshot` 中一并处理（`peakMemoryUsage = max(peakMemoryUsage, allocatedPages * pageSize * dtypeSize)`）。

**方法实现要点**：

- `recordReply`：遍历 replies，对每个消息递增 `totalTokensGenerated`，对 `finished === true` 的消息递增 `completedRequests`（与 §9.11 L3751-3755 一致）
- `recordBatch`：区分 prefill/decode batch，使用增量平均公式更新 `avgPrefillBatchSize`/`avgDecodeBatchSize`（与 §9.11 L3757-3771 一致）。GPU busy ticks 不在此累加，避免与 `recordTick` 双重计数
- `recordTick`：`totalTicks = max(totalTicks, tick + 1)`，`gpuBusyTicks += gpuBusy`，`gpuIdleTicks = max(0, totalTicks - gpuBusyTicks)`，`gpuUtilization = gpuBusyTicks / max(1, totalTicks)`（与 §9.11 L3773-3777 一致）
- `recordRequestLatency`：记录 TTFT = `firstTokenTick - arrivalTick`，TBT = `(finishTick - firstTokenTick) / max(1, decodeSteps)`，E2E = `finishTick - arrivalTick`。分别追加到 `prefillLatencies`、`decodeLatencies`、`requestLatencies`
- `tick`：空操作占位，供 SimulationClock.onTick 注册；未来可用于周期性聚合指标
- `toJSON`：返回包含所有 §4.5 字段 + `parallel.summary()` 的完整 JSON 对象

#### 3. SGHttpApi（`server/src/sglang/api/index.ts`）

**▶ 回应评审偏离 #5：SGHttpApi 独立端口架构论证**

上一轮方案设计 SGHttpApi 独立端口（8000），与现有 SimService（3001）和 HttpService（8888）并存，评审要求论证三端口架构的必要性。

**修正方案：取消独立端口，SGHttpApi 复用 HttpService 端口（8888）**

理由：
1. **HttpService 已有代理能力**：HttpService 已实现 `proxyToSim()` 方法，可将请求代理到 SimService（3001）。`/v1/chat/completions` 端点天然适合走代理路径，因为该请求最终需要注入 Scheduler——而 SimService 已经持有 Scheduler 引用。
2. **避免端口膨胀**：三端口架构增加了部署复杂度和网络配置成本。仿真器为单进程工具，端口数量应最小化。
3. **兼容性**：保留 `/api/internal/metrics` 的现有逻辑不变；新增 `/v1/*` 路由在 HttpService 中直接处理或代理到 SimService。

**修正后的架构**：

```
客户端 → HttpService(:8888)
           ├── /v1/chat/completions   → 代理到 SimService(:3001)
           ├── /v1/internal/metrics   → 直接读取 SimulationMetrics（同 /api/internal/metrics）
           ├── /v1/internal/state     → 代理到 SimService(:3001)
           ├── /api/internal/metrics  → 直接读取 SimulationMetrics（已有）
           └── /api/*                 → 代理到 SimService(:3001)（已有）
```

**SGHttpApi 类重构为无端口的消息处理器**：

```typescript
import type { SimScheduler } from "../scheduler";
import type { SimulationMetrics } from "../metrics";

/**
 * SGHttpApi — SGLang 仿真 HTTP API 消息处理器（§4.3）
 *
 * 不再独立监听端口，而是作为 HttpService 和 SimService 的路由处理器。
 * 在 HttpService 中注册 /v1/* 路由时，委托此类处理请求逻辑。
 */
export class SGHttpApi {
  private scheduler: SimScheduler | null = null;
  private metrics: SimulationMetrics | null = null;

  /** 注入调度器和指标实例 */
  bind(scheduler: SimScheduler, metrics: SimulationMetrics): void;

  /** 处理 POST /v1/chat/completions 请求体，返回 OpenAI 格式占位响应 */
  handleChatCompletions(body: ChatCompletionRequest): ChatCompletionResponse;

  /** 处理 GET /v1/internal/metrics，返回 metrics.toJSON() + 调度器快照 */
  handleInternalMetrics(): Record<string, unknown>;

  /** 处理 GET /v1/internal/state，返回调度器状态 */
  handleInternalState(): Record<string, unknown>;
}
```

**`POST /v1/chat/completions` 实现**：

- 解析请求体为 OpenAI ChatCompletionRequest 格式
- 提取 `messages` → 拼接为文本 → 生成 `inputIds`（简单 token 计数：`text.length` 个 token，值为 0..N-1）
- 构造 `UserMsg { uid, inputIds, samplingParams: { maxTokens: max_tokens ?? 128 } }`
- 通过 `scheduler._incomingQueue.push()` 注入（非 offline 模式）或通过 `SgSimInstance.enqueue()` 注入
- 立即返回 `{ id, object: "chat.completion", ... }` 占位响应（仿真为异步执行，不等待完成）

**`GET /v1/internal/metrics` 实现**：

- 返回 `metrics.toJSON()` + 调度器当前快照（running 请求数、pending 请求数等）

**`GET /v1/internal/state` 实现**：

- 返回调度器状态：`{ runningReqs, pendingReqs, availableTableIndices, cacheSizeInfo, tick }`

#### 4. HttpService 路由扩展（`server/src/http/HttpService.ts`）

在现有 `handleRequest` 方法中新增路由：

```typescript
// 注入 SGHttpApi 实例
private _sgHttpApi: SGHttpApi | null = null;

setSGHttpApi(api: SGHttpApi): void {
  this._sgHttpApi = api;
}

// 在 handleRequest 中，/api/* 路由分支之前新增：
// POST /v1/chat/completions — 代理到 SimService
if (req.method === "POST" && urlPath === "/v1/chat/completions") {
  this.proxyToSim("/v1/chat/completions", req, res);
  return;
}

// GET /v1/internal/metrics — 直接读取 metrics（同 /api/internal/metrics 逻辑）
if (req.method === "GET" && urlPath === "/v1/internal/metrics") {
  if (this._sgHttpApi) {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(this._sgHttpApi.handleInternalMetrics()));
  } else {
    res.writeHead(503, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "SGHttpApi not available" }));
  }
  return;
}

// GET /v1/internal/state — 代理到 SimService
if (req.method === "GET" && urlPath === "/v1/internal/state") {
  this.proxyToSim("/v1/internal/state", req, res);
  return;
}
```

> **设计说明**：保留已有 `/api/internal/metrics` 路由兼容性，新增 `/v1/*` 前缀路由对齐 §4.3 规格。`/v1/chat/completions` 和 `/v1/internal/state` 代理到 SimService 端口；`/v1/internal/metrics` 直接读取内存中的 metrics 实例（与 `/api/internal/metrics` 等效，减少网络往返）。

#### 5. SgSimInstance + createSimulator（`server/src/sglang/Simulator.ts`）

**▶ 回应评审偏离 #6：`createSimulator` / `SgSimInstance` 架构论证 + tick 驱动方式明确**

上一轮评审指出：(a) `SgSimInstance` 超出报告规格但属于 TypeScript 适配的合理设计；(b) `start()` 的 `setInterval` tick 驱动方式需明确 tick 间隔及与 `scheduler.runTick` 的关系。

**修正方案：明确 tick 驱动模型**

`createSimulator` 支持两种运行模式：

1. **在线模式（offlineMode = false）**：`start()` 启动 `setInterval` 循环，每个 interval 调用一次 `scheduler.runTick(incoming)`。
   - **tick 间隔**：由 `SimulatorConfig.tickIntervalMs` 决定（新增字段，默认 `10`ms）。间隔不影响仿真逻辑正确性（仿真时间是离散 tick 计数），仅控制仿真推进速度。
   - **与 scheduler.runTick 的关系**：每次 interval 触发时，`start()` 收集当前 tick 应到达的 `incoming` 消息，调用 `scheduler.runTick(incoming)` 获取响应，然后调用 `metrics.recordReply()` 和 `metrics.recordTick()`。
   - interval 仅驱动 tick 推进节奏，仿真逻辑完全由 `runTick` 决定。

2. **离线模式（offlineMode = true）**：`start()` 不启动 interval，而是预加载全部工作负载请求后一次性运行所有 tick 直到完成。调用方通过 `await sim.start()` 等待仿真完成。这与 §9.10 测试示例中的 `for tick in range(1000)` 模式一致。

**在 SimulatorConfig 中新增字段**：

```typescript
/** tick 间隔（毫秒），仅在线模式生效，默认 10ms */
tickIntervalMs: number;
// DEFAULT: tickIntervalMs: 10
```

```typescript
/** SgSimInstance — 仿真器实例接口（§4.3 导出） */
export interface SgSimInstance {
  readonly ctx: SgSimContext;
  readonly scheduler: SimScheduler;
  readonly metrics: SimulationMetrics;
  readonly workload: WorkloadGenerator;

  /** 启动仿真循环（在线模式启动 interval；离线模式一次性运行） */
  start(): void;

  /** 入队一个请求（在线模式下推入 _incomingQueue，下一 tick 被 runTick 消费） */
  enqueue(msg: SimRequestMsg): void;

  /** 获取当前指标快照 */
  getMetrics(): Record<string, unknown>;

  /** 关闭仿真器（清除 interval，停止循环） */
  shutdown(): void;
}

/** 创建仿真器实例（§4.3 导出函数） */
export function createSimulator(config: SimulatorConfig): SgSimInstance;
```

**实现要点**：
- `createSimulator` 内部创建 `SgSimContext`、`SimScheduler`、`SimulationMetrics`、`WorkloadGenerator`
- `start()` 在线模式：启动 `setInterval`，每 tick 调用 `scheduler.runTick(incoming)` + `metrics.recordReply/tick`
- `start()` 离线模式：预加载 workload 后循环 `for (let t = 0; t < maxTicks; t++)` 调用 `scheduler.runTick`
- `enqueue(msg)` 将消息推入内部 `_incomingQueue`
- `getMetrics()` 返回 `metrics.toJSON()`
- `shutdown()` 清除 interval，停止仿真循环

#### 6. SimService 集成（`server/src/sim/SimService.ts`）

在 SimService 中集成 SGHttpApi：
- 在 `SimService` 构造器或 `start()` 中，创建 `SGHttpApi` 实例
- 将 `SGHttpApi` 绑定到 `SimScheduler` 和 `SimulationMetrics`
- 注册 `/v1/chat/completions` 和 `/v1/internal/state` 路由处理器，委托 `SGHttpApi` 处理
- 将 `SGHttpApi` 实例注入 HttpService（通过 `setSGHttpApi()`）

> **设计决策**：SGHttpApi 不再独立监听端口，而是嵌入 SimService 处理请求逻辑。HttpService 的 `/v1/*` 路由通过代理或直接委托到 SGHttpApi。这样保持现有双端口架构（8888+3001），仅增量添加路由。

#### 7. 导出更新（`server/src/sglang/index.ts`）

新增导出：

```typescript
// S6: WorkloadGenerator
export { WorkloadGenerator, WorkloadConfig, DEFAULT_WORKLOAD_CONFIG } from "./workload";

// S6: SgSimInstance + createSimulator
export { SgSimInstance, createSimulator } from "./Simulator";

// S6: SGHttpApi
export { SGHttpApi } from "./api";
```

### 修改点清单

1. **`server/src/sglang/workload/index.ts`** — 实现 `WorkloadConfig` 接口 + `DEFAULT_WORKLOAD_CONFIG` + `WorkloadGenerator` 类（含 `_sampleLen`、`_sampleArrival`、`_generateTokens` 方法），`samplingParams` 对齐 §4.4 直接赋值
2. **`server/src/sglang/metrics/index.ts`** — 升级 `SimulationMetrics`：补全 §4.5 全部指标字段（移除 `pagesAllocated`/`pagesFree`），实现 `recordReply`、`recordBatch`、`recordTick`、`recordRequestLatency`（[扩展项]）、`recordCacheSnapshot`（[扩展项]）、`tick`、升级 `toJSON`
3. **`server/src/sglang/api/index.ts`** — 实现 `SGHttpApi` 类（重构为无端口消息处理器：`handleChatCompletions`、`handleInternalMetrics`、`handleInternalState`）
4. **`server/src/sglang/Simulator.ts`** — 升级 `Simulator` 为 `SgSimInstance` 接口实现 + 新增 `createSimulator` 工厂函数，明确在线/离线模式的 tick 驱动方式
5. **`server/src/sglang/types.ts`** — SimulatorConfig 新增 `tickIntervalMs` 字段
6. **`server/src/http/HttpService.ts`** — 新增 `/v1/chat/completions`、`/v1/internal/metrics`、`/v1/internal/state` 路由，新增 `setSGHttpApi()` 方法
7. **`server/src/sglang/index.ts`** — 新增 re-export：`WorkloadGenerator`、`WorkloadConfig`、`DEFAULT_WORKLOAD_CONFIG`、`SgSimInstance`、`createSimulator`、`SGHttpApi`
8. **`server/src/test/sglang-s6.test.ts`**（新增）— S6 阶段测试

## 评审偏离回应总结

| # | 偏离项 | 上轮处理 | 本轮修正 |
|---|--------|---------|---------|
| 1 | `samplingParams` 设为 null | 偏离 §4.4 | **修正**：改为对齐 §4.4 直接赋值 `SamplingParams({ maxTokens: outputLen })` |
| 2 | `pagesAllocated` / `pagesFree` 字段 | 超出 §4.5 | **修正**：移除，内存页面信息通过 `/v1/internal/state` 端点暴露 |
| 3 | `recordRequestLatency` 方法 | 必要扩展 | **保留**，标注为 [扩展项]，填补 §4.5 字段定义的填充方法空白 |
| 4 | `recordCacheSnapshot` / `recordMemorySnapshot` | 超出 §9.11 | **保留** `recordCacheSnapshot`（合并内存快照），标注为 [扩展项]；移除 `recordMemorySnapshot`（因 `pagesAllocated`/`pagesFree` 已移除） |
| 5 | SGHttpApi 独立端口 | 需论证 | **修正**：取消独立端口，SGHttpApi 重构为无端口消息处理器，复用 HttpService(:8888) + SimService(:3001) 双端口架构 |
| 6 | `createSimulator` / `SgSimInstance` | 需明确 tick 驱动 | **修正**：在 SimulatorConfig 新增 `tickIntervalMs` 字段；明确在线模式（setInterval）和离线模式（同步循环）两种驱动方式及其与 `scheduler.runTick` 的关系 |

## 测试设计

### 验收测试用例清单

| # | 测试名称 | 验证内容 |
|---|----------|----------|
| 1 | `test_workload_generator_poisson` | Poisson 到达分布生成正确数量的请求，arrivalTick 单调递增 |
| 2 | `test_workload_generator_uniform` | 均匀分布到达的请求 arrivalTick 均匀分布 |
| 3 | `test_workload_generator_trace_replay` | trace 模式直接返回预定义序列 |
| 4 | `test_workload_generator_shared_prefix` | sharedPrefixRatio=0.3 时约 1/3 请求共享前缀 |
| 5 | `test_workload_generator_normal_distribution` | normal 分布下长度值在 [min, max] 范围内 |
| 6 | `test_workload_generator_sampling_params` | 生成的请求 samplingParams.maxTokens 等于 outputLen（对齐 §4.4） |
| 7 | `test_simulation_metrics_record_reply` | recordReply 正确递增 completedRequests 和 totalTokensGenerated |
| 8 | `test_simulation_metrics_record_batch` | recordBatch 正确更新 avgPrefillBatchSize/avgDecodeBatchSize |
| 9 | `test_simulation_metrics_record_tick` | recordTick 正确计算 gpuUtilization、gpuIdleTicks |
| 10 | `test_simulation_metrics_record_request_latency` | recordRequestLatency 正确记录 TTFT/TBT/E2E |
| 11 | `test_simulation_metrics_record_cache_snapshot` | recordCacheSnapshot 正确更新 cache 指标和 peakMemoryUsage |
| 12 | `test_simulation_metrics_to_json` | toJSON 返回包含所有 §4.5 指标字段 + parallel 的完整对象，无 pagesAllocated/pagesFree |
| 13 | `test_simulation_metrics_reset` | reset 清零所有指标字段和 parallel 子结构 |
| 14 | `test_simulation_metrics_tick_clock_integration` | SimulationMetrics.tick() 通过 SimulationClock.onTick 注册后，每 tick 自动触发 recordTick |
| 15 | `test_sg_http_api_chat_completions` | handleChatCompletions 接受请求并注入调度器 |
| 16 | `test_sg_http_api_internal_metrics` | handleInternalMetrics 返回 metrics.toJSON() + 调度器快照 |
| 17 | `test_sg_http_api_internal_state` | handleInternalState 返回调度器状态快照 |
| 18 | `test_create_simulator_online` | createSimulator 在线模式 start/shutdown 正常工作，setInterval 按 tickIntervalMs 触发 |
| 19 | `test_create_simulator_offline` | createSimulator 离线模式 start 同步运行所有 tick 直至完成 |
| 20 | `test_create_simulator_enqueue` | enqueue 注入请求后 runTick 产出响应 |
| 21 | `test_create_simulator_get_metrics` | getMetrics 返回完整指标快照 |
| 22 | `test_http_service_v1_routes` | HttpService /v1/* 路由正确处理请求 |
| 23 | `test_e2e_workload_through_scheduler` | WorkloadGenerator 生成的请求经完整调度循环后产出正确响应 |

### 边界条件覆盖

| 边界条件 | 测试覆盖 |
|----------|----------|
| `arrivalRate = 0` → 所有请求 arrivalTick = 0 | 测试 1 变体 |
| `sharedPrefixRatio = 0` → 无共享前缀 | 测试 4 变体 |
| `numRequests = 0` → 返回空数组 | 测试 1 变体 |
| `totalTicks = 0` → gpuUtilization = 0 | 测试 9 变体 |
| `gpuBusy = 0` → gpuIdleTicks = totalTicks | 测试 9 变体 |
| 空 replies 列表 → recordReply 为 noop | 测试 7 变体 |
| SGHttpApi 未 bind 时 → 返回 503 | 测试 15 变体 |
| `max_tokens` 未指定 → 使用默认 128 | 测试 15 变体 |
| `createSimulator` shutdown 后 enqueue 不崩溃 | 测试 18 变体 |
| `tickIntervalMs = 0` → 使用默认 10ms | 测试 18 变体 |
| 离线模式无 workload → 立即完成 | 测试 19 变体 |

### 新增集成测试（回应评审缺失项）

| # | 测试名称 | 验证内容 |
|---|----------|----------|
| 14 | `test_simulation_metrics_tick_clock_integration` | SimulationMetrics.tick() 注册到 SimulationClock.onTick 后，每次时钟推进自动触发 recordTick |
| 23 | `test_e2e_workload_through_scheduler` | WorkloadGenerator 生成 10 个请求 → 按 arrivalTick 注入 Scheduler → 验证全部完成 + metrics 指标正确 |

## 风险与注意事项

- **兼容性影响**：`SimulationMetrics.toJSON()` 返回字段增多（但移除了 `pagesAllocated`/`pagesFree`），前端消费方需确认是否按需读取。已有 `/api/internal/metrics` 路由返回格式不变，新增 `/v1/internal/metrics` 为新端点。`WorkloadGenerator` 和 `SGHttpApi` 为全新模块，零兼容性破坏。
- **性能影响**：`SimulationMetrics.recordReply` 每次 tick 遍历 replies 列表，开销 O(n)。在仿真场景下 n 通常较小（≤max_running_req），无性能隐患。`WorkloadGenerator.generate` 一次性生成所有请求，不在热路径。
- **回滚方案**：`SimulationMetrics` 新增字段默认值为 0，不影响已有代码逻辑。`toJSON()` 返回更多字段是向后兼容扩展。`WorkloadGenerator` 和 `SGHttpApi` 为独立模块，删除导出即可回滚。`createSimulator` 为新增工厂函数，不影响已有 `Simulator` 类。
- **与 SimulationClock 的集成**：`SimulationMetrics.tick()` 方法通过 `SimulationClock.onTick()` 注册，仅在 `enableOverlap && enableMetrics` 时生效。默认配置下 SimulationClock 为 null，`tick()` 不被调用，不影响正确性。
- **SGHttpApi 无端口设计**：SGHttpApi 不再监听独立端口，减少了端口管理复杂度，但意味着 HttpService 必须在路由层正确委托到 SGHttpApi。如果未来需要独立部署 API 层，可重新引入端口监听能力。
