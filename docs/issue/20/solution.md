---
title: "Issue #20 解决方案"
issue_number: 20
issue_type: Feature
created: 2026-09-02
updated: 2026-09-02
status: draft
review_round: 1
---

# Issue #20 解决方案

## 需求分析

- **问题描述**：Issue #20 要求实现 S6 阶段的三大组件：`WorkloadGenerator`、`SimulationMetrics` 完整指标体系、`SGHttpApi`（仿真 HTTP API），以及顶层导出函数 `createSimulator(config): SgSimInstance`。

- **能力目标**：
  1. **WorkloadGenerator（§4.3 / §9.10）**：Poisson/固定速率请求生成；生成 `SimRequestMsg`（含 uid/inputIds/samplingParams/outputLen）；支持 trace 回放（从预定义序列重放请求）
  2. **SimulationMetrics 完整指标（§4.3 / §4.5）**：
     - 延迟指标：`request_latency_ns` / `ttft_ns` / `tbt_ns`
     - 吞吐指标：`prefill_throughput_tokens_per_s` / `decode_throughput_tokens_per_s`
     - Cache 指标：`hit_rate` / `miss_evictions` / `num_cache_nodes`
     - Memory 指标：`pages_allocated` / `pages_free` / `oom_count`
     - GPU 指标：`gpu_busy_ticks` / `gpu_idle_ticks` / `gpu_utilization`
     - 调度指标：`prefill_batches` / `decode_batches` / `avg_prefill_batch_size` / `avg_decode_batch_size`
     - 方法：`record(stepResult)`、`tick(currentTicks)`、`toJSON()`
     - 已有 `parallel: ParallelMetrics` 子结构保持不变
  3. **SGHttpApi（§4.3）**：
     - `POST /v1/chat/completions`：OpenAI 兼容请求，转成 `SimRequestMsg` 发送到 SchedulerIOMixin
     - `GET /v1/internal/metrics`：返回 `SimulationMetrics.toJSON()` + Snapshot
     - `GET /v1/internal/state`：返回 Scheduler 状态快照
  4. **导出函数 `createSimulator(config): SgSimInstance`**：`start()` / `enqueue(req)` / `getMetrics()` / `shutdown()`

- **影响范围**：修改 `server/src/sglang/workload/index.ts`（WorkloadGenerator 实现）、`server/src/sglang/metrics/index.ts`（SimulationMetrics 完整指标升级）、`server/src/sglang/api/index.ts`（SGHttpApi 实现）、`server/src/http/HttpService.ts`（新增路由）、`server/src/sglang/index.ts`（导出 createSimulator/SgSimInstance）、新增测试文件 `server/src/test/sglang-s6.test.ts`。不修改已有测试代码。

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
/** 工作负载生成器配置（对应 §4.3 WorkloadConfig / §9.10） */
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
        samplingParams: null,  // 由 Simulator 层赋值默认 SamplingParams
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
- `samplingParams` 设为 null，由 `SgSimInstance.enqueue()` 在注入时赋值默认 `SamplingParams`

#### 2. SimulationMetrics 升级（`server/src/sglang/metrics/index.ts`）

**完整指标字段**（对齐 §4.5）：

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

  // ===== 内存指标 =====
  pagesAllocated: number = 0;
  pagesFree: number = 0;
  oomCount: number = 0;
  peakMemoryUsage: number = 0;

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

  /** 记录单个请求完成的延迟数据（TTFT/TBT/E2E） */
  recordRequestLatency(uid: number, arrivalTick: number, firstTokenTick: number,
                       finishTick: number, decodeSteps: number): void;

  /** 记录 cache 指标快照（由 CacheManager 回调） */
  recordCacheSnapshot(hitRate: number, evictionCount: number,
                      utilization: number): void;

  /** 记录内存指标快照（由 CacheManager 回调） */
  recordMemorySnapshot(allocated: number, free: number, peak: number): void;

  /** 时钟 tick 回调（供 SimulationClock.onTick 注册） */
  tick(currentTicks: number): void;

  /** 重置所有指标到默认值 */
  reset(): void;

  /** 序列化为 JSON 可序列化对象 */
  toJSON(): Record<string, unknown>;
}
```

**方法实现要点**：

- `recordReply`：遍历 replies，对 `tag === "resp_done"` 的消息递增 `completedRequests`，对 `tag === "resp_token"` 的消息递增 `totalTokensGenerated`
- `recordBatch`：区分 prefill/decode batch，使用增量平均公式更新 `avgPrefillBatchSize`/`avgDecodeBatchSize`（与 §9.11 伪代码一致）。GPU busy ticks 不在此累加，避免与 `recordTick` 双重计数
- `recordTick`：`totalTicks = max(totalTicks, tick + 1)`，`gpuBusyTicks += gpuBusy`，`gpuIdleTicks = max(0, totalTicks - gpuBusyTicks)`，`gpuUtilization = gpuBusyTicks / max(1, totalTicks)`（与 §9.11 伪代码精确一致）
- `recordRequestLatency`：记录 TTFT = `firstTokenTick - arrivalTick`，TBT = `(finishTick - firstTokenTick) / max(1, decodeSteps)`，E2E = `finishTick - arrivalTick`
- `tick`：空操作占位，供 SimulationClock.onTick 注册；未来可用于周期性聚合指标
- `toJSON`：返回包含所有字段 + `parallel.summary()` 的完整 JSON 对象

#### 3. SGHttpApi（`server/src/sglang/api/index.ts`）

```typescript
import http from "http";
import type { SimScheduler } from "../scheduler";
import type { SimulationMetrics } from "../metrics";

/** SGHttpApi — SGLang 仿真 HTTP API（§4.3） */
export class SGHttpApi {
  private server: http.Server;
  private readonly port: number;
  private scheduler: SimScheduler | null = null;
  private metrics: SimulationMetrics | null = null;

  constructor(port: number = 8000) {
    this.port = port;
    this.server = http.createServer((req, res) => this.handleRequest(req, res));
  }

  /** 注入调度器和指标实例 */
  bind(scheduler: SimScheduler, metrics: SimulationMetrics): void;

  /** 启动 HTTP 服务 */
  start(): void;

  /** 停止 HTTP 服务 */
  stop(): void;

  private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    const url = new URL(req.url || "/", `http://localhost:${this.port}`);
    const path = url.pathname;

    // POST /v1/chat/completions — OpenAI 兼容
    if (req.method === "POST" && path === "/v1/chat/completions") {
      this.handleChatCompletions(req, res);
      return;
    }

    // GET /v1/internal/metrics — 仿真指标
    if (req.method === "GET" && path === "/v1/internal/metrics") {
      this.handleInternalMetrics(req, res);
      return;
    }

    // GET /v1/internal/state — 调度器状态
    if (req.method === "GET" && path === "/v1/internal/state") {
      this.handleInternalState(req, res);
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not Found" }));
  }
}
```

**`POST /v1/chat/completions` 实现**：

- 解析请求体为 OpenAI ChatCompletionRequest 格式
- 提取 `messages` → 拼接为文本 → 生成 `inputIds`（简单 token 计数：`text.length` 个 token，值为 0..N-1）
- 构造 `SimRequestMsg { tag: "req_in", uid, inputIds, samplingParams: null, outputLen: max_tokens }`
- 通过 `scheduler._incomingQueue.push()` 注入（非 offline 模式）或通过 `SgSimInstance.enqueue()` 注入
- 立即返回 `{ id, object: "chat.completion", ... }` 占位响应（仿真为异步执行，不等待完成）

**`GET /v1/internal/metrics` 实现**：

- 返回 `metrics.toJSON()` + 调度器当前快照（running 请求数、pending 请求数等）

**`GET /v1/internal/state` 实现**：

- 返回调度器状态：`{ runningReqs, pendingReqs, availableTableIndices, cacheSizeInfo, tick }`

#### 4. HttpService 路由扩展（`server/src/http/HttpService.ts`）

在现有 `handleRequest` 方法中新增路由：

```typescript
// POST /v1/chat/completions — 代理到 SGHttpApi
if (req.method === "POST" && urlPath === "/v1/chat/completions") {
  this.proxyToSim("/v1/chat/completions", req, res);
  return;
}

// GET /v1/internal/metrics — 已有 /api/internal/metrics，新增 /v1 前缀路由
if (req.method === "GET" && urlPath === "/v1/internal/metrics") {
  this.proxyToSim("/v1/internal/metrics", req, res);
  return;
}

// GET /v1/internal/state — 代理到 SGHttpApi
if (req.method === "GET" && urlPath === "/v1/internal/state") {
  this.proxyToSim("/v1/internal/state", req, res);
  return;
}
```

> **设计说明**：保留已有 `/api/internal/metrics` 路由兼容性，新增 `/v1/*` 前缀路由对齐 §4.3 规格。两者均代理到同一 SimService 端口。

#### 5. SgSimInstance + createSimulator（`server/src/sglang/Simulator.ts`）

```typescript
/** SgSimInstance — 仿真器实例接口（§4.3） */
export interface SgSimInstance {
  readonly ctx: SgSimContext;
  readonly scheduler: SimScheduler;
  readonly metrics: SimulationMetrics;
  readonly workload: WorkloadGenerator;

  /** 启动仿真循环（内部 tick 驱动） */
  start(): void;

  /** 入队一个请求 */
  enqueue(msg: SimRequestMsg): void;

  /** 获取当前指标快照 */
  getMetrics(): Record<string, unknown>;

  /** 关闭仿真器 */
  shutdown(): void;
}

/** 创建仿真器实例（§4.3 导出函数） */
export function createSimulator(config: SimulatorConfig): SgSimInstance;
```

**实现要点**：

- `createSimulator` 内部创建 `SgSimContext`、`SimScheduler`、`SimulationMetrics`、`WorkloadGenerator`
- `start()` 启动内部 `setInterval` tick 循环，每 tick 调用 `scheduler.runTick(incoming)` + `metrics.recordReply/tick`
- `enqueue(msg)` 将消息推入 `scheduler._incomingQueue`
- `getMetrics()` 返回 `metrics.toJSON()`
- `shutdown()` 清除 interval，停止仿真循环

#### 6. SimService 集成（`server/src/sim/SimService.ts`）

在 SimService 中集成 SGHttpApi：

- 在 `SimService` 构造器或 `start()` 中，如果存在 SGLang 仿真配置，创建 `SGHttpApi` 实例
- 将 `SGHttpApi` 绑定到 `SimScheduler` 和 `SimulationMetrics`
- 或选择让 `HttpService` 通过代理路由到 `SGHttpApi` 的端口

> **设计决策**：SGHttpApi 独立端口（默认 8000），与现有 SimService（3001）和 HttpService（8888）并存。HttpService 的 `/v1/*` 路由代理到 SGHttpApi 端口。这样保持现有服务架构不变，仅增量添加。

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

1. **`server/src/sglang/workload/index.ts`** — 实现 `WorkloadConfig` 接口 + `DEFAULT_WORKLOAD_CONFIG` + `WorkloadGenerator` 类（含 `_sampleLen`、`_sampleArrival`、`_generateTokens` 方法）
2. **`server/src/sglang/metrics/index.ts`** — 升级 `SimulationMetrics`：补全 §4.5 全部指标字段，实现 `recordReply`、`recordBatch`、`recordTick`、`recordRequestLatency`、`recordCacheSnapshot`、`recordMemorySnapshot`、`tick`、升级 `toJSON`
3. **`server/src/sglang/api/index.ts`** — 实现 `SGHttpApi` 类（`POST /v1/chat/completions`、`GET /v1/internal/metrics`、`GET /v1/internal/state`）
4. **`server/src/sglang/Simulator.ts`** — 升级 `Simulator` 为 `SgSimInstance` 接口实现 + 新增 `createSimulator` 工厂函数
5. **`server/src/http/HttpService.ts`** — 新增 `/v1/chat/completions`、`/v1/internal/metrics`、`/v1/internal/state` 代理路由
6. **`server/src/sglang/index.ts`** — 新增 re-export：`WorkloadGenerator`、`WorkloadConfig`、`DEFAULT_WORKLOAD_CONFIG`、`SgSimInstance`、`createSimulator`、`SGHttpApi`
7. **`server/src/test/sglang-s6.test.ts`**（新增）— S6 阶段测试

## 测试设计

### 验收测试用例清单

| # | 测试名称 | 验证内容 |
|---|----------|----------|
| 1 | `test_workload_generator_poisson` | Poisson 到达分布生成正确数量的请求，arrivalTick 单调递增 |
| 2 | `test_workload_generator_uniform` | 均匀分布到达的请求 arrivalTick 均匀分布 |
| 3 | `test_workload_generator_trace_replay` | trace 模式直接返回预定义序列 |
| 4 | `test_workload_generator_shared_prefix` | sharedPrefixRatio=0.3 时约 1/3 请求共享前缀 |
| 5 | `test_workload_generator_normal_distribution` | normal 分布下长度值在 [min, max] 范围内 |
| 6 | `test_simulation_metrics_record_reply` | recordReply 正确递增 completedRequests 和 totalTokensGenerated |
| 7 | `test_simulation_metrics_record_batch` | recordBatch 正确更新 avgPrefillBatchSize/avgDecodeBatchSize |
| 8 | `test_simulation_metrics_record_tick` | recordTick 正确计算 gpuUtilization、gpuIdleTicks |
| 9 | `test_simulation_metrics_record_request_latency` | recordRequestLatency 正确记录 TTFT/TBT/E2E |
| 10 | `test_simulation_metrics_to_json` | toJSON 返回包含所有指标字段 + parallel 的完整对象 |
| 11 | `test_simulation_metrics_reset` | reset 清零所有指标字段和 parallel 子结构 |
| 12 | `test_sg_http_api_chat_completions` | POST /v1/chat/completions 接受请求并注入调度器 |
| 13 | `test_sg_http_api_internal_metrics` | GET /v1/internal/metrics 返回 metrics.toJSON() |
| 14 | `test_sg_http_api_internal_state` | GET /v1/internal/state 返回调度器状态快照 |
| 15 | `test_create_simulator` | createSimulator 返回 SgSimInstance，start/shutdown 正常工作 |
| 16 | `test_create_simulator_enqueue` | enqueue 注入请求后 runTick 产出响应 |
| 17 | `test_create_simulator_get_metrics` | getMetrics 返回完整指标快照 |
| 18 | `test_http_service_v1_routes` | HttpService /v1/* 路由正确代理到 SGHttpApi |

### 边界条件覆盖

| 边界条件 | 测试覆盖 |
|----------|----------|
| `arrivalRate = 0` → 所有请求 arrivalTick = 0 | 测试 1 变体 |
| `sharedPrefixRatio = 0` → 无共享前缀 | 测试 4 变体 |
| `numRequests = 0` → 返回空数组 | 测试 1 变体 |
| `totalTicks = 0` → gpuUtilization = 0 | 测试 8 变体 |
| `gpuBusy = 0` → gpuIdleTicks = totalTicks | 测试 8 变体 |
| 空 replies 列表 → recordReply 为 noop | 测试 6 变体 |
| SGHttpApi 未 bind 时 → 返回 503 | 测试 12 变体 |
| `max_tokens` 未指定 → 使用默认 outputLen | 测试 12 变体 |
| `createSimulator` shutdown 后 enqueue 不崩溃 | 测试 15 变体 |

## 风险与注意事项

- **兼容性影响**：`SimulationMetrics.toJSON()` 返回字段增多，前端消费方需确认是否按需读取。已有 `/api/internal/metrics` 路由返回格式不变，新增 `/v1/internal/metrics` 为新端点。`WorkloadGenerator` 和 `SGHttpApi` 为全新模块，零兼容性破坏。
- **性能影响**：`SimulationMetrics.recordReply` 每次 tick 遍历 replies 列表，开销 O(n)。在仿真场景下 n 通常较小（≤max_running_req），无性能隐患。`WorkloadGenerator.generate` 一次性生成所有请求，不在热路径。
- **回滚方案**：`SimulationMetrics` 新增字段默认值为 0，不影响已有代码逻辑。`toJSON()` 返回更多字段是向后兼容扩展。`WorkloadGenerator` 和 `SGHttpApi` 为独立模块，删除导出即可回滚。`createSimulator` 为新增工厂函数，不影响已有 `Simulator` 类。
- **与 SimulationClock 的集成**：`SimulationMetrics.tick()` 方法通过 `SimulationClock.onTick()` 注册，仅在 `enableOverlap && enableMetrics` 时生效。默认配置下 SimulationClock 为 null，`tick()` 不被调用，不影响正确性。
