---
title: "Issue #9 解决方案"
issue_number: 9
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #9 解决方案

## 需求分析

### 问题描述

当前 `server/src/sglang/` 目录不存在。SGLang 仿真模拟器（SGLang Simulator）需要一个 TypeScript 实现的模块骨架，作为后续所有仿真组件（S1: 核心数据结构、K5: 内存预算、P0: 并行基础设施）的承载基础。本 Issue（S0）负责创建整个 `server/src/sglang/` 目录结构，并实现三类顶层基础构件：

1. **顶层配置接口**：`SimulatorConfig`（含运行模式、并行参数、CUDA Graph、Overlap 等）与 `ModelConfig`（含模型架构参数、MoE 特征）
2. **消息类型**：`SimRequestMsg`（req_in/req_resume）、`SimRespMsg`（resp_token/resp_done/resp_reject），对齐 §4.3 接口
3. **全局上下文桩**：`SgSimContext`（§4.2），包含 `tableMgr`/`cacheMgr`/`scheduler`/`tpGroup` 等占位引用，以及 `newId`、`clock` 基础设施

### 能力目标

1. 创建 `server/src/sglang/` 完整目录结构（按 §4.1 模块划分：`core/`、`scheduler/`、`engine/`、`cache/`、`entities/`、`workload/`、`parallel/`、`metrics/`、`api/`、`types.ts`、`Simulator.ts`、`index.ts`）
2. 所有 Python dataclass 转 TypeScript interface/type，纯 strict 模式，零运行时依赖
3. `SimulatorConfig` 支持 `mode: agg | pd-disagg | parallel`；并行参数 `tp_size/dp_size/ep_size/pp_size/cp_size`；CUDA Graph 分桶配置；Overlap 配置；并行带宽/延迟
4. `ModelConfig` 支持 `is_moe/num_experts/moe_top_k/use_mla`、layer/head 尺寸参数
5. 消息类型 `SimRequestMsg` 支持 `req_in`（新请求进入）和 `req_resume`（chunked prefill 续接）两种 tag；`SimRespMsg` 支持 `resp_token`（产出 token）、`resp_done`（请求完成）、`resp_reject`（准入拒绝）三种 tag
6. `SgSimContext` 提供 `tableMgr`/`cacheMgr`/`scheduler`/`tpGroup` 占位引用（初期为 `null`，后续 Issue 赋值）、`newId()` 自增 ID 生成器、`clock` tick 计数器
7. `size=1` 时通信 group 自然退化 noop（无需特殊处理，默认值即可实现）

### 影响范围

| 层 | 路径 | 影响程度 |
|---|---|---|
| 新目录 | `server/src/sglang/` 及全部子目录 | 高 — 新建整个模块骨架 |
| 配置 | `server/src/sglang/types.ts` | 高 — SimulatorConfig、ModelConfig 定义 |
| 消息 | `server/src/sglang/types.ts` | 中 — SimRequestMsg、SimRespMsg 定义 |
| 上下文 | `server/src/sglang/Simulator.ts` | 中 — SgSimContext 类 |
| 导出 | `server/src/sglang/index.ts` | 低 — 统一 re-export |
| 编译 | `server/tsconfig.json` | 无变更（已 include `src/**/*`） |
| 现有代码 | `server/src/sim/`、`server/src/shared/` | 无变更 — 新模块独立 |

## 改造方案

### 总体思路

本 Issue 仅创建骨架和类型定义，不实现仿真逻辑。所有子模块目录均创建 `index.ts` 作为占位入口，后续 Issue 逐步填充。重点确保：

- 类型定义与总体设计文档（§4.2 SimulatorConfig、§2.2 ModelConfig、§4.3 消息类型、§4.2 Context）精确对齐
- TypeScript strict 模式下零编译错误
- 零运行时依赖
- `SgSimContext` 采用"桩 + 延迟赋值"模式，属性初始为 `null`，后续 Issue 通过 setter 或直接赋值注入

### 详细设计

#### 1. 目录结构

```
server/src/sglang/
├── core/
│   └── index.ts               # 占位，S1 填充 SamplingParams/Req/Batch/Context
├── scheduler/
│   └── index.ts               # 占位，S1 后续填充 Scheduler/PrefillManager/DecodeManager 等
├── engine/
│   └── index.ts               # 占位，S1 后续填充 MockEngine/GraphRunner/Sampler
├── cache/
│   └── index.ts               # 占位，S1 后续填充 RadixCache/CacheManager
├── entities/
│   └── index.ts               # 占位，S1 后续填充 ChunkedReq/PendingReq
├── workload/
│   └── index.ts               # 占位，S1 后续填充 WorkloadGenerator/WorkloadConfig
├── parallel/
│   └── index.ts               # 占位，P0 后续填充 SimCommGroup/ParallelTopology
├── metrics/
│   └── index.ts               # 占位，后续填充 SimulationMetrics
├── api/
│   └── index.ts               # 占位，后续填充 API Server stub
├── types.ts                   # ★ SimulatorConfig, ModelConfig, SimRequestMsg, SimRespMsg
├── Simulator.ts               # ★ SgSimContext 类 + Simulator 入口桩
└── index.ts                   # 统一 re-export
```

#### 2. types.ts — 顶层配置与消息类型

##### SimulatorConfig

```typescript
/** 仿真器运行模式 */
export type SimMode = "agg" | "pd-disagg" | "parallel";

/** 仿真器统一配置（对应 §4.2 SimulatorConfig） */
export interface SimulatorConfig {
  // ===== 模型配置 =====
  modelConfig: ModelConfig;

  // ===== 调度配置 =====
  maxRunningReq: number;       // 默认 128
  maxSeqLen: number;           // 默认 8192
  maxExtendTokens: number;     // prefill budget，默认 8192
  cacheType: "radix" | "naive"; // 默认 "radix"

  // ===== KV Cache 配置 =====
  pageSize: number;            // 默认 1
  numPages: number | null;     // null = 自动计算

  // ===== 内存配置 =====
  totalGpuMemory: number;      // 字节，默认 80 * 1024**3
  memoryRatio: number;         // 默认 0.88
  dtypeSize: number;           // 每元素字节数，默认 2

  // ===== CUDA Graph 配置 =====
  enableCudaGraph: boolean;    // 默认 true
  cudaGraphBs: number[] | null; // null = 自动分桶
  cudaGraphMaxBs: number | null;
  graphReplayCostTicks: number; // 默认 1
  eagerForwardCostTicks: number; // 默认 10

  // ===== Overlap Scheduling 配置 =====
  enableOverlap: boolean;      // 默认 true
  cpuScheduleCostTicks: number;     // 默认 1
  cpuProcessResultCostTicks: number; // 默认 1

  // ===== TP 张量并行配置 =====
  tpSize: number;              // 默认 1
  allReduceCostPerByteTicks: number; // 默认 0.001
  allReduceLatencyTicks: number;     // 默认 2
  tpCpuGroupType: string;     // 默认 "gloo"
  tpGpuGroupType: string;     // 默认 "nccl"

  // ===== DP 数据并行配置 =====
  dpSize: number;              // 默认 1
  dpLoadBalanceStrategy: "round_robin" | "shortest_queue"; // 默认 "round_robin"
  enableDpAttention: boolean;  // 默认 false
  dpAttentionAllGatherCostPerByteTicks: number; // 默认 0.0015

  // ===== EP 专家并行配置 =====
  epSize: number;              // 默认 1
  allToAllCostPerByteTicks: number; // 默认 0.002
  allToAllLatencyTicks: number;     // 默认 3
  moeRoutingMode: "mock" | "hash" | "simulated"; // 默认 "mock"
  enableEplb: boolean;         // 默认 false

  // ===== CP Context Parallel 配置 =====
  cpSize: number;              // 默认 1
  cpAllGatherCostPerByteTicks: number; // 默认 0.001

  // ===== PP 流水并行配置 =====
  ppSize: number;              // 默认 1
  ppNumMicroBatches: number;   // 默认 1
  ppSendRecvCostPerByteTicks: number; // 默认 0.0005
  ppPipelineSchedule: "1f1b" | "gpipe" | "interleaved"; // 默认 "1f1b"

  // ===== 通信成本通用配置 =====
  commBandwidthBytesPerTick: number; // 默认 1_000_000
  commOverlapWithCompute: boolean;   // 默认 true

  // ===== 离线模式 =====
  offlineMode: boolean;        // 默认 false

  // ===== Tokenizer =====
  eosTokenId: number;          // 默认 0

  // ===== 采样配置 =====
  mockSampleMode: "random" | "greedy" | "fixed"; // 默认 "random"
  fixedOutputToken: number;    // for "fixed" mode，默认 0

  // ===== 仿真控制 =====
  maxTicks: number | null;     // null = 无限运行
  logLevel: string;            // 默认 "INFO"
  enableMetrics: boolean;      // 默认 true
}

/** SimulatorConfig 默认值 */
export const DEFAULT_SIMULATOR_CONFIG: SimulatorConfig = {
  modelConfig: DEFAULT_MODEL_CONFIG,
  maxRunningReq: 128,
  maxSeqLen: 8192,
  maxExtendTokens: 8192,
  cacheType: "radix",
  pageSize: 1,
  numPages: null,
  totalGpuMemory: 80 * 1024 ** 3,
  memoryRatio: 0.88,
  dtypeSize: 2,
  enableCudaGraph: true,
  cudaGraphBs: null,
  cudaGraphMaxBs: null,
  graphReplayCostTicks: 1,
  eagerForwardCostTicks: 10,
  enableOverlap: true,
  cpuScheduleCostTicks: 1,
  cpuProcessResultCostTicks: 1,
  tpSize: 1,
  allReduceCostPerByteTicks: 0.001,
  allReduceLatencyTicks: 2,
  tpCpuGroupType: "gloo",
  tpGpuGroupType: "nccl",
  dpSize: 1,
  dpLoadBalanceStrategy: "round_robin",
  enableDpAttention: false,
  dpAttentionAllGatherCostPerByteTicks: 0.0015,
  epSize: 1,
  allToAllCostPerByteTicks: 0.002,
  allToAllLatencyTicks: 3,
  moeRoutingMode: "mock",
  enableEplb: false,
  cpSize: 1,
  cpAllGatherCostPerByteTicks: 0.001,
  ppSize: 1,
  ppNumMicroBatches: 1,
  ppSendRecvCostPerByteTicks: 0.0005,
  ppPipelineSchedule: "1f1b",
  commBandwidthBytesPerTick: 1_000_000,
  commOverlapWithCompute: true,
  offlineMode: false,
  eosTokenId: 0,
  mockSampleMode: "random",
  fixedOutputToken: 0,
  maxTicks: null,
  logLevel: "INFO",
  enableMetrics: true,
};
```

##### ModelConfig

```typescript
/** 模型特征描述（对应 §2.2 + §4.2 ModelConfig） */
export interface ModelConfig {
  numLayers: number;
  hiddenSize: number;
  numKvHeads: number;
  headDim: number;
  vocabSize: number;
  isMoe: boolean;              // 默认 false
  numExperts: number;           // MoE only，默认 0
  moeIntermediateSize: number;  // MoE only，默认 0
  moeTopK: number;             // MoE 每个 token 选择专家数，默认 1
  intermediateSize: number;    // dense MLP，默认 0
  numAttentionHeads: number;   // 默认 0
  rmsNormEps: number;          // 默认 1e-6
  ropeTheta: number;           // 默认 10000.0
  maxPositionEmbeddings: number; // 默认 8192
  useMla?: boolean;            // MLA 注意力模式（可选扩展）
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  numLayers: 32,
  hiddenSize: 4096,
  numKvHeads: 8,
  headDim: 128,
  vocabSize: 128256,
  isMoe: false,
  numExperts: 0,
  moeIntermediateSize: 0,
  moeTopK: 1,
  intermediateSize: 0,
  numAttentionHeads: 0,
  rmsNormEps: 1e-6,
  ropeTheta: 10000.0,
  maxPositionEmbeddings: 8192,
};
```

##### SimRequestMsg

```typescript
/** 请求消息标签 */
export type SimRequestMsgTag = "req_in" | "req_resume";

/** 请求进入/续接消息（对应 §4.3 接口） */
export interface SimRequestMsg {
  tag: SimRequestMsgTag;
  uid: number;
  inputIds: number[];          // 仿真中用 number[] 替代 tensor
  samplingParams: SamplingParams | null; // req_resume 时为 null（复用已有参数）
  outputLen: number;           // 预期输出长度
}
```

> **设计说明**：`req_in` 表示新请求首次进入调度器，`samplingParams` 和 `outputLen` 必填；`req_resume` 表示 chunked prefill 续接，`samplingParams` 为 `null`（复用首次绑定参数），`outputLen` 仍需携带以便 scheduler 快速判断 `remain_len`。

##### SimRespMsg

```typescript
/** 响应消息标签 */
export type SimRespMsgTag = "resp_token" | "resp_done" | "resp_reject";

/** 响应消息（对应 §4.3 接口） */
export interface SimRespMsg {
  tag: SimRespMsgTag;
  uid: number;
  nextToken: number | null;    // resp_done/resp_reject 时为 null
  finished: boolean;           // resp_done 时为 true，其余为 false
  reason?: string;             // resp_reject 时的拒绝原因
}
```

> **设计说明**：
> - `resp_token`：decode 阶段每步产出，`nextToken` 为采样结果，`finished = false`
> - `resp_done`：请求完成，`nextToken` 为最后一个 token 或 `null`，`finished = true`
> - `resp_reject`：准入拒绝（如 KV 容量不足），`nextToken = null`，`finished = true`，附带 `reason`

##### SamplingParams（S0 骨架，S1 完善）

```typescript
/** 采样参数（对应 §2.2.1，S0 仅定义接口，S1 实现完整逻辑） */
export interface SamplingParams {
  temperature: number;         // 默认 0.0
  topK: number;                // 默认 -1（不限制）
  topP: number;                // 默认 1.0（不限制）
  ignoreEos: boolean;          // 默认 false
  maxTokens: number;           // 默认 1024
}
```

#### 3. Simulator.ts — SgSimContext 类

```typescript
import type { SimulatorConfig } from "./types";

/**
 * SGLang 仿真全局上下文（对应 §4.2 Context + 全局上下文工具函数）
 *
 * 设计原则：
 * - 属性初始为 null，后续 Issue 逐步注入实际实例
 * - newId() 严格单调递增，用于生成全局唯一请求 ID
 * - clock 为离散 tick 计数器，由外部 run_tick 驱动
 */
export class SgSimContext {
  readonly config: SimulatorConfig;

  // ===== 占位引用（后续 Issue 赋值） =====
  tableMgr: TableManager | null = null;    // S1 赋值
  cacheMgr: CacheManager | null = null;    // S1 赋值
  scheduler: SimScheduler | null = null;   // S1 赋值
  tpGroup: SimCommGroup | null = null;     // P0 赋值

  // ===== 基础设施 =====
  private _nextId: number = 0;
  private _clock: number = 0;

  constructor(config: SimulatorConfig) {
    this.config = config;
  }

  /** 生成全局唯一 ID（严格单调递增） */
  newId(): number {
    return ++this._nextId;
  }

  /** 当前 tick */
  get clock(): number {
    return this._clock;
  }

  /** 推进时钟（由外部 run_tick 调用） */
  advanceClock(ticks: number = 1): void {
    this._clock += ticks;
  }

  /** 重置上下文状态（用于 reset 场景） */
  reset(): void {
    this._nextId = 0;
    this._clock = 0;
    this.tableMgr = null;
    this.cacheMgr = null;
    this.scheduler = null;
    this.tpGroup = null;
  }
}

/**
 * Simulator 入口桩（S0 仅定义接口，S1 实现完整调度循环）
 */
export class Simulator {
  readonly ctx: SgSimContext;

  constructor(config: SimulatorConfig) {
    this.ctx = new SgSimContext(config);
  }

  /**
   * 执行一个调度 tick
   * @param incoming 进入的请求消息列表
   * @returns 响应消息列表
   */
  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // S1 实现：消息分发 → 调度 → forward → 结果处理
    return [];
  }

  /** 重置仿真器状态 */
  reset(): void {
    this.ctx.reset();
  }
}
```

> **关键设计决策**：
> 1. `SgSimContext` 使用类而非 interface，因为需要封装 `newId()` 和 `advanceClock()` 的有状态行为
> 2. 占位引用声明为具体类型的 nullable（`TableManager | null`），而非 `any`，确保后续赋值时有类型检查
> 3. `Simulator.runTick()` 签名与 §3.3.1 `SimScheduler.run_tick()` 对齐，S1 实现内部转发
> 4. `size=1` 时 `tpGroup` 保持 `null`，通信调用方需检查——这与 SGLang 源码中 `tp_size=1` 时 TP group 为 noop 的语义一致

#### 4. 占位类型前向声明

为使 `SgSimContext` 的占位引用有类型而非 `any`，在 `types.ts` 中声明以下接口桩（S1/P0 实现）：

```typescript
// ===== 占位接口（后续 Issue 实现） =====

/** 表管理器桩（S1 实现） */
export interface TableManager {
  allocate(): number;
  free(tableIdx: number): void;
  readonly availableSize: number;
}

/** 缓存管理器桩（S1 实现） */
export interface CacheManager {
  readonly availableSize: number;
  matchReq(req: unknown): unknown;
  lockReq(handle: unknown): void;
  unlockReq(handle: unknown): void;
}

/** 调度器桩（S1 实现） */
export interface SimScheduler {
  runTick(incoming: SimRequestMsg[]): SimRespMsg[];
}

/** 通信组桩（P0 实现，size=1 时为 noop） */
export interface SimCommGroup {
  allReduce(dataBytes: number): number;
  allToAll(dataBytes: number): number;
  sendRecv(dataBytes: number): number;
  barrier(): void;
}
```

### 修改点清单

1. **新建** `server/src/sglang/` 目录及全部子目录
2. **新建** `server/src/sglang/types.ts` — 定义 `SimMode`、`SimulatorConfig`（含默认值常量）、`ModelConfig`（含默认值常量）、`SamplingParams`、`SimRequestMsg`、`SimRespMsg`、占位接口 `TableManager`/`CacheManager`/`SimScheduler`/`SimCommGroup`
3. **新建** `server/src/sglang/Simulator.ts` — 定义 `SgSimContext` 类和 `Simulator` 入口桩
4. **新建** `server/src/sglang/index.ts` — 统一 re-export 所有公共类型和类
5. **新建** 各子模块 `index.ts` — 空文件或仅含注释占位（`core/`、`scheduler/`、`engine/`、`cache/`、`entities/`、`workload/`、`parallel/`、`metrics/`、`api/`）
6. **验证** `npx tsc --noEmit` 通过 strict 编译零错误

## 测试设计

### 验收测试用例清单

| 编号 | 测试场景 | 预期结果 |
|------|---------|---------|
| T1 | `DEFAULT_SIMULATOR_CONFIG` 所有字段均有默认值 | 无 undefined 字段 |
| T2 | `DEFAULT_MODEL_CONFIG` 所有字段均有默认值 | 无 undefined 字段 |
| T3 | `SgSimContext.newId()` 连续调用返回 1, 2, 3... | 严格单调递增 |
| T4 | `SgSimContext.clock` 初始为 0 | 正确 |
| T5 | `SgSimContext.advanceClock(5)` 后 clock 为 5 | 正确 |
| T6 | `SgSimContext.reset()` 后 clock=0, _nextId=0, 所有占位引用=null | 正确 |
| T7 | `SimRequestMsg` tag="req_in" 时构造正确 | 类型推断无误 |
| T8 | `SimRequestMsg` tag="req_resume" 时 samplingParams=null 合法 | 编译通过 |
| T9 | `SimRespMsg` tag="resp_reject" 时 reason 字段可选存在 | 类型推断无误 |
| T10 | `SimulatorConfig.tpSize=1` 时字段值正确，无特殊逻辑依赖 | 正确 |
| T11 | `Simulator.runTick([])` 返回空数组 | 正确 |
| T12 | TypeScript strict 编译零错误 | 通过 |
| T13 | 各子模块 `index.ts` 存在且可导入 | 无运行时错误 |

### 边界条件覆盖

- `SimulatorConfig.numPages = null`（自动计算 vs 显式指定）
- `SimulatorConfig.maxTicks = null`（无限运行）
- `SimulatorConfig.cudaGraphBs = null` vs `number[]`（自动分桶 vs 手动指定）
- `ModelConfig.isMoe = false` 时 MoE 字段应仍有默认值 0
- `SimRespMsg.tag = "resp_reject"` 时 `nextToken = null`、`finished = true`

## 风险与注意事项

### 兼容性影响

- 新增 `server/src/sglang/` 目录完全独立于现有 `server/src/sim/` 和 `server/src/shared/`，**零兼容性风险**
- `tsconfig.json` 已配置 `include: ["src/**/*"]`，自动覆盖新目录，无需修改

### 性能影响

- S0 仅创建类型和骨架类，无运行时逻辑，**零性能影响**
- `SgSimContext.newId()` 使用简单自增，O(1) 无性能隐患

### 回滚方案

- 删除 `server/src/sglang/` 整个目录即可回滚，无其他文件受影响
