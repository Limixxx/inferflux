---
title: "Issue #1 解决方案"
issue_number: 1
issue_type: Feature
created: 2026-08-20
updated: 2026-08-20
status: revised
review_round: 2
---

# Issue #1 解决方案

## 需求分析

### 问题描述

当前 `server/` TypeScript 项目仅支持 **PD-disaggregated（PD 分离）** 模式模拟：Prefill（P）和 Decode（D）运行在物理分离的 worker 实例上，通过 KV cache transfer link 连接。但最常见的部署模式——**aggregated（聚合）** 模式（单个 worker 同时处理 prefill 和 decode，共享一个 KV 缓存池，无跨实例传输）完全缺失。

用户无法在相同负载条件下对比 PD 分离与聚合架构的性能差异，无法回答诸如"PD 分离对我的负载是否值得传输开销"等关键问题。

### 能力目标

1. 在现有 SimEngine 中新增 `agg` 模式，使模拟器支持 `pd-disagg` 和 `agg` 两种部署架构
2. Agg 模式下完整模拟 SGLang 聚合架构的请求生命周期，涵盖以下关键机制：
   - **RadixCache 前缀复用**：通过 `cacheHitRate` → `cachedLen`/`uncachedLen` 模拟前缀匹配，命中部分无需 prefill 计算
   - **BlockManager 预分配**：请求准入时一次性预分配完整 `inputLen` 对应的 KV 物理块（含已缓存部分），分块 prefill 不降低总显存占用
   - **make_batch 混合批处理**：同一 GPU 迭代中 Prefill 与 Decode 请求共存，受 token 预算控制
   - **Chunked Prefill 可选化**：新增 `chunkedPrefill` 开关，支持非分块（单轮 prefill 即产出首 token）与分块（多轮 prefill，全部分片完成后产出首 token）两种子模式
   - **LPM 聚合**：多个小 prefill 请求可聚合入同一批次
3. 状态机适配两种 prefill 子模式：`PREFILL → DECODE`（非分块）vs `CHUNKED_PREFILL → DECODE`（分块）
4. 指标体系（TTFT 分解、Gauges、时序图）自适应模式切换
5. 前端可视化适配 agg 模式（单列 worker 布局、侧栏参数切换、breakdown 图表适配）
6. 保持 pd-disagg 模式完全兼容，默认值不变

### 影响范围

| 层 | 文件 | 影响程度 |
|---|---|---|
| 类型定义 | `shared/types.ts` | 中 — 新增字段和类型 |
| 常量 | `shared/constants.ts` | 低 — 新增 agg BD_KEYS |
| 实体 | 新建 `entities/WorkerInstance.ts` | 高 — 新实体 |
| 引擎 | `sim/SimEngine.ts` | 高 — 新增 stepAgg 分支 |
| 指标 | `sim/MetricsCollector.ts` | 中 — 模式感知 breakdown |
| 服务 | `sim/SimService.ts` | 中 — 模式感知 getRenderState |
| 预设 | `shared/presets.ts` | 低 — 新增 mode 字段和 agg 预设 |
| 国际化 | `shared/i18n.ts` | 低 — 新增 agg 相关 label |
| 前端 | `public/pd-disagg.html` | 高 — 模式切换 UI + agg 渲染 |
| 入口 | `src/index.ts` | 无变更 |
| HTTP | `http/HttpService.ts` | 无变更 |

## 评审意见响应

上一轮评审（review_round 1）提出"根据 SGLang 端到端请求流程分析优化仿真流程"。以下逐条响应：

### 1. Chunked Prefill 应为可选优化，而非强制

**问题**：v1 方案中 agg 模式始终使用 `chunkSize` 进行分块 prefill，未区分 SGLang 中"不开启 Chunked Prefill"与"开启 Chunked Prefill"两条路径。

**响应**：新增 `chunkedPrefill: boolean` 参数（agg 模式专用）。两种子模式的行为差异：

| 维度 | `chunkedPrefill=false` | `chunkedPrefill=true` |
|---|---|---|
| 状态机 | `w_prefill → w_decode` | `w_chunked_prefill → w_decode` |
| Prefill 轮次 | 单轮 GPU 迭代，一次性计算全部 `uncachedLen` | 多轮 GPU 迭代，每轮计算 `chunkSize` 个 token |
| 首 Token 时机 | prefill 单轮结束即采样产出 | 全部分片完成后才采样产出 |
| KV 预分配 | 一次性预分配完整 `inputLen`（与分块无关） | 同左 |
| 适用场景 | 短 Prompt、追求低首字延迟 | 超长 Prompt、平摊算力峰值 |

### 2. RadixCache 前缀复用已存在，需在 agg 模式中显式复用

**问题**：v1 方案未提及 RadixCache。

**响应**：当前代码已通过 `cacheHitRate` → `cachedLen`/`uncachedLen` 机制模拟 RadixCache 前缀复用（见 `entities/Request.ts:makeRequest`）。agg 模式直接复用此机制：
- `cachedLen = inputLen * cacheHitRate`：前缀匹配命中的 token 数，其 KV 已在缓存中，无需 prefill 计算
- `uncachedLen = inputLen - cachedLen`：需要 prefill 计算的 token 数
- Prefill 计算量仅取决于 `uncachedLen`，但 KV 占用为完整 `inputLen`（命中部分复用已有 KV 块，不重新计算但仍占显存）

### 3. BlockManager 预分配已存在，需在 agg 模式中显式复用

**问题**：v1 方案未提及 BlockManager 预分配。

**响应**：当前 pd-disagg 代码在 PrefillInstance 准入时执行 `p.kvUsed += r.inputLen`（完整 inputLen），已正确模拟 BlockManager 一次性预分配全部 KV 物理块的行为。agg 模式复用此逻辑：请求从 `waitingQ` 准入到 `running` 批次时，`w.kvUsed += r.inputLen`，无论是否开启 chunked prefill，总显存占用不变。

### 4. make_batch 混合批处理 + LPM

**问题**：v1 方案中 prefill 和 decode 分处不同队列（slots + prefillDone + running），未模拟 SGLang make_batch 的混合批处理能力。

**响应**：修订 WorkerInstance 为**统一运行批次**设计。`running` 数组中同时包含 PREFILL/CHUNKED_PREFILL 和 DECODE 请求，同一 GPU 迭代内共同计算：

- **准入（make_batch）**：每步从 `waitingQ` 尝试准入新请求，受 `maxRunning` 和 KV 容量约束。多个小 prefill 请求可同时准入（LPM 聚合）
- **步延迟**：`step = max(prefill_compute_time, decode_step_time)`，模拟 prefill（计算密集）与 decode（访存密集）在 GPU 上的并行重叠
- **配额管理**：`maxRunning` 限制批次内总请求数，间接为 decode 保留算力配额，避免长 prefill 挤占生成

### 5. 状态机与首 Token 时机

**问题**：v1 方案状态机过于简化（`w_waiting → w_prefill → w_decode`），未区分两种 prefill 路径的首 token 时机差异。

**响应**：修订状态机为四阶段（不含公共的 response/done）：

```
非分块: tokenize → w_waiting → w_prefill → w_decode → response → done
                            (单轮 prefill,  首token在此步产出)
分  块: tokenize → w_waiting → w_chunked_prefill → w_decode → response → done
                            (多轮 prefill,   全部分片完成后产出首token)
```

首 Token 时间戳 `firstToken = wPrefillDone + detokenizeMs`：
- 非分块：`wPrefillDone` = 单轮 prefill 结束时刻
- 分块：`wPrefillDone` = 最后一个分片完成时刻

### 6. Decode 阶段一致性

**问题**：（确认）Decode 阶段在两种 prefill 子模式下完全一致。

**响应**：agg 模式的 decode 逻辑直接复用 `DecodeInstance` 的 `decodeStepMs`、`retractDecode`、`reservedOffset`、ntr 衰减等机制，确保与 pd-disagg 的 decode 行为一致。

## 改造方案

### 总体思路

采用 Issue 中推荐的 **Alternative B: Mode flag inside existing SimEngine**。在 `SimParams` 中增加 `mode: "pd-disagg" | "agg"` 字段，`SimEngine.step()` 根据 mode 分支执行不同逻辑。

v1 方案采用 `slots + prefillDone + running` 三队列设计，本次修订改为**统一 `running` 批次**设计，以准确模拟 SGLang make_batch 的混合批处理能力。

### 详细设计

#### 1. 类型变更 (`shared/types.ts`)

**1.1 新增 `SimMode` 类型**

```typescript
export type SimMode = "pd-disagg" | "agg";
```

**1.2 `SimParams` 新增字段**

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | `SimMode` | 部署模式，默认 `"pd-disagg"` |
| `numWorkers` | `number` | agg 模式下 worker 实例数（取代 numP + numD） |
| `kvGb` | `number` | agg 模式下统一 KV 预算 GB（取代 kvGbP + kvGbD） |
| `chunkedPrefill` | `boolean` | agg 模式下是否开启分块 prefill，默认 `false` |

**1.3 `ReqStage` 扩展**

新增 agg 模式专用阶段：

```typescript
export type ReqStage =
  // 现有 pd-disagg 阶段
  | "tokenize" | "p_bootstrap" | "p_waiting" | "p_prefill"
  | "p_transfer" | "d_waiting" | "d_running" | "response" | "done"
  // 新增 agg 阶段
  | "w_waiting"          // 等待 make_batch 准入
  | "w_prefill"          // 非分块 prefill（单轮迭代）
  | "w_chunked_prefill"  // 分块 prefill（多轮迭代）
  | "w_decode";          // decode 阶段
```

**1.4 `RequestStamps` 扩展**

新增 agg 模式时间戳字段：

| 字段 | 说明 |
|---|---|
| `wQueueExit` | agg 模式下离开 waitingQ 时刻（make_batch 准入，KV 预分配完成） |
| `wPrefillDone` | agg 模式下 prefill 完成时刻（首 token 采样时刻） |

这些字段在 pd-disagg 模式下保持 `NaN`，不影响现有逻辑。

**1.5 `SimRequest` 新增字段**

| 字段 | 类型 | 说明 |
|---|---|---|
| `w` | `any` (WorkerInstance) | agg 模式下所属 worker 引用 |
| `chunkOffset` | `number` | 分块 prefill 当前偏移量（非分块模式不用） |

**1.6 `Gauges` 扩展**

```typescript
export interface Gauges {
  // 现有字段
  pQueue: number; dQueue: number; running: number;
  kvP: number; kvD: number; kvDpre: number;
  link: number; inflight: number;
  // 新增 agg 字段
  wQueue: number;   // agg 模式下 worker 等待队列总深度
  kvW: number;      // agg 模式下 worker KV 利用率
}
```

pd-disagg 模式下 `wQueue` 和 `kvW` 为 0。

#### 2. 常量变更 (`shared/constants.ts`)

新增 agg 模式专用 breakdown keys（4 列，对应 SGLang 聚合模式的 TTFT 分解）：

```typescript
export const BD_KEYS_DISAGG = ["tokenize","bootstrap","pQueue","prefill","transfer","dQueue","detok"] as const;
export const BD_KEYS_AGG    = ["tokenize","queue","prefill","detok"] as const;
// 保持向后兼容
export const BD_KEYS = BD_KEYS_DISAGG;
```

agg 模式 4 列的含义：
- `tokenize`：Router 分词耗时
- `queue`：waitingQ 等待 + RadixCache 匹配 + BlockManager 预分配 + make_batch 准入
- `prefill`：GPU prefill 计算（非分块为单轮，分块为多轮累计）
- `detok`：首 token detokenize 耗时

注：v1 方案有 5 列（含 `decode` 列），但 agg 模式下 prefill 完成后 decode 立即在同一批次开始，`decode` 列恒为 0，故移除。首 token 由 prefill 末轮采样产出（而非独立 decode 步骤），TTFT = tokenize + queue + prefill + detok。

`MetricsCollector` 根据 `engine.P.mode` 选择使用哪个 BD_KEYS 变体。

`SERIES_KEYS` 扩展：

```typescript
export const SERIES_KEYS = [
  "ttft","tpot","e2e","rps","tps",
  "pQueue","dQueue","running","kvP","kvD","kvDpre","dHandshake","link","inflight",
  "wQueue","kvW"  // 新增
] as const;
```

#### 3. 新实体 `WorkerInstance` (`entities/WorkerInstance.ts`)

agg 模式下的统一 worker，采用**统一 running 批次**设计（替代 v1 的三队列设计），模拟 SGLang make_batch 混合批处理。

```typescript
export class WorkerInstance {
  id: number;
  waitingQ: SimRequest[] = [];      // 等待 make_batch 准入（含 RadixCache 匹配 + KV 预分配检查）
  running: SimRequest[] = [];       // 统一运行批次：混合 w_prefill / w_chunked_prefill / w_decode
  kvUsed = 0;
  draining = false;
  ntr = 0;
  retractGlow = -1e9;
  nextStepAt: number | null = null;

  constructor(id: number) { this.id = id; }

  maxTokens(P: SimParams): number {
    return Math.max(1, Math.floor(P.kvGb * 2**30 / cellSizeOf(P)));
  }

  /** Decode 步延迟：仅统计 w_decode 请求的 KV 读取代价（复用 DecodeInstance 逻辑） */
  decodeStepMs(P: SimParams): number {
    const perTokMs = P.decodeMsPerReq / PERREQ_REF_CTX;
    let kvMs = 0;
    for (const r of this.running)
      if (r.stage === "w_decode") kvMs += perTokMs * (r.inputLen + r.tokensOut);
    return P.decodeMsBase + kvMs;
  }

  /**
   * 统一步延迟（make_batch 的 GPU 迭代时间）：
   * - 若批次含 prefill 请求：step = max(prefill_compute, decode_step)
   *   模拟 prefill（计算密集）与 decode（访存密集）在 GPU 上的并行重叠
   * - 若批次仅含 decode 请求：step = decode_step
   */
  stepLatencyMs(P: SimParams): number {
    let prefillMs = 0;
    for (const r of this.running) {
      if (r.stage === "w_prefill") {
        // 非分块：单轮处理全部 uncachedLen
        prefillMs = Math.max(prefillMs, fullPrefillMs(P, r));
      } else if (r.stage === "w_chunked_prefill") {
        // 分块：当前分片的计算时间
        prefillMs = Math.max(prefillMs, chunkPrefillMs(P, r, r.chunkOffset));
      }
    }
    const decMs = this.decodeStepMs(P);
    return prefillMs > 0 ? Math.max(prefillMs, decMs) : decMs;
  }

  pendingLoad(): number {
    return this.waitingQ.length + this.running.length;
  }

  isEmpty(): boolean {
    return this.waitingQ.length === 0 && this.running.length === 0;
  }

  /** KV headroom reserved for in-flight decodes（复用 DecodeInstance 逻辑） */
  reservedOffset(): number {
    let off = 0;
    for (const r of this.running)
      if (r.stage === "w_decode")
        off += Math.min(r.outputLen - r.tokensOut, NTR_CLIP) * this.ntr;
    return off;
  }

  /** SGLang retract_decode：抢占 decode 最少的请求（复用 DecodeInstance 逻辑） */
  retractDecode(P: SimParams, engine: ISimEngine, now: number): number {
    const cap = this.maxTokens(P);
    let count = 0;
    while (this.running.length > 1 && this.kvUsed + this.running.length > cap) {
      let vi = 0;
      for (let i = 1; i < this.running.length; i++) {
        const a = this.running[i], b = this.running[vi];
        // 仅抢占 w_decode 请求；prefill 请求不可抢占（正在计算中）
        if (a.stage !== "w_decode") continue;
        if (b.stage !== "w_decode") { vi = i; continue; }
        if (a.tokensOut < b.tokensOut ||
            (a.tokensOut === b.tokensOut && a.inputLen > b.inputLen)) vi = i;
      }
      if (this.running[vi].stage !== "w_decode") break;
      const victim = this.running[vi];
      this.kvUsed -= (victim.inputLen + victim.tokensOut);
      this.running.splice(vi, 1);
      victim.retracted = true;
      victim.stage = "w_waiting";
      this.waitingQ.unshift(victim);
      count++;
    }
    if (count) {
      this.raiseNtrAfterRetract();
      this.retractGlow = now;
      engine.retractTotal += count;
    }
    return count;
  }

  raiseNtrAfterRetract(): void { /* 同 DecodeInstance */ }
}
```

核心设计要点：
- **统一 running 批次**：替代 v1 的 `slots + prefillDone + running` 三队列。PREFILL/CHUNKED_PREFILL 和 DECODE 请求共处同一 `running` 数组，模拟 make_batch 混合批处理
- **无 TransferLink**：prefill 完成后请求状态直接在 `running` 内从 `w_prefill`/`w_chunked_prefill` 切换为 `w_decode`，KV 留在原池中
- **步延迟并行模型**：`step = max(prefill_compute, decode_step)`，模拟 SGLang 混合批处理中 prefill 与 decode 的 GPU 并行
- **retract 逻辑**：同 DecodeInstance，仅抢占 `w_decode` 请求（prefill 请求正在计算中，不可抢占）

#### 4. 引擎变更 (`sim/SimEngine.ts`)

**4.1 新增字段**

```typescript
wList: WorkerInstance[] = [];
rrW: RRCounter = { i: 0 };
wSeq = 0;
```

**4.2 `syncTopology()` 扩展**

```typescript
syncTopology(): void {
  const P = this.P;
  if (P.mode === "agg") {
    while (this.wList.length < P.numWorkers) this.wList.push(new WorkerInstance(this.wSeq++));
    for (let i = 0; i < this.wList.length; i++) this.wList[i].draining = i >= P.numWorkers;
    while (this.wList.length > P.numWorkers && this.wList[this.wList.length - 1].isEmpty())
      this.wList.pop();
  } else {
    // 现有 pList/dList 同步逻辑不变
    while (this.pList.length < P.numP) this.pList.push(new PrefillInstance(this.pSeq++));
    while (this.dList.length < P.numD) this.dList.push(new DecodeInstance(this.dSeq++));
    // ... 现有 draining / pop 逻辑
  }
}
```

**4.3 `step()` 分支**

```typescript
step(): void {
  this.syncTopology();
  if (this.P.mode === "agg") {
    this.stepAgg();
  } else {
    this.stepDisagg();  // 现有逻辑原封不动移入此方法
  }
  this.inflightIntegral += this.allActive.size * TICK;
  this.metrics.tick(this.now, this);
}
```

将现有 `step()` 体（不含末尾 inflightIntegral / metrics.tick）重命名为 `stepDisagg()`，不做任何修改。

**4.4 `stepAgg()` 核心逻辑**

```
1. 请求到达 → tokenize → 选择 worker（LB） → worker.waitingQ
   - 无 bootstrap 握手，无 D 端预分配，无 TransferLink
   - RadixCache 前缀匹配已在 makeRequest 中完成（cachedLen/uncachedLen）
   - 请求仅关联 w（worker），不关联 p/d

2. Per worker make_batch + GPU step:
   2a. retract 检查（KV 超限时抢占 w_decode 请求）

   2b. make_batch 准入：从 waitingQ 取请求 → 检查 KV 容量 + maxRunning
       - KV 准入条件：kvUsed + inputLen <= cap（BlockManager 一次性预分配完整 inputLen）
       - 准入后设置 wQueueExit 时间戳
       - chunkedPrefill=false: stage = "w_prefill", chunksTotal = 1
       - chunkedPrefill=true:  stage = "w_chunked_prefill", chunkOffset = 0,
                               chunksTotal = ceil(uncachedLen / chunkSize)

   2c. GPU step 循环（while nextStepAt <= now）:
       - 计算 stepLatencyMs = max(prefill_compute, decode_step)
       - 遍历 running 中每个请求：
         * w_prefill: 单轮 prefill 完成 → 采样首 token → stage = "w_decode"
           wPrefillDone = stepAt; firstToken = stepAt + detokenizeMs; tokensOut = 1
         * w_chunked_prefill: chunkOffset++ → 若 >= chunksTotal:
           全部分片完成 → 采样首 token → stage = "w_decode"
           wPrefillDone = stepAt; firstToken = stepAt + detokenizeMs; tokensOut = 1
         * w_decode: tokensOut++ → 若 >= outputLen:
           lastToken = stepAt + detokenizeMs; stage = "response"; 移入 responding
           kvUsed -= (inputLen + tokensOut)
       - ntr 衰减（无 retract 时）
       - 再次 make_batch 准入（同 2b）
       - nextStepAt = stepAt + stepLatencyMs

3. 响应完成:
   - 同现有逻辑，detok → done → 从 allActive 移除 → metrics.record(r, now, "agg")
```

关键设计要点：
- **make_batch 与 GPU step 耦合**：每次 GPU step 前执行 make_batch 准入，模拟 SGLang 每轮调度的 make_batch → forward → state_update 循环
- **混合批处理步延迟**：`max(prefill_compute, decode_step)` 模拟 prefill 与 decode 在同一 GPU 迭代中的并行重叠
- **首 token 时机**：无论非分块还是分块，首 token 均在 prefill 完成（`wPrefillDone`）后采样，加 detokenize 延迟后到达客户端

**4.5 `sampleGauges()` 扩展**

```typescript
sampleGauges(): Gauges {
  if (this.P.mode === "agg") {
    let wQueue = 0, running = 0, kvWu = 0, kvWc = 0;
    for (const w of this.wList) {
      wQueue += w.waitingQ.length;
      running += w.running.length;
      kvWu += w.kvUsed; kvWc += w.maxTokens(this.P);
    }
    return {
      pQueue: 0, dQueue: 0, running,
      kvP: 0, kvD: 0, kvDpre: 0,
      link: 0, inflight: this.allActive.size,
      wQueue, kvW: kvWc ? kvWu / kvWc : 0,
    };
  }
  // 现有 pd-disagg 逻辑不变...
}
```

#### 5. 请求创建变更 (`entities/Request.ts`)

`makeRequest()` 新增字段初始化：

```typescript
return {
  ...现有字段,
  w: null,
  chunkOffset: 0,
  stamps: { ...现有字段,
            wQueueExit: NaN, wPrefillDone: NaN },
};
```

agg 模式下 `r.w` 由 `stepAgg` 设置，`r.p`/`r.d` 保持 `null`。

#### 6. 指标变更 (`sim/MetricsCollector.ts`)

**6.1 `record()` 模式感知**

```typescript
record(r: SimRequest, now: number, mode: SimMode): void {
  const s = r.stamps;
  const ttft = s.firstToken - s.recv;
  const tpot = r.outputLen > 1 ? (s.lastToken - s.firstToken) / (r.outputLen - 1) : 0;
  const e2e = s.detokDone - s.recv;

  let bd: number[];
  if (mode === "agg") {
    bd = [
      s.tokenized - s.recv,           // tokenize
      s.wQueueExit - s.tokenized,     // queue (waiting + RadixCache + pre-alloc + make_batch)
      s.wPrefillDone - s.wQueueExit,  // prefill (单轮或分块累计)
      s.firstToken - s.wPrefillDone,  // detok (首 token detokenize)
    ];
  } else {
    bd = [ /* 现有 7 列计算不变 */ ];
  }
  // dHandshake 仅 pd-disagg 模式有意义
  const dHandshake = mode === "agg" ? 0 : s.preallocDone - s.tokenized;
  // ... 其余 ring/bucket 逻辑不变
}
```

**6.2 `tick()` 和 `recentBreakdown()` 适配**

- `tick()` 调用 `engine.sampleGauges()` 时根据 mode 写入对应 series keys：
  - agg 模式：`put("wQueue", g.wQueue); put("kvW", g.kvW);` 替代 `pQueue/dQueue/kvP/kvD`
  - agg 模式下 `pQueue`/`dQueue`/`kvP`/`kvD`/`kvDpre`/`link`/`dHandshake` 写入 0
- `bdSeries` 数组长度根据 mode 对应的 BD_KEYS 长度动态确定（agg=4, pd-disagg=7）
- `reset()` 时根据 mode 选择对应 BD_KEYS 初始化 `bkBd` 和 `bdSeries`

注：`MetricsCollector` 需在 mode 切换时重新 reset，由 `SimService.handleParams` 检测 mode 变更后触发 `engine.reset()` 保证。

#### 7. 服务层变更 (`sim/SimService.ts`)

**7.1 `getRenderState()` 模式感知**

agg 模式下序列化 `wList` 替代 `pList/dList`：

```typescript
getRenderState(): any {
  const eng = this.engine;
  if (eng.P.mode === "agg") {
    const sreq = (r: SimRequest) => serializeReqAgg(r);
    return {
      now: eng.now, P: eng.P, retractTotal: eng.retractTotal,
      wList: eng.wList.map(w => ({
        id: w.id, kvUsed: w.kvUsed, draining: w.draining,
        maxTokens: w.maxTokens(eng.P), ntr: w.ntr, retractGlow: w.retractGlow,
        nextStepAt: w.nextStepAt,
        waitingQ: w.waitingQ.map(sreq),
        running: w.running.map(sreq),
      })),
      pList: [], dList: [],
      allActive: Array.from(eng.allActive).map(sreq),
      responding: eng.responding.map(sreq),
    };
  }
  // 现有 pd-disagg 逻辑不变
}
```

`serializeReqAgg` 序列化 agg 模式请求（含 stage、chunkOffset、tokensOut 等）。

**7.2 `handleParams()` 增强**

当 `mode` 参数变更时，需要 `reset()` + `syncTopology()`，因为拓扑结构根本改变：

```typescript
private handleParams(body: SimParamsRequest): void {
  const prevMode = this.engine.P.mode;
  Object.assign(this.engine.P, body.params);
  if (body.params.mode && body.params.mode !== prevMode) {
    this.engine.reset();  // 模式切换必须重置（含 metrics.reset 重新初始化 bdSeries）
  }
}
```

#### 8. 预设变更 (`shared/presets.ts`)

**8.1 DEFAULTS 扩展**

```typescript
export const DEFAULTS: SimParams = {
  ...现有字段,
  mode: "pd-disagg",       // 向后兼容默认
  numWorkers: 2,            // agg 模式默认 worker 数
  kvGb: 99,                // agg 模式默认 KV 预算
  chunkedPrefill: false,   // agg 模式默认非分块
};
```

**8.2 新增 agg 预设**

```typescript
aggBalanced: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 2, kvGb: 99, chunkedPrefill: false,
  lbPolicyP: "least" as LBPolicy,  // 复用为 worker LB
},
aggChunkedPrefill: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 2, kvGb: 99, chunkedPrefill: true,
  chunkSize: 8192, inputLenMean: 8192,
},
aggDecodeHeavy: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 1, kvGb: 141, outputLenMean: 1024, chunkedPrefill: false,
},
aggHighQps: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 4, kvGb: 99, qps: 16, chunkedPrefill: false,
},
```

**8.3 PARAM_DEFS 扩展**

新增以下参数定义：

```typescript
{group:"topology", key:"mode", type:"select", options:["pd-disagg","agg"], i18nPrefix:"mode."},
{group:"topology", key:"numWorkers", min:1, max:8, step:1, fmt:v=>v.toFixed(0)},
{group:"kv", key:"kvGb", min:1, max:288, step:1, fmt:v=>v.toFixed(0)},
{group:"compute", key:"chunkedPrefill", type:"toggle", i18nPrefix:"prefill."},
```

前端侧栏根据 mode 显示/隐藏相关参数：
- `pd-disagg` 模式：显示 `numP`, `numD`, `lbPolicyP`, `lbPolicyD`, `kvGbP`, `kvGbD`, `bandwidthGBs`, `transferOverheadMs`
- `agg` 模式：显示 `numWorkers`, `lbPolicyP`（复用为 worker LB）, `kvGb`, `chunkedPrefill`；隐藏上述 pd-disagg 专属参数

#### 9. 国际化变更 (`shared/i18n.ts`)

新增 label：

| Key | zh | en |
|---|---|---|
| `mode.pd-disagg` | PD 分离 | PD Disaggregated |
| `mode.agg` | 聚合 | Aggregated |
| `p.mode` | 部署模式 | Deploy mode |
| `p.numWorkers` | Worker 实例数 | # Worker instances |
| `p.kvGb` | KV 显存 (GB) | KV memory (GB) |
| `p.chunkedPrefill` | 分块 Prefill | Chunked Prefill |
| `prefill.on` | 开启 | Enabled |
| `prefill.off` | 关闭 | Disabled |
| `bd.queue` | 排队 | Queue |
| `bd.prefill` | Prefill 计算 | Prefill compute |
| `bd.detok` | Detok | Detok |
| `n.worker` | Worker | Worker |
| `m.wQueue` | Worker 排队 | Worker queue |
| `m.kvW` | KV 利用率 (W) | KV util (Worker) |
| `s.w_prefill` | Prefill | Prefill |
| `s.w_chunked_prefill` | 分块 Prefill | Chunked Prefill |
| `s.w_decode` | Decode | Decode |
| `s.w_waiting` | 等待中 | Waiting |
| `preset.aggBalanced` | 聚合-均衡 | Agg-Balanced |
| `preset.aggChunkedPrefill` | 聚合-分块Prefill | Agg-Chunked-Prefill |
| `preset.aggDecodeHeavy` | 聚合-Decode密集 | Agg-Decode-Heavy |
| `preset.aggHighQps` | 聚合-高QPS | Agg-High-QPS |
| `info.capW` | Worker 容量 | Worker capacity |

#### 10. 前端变更 (`public/pd-disagg.html`)

**10.1 Topbar 新增模式切换**

在 topbar 中增加 mode 切换按钮组（`pd-disagg` | `agg`），切换时调用 `/api/params` 更新 mode 并触发重置。

**10.2 Renderer 适配**

现有 `Renderer.draw()` 根据 `sim.P.mode` 分支：
- `pd-disagg`：保持现有双列 P/D + TransferLink 动画布局
- `agg`：单列 Worker 布局，每个 Worker 实例内展示 `waitingQ → running` 视图，running 内按 stage 着色区分 prefill/chunked_prefill/decode

新增 `drawWorker(sim, w, r, idx, now)` 方法，绘制单个 Worker 实例：
- 标题行：`Worker {id}` + ntr 指标
- 队列行：waitingQ（灰色） → running（按 stage 着色：w_prefill=蓝, w_chunked_prefill=紫, w_decode=绿）
- 底部：KV 利用率 gauge

**10.3 侧栏参数联动**

根据 `sim.P.mode` 动态显示/隐藏参数行：
- `mode === "agg"` 时隐藏：`numP`, `numD`, `lbPolicyD`, `kvGbP`, `kvGbD`, `bandwidthGBs`, `transferOverheadMs`
- `mode === "agg"` 时显示：`numWorkers`, `kvGb`, `chunkedPrefill`
- `mode === "pd-disagg"` 时反之

**10.4 Breakdown 图表适配**

`drawBreakdown()` 根据 `sim.P.mode` 选择 BD_KEYS 和 COLORS：
- `pd-disagg`：7 列 `["tokenize","bootstrap","pQueue","prefill","transfer","dQueue","detok"]`
- `agg`：4 列 `["tokenize","queue","prefill","detok"]`

**10.5 Metrics 卡片适配**

agg 模式下：
- 隐藏 `kvP`, `kvD`, `kvDpre`, `dHandshake`, `link` 卡片
- 显示 `wQueue`, `kvW` 卡片
- `running` 卡片不变（统计 w_prefill + w_chunked_prefill + w_decode 总数）

**10.6 Preset 按钮扩展**

新增 agg 预设按钮组（aggBalanced / aggChunkedPrefill / aggDecodeHeavy / aggHighQps），与现有 pd-disagg 预设按钮区分。

**10.7 请求到达动画**

agg 模式下：请求到达 → router → 单列 Worker（无 P/D 双路由动画，无 TransferLink 动画）。

### 修改点清单

1. `shared/types.ts` — 新增 `SimMode` 类型；`SimParams` 新增 `mode/numWorkers/kvGb/chunkedPrefill`；`ReqStage` 新增 `w_waiting/w_prefill/w_chunked_prefill/w_decode`；`RequestStamps` 新增 `wQueueExit/wPrefillDone`；`SimRequest` 新增 `w/chunkOffset`；`Gauges` 新增 `wQueue/kvW`
2. `shared/constants.ts` — 新增 `BD_KEYS_AGG`（4 列）；`SERIES_KEYS` 新增 `wQueue/kvW`；保持 `BD_KEYS = BD_KEYS_DISAGG` 向后兼容
3. `entities/WorkerInstance.ts` — **新建**，统一 Worker 实体：`waitingQ` + `running`（混合批次），含 `stepLatencyMs`/`decodeStepMs`/`retractDecode`/`reservedOffset`
4. `sim/SimEngine.ts` — 新增 `wList/rrW/wSeq` 字段；`syncTopology()` 增加 agg 分支；`step()` 拆分为 `stepDisagg()/stepAgg()`；`stepAgg()` 实现 make_batch + 混合 GPU step + 状态转换；`sampleGauges()` 增加 agg 分支
5. `entities/Request.ts` — `makeRequest` 新增 `w: null`、`chunkOffset: 0`、stamps 新增 `wQueueExit/wPrefillDone` 初始值
6. `sim/MetricsCollector.ts` — `record()` 增加 mode 参数和 agg breakdown 计算（4 列）；`tick()` 增加 agg series 写入；`reset()` 根据 mode 初始化 bdSeries
7. `sim/SimService.ts` — `getRenderState()` 增加 agg 分支序列化 wList；`handleParams()` 检测 mode 变更触发 reset
8. `shared/presets.ts` — DEFAULTS 新增 `mode/numWorkers/kvGb/chunkedPrefill`；新增 4 个 agg 预设；PARAM_DEFS 新增 `mode/numWorkers/kvGb/chunkedPrefill` 定义
9. `shared/i18n.ts` — 新增 agg 相关 i18n label
10. `public/pd-disagg.html` — topbar 增加 mode 切换；Renderer 增加 `drawWorker()` 方法及 agg 渲染分支；侧栏参数联动；breakdown/metrics 卡片适配；预设按钮扩展

## 测试设计

### 验收测试用例清单

| # | 用例 | 验证点 |
|---|---|---|
| T1 | 默认启动 → 模式为 pd-disagg | DEFAULTS.mode === "pd-disagg"，所有现有功能正常 |
| T2 | 切换为 agg 模式（非分块） | engine 重置，wList 创建，pList/dList 为空，chunkedPrefill=false |
| T3 | agg 非分块 prefill 仿真 | 请求走 tokenize → w_waiting → w_prefill → w_decode → response → done；首 token 在 w_prefill 单步后产出 |
| T4 | agg 分块 prefill 仿真 | chunkedPrefill=true，请求走 w_chunked_prefill（多轮）→ w_decode；首 token 在全部分片完成后产出 |
| T5 | agg KV 容量约束 | 当 KV 满时请求排队等待，不超分配（kvUsed <= maxTokens） |
| T6 | agg retract | 当 decode 批次 KV 不足时触发抢占，被抢占请求回到 waitingQ；w_prefill/w_chunked_prefill 请求不被抢占 |
| T7 | agg metrics 正确性（非分块） | TTFT = tokenize + queue + prefill + detok，BD_KEYS 为 4 列 |
| T8 | agg metrics 正确性（分块） | TTFT = tokenize + queue + prefill(多轮累计) + detok |
| T9 | agg gauges 正确性 | wQueue/kvW 正确反映队列深度和 KV 利用率 |
| T10 | RadixCache 前缀复用 | cacheHitRate > 0 时 uncachedLen < inputLen，prefill 计算量减少但 KV 占用仍为完整 inputLen |
| T11 | BlockManager 预分配 | 请求准入时 kvUsed += inputLen（非 uncachedLen），分块 prefill 不降低总显存占用 |
| T12 | make_batch 混合批处理 | running 中同时存在 w_prefill/w_chunked_prefill 和 w_decode 请求，stepLatencyMs = max(prefill, decode) |
| T13 | 切换回 pd-disagg | engine 重置，pList/dList 重建，wList 为空，现有功能正常 |
| T14 | agg 预设加载 | 加载 aggBalanced/aggChunkedPrefill/aggDecodeHeavy/aggHighQps 后参数正确 |
| T15 | agg 模式 getRenderState | 返回 wList 序列化，pList/dList 为空数组 |
| T16 | agg 模式侧栏联动 | agg 模式下隐藏 numP/numD 等，显示 numWorkers/kvGb/chunkedPrefill |
| T17 | agg 模式 Renderer | 单列 Worker 布局，running 内按 stage 着色，无 TransferLink 动画 |
| T18 | agg 模式 breakdown | 4 列堆叠图，label 正确 |
| T19 | 多 worker 负载均衡 | agg 模式下 least/round_robin/power_of_two/random 均正常分发 |
| T20 | 分块 vs 非分块 TTFT 对比 | 相同负载下，非分块 TTFT < 分块 TTFT（长 Prompt 场景） |

### 边界条件覆盖

| # | 边界条件 | 预期行为 |
|---|---|---|
| B1 | `numWorkers = 1` 单 worker | 所有请求进同一 worker，KV 可能更早饱和 |
| B2 | `numWorkers` 动态增大/缩小 | syncTopology 正确标记 draining/新增 worker |
| B3 | `kvGb` 极小（如 1 GB） | KV 快速饱和，大量请求排队或 retract |
| B4 | 高 QPS + 长 output（agg decode 重载） | retract 频繁，ntr 上升，throughput 受限 |
| B5 | 模式切换时请求进行中 | reset() 清空所有状态，重新开始 |
| B6 | `mode` 参数非法值 | 忽略，保持当前模式 |
| B7 | agg 模式下 `chunkSize` = 1 token（分块开启） | 极小 chunk，prefill 步骤极多但不崩溃 |
| B8 | `chunkedPrefill` 切换 | 从 false → true 时 reset，分块逻辑生效 |
| B9 | `cacheHitRate = 1.0`（全命中） | uncachedLen = 0，prefill 瞬间完成，首 token 仅需 detok |
| B10 | `cacheHitRate = 0`（全未命中） | uncachedLen = inputLen，prefill 计算量最大 |
| B11 | 混合批次中仅 prefill 请求 | stepLatencyMs = prefill_compute（decode_step 无请求时仅 decodeMsBase） |
| B12 | 混合批次中仅 decode 请求 | stepLatencyMs = decodeStepMs（同 DecodeInstance） |

## 风险与注意事项

### 兼容性影响

- **向后兼容**：`mode` 默认值为 `"pd-disagg"`，现有行为完全不变。所有新增 `SimParams` 字段都有默认值，旧代码不传新字段也不受影响
- **API 兼容**：`/state` 和 `/render` 返回结构新增 `wList` 等字段，现有前端字段不变；agg 模式下 `pList/dList` 返回空数组而非省略，保证消费端不报错
- **前端兼容**：mode 切换是 UI 新增功能，不影响现有 pd-disagg 页面
- **Metrics 兼容**：`MetricsCollector.record()` 新增 `mode` 参数为可选参数（默认 `"pd-disagg"`），现有调用不需修改

### 性能影响

- `stepAgg()` 单步计算量与 `stepDisagg()` 相当，`stepLatencyMs` 计算为 O(running.length)，无额外性能开销
- `sampleGauges()` 仅多遍历 wList 或 pList/dList，O(N) 无变化
- MetricsCollector 的 `record()` 增加 mode 判断分支，开销可忽略

### 回滚方案

- 所有变更通过 `mode` 字段控制，设为 `"pd-disagg"` 即完全回退到现有行为
- 新增的 `WorkerInstance.ts` 文件仅 agg 模式引用，不影响现有代码路径
- 若需完全回滚，删除 `WorkerInstance.ts`，移除 `SimParams` 中 4 个新字段及 `stepAgg()` 分支即可
