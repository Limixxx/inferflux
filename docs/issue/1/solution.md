---
title: "Issue #1 解决方案"
issue_number: 1
issue_type: Feature
created: 2026-08-20
updated: 2026-08-20
status: draft
review_round: 1
---

# Issue #1 解决方案

## 需求分析

### 问题描述

当前 `server/` TypeScript 项目仅支持 **PD-disaggregated（PD 分离）** 模式模拟，即 Prefill（P）和 Decode（D）运行在物理分离的 worker 实例上，通过 KV cache transfer link 连接。但最常见的部署模式——**aggregated（聚合）** 模式（单个 worker 同时处理 prefill 和 decode，共享一个 KV 缓存池，无跨实例传输）完全缺失。

用户无法在相同负载条件下对比 PD 分离与聚合架构的性能差异，无法回答诸如"PD 分离对我的负载是否值得传输开销"等关键问题。

### 能力目标

1. 在现有 SimEngine 中新增 `agg` 模式，使模拟器支持 `pd-disagg` 和 `agg` 两种部署架构
2. Agg 模式下请求生命周期简化为 6 阶段：`tokenize → waiting → prefill → decode → response → done`，无 bootstrap 握手、无 KV 传输、无 P/D 分离
3. 指标体系（TTFT 分解、Gauges、时序图）自适应模式切换
4. 前端可视化适配 agg 模式（单列 worker 布局、侧栏参数切换、breakdown 图表适配）
5. 保持 pd-disagg 模式完全兼容，默认值不变

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

## 改造方案

### 总体思路

采用 Issue 中推荐的 **Alternative B: Mode flag inside existing SimEngine**。在 `SimParams` 中增加 `mode: "pd-disagg" | "agg"` 字段，`SimEngine.step()` 根据 mode 分支执行不同逻辑。这种方式最大化代码复用（metrics、LB、HTTP、前端基础设施全部共享），单一真相源，且通过一个参数切换即可实现 A/B 对比。

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

**1.3 `ReqStage` 扩展**

新增 agg 模式专用阶段：

```typescript
export type ReqStage =
  // 现有 pd-disagg 阶段
  | "tokenize" | "p_bootstrap" | "p_waiting" | "p_prefill"
  | "p_transfer" | "d_waiting" | "d_running" | "response" | "done"
  // 新增 agg 阶段
  | "w_waiting" | "w_prefill" | "w_decode";
```

**1.4 `RequestStamps` 扩展**

新增 agg 模式时间戳字段：

| 字段 | 说明 |
|---|---|
| `wQueueExit` | agg 模式下离开等待队列时刻 |
| `wPrefillDone` | agg 模式下 prefill 完成时刻 |
| `wDecodeExit` | agg 模式下进入 decode 时刻（即 firstToken 时刻） |

这些字段在 pd-disagg 模式下保持 `NaN`，不影响现有逻辑。

**1.5 `SimRequest` 新增字段**

| 字段 | 类型 | 说明 |
|---|---|---|
| `w` | `any` (WorkerInstance) | agg 模式下所属 worker 引用 |

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

新增 agg 模式专用 breakdown keys：

```typescript
export const BD_KEYS_DISAGG = ["tokenize","bootstrap","pQueue","prefill","transfer","dQueue","detok"] as const;
export const BD_KEYS_AGG    = ["tokenize","queue","prefill","decode","detok"] as const;
// 保持向后兼容
export const BD_KEYS = BD_KEYS_DISAGG;
```

`MetricsCollector` 根据 engine.P.mode 选择使用哪个 BD_KEYS 变体。

#### 3. 新实体 `WorkerInstance` (`entities/WorkerInstance.ts`)

agg 模式下的统一 worker，同时负责 prefill 和 decode，共享单一 KV 缓存池。

```typescript
export class WorkerInstance {
  id: number;
  waitingQ: SimRequest[] = [];      // 等待 prefill 的队列
  slots: PrefillSlot[] = [];        // 活跃 prefill 槽位（复用现有 PrefillSlot 接口）
  prefillDone: SimRequest[] = [];   // prefill 完成，等待 decode 批次准入
  running: SimRequest[] = [];       // 活跃 decode 批次
  kvUsed = 0;
  draining = false;
  ntr = 0;
  retractGlow = -1e9;
  nextStepAt: number | null = null;

  constructor(id: number) { this.id = id; }

  maxTokens(P: SimParams): number {
    return Math.max(1, Math.floor(P.kvGb * 2**30 / cellSizeOf(P)));
  }

  decodeStepMs(P: SimParams): number {
    // 复用 DecodeInstance.decodeStepMs 的逻辑
    const perTokMs = P.decodeMsPerReq / PERREQ_REF_CTX;
    let kvMs = 0;
    for (const r of this.running) kvMs += perTokMs * (r.inputLen + r.tokensOut);
    return P.decodeMsBase + kvMs;
  }

  pendingLoad(): number {
    return this.waitingQ.length + this.slots.length
         + this.prefillDone.length + this.running.length;
  }

  isEmpty(): boolean {
    return this.waitingQ.length === 0 && this.slots.length === 0
        && this.prefillDone.length === 0 && this.running.length === 0;
  }

  reservedOffset(): number {
    // 同 DecodeInstance.reservedOffset
    let off = 0;
    for (const r of this.running)
      off += Math.min(r.outputLen - r.tokensOut, NTR_CLIP) * this.ntr;
    return off;
  }
}
```

核心设计要点：
- **无 TransferLink**：prefill 完成后请求直接移入 `prefillDone` 队列，等待 decode 批次准入
- **单一 KV 池**：prefill 消耗 KV → decode 复用同一 KV → 请求完成后整体释放
- **retract 逻辑**：同 DecodeInstance，当 KV 不足时抢占 decode 最少的请求

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
  if (this.P.mode === "agg") {
    while (this.wList.length < P.numWorkers) this.wList.push(new WorkerInstance(this.wSeq++));
    for (let i = 0; i < this.wList.length; i++) this.wList[i].draining = i >= P.numWorkers;
    while (this.wList.length > P.numWorkers && this.wList[this.wList.length - 1].isEmpty())
      this.wList.pop();
  } else {
    // 现有 pList/dList 同步逻辑不变
  }
}
```

**4.3 `step()` 分支**

```typescript
step(): void {
  this.syncTopology();
  if (this.P.mode === "agg") this.stepAgg();
  else this.stepDisagg();  // 现有逻辑原封不动移入此方法
}
```

将现有 `step()` 体重命名为 `stepDisagg()`，不做任何修改。

**4.4 `stepAgg()` 核心逻辑**

```
1. 请求到达 → tokenize → 选择 worker（LB） → worker.waitingQ
   - 无 bootstrap 握手，无 D 端预分配
   - 请求仅关联 w（worker），不关联 p/d

2. Per worker prefill:
   - 从 waitingQ 取请求 → 检查 KV 容量 → 放入 slots
   - chunked prefill 同 pd-disagg（复用 chunkPrefillMs）
   - prefill 完成 → 请求移入 prefillDone 队列
   - KV 仍留在当前 worker 的池中（无传输）

3. Per worker decode 准入:
   - 从 prefillDone 取请求 → 检查 running 容量 + KV 容量
   - 准入条件：running.length < maxRunning && kvUsed + joinKv <= cap
   - 准入后设置 firstToken 时间戳

4. Per worker decode 循环:
   - 同 DecodeInstance 的 decode 循环
   - 包含 retract 逻辑（KV 不足时抢占）
   - ntr 衰减同现有逻辑
   - 请求完成 → 从 running 移除 → 释放 KV → 加入 responding

5. 响应完成:
   - 同现有逻辑，detok → done → 从 allActive 移除 → metrics.record()
```

**4.5 `sampleGauges()` 扩展**

```typescript
sampleGauges(): Gauges {
  if (this.P.mode === "agg") {
    let wQueue = 0, running = 0, kvWu = 0, kvWc = 0;
    for (const w of this.wList) {
      wQueue += w.waitingQ.length + w.prefillDone.length;
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

`makeRequest()` 无需大幅修改。agg 模式下：
- `r.w = worker`（由 stepAgg 设置）
- `r.p = null; r.d = null`
- `r.kvPoll = null`（agg 无 KV poll）
- `r.dPrealloc = true`（始终为 true，agg 模式下无 prealloc 等待）
- `stamps` 新增字段初始为 `NaN`

#### 6. 指标变更 (`sim/MetricsCollector.ts`)

**6.1 `record()` 模式感知**

```typescript
record(r: SimRequest, now: number, mode: SimMode): void {
  // TTFT/TPOT/E2E 计算通用
  const s = r.stamps;
  const ttft = s.firstToken - s.recv;
  const tpot = r.outputLen > 1 ? (s.lastToken - s.firstToken) / (r.outputLen - 1) : 0;
  const e2e = s.detokDone - s.recv;

  let bd: number[];
  if (mode === "agg") {
    bd = [
      s.tokenized - s.recv,           // tokenize
      s.wQueueExit - s.tokenized,     // queue
      s.wPrefillDone - s.wQueueExit,  // prefill
      s.wDecodeExit - s.wPrefillDone, // decode
      s.detokDone - s.lastToken,      // detok
    ];
  } else {
    bd = [ /* 现有 7 列计算不变 */ ];
  }
  // 其余逻辑不变
}
```

**6.2 `tick()` 和 `recentBreakdown()` 适配**

`tick()` 调用 `engine.sampleGauges()` 时需根据 mode 写入对应 series keys：
- agg 模式下 `put("wQueue", g.wQueue); put("kvW", g.kvW);` 替代 `pQueue/dQueue/kvP/kvD`
- `bdSeries` 数组长度根据 mode 对应的 BD_KEYS 长度动态确定

**6.3 `SERIES_KEYS` 扩展**

在 `constants.ts` 中：
```typescript
export const SERIES_KEYS = [
  "ttft","tpot","e2e","rps","tps",
  "pQueue","dQueue","running","kvP","kvD","kvDpre","dHandshake","link","inflight",
  "wQueue","kvW"  // 新增
] as const;
```

MetricsCollector 在 reset 时为所有 SERIES_KEYS 初始化 series 数组。

#### 7. 服务层变更 (`sim/SimService.ts`)

**7.1 `getRenderState()` 模式感知**

agg 模式下序列化 `wList` 替代 `pList/dList`：

```typescript
getRenderState(): any {
  const eng = this.engine;
  if (eng.P.mode === "agg") {
    return {
      now: eng.now, P: eng.P, retractTotal: eng.retractTotal,
      wList: eng.wList.map(w => ({
        id: w.id, kvUsed: w.kvUsed, draining: w.draining,
        maxTokens: w.maxTokens(eng.P), ntr: w.ntr, retractGlow: w.retractGlow,
        waitingQ: w.waitingQ.map(sreq),
        slots: w.slots.map(s => ({ req: sreq(s.req), busyUntil: s.busyUntil })),
        prefillDone: w.prefillDone.map(sreq),
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

**7.2 `handleParams()` 增强**

当 `mode` 参数变更时，需要 `reset()` + `syncTopology()`，因为拓扑结构根本改变：

```typescript
private handleParams(body: SimParamsRequest): void {
  const prevMode = this.engine.P.mode;
  Object.assign(this.engine.P, body.params);
  if (body.params.mode && body.params.mode !== prevMode) {
    this.engine.reset();  // 模式切换必须重置
  }
}
```

**7.3 `handlePreset()` 增强**

预设切换时自动处理 mode。

#### 8. 预设变更 (`shared/presets.ts`)

**8.1 DEFAULTS 扩展**

```typescript
export const DEFAULTS: SimParams = {
  ...现有字段,
  mode: "pd-disagg",     // 向后兼容默认
  numWorkers: 2,          // agg 模式默认 worker 数
  kvGb: 99,              // agg 模式默认 KV 预算
};
```

**8.2 新增 agg 预设**

```typescript
aggBalanced: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 2, kvGb: 99,
  lbPolicyP: "least" as LBPolicy,  // 复用为 worker LB
},
aggDecodeHeavy: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 1, kvGb: 141, outputLenMean: 1024,
},
aggHighQps: {
  ...DEFAULTS,
  mode: "agg", numWorkers: 4, kvGb: 99, qps: 16,
},
```

**8.3 PARAM_DEFS 扩展**

新增以下参数定义：

```typescript
{group:"topology", key:"mode", type:"select", options:["pd-disagg","agg"], i18nPrefix:"mode."},
{group:"topology", key:"numWorkers", min:1, max:8, step:1, fmt:v=>v.toFixed(0)},
{group:"kv", key:"kvGb", min:1, max:288, step:1, fmt:v=>v.toFixed(0)},
```

前端侧栏根据 mode 显示/隐藏相关参数：
- `pd-disagg` 模式：显示 `numP`, `numD`, `lbPolicyP`, `lbPolicyD`, `kvGbP`, `kvGbD`, `bandwidthGBs`, `transferOverheadMs`
- `agg` 模式：显示 `numWorkers`, `lbPolicyP`（复用为 worker LB）, `kvGb`；隐藏 `numP`, `numD`, `lbPolicyD`, `kvGbP`, `kvGbD`, `bandwidthGBs`, `transferOverheadMs`

#### 9. 国际化变更 (`shared/i18n.ts`)

新增 label：

| Key | zh | en |
|---|---|---|
| `mode.pd-disagg` | PD 分离 | PD Disaggregated |
| `mode.agg` | 聚合 | Aggregated |
| `p.mode` | 部署模式 | Deploy mode |
| `p.numWorkers` | Worker 实例数 | # Worker instances |
| `p.kvGb` | KV 显存 (GB) | KV memory (GB) |
| `bd.queue` | 排队 | Queue |
| `bd.prefill` | Prefill 计算 | Prefill compute |
| `bd.decode` | Decode 计算 | Decode compute |
| `n.worker` | Worker | Worker |
| `m.wQueue` | Worker 排队 | Worker queue |
| `m.kvW` | KV 利用率 (W) | KV util (Worker) |
| `q.prefillDone` | 等待 decode | Pending decode |
| `preset.aggBalanced` | 聚合-均衡 | Agg-Balanced |
| `preset.aggDecodeHeavy` | 聚合-Decode 密集 | Agg-Decode-Heavy |
| `preset.aggHighQps` | 聚合-高 QPS | Agg-High-QPS |
| `info.capW` | Worker 容量 | Worker capacity |

#### 10. 前端变更 (`public/pd-disagg.html`)

**10.1 Topbar 新增模式切换**

在 topbar 中增加 mode 切换按钮组（`pd-disagg` | `agg`），切换时调用 `/api/params` 更新 mode 并触发重置。

**10.2 Renderer 适配**

现有 `Renderer.draw()` 根据 `sim.P.mode` 分支：
- `pd-disagg`：保持现有双列 P/D + TransferLink 动画布局
- `agg`：单列 Worker 布局，每个 Worker 实例内展示 `waitingQ → slots → prefillDone → running` 的堆叠视图，无 TransferLink 动画

新增 `drawWorker(sim, w, r, idx, now)` 方法，绘制单个 Worker 实例：
- 标题行：`Worker {id}` + ntr 指标
- 队列行：waiting → prefill → prefillDone → running
- 底部：KV 利用率 gauge

**10.3 侧栏参数联动**

根据 `sim.P.mode` 动态显示/隐藏参数行：
- `mode === "agg"` 时隐藏：`numP`, `numD`, `lbPolicyD`, `kvGbP`, `kvGbD`, `bandwidthGBs`, `transferOverheadMs`
- `mode === "agg"` 时显示：`numWorkers`, `kvGb`
- `mode === "pd-disagg"` 时反之

**10.4 Breakdown 图表适配**

`drawBreakdown()` 根据 `sim.P.mode` 选择 BD_KEYS 和 COLORS：
- `pd-disagg`：7 列 `["tokenize","bootstrap","pQueue","prefill","transfer","dQueue","detok"]`
- `agg`：5 列 `["tokenize","queue","prefill","decode","detok"]`

**10.5 Metrics 卡片适配**

agg 模式下：
- 隐藏 `kvP`, `kvD`, `kvDpre`, `dHandshake`, `link` 卡片
- 显示 `wQueue`, `kvW` 卡片
- `queue` 卡片显示 `wQueue` 值

**10.6 Preset 按钮扩展**

新增 agg 预设按钮组，与现有 pd-disagg 预设按钮区分。

**10.7 请求到达动画**

agg 模式下：请求到达 → router → 单列 Worker（无 P/D 双路由动画）

### 修改点清单

1. `shared/types.ts` — 新增 `SimMode` 类型，`SimParams` 新增 `mode/numWorkers/kvGb`，`ReqStage` 新增 agg 阶段，`RequestStamps` 新增 agg 时间戳，`SimRequest` 新增 `w` 字段，`Gauges` 新增 `wQueue/kvW`
2. `shared/constants.ts` — 新增 `BD_KEYS_AGG`，`SERIES_KEYS` 新增 `wQueue/kvW`
3. `entities/WorkerInstance.ts` — **新建**，实现统一 Worker 实体（waitingQ/slots/prefillDone/running/kvUsed/ntr/retract）
4. `sim/SimEngine.ts` — 新增 `wList/rrW/wSeq` 字段，`syncTopology()` 增加 agg 分支，`step()` 拆分为 `stepDisagg()/stepAgg()`，`sampleGauges()` 增加 agg 分支
5. `entities/Request.ts` — `makeRequest` 的 stamps 新增字段初始值
6. `sim/MetricsCollector.ts` — `record()` 增加 mode 参数和 agg breakdown 计算，`tick()` 增加 agg series 写入，`recentBreakdown()` 适配动态 BD_KEYS 长度
7. `sim/SimService.ts` — `getRenderState()` 增加 agg 分支序列化 wList，`handleParams()` 检测 mode 变更触发 reset
8. `shared/presets.ts` — DEFAULTS 新增 `mode/numWorkers/kvGb`，新增 3 个 agg 预设，PARAM_DEFS 新增 `mode/numWorkers/kvGb` 定义
9. `shared/i18n.ts` — 新增 agg 相关 i18n label
10. `public/pd-disagg.html` — topbar 增加 mode 切换，Renderer 增加 `drawWorker()` 方法及 agg 渲染分支，侧栏参数联动，breakdown/metrics 卡片适配，预设按钮扩展

## 测试设计

### 验收测试用例清单

| # | 用例 | 验证点 |
|---|---|---|
| T1 | 默认启动 → 模式为 pd-disagg | DEFAULTS.mode === "pd-disagg"，所有现有功能正常 |
| T2 | 切换为 agg 模式 | engine 重置，wList 创建，pList/dList 为空 |
| T3 | agg 模式基本仿真 | 请求走 tokenize → w_waiting → w_prefill → w_decode → response → done |
| T4 | agg 模式 KV 容量约束 | 当 KV 满时请求排队等待，不超分配 |
| T5 | agg 模式 retract | 当 decode 批次 KV 不足时触发抢占，被抢占请求回到 waitingQ |
| T6 | agg 模式 metrics 正确性 | TTFT = tokenize + queue + prefill + decode + detok，BD_KEYS 为 5 列 |
| T7 | agg 模式 gauges 正确性 | wQueue/kvW 正确反映队列深度和 KV 利用率 |
| T8 | 切换回 pd-disagg | engine 重置，pList/dList 重建，wList 为空，现有功能正常 |
| T9 | agg 预设加载 | 加载 aggBalanced/aggDecodeHeavy/aggHighQps 后参数正确 |
| T10 | agg 模式 getRenderState | 返回 wList 序列化，pList/dList 为空数组 |
| T11 | agg 模式侧栏联动 | agg 模式下隐藏 numP/numD 等，显示 numWorkers/kvGb |
| T12 | agg 模式 Renderer | 单列 Worker 布局，无 TransferLink 动画 |
| T13 | agg 模式 breakdown | 5 列堆叠图，label 正确 |
| T14 | 多 worker 负载均衡 | agg 模式下 least/round_robin/power_of_two/random 均正常分发 |

### 边界条件覆盖

| # | 边界条件 | 预期行为 |
|---|---|---|
| B1 | `numWorkers = 1` 单 worker | 所有请求进同一 worker，KV 可能更早饱和 |
| B2 | `numWorkers` 动态增大/缩小 | syncTopology 正确标记 draining/新增 worker |
| B3 | `kvGb` 极小（如 1 GB） | KV 快速饱和，大量请求排队或 retract |
| B4 | 高 QPS + 长 output（agg decode 重载） | retract 频繁，ntr 上升，throughput 受限 |
| B5 | 模式切换时请求进行中 | reset() 清空所有状态，重新开始 |
| B6 | `mode` 参数非法值 | 忽略，保持当前模式 |
| B7 | agg 模式下 `chunkSize` = 1 token | 极小 chunk，prefill 步骤极多但不崩溃 |

## 风险与注意事项

### 兼容性影响

- **向后兼容**：`mode` 默认值为 `"pd-disagg"`，现有行为完全不变。所有新增 `SimParams` 字段都有默认值，旧代码不传新字段也不受影响
- **API 兼容**：`/state` 和 `/render` 返回结构新增 `wList` 等字段，现有前端字段不变；agg 模式下 `pList/dList` 返回空数组而非省略，保证消费端不报错
- **前端兼容**：mode 切换是 UI 新增功能，不影响现有 pd-disagg 页面

### 性能影响

- `stepAgg()` 单步计算量与 `stepDisagg()` 相当，无额外性能开销
- `sampleGauges()` 仅多遍历 wList 或 pList/dList，O(N) 无变化
- MetricsCollector 的 `record()` 增加 mode 判断分支，开销可忽略

### 回滚方案

- 所有变更通过 `mode` 字段控制，设为 `"pd-disagg"` 即完全回退到现有行为
- 新增的 `WorkerInstance.ts` 文件仅 agg 模式引用，不影响现有代码路径
- 若需完全回滚，删除 `WorkerInstance.ts`，移除 `SimParams` 中 3 个新字段及 `stepAgg()` 分支即可
