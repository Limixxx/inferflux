# AGENT.md — TS ↔ pd-disagg.html 对应关系

本文档记录 `server/src/` 下 TypeScript 文件与原始 `pd-disagg.html` 的逐段对应关系。当 HTML 更新后，Agent 可据此定位需要同步修改的 TS 文件。

> **行号基于 pd-disagg.html 原文**。HTML 更新后行号可能漂移，请以函数名/类名/标记为锚点。

---

## 1. 全局映射总览

| HTML 行号范围 | HTML 内容 | TS 文件 | 状态 |
|---|---|---|---|
| 1–161 | HTML head + CSS + body 结构 | — | 未迁移（前端） |
| 162 | `<script>` 开始 | — | — |
| 175–283 | I18N 词典 (DICT, LANG, t) | `src/shared/i18n.ts` | 已迁移 |
| 285–413 | 参数/预设 (MODEL_PRESETS … PARAM_DEFS) | `src/shared/presets.ts` + `src/shared/utils.ts` | 已迁移 |
| 415 | `/*ENGINE-START*/` | — | 标记 |
| 416–471 | RNG、分布、cellSizeOf、KVPOLL | `src/shared/rng.ts` + `src/shared/utils.ts` + `src/shared/constants.ts` + `src/shared/types.ts` | 已迁移 |
| 473–499 | makeRequest | `src/sim/entities/Request.ts` | 已迁移 |
| 500–524 | chunkTokens, chunkPrefillMs | `src/shared/utils.ts` | 已迁移 |
| 526–638 | TransferLink, PrefillInstance, DecodeInstance | `src/sim/entities/*.ts` | 已迁移 |
| 640–668 | selectByPolicy | `src/sim/LoadBalancer.ts` | 已迁移 |
| 670–804 | MetricsCollector | `src/sim/MetricsCollector.ts` | 已迁移 |
| 806–1133 | SimEngine | `src/sim/SimEngine.ts` | 已迁移 |
| 1135–1141 | runHeadless | `src/sim/SimEngine.ts` | 已迁移 |
| 1142–1302 | runSelfTest | — | 未迁移（浏览器测试） |
| 1303 | `/*ENGINE-END*/` | — | 标记 |
| 1305–1685 | Renderer 类 | — | 未迁移（Canvas 渲染，浏览器端） |
| 1687–1758 | drawSpark, drawBreakdown, fmtMs, fmtPct, fmtNum | `src/shared/utils.ts`（fmtMs/fmtPct/fmtNum 部分迁移） | 部分迁移 |
| 1762–2102 | UI 状态、侧边栏、主循环、boot | — | 未迁移（前端 UI） |
| 2103–2105 | `</script></body></html>` | — | — |
| — | — | `src/sim/SimService.ts` | 新增（HTTP API 层） |
| — | — | `src/http/HttpService.ts` | 新增（静态文件服务） |
| — | — | `src/index.ts` | 新增（主入口） |

---

## 2. 逐文件详细对应

### `src/shared/types.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 470–471 | `const KVPOLL = {...}` | `export const KVPOLL` (as const) |
| — | — | `SimParams` 接口（从 DEFAULTS 对象结构推导，lines 314–325） |
| 474–499 | `makeRequest` 返回对象结构 | `SimRequest` 接口 |
| 495–497 | `stamps: {...}` | `RequestStamps` 接口 |
| — | — | `MetricSnapshot`, `Gauges`, `ISimEngine` 等接口（新增） |
| — | — | `SimStateResponse`, `SimCommandRequest` 等 API 类型（新增） |

**注意**：`SimRequest.p` 和 `.d` 在 TS 中声明为 `any`（非 `PrefillInstance | null`），以避免 types.ts ↔ entity 文件之间的循环导入。

### `src/shared/constants.ts`

| HTML 行号 | HTML 常量 | TS 对应 |
|---|---|---|
| 417 | `const TICK = 5` | `export const TICK` |
| 418 | `const RING_MAX = 4096` | `export const RING_MAX` |
| 419 | `const SERIES_LEN = 300` | `export const SERIES_LEN` |
| 420 | `const BUCKET_MS = 1000` | `export const BUCKET_MS` |
| 426 | `const NTR_MAX = 1.0` | `export const NTR_MAX` |
| 427 | `const NTR_CLIP = 4096` | `export const NTR_CLIP` |
| 428 | `const NTR_DECAY_STEPS = 600` | `export const NTR_DECAY_STEPS` |
| 433 | `const HANDSHAKE_RTT_MS = 2` | `export const HANDSHAKE_RTT_MS` |
| 312 | `const PERREQ_REF_CTX = 4096` | `export const PERREQ_REF_CTX` |
| 671–672 | `const SERIES_KEYS = [...]` | `export const SERIES_KEYS` (as const) |
| 673 | `const BD_KEYS = [...]` | `export const BD_KEYS` (as const) |

### `src/shared/utils.ts`

| HTML 行号 | HTML 函数 | TS 对应 | 签名变化 |
|---|---|---|---|
| 311 | `gpuKvBudget(hbm)` | `export function gpuKvBudget` | 无变化 |
| 339–343 | `fmtTokens(v)` | `export function fmtTokens` | 无变化 |
| 347–348 | `LOG_STEPS`, `SNAP_GRID` | `export const LOG_STEPS`, `export const SNAP_GRID` | 无变化 |
| 349–361 | `logSliderToVal(def, pos)` | `export function logSliderToVal` | 无变化 |
| 362–369 | `nearestStepIdx(def, val)` | `export function nearestStepIdx` | 无变化 |
| 370–374 | `logValToSlider(def, val)` | `export function logValToSlider` | 无变化 |
| 435 | `clamp(v, lo, hi)` | `export function clamp` | 无变化 |
| 463–467 | `cellSizeOf(P)` | `export function cellSizeOf` | 无变化 |
| 500–502 | `chunkTokens(engine, r, idx)` | `export function chunkTokens(P, uncachedLen, idx)` | **签名变化**：`(engine, r, idx)` → `(P: SimParams, uncachedLen: number, idx: number)` |
| 511–524 | `chunkPrefillMs(P, r, idx)` | `export function chunkPrefillMs` | `r` 参数类型改为 `{ uncachedLen: number; cachedLen: number }` |
| 1753–1758 | `fmtMs(v)` | `export function fmtMs` | 无变化 |
| 1759 | `fmtPct(v)` | `export function fmtPct` | 无变化 |
| 1760 | `fmtNum(v)` | `export function fmtNum` | 无变化 |

### `src/shared/rng.ts`

| HTML 行号 | HTML 函数 | TS 对应 |
|---|---|---|
| 436–444 | `mulberry32(seed)` | `export function mulberry32` |
| 445 | `expSample(rng, mean)` | `export function expSample` |
| 446–451 | `lognormalSample(rng, mean, sigma)` | `export function lognormalSample` |
| 452–458 | `sampleLen(rng, mean, dist)` | `export function sampleLen` |

新增类型：`export type RNG = () => number`

### `src/shared/presets.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 292–299 | `MODEL_PRESETS` | `export const MODEL_PRESETS: Record<string, ModelPreset>` |
| 303–310 | `GPU_PRESETS` | `export const GPU_PRESETS: Record<string, GpuPreset>` |
| 314–325 | `DEFAULTS` | `export const DEFAULTS: SimParams` |
| 327–337 | `PRESETS` | `export const PRESETS: Record<string, Partial<SimParams>>` |
| 377–412 | `PARAM_DEFS` | `export const PARAM_DEFS: ParamDef[]` + `export interface ParamDef` |
| 413 | `GROUPS` | `export const GROUPS` |

### `src/shared/i18n.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 176–228 | `DICT.zh` | `DICT.zh` |
| 229–280 | `DICT.en` | `DICT.en` |
| 282 | `let LANG = ...` | `let LANG` + `detectLang()` |
| 283 | `function t(k)` | `export function t(k)` |

新增：`export type Lang`, `export function setLang/getLang/detectLang`

### `src/sim/entities/Request.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 474 | `let REQ_SEQ = 0` | `let REQ_SEQ = 0` |
| 475–499 | `function makeRequest(engine, arrivalT)` | `export function makeRequest(engine: ISimEngine, arrivalT: number): SimRequest` |
| — | — | `export function resetReqSeq()` (新增，在 reset 时调用) |

### `src/sim/entities/TransferLink.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 526–529 | `class TransferLink` | `export class TransferLink` |

### `src/sim/entities/PrefillInstance.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 531–555 | `class PrefillInstance` | `export class PrefillInstance` |
| — | slots 内联对象 `{ req, busyUntil }` | `export interface PrefillSlot` (新增接口) |

### `src/sim/entities/DecodeInstance.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 557–638 | `class DecodeInstance` | `export class DecodeInstance` |

包含方法：`maxTokens`, `decodeStepMs`, `pendingLoad`, `isEmpty`, `reservedOffset`, `raiseNtrAfterRetract`, `retractDecode`

### `src/sim/LoadBalancer.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 645–668 | `function selectByPolicy(policy, pool, loadOf, rng, rr)` | `export function selectByPolicy<T>(...)` (泛型化) |

新增：`export interface RRCounter { i: number }`

### `src/sim/MetricsCollector.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 675–804 | `class MetricsCollector` | `export class MetricsCollector` |

方法：`reset`, `record`, `tick`, `snapshot`, `latestSeries`, `recentBreakdown`

新增内部接口：`RingEntry`

### `src/sim/SimEngine.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 807–832 | `constructor` + `reset()` | `constructor` + `reset()` |
| 833–843 | `syncTopology()` | `syncTopology()` |
| 844–850 | `advance(dtSim)` | `advance(dtSim)` |
| 851–1079 | `step()` | `step()` |
| 1083–1090 | `enqueueReadyChunks(p, r)` | `enqueueReadyChunks(p, r)` |
| 1091–1103 | `completeTransfer(p, r, doneAt)` | `completeTransfer(p, r, doneAt)` |
| 1105–1132 | `sampleGauges()` | `sampleGauges(): Gauges` |
| 1136–1141 | `runHeadless(overrides, seconds, seed)` | `export function runHeadless` |

**关键变更**：
- `enqueueReadyChunks` 中 `chunkTokens(this, r, idx)` → `chunkTokens(this.P, r.uncachedLen, idx)`（签名变化）
- 实现 `ISimEngine` 接口
- `reset()` 中新增 `resetReqSeq()` 调用

### `src/sim/SimService.ts` (新增)

无 HTML 对应。封装 SimEngine 为独立 HTTP 服务：
- 16ms 帧率模拟循环（`setInterval`）
- REST API: `/state`, `/command`, `/params`, `/preset`, `/health`
- CORS 支持

### `src/http/HttpService.ts` (新增)

无 HTML 对应。独立静态文件 HTTP 服务器：
- 从项目根目录提供静态文件
- `/api/*` 代理到 SimService（端口 3001）

### `src/index.ts` (新增)

无 HTML 对应。主入口，启动两个服务 + 优雅关闭。

---

## 3. 未迁移部分（保留在 HTML 中）

以下逻辑仍在 `pd-disagg.html` 中，属于浏览器端渲染/UI 层：

| HTML 行号 | 内容 | 原因 |
|---|---|---|
| 1–161 | HTML 结构 + CSS | 前端布局 |
| 1305–1314 | COLORS, kvGaugeColor, BOX_H 等渲染常量 | Canvas 渲染参数 |
| 1316–1685 | `class Renderer` | Canvas 2D 绘制，需 DOM API |
| 1688–1714 | `drawSpark()` | Canvas sparkline 绘制 |
| 1716–1751 | `drawBreakdown()` | Canvas breakdown 柱状图绘制 |
| 1763–1804 | 全局状态 `P`, `sim`, `renderer`, `METRIC_DEFS` | 前端运行时状态 |
| 1806–1822 | `seriesMean()`, `lastSeriesVal()` | 前端数据查询 |
| 1824–1852 | `buildMetricCards()`, `updatePanels()` | DOM 操作 |
| 1855–1995 | `buildSidebar()`, `buildRow()`, `applyModelPreset()` 等 | DOM 操作 |
| 1998–2048 | `applyI18n()`, `wireTopbar()` | DOM 操作 |
| 2050–2061 | `frame()` 主循环 | `requestAnimationFrame` |
| 2063–2102 | `boot()` | 页面加载初始化 |

---

## 4. 同步更新指南

当 `pd-disagg.html` 更新后，按以下步骤同步 TS 代码：

1. **定位变更区域**：在 HTML 中找到修改的行号范围
2. **查上表**：确定对应的 TS 文件
3. **同步修改**：注意签名差异（见各表「签名变化」列）
4. **检查编译**：确保 TypeScript strict 模式零错误

### 最常需同步的文件

| 修改场景 | 需同步的 TS 文件 |
|---|---|
| 调整模拟参数（DEFAULTS/PRESETS） | `src/shared/presets.ts` |
| 修改引擎常量（TICK/NTR_*等） | `src/shared/constants.ts` |
| 修改 RNG/分布逻辑 | `src/shared/rng.ts` |
| 修改 cellSizeOf/chunkPrefillMs | `src/shared/utils.ts` |
| 修改请求结构 | `src/shared/types.ts` + `src/sim/entities/Request.ts` |
| 修改 Prefill/Decode 实例逻辑 | `src/sim/entities/PrefillInstance.ts` 或 `DecodeInstance.ts` |
| 修改 step() 调度逻辑 | `src/sim/SimEngine.ts` |
| 修改指标采集逻辑 | `src/sim/MetricsCollector.ts` |
| 修改负载均衡策略 | `src/sim/LoadBalancer.ts` |
| 新增/修改 I18N 词条 | `src/shared/i18n.ts` |
| 修改侧边栏参数定义 | `src/shared/presets.ts` (PARAM_DEFS) |
