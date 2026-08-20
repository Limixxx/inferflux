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
| — | — | `src/sim/entities/WorkerInstance.ts` | 新增（agg 模式统一 Worker） |
| — | — | `src/http/HttpService.ts` | 新增（静态文件服务） |
| — | — | `src/index.ts` | 新增（主入口） |
| — | — | `src/test/agg.test.ts` | 新增（agg 模式验收测试） |

---

## 2. 逐文件详细对应

### `src/shared/types.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 470–471 | `const KVPOLL = {...}` | `export const KVPOLL` (as const) |
| — | — | `SimMode` 类型：`"pd-disagg" \| "agg"` — **新增**，部署模式选择 |
| — | — | `SimParams` 接口（从 DEFAULTS 对象结构推导，lines 314–325） |
| 474–499 | `makeRequest` 返回对象结构 | `SimRequest` 接口 |
| 495–497 | `stamps: {...}` | `RequestStamps` 接口 |
| — | — | `MetricSnapshot`, `Gauges`, `ISimEngine` 等接口（新增） |
| — | — | `SimStateResponse`, `SimCommandRequest` 等 API 类型（新增） |

**Agg 模式新增字段**：

| 接口 | 新增字段 | 说明 |
|---|---|---|
| `ReqStage` | `w_waiting`, `w_prefill`, `w_chunked_prefill`, `w_decode` | agg 模式请求阶段 |
| `SimParams` | `mode: SimMode`, `numWorkers: number`, `kvGb: number`, `chunkedPrefill: boolean` | agg 拓扑参数，默认 `mode:"pd-disagg"` 向后兼容 |
| `SimRequest` | `w: any`, `chunkOffset: number` | WorkerInstance 引用 + 分块偏移量 |
| `RequestStamps` | `wQueueExit: number`, `wPrefillDone: number` | agg 模式时间戳（pd-disagg 模式下为 NaN） |
| `Gauges` | `wQueue: number`, `kvW: number` | agg 模式指标（pd-disagg 模式下为 0） |

**注意**：`SimRequest.p` 和 `.d` 在 TS 中声明为 `any`（非 `PrefillInstance | null`），以避免 types.ts ↔ entity 文件之间的循环导入。同理 `SimRequest.w` 声明为 `any`。

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
| 671–672 | `const SERIES_KEYS = [...]` | `export const SERIES_KEYS` (as const) — **已扩展**：新增 `"wQueue"`, `"kvW"` |
| 673 | `const BD_KEYS = [...]` | `export const BD_KEYS_DISAGG` — 原有 7 列（tokenize/bootstrap/pQueue/prefill/transfer/dQueue/detok） |
| — | — | `export const BD_KEYS_AGG` — **新增** 4 列（tokenize/queue/prefill/detok） |
| — | — | `export const BD_KEYS = BD_KEYS_DISAGG` — 向后兼容别名 |

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
| — | — | `export function fullPrefillMs(P, r)` | **新增**：agg 非分块模式，一次 GPU 迭代处理全部 uncachedLen |
| — | — | `function prefillCoreMs(P, r, q, ctx)` | **新增**：内部共享函数，gemm + per-layer attention 成本模型 |
| 1753–1758 | `fmtMs(v)` | `export function fmtMs` | 无变化 |
| 1759 | `fmtPct(v)` | `export function fmtPct` | 无变化 |
| 1760 | `fmtNum(v)` | `export function fmtNum` | 无变化 |

**新增函数关系**：`fullPrefillMs` 和 `chunkPrefillMs` 均调用内部 `prefillCoreMs`——提取了 gemm + 全注意力/SWA 注意力成本计算为共享逻辑。

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
| 314–325 | `DEFAULTS` | `export const DEFAULTS: SimParams` — **新增 agg 字段**：`mode:"pd-disagg"`, `numWorkers:2`, `kvGb:99`, `chunkedPrefill:false` |
| 327–337 | `PRESETS` | `export const PRESETS: Record<string, Partial<SimParams>>` — **新增 4 个 agg 预设** |
| 377–412 | `PARAM_DEFS` | `export const PARAM_DEFS: ParamDef[]` + `export interface ParamDef` |
| 413 | `GROUPS` | `export const GROUPS` |

**新增 agg 预设**：

| 预设名 | 关键参数 |
|---|---|
| `aggBalanced` | `mode:"agg"`, `numWorkers:2`, `kvGb:99`, `chunkedPrefill:false` |
| `aggChunkedPrefill` | `mode:"agg"`, `chunkedPrefill:true`, `chunkSize:8192`, `inputLenMean:8192` |
| `aggDecodeHeavy` | `mode:"agg"`, `numWorkers:1`, `kvGb:141`, `outputLenMean:1024` |
| `aggHighQps` | `mode:"agg"`, `numWorkers:4`, `kvGb:99`, `qps:16` |

**新增 PARAM_DEFS 条目**：

| key | type | group | 说明 |
|---|---|---|---|
| `mode` | `select`, options `["pd-disagg","agg"]` | topology | 部署模式切换 |
| `numWorkers` | slider, min:1, max:8 | topology | agg Worker 实例数 |
| `kvGb` | slider, min:1, max:288 | kv | agg 统一 KV 显存 |
| `chunkedPrefill` | `toggle` | compute | 分块 Prefill 开关 |

### `src/shared/i18n.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 176–228 | `DICT.zh` | `DICT.zh` |
| 229–280 | `DICT.en` | `DICT.en` |
| 282 | `let LANG = ...` | `let LANG` + `detectLang()` |
| 283 | `function t(k)` | `export function t(k)` |

新增：`export type Lang`, `export function setLang/getLang/detectLang`

**Agg 模式新增 i18n 条目**（zh/en 双语）：

| 键前缀 | 用途 |
|---|---|
| `preset.agg*` (4个) | agg 预设名称 |
| `p.mode`, `mode.pd-disagg`, `mode.agg` | 部署模式标签 |
| `p.numWorkers` | Worker 实例数标签 |
| `p.chunkedPrefill`, `prefill.on`, `prefill.off` | 分块 Prefill 开关标签 |
| `p.kvGb` | agg 统一 KV 显存标签 |
| `s.w_waiting`, `s.w_prefill`, `s.w_chunked_prefill`, `s.w_decode` | agg 请求阶段标签 |
| `bd.queue` | agg TTFT breakdown "排队" 列 |
| `m.wQueue`, `m.kvW` | agg 指标卡片标签 |
| `n.worker`, `inst.pipeline` | 实体名称标签 |
| `info.capW` | Worker 容量信息标签 |
| `lg.wPrefill`, `lg.wChunkedPrefill`, `lg.wDecode` | agg 实体日志标签 |

### `src/sim/entities/Request.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 474 | `let REQ_SEQ = 0` | `let REQ_SEQ = 0` |
| 475–499 | `function makeRequest(engine, arrivalT)` | `export function makeRequest(engine: ISimEngine, arrivalT: number): SimRequest` |
| — | — | `export function resetReqSeq()` (新增，在 reset 时调用) |

**Agg 模式新增初始值**：

| 字段 | 初始值 | 说明 |
|---|---|---|
| `w` | `null` | WorkerInstance 引用（agg 模式赋值，pd-disagg 恒为 null） |
| `chunkOffset` | `0` | 当前分块 prefill 的分片索引 |
| `stamps.wQueueExit` | `NaN` | make_batch 入场时间 |
| `stamps.wPrefillDone` | `NaN` | prefill 完成时间 |

### `src/sim/entities/TransferLink.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 526–529 | `class TransferLink` | `export class TransferLink` |

**注意**：TransferLink 仅在 pd-disagg 模式使用，agg 模式无传输链路。

### `src/sim/entities/PrefillInstance.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 531–555 | `class PrefillInstance` | `export class PrefillInstance` |
| — | slots 内联对象 `{ req, busyUntil }` | `export interface PrefillSlot` (新增接口) |

**注意**：PrefillInstance 仅在 pd-disagg 模式使用。

### `src/sim/entities/DecodeInstance.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 557–638 | `class DecodeInstance` | `export class DecodeInstance` |

包含方法：`maxTokens`, `decodeStepMs`, `pendingLoad`, `isEmpty`, `reservedOffset`, `raiseNtrAfterRetract`, `retractDecode`

**注意**：DecodeInstance 仅在 pd-disagg 模式使用。

### `src/sim/entities/WorkerInstance.ts` (新增)

无 HTML 对应。Agg 模式统一 Worker 实体，模拟 SGLang make_batch 混合批调度。

**类结构**：

| 字段/方法 | 类型 | 说明 |
|---|---|---|
| `id` | `number` | 实例编号 |
| `waitingQ` | `SimRequest[]` | 等待 make_batch 入场的请求队列 |
| `running` | `SimRequest[]` | 统一批次：混合 w_prefill / w_chunked_prefill / w_decode |
| `kvUsed` | `number` | 已用 KV token 数 |
| `draining` | `boolean` | 拓扑缩容标记 |
| `ntr` | `number` | new_token_ratio 当前值 |
| `retractGlow` | `number` | 抢占视觉反馈时间戳 |
| `nextStepAt` | `number \| null` | 下一步 GPU 迭代时间 |
| `maxTokens(P)` | 方法 | KV 池容量（tokens），使用 `kvGb` 计算 |
| `decodeStepMs(P)` | 方法 | 仅统计 w_decode 请求的 KV 读取成本 |
| `stepLatencyMs(P)` | 方法 | `max(prefill_compute, decode_step)` — 混合批重叠 |
| `pendingLoad()` | 方法 | `waitingQ.length + running.length` |
| `isEmpty()` | 方法 | 队列和批次均为空 |
| `reservedOffset()` | 方法 | decode 请求的 KV 预留量（镜像 DecodeInstance） |
| `raiseNtrAfterRetract()` | 方法 | 抢占后提升 NTR（镜像 SGLang） |
| `retractDecode(P, engine, now)` | 方法 | 驱逐最少已解码 token 的 w_decode 请求；prefill 请求不可驱逐 |

**与 DecodeInstance 的关键差异**：
- `running` 数组混合 prefill + decode（DecodeInstance 仅有 decode）
- `stepLatencyMs` 使用 `max()` 重叠语义（DecodeInstance 仅有 `decodeStepMs`）
- `maxTokens` 使用 `kvGb` 而非 `kvGbD`
- prefill 请求不可被 retract（GPU 正在计算中）

### `src/sim/LoadBalancer.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 645–668 | `function selectByPolicy(policy, pool, loadOf, rng, rr)` | `export function selectByPolicy<T>(...)` (泛型化) |

新增：`export interface RRCounter { i: number }`

**注意**：LoadBalancer 在两种模式下均使用——pd-disagg 选 P/D 实例，agg 选 Worker 实例。

### `src/sim/MetricsCollector.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 675–804 | `class MetricsCollector` | `export class MetricsCollector` |

方法：`reset`, `record`, `tick`, `snapshot`, `latestSeries`, `recentBreakdown`

新增内部接口：`RingEntry`

**Agg 模式扩展**：

| 新增 | 说明 |
|---|---|
| `mode: SimMode` 字段 | 当前部署模式，默认 `"pd-disagg"` |
| `setMode(mode)` 方法 | 设置模式；若模式变更，重新初始化 `bdBd[]` 和 `bdSeries[]` 数组长度 |
| `bdKeysFor(mode)` 辅助函数 | 返回 `BD_KEYS_DISAGG`(7列) 或 `BD_KEYS_AGG`(4列) |
| `record(r, now, mode)` | 第三个参数 `mode` 决定 4 列 vs 7 列 breakdown 计算；agg 无 dHandshake |
| `tick()` | 模式感知 gauge 路由——agg 模式下 `pQueue/dQueue/kvP/kvD/kvDpre/link/dHandshake` 归零，`wQueue/kvW` 写入实际值；pd-disagg 反之 |

**record() breakdown 计算差异**：

| 模式 | 列数 | 计算公式 |
|---|---|---|
| pd-disagg | 7 | `[tokenized-recv, bootstrapDone-tokenized, pQueueExit-bootstrapDone, prefillDone-pQueueExit, transferDone-prefillDone, dQueueExit-transferDone, firstToken-dQueueExit]` |
| agg | 4 | `[tokenized-recv, wQueueExit-tokenized, wPrefillDone-wQueueExit, firstToken-wPrefillDone]` |

### `src/sim/SimEngine.ts`

| HTML 行号 | HTML 内容 | TS 对应 |
|---|---|---|
| 807–832 | `constructor` + `reset()` | `constructor` + `reset()` |
| 833–843 | `syncTopology()` | `syncTopology()` — **模式感知**：agg 管理 wList，pd-disagg 管理 pList/dList |
| 844–850 | `advance(dtSim)` | `advance(dtSim)` |
| 851–1079 | `step()` | `step()` — **模式分派**：`stepAgg()` 或 `stepDisagg()` |
| — | — | `stepDisagg()` — 原 step() 逻辑不变 |
| — | — | `stepAgg()` — **新增**：agg 模式完整生命周期 |
| 1083–1090 | `enqueueReadyChunks(p, r)` | `enqueueReadyChunks(p, r)` |
| 1091–1103 | `completeTransfer(p, r, doneAt)` | `completeTransfer(p, r, doneAt)` |
| 1105–1132 | `sampleGauges()` | `sampleGauges()` — **模式感知**：agg 返回 wQueue/kvW，pd-disagg 返回 pQueue/kvP/kvD 等 |
| 1136–1141 | `runHeadless(overrides, seconds, seed)` | `export function runHeadless` |

**关键变更**：
- `enqueueReadyChunks` 中 `chunkTokens(this, r, idx)` → `chunkTokens(this.P, r.uncachedLen, idx)`（签名变化）
- 实现 `ISimEngine` 接口
- `reset()` 中新增 `resetReqSeq()` 调用和 `metrics.setMode(this.P.mode)` 调用

**Agg 模式新增字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `wList` | `WorkerInstance[]` | Worker 实例列表 |
| `rrW` | `RRCounter` | Worker 轮询计数器 |
| `wSeq` | `number` | Worker ID 序列 |

**stepAgg() 流程**：
1. Arrivals → tokenize → LB → `worker.waitingQ`（无 bootstrap，无 transfer link）
2. Per worker: make_batch admission（KV 预分配全部 inputLen）→ GPU step loop:
   - `w_prefill`：单次迭代完成 → 采样首 token → `w_decode`
   - `w_chunked_prefill`：多迭代；全部 chunk 完成后才采样首 token → `w_decode`
   - `w_decode`：每步发射 token；完成释放 KV
   - Step latency = `max(prefill_compute, decode_step)`
3. Response → done（镜像 disagg detok 路径，无 D 端握手）

**syncTopology() 模式分支**：
- `mode === "agg"`：按 `numWorkers` 增删 `wList`，标记 `draining`
- 否则：按 `numP`/`numD` 增删 `pList`/`dList`，标记 `draining`

**sampleGauges() 模式分支**：
- `mode === "agg"`：聚合所有 worker 的 `waitingQ.length`、`running.length`、`kvUsed`/`maxTokens`；pd-disagg gauge 全归零
- 否则：原有逻辑不变；agg gauge 全归零

### `src/sim/SimService.ts` (新增，agg 扩展)

无 HTML 对应。封装 SimEngine 为独立 HTTP 服务。

**原有 API**：`/state`, `/command`, `/params`, `/preset`, `/health`

**Agg 模式扩展**：

| 新增/变更 | 说明 |
|---|---|
| `GET /render` | **新增端点**：返回前端渲染所需的实体级状态（模式感知序列化） |
| `serializeReqAgg(r, wIdx)` | **新增函数**：agg 模式请求序列化，输出 `wIdx` 而非 `pIdx`/`dIdx` |
| `getRenderState()` | **新增方法**：根据 `P.mode` 返回不同结构——agg 输出 `wList[]`（含 `waitingQ`/`running`），pd-disagg 输出 `pList[]`/`dList[]`（含完整链路状态） |
| `handleParams()` | **扩展**：`mode` 或 `chunkedPrefill` 变更时触发 `engine.reset()`（拓扑/状态机根本变化） |
| `SERIES_KEYS` 本地常量 | 已扩展：新增 `"wQueue"`, `"kvW"` |

**`/render` 响应结构差异**：

| 字段 | pd-disagg | agg |
|---|---|---|
| `mode` | `"pd-disagg"` | `"agg"` |
| `pList` | PrefillInstance[] | `[]` |
| `dList` | DecodeInstance[] | `[]` |
| `wList` | `[]` | WorkerInstance[]（含 `waitingQ`/`running`） |
| 请求序列化 | `pIdx`, `dIdx`, `kvPoll`, `chunksQueued/Transferred` 等 | `wIdx`, `chunkOffset` |

### `src/http/HttpService.ts` (新增)

无 HTML 对应。独立静态文件 HTTP 服务器：
- 从项目根目录提供静态文件
- `/api/*` 代理到 SimService（端口 3001）

### `src/index.ts` (新增)

无 HTML 对应。主入口，启动两个服务 + 优雅关闭。

### `src/test/agg.test.ts` (新增)

无 HTML 对应。16 个验收测试用例，覆盖：

| 用例 | 覆盖范围 |
|---|---|
| T1 | 默认模式为 pd-disagg |
| T2 | agg 模式创建 wList |
| T3 | 非分块 prefill 完整生命周期 |
| T4 | 分块 prefill 多迭代 |
| T5 | KV 容量约束 |
| T7 | 4 列 breakdown |
| T9 | gauges wQueue/kvW |
| T10 | RadixCache 命中 |
| T11 | BlockManager 预分配 |
| T12 | 混合批次（prefill+decode 共存） |
| T13 | 模式切换（pd-disagg ↔ agg） |
| T14 | agg 预设加载 |
| T15 | /render 端点状态 |
| B1 | 单 Worker 边界 |
| B3 | 微小 KV 容量 |
| B9 | 全缓存命中 |

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
| — | agg 前端适配（drawAgg/drawWorker、4 个 agg 预设按钮、参数联动、4 列 breakdown 渲染） | Canvas + DOM，需浏览器 API |

---

## 4. 同步更新指南

当 `pd-disagg.html` 更新后，按以下步骤同步 TS 代码：

1. **定位变更区域**：在 HTML 中找到修改的行号范围
2. **查上表**：确定对应的 TS 文件
3. **同步修改**：注意签名差异（见各表「签名变化」列）
4. **检查编译**：确保 TypeScript strict 模式零错误
5. **模式感知**：如果改动涉及 pd-disagg 专有逻辑，检查是否需要在 agg 模式添加对应逻辑或显式归零

### 最常需同步的文件

| 修改场景 | 需同步的 TS 文件 |
|---|---|
| 调整模拟参数（DEFAULTS/PRESETS） | `src/shared/presets.ts` |
| 修改引擎常量（TICK/NTR_*等） | `src/shared/constants.ts` |
| 修改 RNG/分布逻辑 | `src/shared/rng.ts` |
| 修改 cellSizeOf/chunkPrefillMs | `src/shared/utils.ts`（同步更新 `fullPrefillMs` / `prefillCoreMs`） |
| 修改请求结构 | `src/shared/types.ts` + `src/sim/entities/Request.ts` |
| 修改 Prefill/Decode 实例逻辑 | `src/sim/entities/PrefillInstance.ts` 或 `DecodeInstance.ts`；检查 `WorkerInstance.ts` 是否需同步 |
| 修改 step() 调度逻辑 | `src/sim/SimEngine.ts`（`stepDisagg` 和 `stepAgg` 分别同步） |
| 修改指标采集逻辑 | `src/sim/MetricsCollector.ts`（注意 4 列 vs 7 列 breakdown） |
| 修改负载均衡策略 | `src/sim/LoadBalancer.ts` |
| 新增/修改 I18N 词条 | `src/shared/i18n.ts` |
| 修改侧边栏参数定义 | `src/shared/presets.ts` (PARAM_DEFS) |
| 修改 agg Worker 调度逻辑 | `src/sim/entities/WorkerInstance.ts` + `src/sim/SimEngine.ts`（`stepAgg`） |
| 修改 agg KV 容量模型 | `src/shared/utils.ts`（`cellSizeOf`）+ `src/sim/entities/WorkerInstance.ts`（`maxTokens`） |
| 修改 TTFT breakdown 列 | `src/shared/constants.ts`（`BD_KEYS_*`）+ `src/sim/MetricsCollector.ts`（`record`） |
| 修改前端渲染状态格式 | `src/sim/SimService.ts`（`getRenderState`、`serializeReq`/`serializeReqAgg`） |
