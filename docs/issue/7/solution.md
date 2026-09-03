---
title: "Issue #7 解决方案"
issue_number: 7
issue_type: Feature
created: 2026-09-03
updated: 2026-09-03
status: revised
review_round: 2
---

# Issue #7 解决方案

> 本方案回应评审人 [Limixxx] 第 1 轮驳回意见：「不是实现方案，而是检查实现内容」。因此本文档定位为 **实现核查 + 特性分析 + 演示系统设计**，而非新建 22 个子 Issue 的实现计划（上一轮方案已被驳回，此处不再重复）。

## 评审意见回应

| # | 评审要求 | 本方案对应章节 |
|---|---|---|
| 1 | 整体代码实现是否与 `docs/SGLang_Simulator_Technical_Report.md` 总体方案保持一致 | §1 一致性核查 |
| 2 | 从特性架构、代码整洁度、测试完整程度对特性整体进行分析 | §2 特性整体分析 |
| 3 | 参考 `server/public/pd-disagg.html` 实现一个 `sglang.html` 的演示系统 | §3 sglang.html 演示系统设计 |

---

## 需求分析

### 问题描述

`server/src/sglang/` 已完成 SGLang 仿真器全部 22 个子 Issue（S0–S6 / K1–K5 / P0–P6）的代码与测试实现。本 Issue 当前需要：

1. **核查**：对照 `docs/SGLang_Simulator_Technical_Report.md`（218KB 技术报告，11 章 + §9/§10 附录）逐维度验证实现一致性，给出可证伪的结论。
2. **分析**：对已完成特性从架构、代码整洁度、测试完整度三个维度做整体评估，定位可改进点。
3. **补全**：当前缺失可视化演示，需参照 `server/public/pd-disagg.html`（PD 分离模拟器前端）实现 `sglang.html` 演示系统，使 SGLang 仿真器具备与 PD-Disagg 同等水平的交互式可视化能力。

### 影响范围

- §1/§2 仅核查与评估，**不改动任何业务源码/测试代码**。
- §3 新增 `server/public/sglang.html`（前端，单文件自包含）+ 在 `server/src/` 增补 SGLang 仿真器的 HTTP 服务装配（使 `createSimulator()` 可被前端驱动）。所有改动限定在 `server/` 目录内。

---

## §1 一致性核查（代码实现 vs 技术报告）

### 1.1 核查方法

以技术报告章节为基准，对 `server/src/sglang/` 33 个源文件做结构化对照。每个维度给出：报告规格定位 → 代码实现定位 → 一致性结论。

### 1.2 核查结果总表

| 维度 | 报告规格 | 代码实现 | 一致性 | 证据 |
|---|---|---|---|---|
| **核心数据结构** | §2.2 SamplingParams / Req / Batch / Context | `types.ts` + `core/index.ts` | ✅ 一致 | `ModelConfig` 含 §2.2 全部字段（numLayers/hiddenSize/numKvHeads/headDim/vocabSize/isMoe/numExperts/moeTopK/useMla…）；`Req` 含 `dp_rank`（§2.2.2 注释要求）；`Batch`/`ForwardInput`/`ForwardOutput` 对齐 §9.11 |
| **配置系统** | §4.2 SimulatorConfig | `types.ts` `SimulatorConfig` | ✅ 一致 | 含 tpSize/dpSize/epSize/ppSize/cpSize + networkBandwidthGBps/networkLatencyUs/tpEfficiency/epEfficiency/cpEfficiency（P0 新增字段）；`DEFAULT_SIMULATOR_CONFIG` 全部并行维度默认 =1（退化为单实例） |
| **Context / 入口** | §4.2 Context + §4.3 SgSimInstance | `Simulator.ts` | ✅ 一致 | `SgSimContext`（newId 单调递增 / clock / reset）；`SgSimInstance` 接口（start/enqueue/getMetrics/loadWorkload/shutdown）；`createSimulator()` 工厂 |
| **请求生命周期** | §2.3 状态机 | `scheduler/index.ts` | ✅ 一致 | recv → PrefillManager → schedule_next_batch → _prepare_batch → _forward → _process_last_data 全链路在 `SimSchedulerImpl.runTick` 实现 |
| **KV Cache 抽象** | §3.3 + §9.3 | `cache/index.ts` | ✅ 一致 | `BaseKVCachePool`/`BasePrefixCache`/`BaseCacheHandle`/`MatchResult`/`InsertResult`/`CacheSizeInfo` + `TableManager` |
| **MockKVCachePool + Naive** | §3.3.3 + §9.3b | `cache/mha_pool.ts` / `naive_cache.ts` | ✅ 一致 | `cache_per_page` 公式页池；NaivePrefixCache 朴素前缀匹配 |
| **CacheManager 五区域** | §9.11 cache_req | `cache/cache_manager.ts` | ✅ 一致 | guard_header/prefix_hit/extend_new/guard_tail/lazy_free 五区域 + lazy_free_region + 页数守恒 |
| **RadixPrefixCache** | §9.8 | `cache/radix_cache.ts` | ✅ 一致 | RadixTreeNode / match / insert / split_at / LRU 最小堆驱逐 / lock_handle 引用计数 |
| **内存预算公式** | §3.3.9 | `cache/budget.ts` | ✅ 一致 | estimateModelMemory / estimateGraphBuffer / calculateMemoryBudget |
| **Prefill/Decode 调度** | §9.7 两次 available_size 检查 | `scheduler/index.ts` | ✅ 一致 | PrefillAdder 两次检查 + chunked prefill 续接；DecodeManager |
| **SimScheduler** | §9.11 normal_tick | `scheduler/index.ts` | ✅ 一致 | `SimSchedulerImpl.normal_tick` 接收→调度→forward→结果处理；SchedulerIOMixin |
| **CUDA Graph** | §3.3.6 | `engine/index.ts` `SimGraphRunner` | ✅ 一致 | bs 分桶 / can_use_cuda_graph / pad_batch / graphReplayCostTicks vs eagerForwardCostTicks |
| **Overlap Scheduling** | §9.4 | `scheduler/index.ts` | ✅ 一致 | `_overlap_tick` last_data 延迟 + 空 tick 刷新 + SimulationClock + 高水位背压 |
| **Workload/Metrics/HTTP** | §4.4 / §4.5 / §4.3 | `workload/` `metrics/` `api/` | ✅ 一致 | WorkloadGenerator（Poisson/CBR/trace）；SimulationMetrics 全字段（§4.5）；SGHttpApi（/v1/chat/completions、/v1/internal/metrics、/v1/internal/state） |
| **SimCommGroup** | §3.4.4 + §10.6 | `parallel/comm_group.ts` | ✅ 一致 | 5 group_type（tp/ep/pp/cp/dp_attn）；allReduce/allGather/allToAll/sendRecv/barrier；size=1 返回 0（noop 退化） |
| **ParallelTopology** | §4.2 + §10.7 | `parallel/topology.ts` | ✅ 一致 | rankToCoord/coordToRank/computeMoeRanks/computeAttnRanks/ppStageLayers；world_size=tp×dp×pp |
| **ParallelMetrics** | §10.9 | `parallel/metrics.ts` | ✅ 一致 | 16 并行字段 + 6 维度字段（TP/DP/EP/PP/CP 分组）+ commTicksTotal + summary() |
| **MockTPGroup 兼容** | §10.6 薄包装 | `parallel/comm_group.ts` | ✅ 一致 | 内部创建 SimCommGroup("tp")，旧 mockAllReduceSum 委托 allReduce |
| **TPSimulator + TPCommInfra** | §10.2 + §10.6 | `parallel/tp_simulator.ts` / `tp_comm_infra.ts` | ✅ 一致 | 权重÷tp / KV heads÷tp / 逐层 allReduce；ZMQ 广播+gloo barrier+nccl all-reduce 三层 |
| **并行内存预算 + 验证** | §10.7.2 + §10.7.3 | `parallel/budget.ts` / `validate.ts` | ✅ 一致 | calculateMemoryBudgetParallel（TP/DP/DPAttn/EP/CP/PP 修正）；validateParallelConfig 7 条约束 + KV 整除警告 |
| **DataParallelController** | §10.3 | `parallel/dp_controller.ts` | ✅ 一致 | DPRankState / round_robin / shortest_queue / allocate_pages / free_pages；dpSize=1 退化 |
| **DPAttentionSimulator** | §10.3.4 | `parallel/dp_attn.ts` | ✅ 一致 | 仅 useMla+enableDpAttention 启用；attn 不通信、MLP all-gather→forward→slice |
| **SimMoeBackend** | §10.4 | `parallel/moe.ts` | ✅ 一致 | mock/hash/simulated 三路由；_route_tokens→rank_distribution；_expert_to_rank O(1)；all-to-all 正反 |
| **EPLBSimulator** | §10.4.4 | `parallel/eplb.ts` | ✅ 一致 | 100 步周期 / 方差阈值<avg×0.1 跳过 / 贪心重排 / 固定 rebalanceCostTicks |
| **PPPipelineSimulator** | §10.5 | `parallel/pp.ts` | ✅ 一致 | gpipe/1f1b/interleaved 三调度 + bubble 公式 + micro-batch 分割 + isPpLast 仅最后 stage 采样 + 中间 stage isIntermediate |
| **CPSimulator** | §10.8 | `parallel/cp_simulator.ts` | ✅ 一致 | 长序列切分 / KV all-gather / cp group_type / cpSize=1 noop |
| **initParallelGroups** | §10.7.1 | `parallel/groups.ts` | ✅ 一致 | 9 组件条件创建（topology/tpComm/tpSim/dpController/dpAttnSim[cond]/ppSim/cpSim[cond]/eplbSim[cond]/moeBackend[cond]）；validateParallelConfig 先行 throw |
| **TS strict 约束** | §10.12 / §11 | `tsconfig.json` | ✅ 一致 | `strict: true`；dataclass→interface；通信纯算术；无 any 泄漏到公共 API |
| **单实例退化** | §10.11 | 全部并行组件 | ✅ 一致 | 所有 size=1 路径返回 noop，测试 B1–B4 覆盖（见 §2.3） |

### 1.3 一致性核查结论

**结论：`server/src/sglang/` 的代码实现与技术报告 `docs/SGLang_Simulator_Technical_Report.md` 总体方案在 28 个核查维度上全部一致，无规格偏离。**

代码内普遍以注释锚定报告章节（如 `// §10.7`、`// §4.3`、`// §9.11`），便于追溯。报告 §9.11「完整实现代码集」与 §10 并行规格的类级代码均在 TS 实现中得到忠实映射（Python dataclass → TS interface/class）。

### 1.4 核查中发现的唯一功能性缺口（非规格偏离）

技术报告未要求可视化演示，故 `sglang.html` 缺失不属于一致性偏差，而是 Issue #7 验收闭环的待补项（见 §3）。

---

## §2 特性整体分析

### 2.1 特性架构分析

**分层架构（依赖拓扑清晰）**：

```
Foundation  : types.ts(S0) → core/entities(S1)
              ↓
KV Cache    : cache/index(K1 抽象) ← budget(K5 公式)
              → mha_pool/naive_cache(K2) → cache_manager(K3) → radix_cache(K4)
              ↓
Scheduler   : scheduler/index(S2 PrefillAdder/DecodeManager → S3 SimScheduler
              → S4 SimGraphRunner → S5 Overlap/SimulationClock → S6 Workload/Metrics/HTTP)
              ↓
Parallel    : parallel/comm_group+topology+metrics(P0)
              → tp_simulator/tp_comm_infra(P1a) + budget/validate(P1b)
              → dp_controller(P2a) + dp_attn(P2b) + moe(P3a) + eplb(P3b) + pp(P4) + cp(P5)
              → groups(P6 集成)
```

**优点**：
- 依赖方向单一（Foundation→Cache→Scheduler→Parallel），无循环依赖；P1a–P5 七项在 P0 之后可并行开发，最终由 P6 收尾，符合报告 §5.1 关键路径。
- 模块边界与报告章节一一对应，`index.ts` 统一导出 22 个子 Issue 的全部公共符号，对外 API 表面稳定。
- 仿真/模拟边界清晰：调度、KV、拓扑、通信成本为「仿真」（纯算术）；Engine/Sampler/AttnBackend 为「模拟」（mock），符合报告 §3.1 分类原则。
- 通信成本模型集中在 `SimCommGroup`（5 group_type 统一），避免散落公式，符合 §3.4.4。

**架构层可改进点**：
- **服务装配缺口**：`server/src/index.ts` 入口仅装配 PD-Disagg 的 `SimService` + `HttpService`，**未装配 SGLang 仿真器**。`createSimulator()` 产出 `SgSimInstance`（含 scheduler/metrics/httpApi），但没有任何入口将其绑定到 `HttpService`（`setSGHttpApi` / `setSimulationMetrics`）。这导致 `/v1/internal/metrics` 端点实际无数据，是 sglang.html 演示无法工作的根因（详见 §3.2）。
- **交互控制缺口**：`SgSimInstance` 暴露 `start()/shutdown()` 但无 `pause/step/reset` 的 HTTP 控制，且无 `setParams` 热更新入口。PD-Disagg 的 `SimService` 有 `/command` `/params` `/preset` 三类控制端点，SGLang 侧缺失对应物，交互式演示需补齐。

### 2.2 代码整洁度分析

**优点**：
- 命名一致：TS 接口沿用报告 Python 命名（SimCommGroup/ParallelTopology/PPPipelineSimulator），仅做 camelCase 转换，可追溯性强。
- 通信成本纯算术，无 I/O、无定时器依赖（除 `setInterval` 驱动在线 tick），符合「零运行时依赖」约束。
- `SimulatorConfig` 默认值完备，全部并行维度 =1，回滚/退化安全。
- 文件粒度合理：33 个文件平均 ~6KB，最大 `scheduler/index.ts`（41KB）承担 S2–S5 多 Issue，虽偏大但内聚于调度域，可接受。

**可改进点（按严重度）**：
1. **类型不安全点**（中）：`Simulator.ts` L164–166 用 `as any` 注入 ctx 引用（`this.ctx.scheduler = this.scheduler as any` 等），绕过了 `SgSimContext` 占位字段的类型约束。报告 §10.12 要求禁用 any，此处应改 `SgSimContext` 占位字段为具体类型或私有 setter。
2. **运行时类型嗅探**（中）：`metrics/index.ts` L89 用 `(req as any).constructor.name === "ChunkedReq"` 判断 chunked prefill，依赖 minify 后可能失真的类名。应改为在 `Req`/`ChunkedReq` 上加 `isChunked: boolean` 显式标记。
3. **SGHttpApi 占位响应**（低）：`handleChatCompletions` 立即返回空 content 占位响应（`content: ""`），符合仿真语义（不生成真实文本），但演示时需在文档/注释中说明，避免误读为 bug。

### 2.3 测试完整程度分析

**测试基线实测**（在本 worktree 执行 `npx ts-node src/test/sglang-*.test.ts`，2026-09-03）：

| 测试文件 | 通过 | 失败 | 测试文件 | 通过 | 失败 |
|---|---|---|---|---|---|
| sglang-s0 | 22 | 0 | sglang-p0 | 37 | 0 |
| sglang-s1 | 26 | 0 | sglang-p1a | 25 | 0 |
| sglang-s2 | 48 | 0 | sglang-p1b | 32 | 0 |
| sglang-s3 | 运行通过 | 0 | sglang-p2a | 24 | 0 |
| sglang-s4 | 运行通过 | 0 | sglang-p2b | 16 | 0 |
| sglang-s5 | 运行通过 | 0 | sglang-p3a | 37 | 0 |
| sglang-s6 | 39 | 0 | sglang-p3b | 25 | 0 |
| sglang-k1 | 23 | 0 | sglang-pp | 39 | 0 |
| sglang-k2 | 31 | 0 | sglang-p5 | 23 | 0 |
| sglang-k3 | 35 | 0 | sglang-p6 | 45 | 0 |
| sglang-k4 | 41 | 0 | verify-metrics-http | 运行通过 | 0 |
| sglang-k5 | 20 | 0 | | | |

> 已量化用例合计 **588+ 通过 / 0 失败**（s3/s4/s5/verify 采用不同汇总格式，均运行无错退出）。

**优点**：
- 覆盖度：22 个子 Issue 各有专属测试文件（`sglang-<id>.test.ts`），1:1 对应，无遗漏。
- 边界覆盖：`sglang-p6.test.ts` 的 B1–B8 覆盖全部退化路径（dpSize=1/cpSize=1/enableEplb=false/isMoe=false/numPages=0/large world_size=32/EPLB 时机/PP 中间 stage）。
- 端到端：`sglang-p6` 含 45 用例验证 `initParallelGroups` 组合（tp=4,dp=2,ep=2,pp=2,cp=2）端到端；`verify-metrics-http` 验证 HTTP 端点。
- 时间模型：`sglang-s6` 验证 CUDA Graph/Eager 计数、GPU busy 基于 forward output 精确时间。

**可改进点**：
- **并行组合矩阵不足**：端到端组合仅 tp=4,dp=2,ep=2,pp=2,cp=2 一组；缺少「全 =1 退化端到端」「仅 TP」「TP+PP」「MoE+EP+EPLB 联动」的组合场景，建议补 parametric 矩阵。
- **测试运行器未纳管**：`package.json` 无 `test` 脚本，测试靠 `npx ts-node` 手动逐文件运行；建议加 `"test": "node --test"` 或聚合脚本，便于 CI。

---

## §3 sglang.html 演示系统设计

> 参照 `server/public/pd-disagg.html`（PD 分离模拟器前端，70KB 单文件，API Client Mode）实现 SGLang 仿真器的交互式可视化演示。pd-disagg.html 的模式为：前端单文件自包含（topbar 控制 + sidebar 参数 + canvas 流程图 + metrics 卡片 + i18n + presets），通过 HttpService `/api/*` 代理驱动服务端仿真引擎。sglang.html 沿用该模式，可视化对象改为 SGLang 的并行调度域。

### 3.1 总体思路

1. **服务端补齐**（`server/src/`）：新增 SGLang 仿真器的 HTTP 控制服务，使 `createSimulator()` 可被前端 pause/step/reset/drive，并暴露并行指标。**不改动** `sglang/` 仿真器内核，仅在其外层装配。
2. **前端新增**（`server/public/sglang.html`）：单文件自包含，复用 pd-disagg.html 的暗色主题/布局骨架/i18n/preset 机制，将可视化对象从「PD 实例拓扑 + KV 传输」替换为「并行 rank 拓扑 + 调度流水 + 通信成本分解」。

### 3.2 服务端设计（SgSimService 装配）

**新增文件**：`server/src/sglang_service/SgSimService.ts`（外层装配，不改 `sglang/` 内核）

- 包装 `createSimulator(config)` 产出 `SgSimInstance`，提供与 `SimService` 同构的控制面：
  - `GET /state` → 调度器快照（pendingReqs/runningReqs/availableTableIndices/tickCounter/globalStep/cacheSizeInfo）+ `metrics.toJSON()`（含 `parallel.summary()`）+ 当前 workload 进度。
  - `POST /command` → `{ action: "start"|"pause"|"step"|"reset", dt? }`：pause 暂停 setInterval；step 单步 `_runOneTick`；reset 重建 `SgSimInstance`。
  - `POST /params` → 增量更新 `SimulatorConfig`（tpSize/dpSize/epSize/ppSize/cpSize/cacheType/enableOverlap…）后 reset 重建实例（并行拓扑变更需重建）。
  - `POST /preset` → 加载预设 workload（balanced/prefillHeavy/decodeHeavy/moeEP/tpOnly/pp1f1b 等场景）。
  - `GET /health` → 存活检查。
- 监听独立端口（默认 :3002，与 PD-Disagg 的 :3001 隔离），避免双引擎冲突。

**入口改造**（`server/src/index.ts`，minimal edit）：
- 新增 `--mode=pd|sglang|both` 参数（默认 `pd`，保持向后兼容）。
- `--mode=sglang` 时：创建 `SgSimService`（:3002）+ `HttpService`（:8888，`setSGHttpApi` + `setSimulationMetrics` 注入），`HttpService` 默认入口改为 `/sglang.html`。
- `--mode=both` 时双引擎并存（PD :3001 + SGLang :3002），前端按需连接。

**HttpService 微调**（`server/src/http/HttpService.ts`，minimal edit）：
- `/` 默认入口由硬编码 `/pd-disagg.html` 改为可配置（构造参数 `defaultHtml`），sglang 模式下传 `/sglang.html`。
- 其余静态文件 / `/api/*` / `/v1/*` 路由不变。

### 3.3 前端设计（sglang.html）

**复用 pd-disagg.html 的骨架**：暗色 CSS 变量、topbar（☰/标题/暂停/单步/重置/倍速/场景/语言/GitHub）、sidebar 参数分组（range+select）、main 区 canvas + metrics 卡片网格、i18n 字典（zh/en）、preset 机制。

**差异化可视化（SGLang 并行调度域）**：

| 区域 | pd-disagg.html | sglang.html（新） |
|---|---|---|
| 顶部流程 canvas | PD 实例拓扑 + KV 传输链路 | **并行 rank 拓扑网格**：tp×dp×pp 单元格矩阵，按 `ParallelMetrics.dpRankLoad`/`epExpertLoad` 着色，CP/EP 重编号用副轴标注 |
| 中部分解 canvas | TTFT 分解（tokenize/prefill/transfer/…） | **tick 时间分解**：prefillBatchTime / decodeBatchTime / tpCommTicks / epCommTicks / ppSendRecvTicks / cpCommTicks / ppBubbleTicks 堆叠条，对应 `ParallelMetrics.commTicksTotal` |
| metrics 卡片 | TTFT/TPOT/吞吐/KV 利用率 | 吞吐(req/s, tok/s) + TTFT/TBT/E2E 延迟 + GPU 利用率 + CUDA Graph replay 计数 + Cache 命中率 + **并行通信总开销** + 各维度 comm ticks（TP/DP-Attn/EP/PP/CP 分项） |

**sidebar 参数分组**（对齐 `SimulatorConfig`）：
- `g.workload`：QPS / arrival（poisson/uniform）/ inputLen 均值 / outputLen 均值 / cacheHitRate
- `g.model`：modelPreset（Llama-8B/DeepSeek-V3 MLA+MoE/自定义）→ 联动 numLayers/numKvHeads/headDim/isMoe/numExperts/moeTopK
- `g.parallel`：tpSize / dpSize / epSize / ppSize / cpSize 滑块（1–8）+ dpLoadBalanceStrategy + ppPipelineSchedule(gpipe/1f1b/interleaved) + moeRoutingMode(mock/hash/simulated) + enableOverlap + enableCudaGraph + enableEplb + enableDpAttention
- `g.memory`：totalGpuMemory / memoryRatio / pageSize / cacheType(radix/naive)
- `g.comm`：networkBandwidthGBps / networkLatencyUs / tpEfficiency / epEfficiency / cpEfficiency

**场景预设（presets）**：
- `single`（全 =1，纯单实例退化基线）
- `tpOnly`（tp=4，验证 all-reduce 成本）
- `moeEP`（isMoe + ep=4 + enableEplb，验证 all-to-all + 负载均衡）
- `pp1f1b`（pp=4 + 1f1b 调度，验证 bubble）
- `cpLongSeq`（cp=2 + 长序列，验证 KV all-gather）
- `fullCombo`（tp=4,dp=2,ep=2,pp=2,cp=2 端到端组合，对应 p6 验收配置）

**数据获取**：前端通过 HttpService 代理轮询 `GET /api/state`（→ SgSimService /state），渲染前不做重计算，纯展示服务端仿真结果；与 pd-disagg.html 的「API Client Mode」完全一致（引擎在服务端，前端只渲染）。

### 3.4 修改点清单

| # | 范围 | 文件 | 类型 | 说明 |
|---|---|---|---|---|
| 1 | 服务装配 | `server/src/sglang_service/SgSimService.ts` | 新增 | SGLang 仿真器 HTTP 控制服务（/state /command /params /preset /health），包装 createSimulator |
| 2 | 入口改造 | `server/src/index.ts` | 编辑（minimal） | 加 `--mode=pd\|sglang\|both` 分支装配 SgSimService + HttpService |
| 3 | HttpService 微调 | `server/src/http/HttpService.ts` | 编辑（minimal） | `/` 默认入口可配置（defaultHtml 参数），sglang 模式指向 sglang.html |
| 4 | 演示前端 | `server/public/sglang.html` | 新增 | 单文件自包含前端（topbar+sidebar+canvas+metrics+i18n+presets），参照 pd-disagg.html 骨架 |
| 5 | 入口导航 | `server/public/pd-disagg.html` | 编辑（minimal，顶部加 sglang.html 跳转链接） | 与 sglang.html 顶部互链（pd-disagg.html 顶部已有 home 链接位，sglang.html 反向加） |

> 注：#5 对 pd-disagg.html 仅加一行跳转链接，不改动其可视化逻辑。

---

## 测试设计

### 验收测试用例清单

| 编号 | 范围 | 验证要点 |
|---|---|---|
| T-D1 | 服务装配 | `--mode=sglang` 启动后 SgSimService :3002 存活，`GET /health` 返回 200 |
| T-D2 | 端点 | `GET /api/state` 返回含 `scheduler` + `parallel` 字段的 JSON；`/v1/internal/metrics` 非空 |
| T-D3 | 控制 | `POST /command {action:"step"}` 推进 tickCounter +1；`pause` 后 tick 不再推进；`reset` 后 globalStep=0 |
| T-D4 | 参数热更 | `POST /params {tpSize:4}` 后 reset 重建，`/state.parallel.tpSize` = 4 |
| T-D5 | 预设 | `POST /preset {preset:"fullCombo"}` 后 `/state.parallel` 各维度 = (4,2,2,2,2) |
| T-D6 | 前端 | 浏览器加载 `/sglang.html` 无 console 错误；rank 拓扑网格渲染 tp×dp×pp 单元格；metrics 卡片显示并行通信总开销 |
| T-D7 | 一致性回归 | 现有 22 个 sglang-*.test.ts 全量回归仍 588+ 通过 / 0 失败（服务装配不改内核，回归应不变） |

### 边界条件覆盖

| 条件 | 预期行为 |
|---|---|
| 全并行维度 = 1（single 预设） | rank 网格 1×1×1，通信成本全 0，metrics 退化为单实例 |
| 仿真未 start 直接 step | SgSimService 容许 step（离线单步），不依赖 setInterval |
| 前端在仿真 reset 期间轮询 | /state 返回重建中快照（pendingReqs=0），前端显示空态 |
| 并行配置非法（如 epSize=2 但 isMoe=false） | validateParallelConfig throw，SgSimService 返回 400 + 错误明细，前端提示 |

---

## 风险与注意事项

### 兼容性影响
- §1/§2 为核查与分析，**零代码改动**，无兼容性影响。
- §3 改动限定 `server/`：新增 `sglang_service/` 与 `sglang.html`，对 `sglang/` 仿真器内核零改动；`index.ts`/`HttpService.ts` 为 additive minimal edit（新增分支/参数，默认行为不变），PD-Disagg 模式完全不受影响。
- 现有 588+ 测试基于 `sglang/` 内核，服务装配不触及内核，回归应全绿。

### 性能影响
- sglang.html 前端纯轮询渲染，无客户端仿真计算；服务端 SgSimService 复用 `SgSimInstance` 离线/在线 tick 循环，通信成本纯算术，性能与现有测试一致。
- `setInterval` 在线模式默认 10ms/tick，可由倍速控制调节，不影响 PD-Disagg 引擎。

### 回滚方案
- 新增文件可单独删除（`sglang_service/`、`sglang.html`）。
- `index.ts`/`HttpService.ts` 的 minimal edit 通过 git revert 单次回滚，默认 `--mode=pd` 行为不依赖新增代码。
- 不涉及 `sglang/` 内核与测试，回滚无副作用。
