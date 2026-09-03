---
title: "Issue #7 解决方案"
issue_number: 7
issue_type: Feature
created: 2026-09-03
updated: 2026-09-03
status: draft
review_round: 1
---

# Issue #7 解决方案

## 需求分析

### 问题描述

在 `server/src/sglang/` 目录下，使用 TypeScript strict 模式、零运行时依赖，构建忠实于技术报告规格的 **SGLang 单实例仿真器 + TP/DP/EP/PP/CP 并行仿真**。核心要求：

1. **单实例核心**：Scheduler / Prefill / Decode / Cache / Table / Radix / Graph / Overlap / Memory 全部组件
2. **通信基础设施**：`SimCommGroup` 统一 5 种 group_type（tp/ep/pp/cp/dp_attn）成本模型
3. **拓扑**：`ParallelTopology` rank↔(tp,dp,pp) 坐标互转、MoE/Attention 层级 rank 推导
4. **TP/DP/EP/PP/CP 五种并行策略**完整仿真
5. **并行组合集成**：`initParallelGroups` + `calculateMemoryBudgetParallel` + `validateParallelConfig`
6. **TS strict 约束**：所有 Python dataclass → TS interface，通信成本纯算术，size=1 时完全退化为 noop

### 能力目标

| 维度 | 目标 |
|---|---|
| 单实例调度 | 支持 prefill/decode batch 分离、chunked prefill、overlap scheduling |
| KV Cache | 页式管理、前缀缓存（naive + radix）、LRU 驱逐、内存预算公式 |
| CUDA Graph | batch size 分桶、replay vs eager 时间模型、graph invalidation |
| 并行仿真 | TP/DP/EP/PP/CP 五种策略 + 组合修正 + 7 条配置约束验证 |
| 仿真闭环 | WorkloadGenerator → SimScheduler tick 循环 → Metrics → HTTP API |

### 影响范围

所有代码限定在 `server/src/sglang/` 目录，涉及 22 个子 Issue 的实现：

- **Foundation (2)**：S0 骨架配置、S1 数据结构
- **KVCache (5)**：K1-K5
- **调度核心 (5)**：S2-S6
- **并行基础设施 (1)**：P0
- **并行策略 (8)**：P1a, P1b, P2a, P2b, P3a, P3b, P4, P5
- **组合集成 (1)**：P6

## 改造方案

### 总体思路

采用分层依赖拓扑逐步构建，遵循 S0→S1→K1→K5→K2→K3→K4→S2→S3→S4→S5→S6→P0→P1a~P5→P6 的关键路径。所有并行策略在 P0 之后可并行开发，最终由 P6 收尾集成。

### 详细设计

#### 1. Foundation 层 (S0 + S1)

**S0 — 骨架与配置**

- `types.ts`：定义 `SimMode`, `ModelConfig`, `SimulatorConfig`, 消息类型（`SimRequestMsg`, `SimRespMsg`, `SchedulerMsg`）
- `Simulator.ts`：`SgSimContext`（全局上下文，newId/clock/reset）、`Simulator`（入口桩）、`SgSimInstance`（完整实例接口）
- `index.ts`：统一导出

关键接口：
```typescript
interface SimulatorConfig {
  modelConfig: ModelConfig;
  tpSize: number; dpSize: number; epSize: number; ppSize: number; cpSize: number;
  networkBandwidthGBps: number; networkLatencyUs: number;
  // ... 调度/CUDA Graph/Overlap/内存/采样/仿真控制参数
}
```

**S1 — 数据结构与工具函数**

- `core/index.ts`：`SamplingParams` class、`Req` class（含 `dpRank` 字段）、`Batch` class、`ForwardOutput`、`BatchSamplingArgs`
- `entities/index.ts`：`ChunkedReq`、`PendingReq`
- 工具函数：`alignDown`, `divCeil`, `divEven`, `bytesPerElement`

#### 2. KVCache 子系统 (K1-K5)

**K1 — 抽象层**
- `BaseKVCachePool`, `BasePrefixCache`, `BaseCacheHandle`, `MatchResult`, `InsertResult`, `CacheSizeInfo`
- `TableManager`：page_table 行分配管理

**K5 — 内存预算基础公式**
- `estimateModelMemory()`：权重 + embedding + 附加开销
- `estimateGraphBuffer()`：CUDA Graph buffer 估算
- `calculateMemoryBudget()`：`numPages = floor(available / (bytesPerToken × pageSize))`

**K2 — MockKVCachePool + NaivePrefixCache**
- `MockKVCachePool`：基于 `cache_per_page` 公式的页式 KV 池
- `NaivePrefixCache`：朴素前缀匹配 + `NaiveCacheHandle`

**K3 — CacheManager**
- `cache_req` 五区域分配：guard_header / prefix_hit / extend_new / guard_tail / lazy_free
- `lazy_free_region`：延迟释放机制
- 页数守恒验证

**K4 — RadixPrefixCache**
- `RadixTreeNode`：前缀树节点
- `match / insert / split_at` 操作
- LRU 最小堆驱逐策略
- `lock_handle` 引用计数

#### 3. 调度与仿真核心 (S2-S6)

**S2 — PrefillManager + DecodeManager**
- `PrefillAdder`：两次 `available_size` 检查 + chunked prefill 续接
- `DecodeManager`：decode batch 组装

**S3 — SimScheduler**
- `normal_tick`：接收消息 → 调度决策 → forward → 结果处理
- `SchedulerIOMixin`：消息收发
- `MockEngine` / `MockSampler` / `MockAttnBackend`

**S4 — SimGraphRunner**
- CUDA Graph batch size 分桶（1,2,4,8,16,...,maxBs）
- `graphReplayCostTicks = base × (1 + 0.05 × bs/128)`
- `eagerForwardCostTicks`：prefill 按 token 数线性、decode 按 bs 缩放

**S5 — Overlap Scheduling**
- `_overlap_tick`：last_data 延迟 + 空 tick 刷新
- `SimulationClock`：GPU 时序追踪、重叠检测
- 高水位背压机制

**S6 — WorkloadGenerator + SimulationMetrics + HTTP API**
- `WorkloadGenerator`：Poisson/CBR/trace 驱动
- `SimulationMetrics`：吞吐/延迟/Cache/GPU/CUDA Graph/并行指标
- `SGHttpApi`：简化 HTTP 服务

#### 4. 并行基础设施 (P0)

**SimCommGroup — 统一通信成本模型**

```typescript
class SimCommGroup {
  constructor(opts: SimCommGroupOpts)  // groupType, size, bandwidth, latency, efficiency
  allReduce(tensorBytes): number   // 2×bytes×(size-1)/size/bw + latency
  allGather(sizes[]): number       // bytes×(size-1)/bw + latency
  allToAll(send[], recv[]): number // bytes×size/bw + latency×size
  sendRecv(bytes, peer): number    // bytes/bw + latency
  barrier(): void                  // noop
}
```
size=1 时 allReduce/allGather/allToAll 返回 0（noop 退化）。

**ParallelTopology — 拓扑映射**

```typescript
class ParallelTopology {
  rankToCoord(rank): [tpIdx, dpIdx, ppIdx]
  coordToRank(tpIdx, dpIdx, ppIdx): rank
  computeMoeRanks(tpRank): [moeDpRank, moeEpRank, moeTpRank]
  computeAttnRanks(tpRank): [attnCpRank, attnTpRank]
  ppStageLayers(numLayers): Array<{start, end}>
}
```

**ParallelMetrics — 18+ 字段并行指标子结构**

TP/DP/EP/PP/CP 分组指标 + 通用维度字段，嵌入 `SimulationMetrics.parallel`。

**MockTPGroup — 向后兼容薄包装**

内部创建 `SimCommGroup("tp")`，旧 `mockAllReduceSum` 委托 `allReduce`。

#### 5. 并行策略 (P1a-P5)

**P1a — TPSimulator + TPCommInfraSimulator**
- `TPSimulator`：权重÷tp、KV heads÷tp、逐层 allReduceAfterAttn/allReduceAfterMlp
- `TPCommInfraSimulator`：ZMQ 广播 + gloo barrier + nccl all-reduce 三层

**P1b — calculateMemoryBudgetParallel + validateParallelConfig**
- 并行组合内存预算：TP 权重修正 → EP MoE 权重修正 → PP stage 切分 → DP KV pool 划分 → CP KV 倍增 → DP-Attn KV 倍增
- 7 条约束验证：world_size 一致性、EP 仅 MoE、CP 整除 TP、EP 整除 TP/CP、PP 层数覆盖、DP-Attn 需要 MLA、memoryRatio 范围
- KV 整除警告

**P2a — DataParallelController**
- `DPRankState`：rank/load/pages_allocated/pages_capacity
- `round_robin` / `shortest_queue` 两种分发策略
- `allocate_pages` / `free_pages` 页管理
- dpSize=1 时退化为始终选 rank 0

**P2b — DPAttentionSimulator**
- 仅 `useMla && enableDpAttention && dpSize > 1` 时启用
- Attention 层不通信（MLA KV cache 每 rank 自维护）
- MLP 层 all-gather → forward → slice

**P3a — SimMoeBackend**
- 三种路由模式：mock（均匀轮转）/ hash（splitmix32 确定性）/ simulated（mulberry32 伪随机）
- `_routeTokens` → rank_distribution + expert_counts
- `_expertToRank`：O(1) 查表
- forward：路由 → 正向 all-to-all → mock forward → 反向 all-to-all

**P3b — EPLBSimulator**
- 100 步周期检查
- 方差阈值 `<avg×0.1` 跳过
- 贪心重排：过载 rank 搬最小负载 expert 到欠载 rank
- 固定 `rebalanceCostTicks`

**P4 — PPPipelineSimulator**
- 三种调度：gpipe（bubble = (pp-1)×mb×numMB）/ 1f1b（bubble = (pp-1)×mb）/ interleaved（bubble = (pp-1)×chunks×mb）
- micro-batch 分割（ceil 分配，不 padding）
- `_stageSendRecvCost`：dataBytes = microBatchSize × ceil(hidden/tp) × dtypeSize × 2
- 通信计算重叠折算：`max(0, rawCost - microBatchTicks)`
- `isPpLastStage`：仅最后 stage 采样

**P5 — CPSimulator**
- 长序列切分：`seqLenPerRank = divCeil(seqLen, cpSize)`
- 每层 attention 后 KV all-gather：`kvBytesPerRank = seqLenPerRank × numKvHeads × headDim × dtypeSize × 2`
- MLP 层不通信

#### 6. 并行组合集成 (P6)

**initParallelGroups — 统一初始化**

```typescript
function initParallelGroups(opts): ParallelGroups {
  // 1. validateParallelConfig
  // 2. 创建 ParallelTopology
  // 3. 按条件创建 9 组件：
  //    topology, tpComm, tpSim, dpController,
  //    dpAttnSim (条件), ppSim, cpSim (条件),
  //    eplbSim (条件), moeBackend (条件)
}
```

**MockEngine 完整层循环**

```
层循环前：ZMQ 广播 token IDs
每层：
  Attention + CP KV all-gather
  → TP all-reduce after attn
  → MLP / MoE (EP all-to-all)
  → TP all-reduce after MLP (非 MoE)
  → DP-Attn all-gather after MLP
层循环后：
  CPU barrier
  → PP 通信仿真（CUDA Graph replay 时跳过）
  → TP 通信指标汇总
  → 采样（仅最后 PP stage）
```

### 修改点清单

| # | 范围 | 文件 | 说明 |
|---|---|---|---|
| 1 | S0 | `types.ts`, `Simulator.ts`, `index.ts` | 骨架配置 + 消息类型 + Context |
| 2 | S1 | `core/index.ts`, `entities/index.ts` | 数据结构 + 工具函数 |
| 3 | K1 | `cache/index.ts` | 抽象层 + TableManager |
| 4 | K5 | `cache/budget.ts` | 内存预算基础公式 |
| 5 | K2 | `cache/mha_pool.ts`, `cache/naive_cache.ts` | MockKVCachePool + NaivePrefixCache |
| 6 | K3 | `cache/cache_manager.ts` | CacheManager 五区域分配 |
| 7 | K4 | `cache/radix_cache.ts` | RadixPrefixCache + LRU 驱逐 |
| 8 | S2 | `scheduler/index.ts` | PrefillManager + DecodeManager |
| 9 | S3 | `scheduler/index.ts`, `engine/index.ts` | SimScheduler + MockEngine |
| 10 | S4 | `engine/index.ts` | SimGraphRunner |
| 11 | S5 | `scheduler/index.ts` | Overlap Scheduling + SimulationClock |
| 12 | S6 | `metrics/index.ts`, `workload/index.ts`, `api/index.ts` | Metrics + Workload + HTTP |
| 13 | P0 | `parallel/comm_group.ts`, `parallel/topology.ts`, `parallel/metrics.ts` | 通信成本 + 拓扑 + 指标 |
| 14 | P1a | `parallel/tp_simulator.ts`, `parallel/tp_comm_infra.ts` | TP 仿真 + 通信基础设施 |
| 15 | P1b | `parallel/budget.ts`, `parallel/validate.ts` | 并行内存预算 + 配置验证 |
| 16 | P2a | `parallel/dp_controller.ts` | 标准 DP 分发器 |
| 17 | P2b | `parallel/dp_attn.ts` | DP-Attn 仿真器 |
| 18 | P3a | `parallel/moe.ts` | MoE 路由 + all-to-all |
| 19 | P3b | `parallel/eplb.ts` | EP 负载均衡 |
| 20 | P4 | `parallel/pp.ts` | PP 流水并行 |
| 21 | P5 | `parallel/cp_simulator.ts` | CP Context Parallel |
| 22 | P6 | `parallel/groups.ts`, `engine/index.ts` | 并行组件集成 + 层循环 |

## 测试设计

### 验收测试用例清单

| 编号 | 测试范围 | 验证要点 |
|---|---|---|
| T-S0 | S0 骨架 | SimulatorConfig 默认值完整、SgSimContext newId 单调递增、消息类型联合完备 |
| T-S1 | S1 数据结构 | SamplingParams 默认值、Req 创建/completeOne、Batch 增删、divEven/alignDown/divCeil 边界 |
| T-K1 | K1 抽象层 | TableManager allocate/free 可用性、MatchResult/InsertResult 结构完整性 |
| T-K2 | K2 实现 | MockKVCachePool 页分配回收、NaivePrefixCache match/insert |
| T-K3 | K3 CacheManager | cache_req 五区域分配、lazy_free_region、页数守恒 |
| T-K4 | K4 RadixPrefixCache | match/insert/split_at 正确性、LRU 驱逐触发、lock_handle 引用计数 |
| T-K5 | K5 内存预算 | calculateMemoryBudget 返回合理 numPages、OOM 边界 |
| T-S2 | S2 调度 | PrefillAdder 两次 available_size 检查、chunked prefill 续接 |
| T-S3 | S3 调度器 | normal_tick 完整循环、消息分发、forward→结果处理 |
| T-S4 | S4 Graph | SimGraphRunner 分桶策略、replay vs eager 时间模型 |
| T-S5 | S5 Overlap | last_data 延迟处理、空 tick 刷新、高水位背压 |
| T-S6 | S6 闭环 | WorkloadGenerator 生成请求、Metrics 记录、HTTP 端点响应 |
| T-P0 | P0 并行基础 | SimCommGroup size=1 noop、size>1 成本公式、ParallelTopology 坐标映射、computeMoeRanks/computeAttnRanks |
| T-P1a | P1a TP | TPSimulator allReduce 成本、TPCommInfra ZMQ/barrier/nccl 三层 |
| T-P1b | P1b 预算+验证 | calculateMemoryBudgetParallel 修正公式、validateParallelConfig 7 条约束 + KV 警告 |
| T-P2a | P2a DP | round_robin/shortest_queue 分发、allocate_pages/free_pages 页管理 |
| T-P2b | P2b DP-Attn | 仅 useMla+enableDpAttention 启用、simulateMlpForward 成本 |
| T-P3a | P3a EP | 三种路由模式、all-to-all 正反、expertToRank 一致性 |
| T-P3b | P3b EPLB | 100 步周期、方差阈值、贪心重排 |
| T-P4 | P4 PP | 三种调度 bubble 公式、micro-batch 分割、isPpLast 采样逻辑 |
| T-P5 | P5 CP | KV all-gather 字节数公式、cpSize=1 noop |
| T-P6 | P6 集成 | initParallelGroups 条件创建、完整层循环通信成本、端到端 tp=4,dp=2,ep=2,pp=2,cp=2 组合 |

### 边界条件覆盖

| 条件 | 预期行为 |
|---|---|
| 所有并行维度 = 1 | 完全退化为纯单实例，所有通信成本为 0 |
| tpSize > numKvHeads | allowReplicate=true 允许 head 复制，部分 rank KV heads = 0 |
| ppSize > numLayers | validateParallelConfig 报错（约束 5） |
| epSize > 1 但 isMoe=false | validateParallelConfig 报错（约束 2） |
| cpSize 不整除 tpSize | validateParallelConfig 报错（约束 3） |
| enableDpAttention 但 useMla=false | validateParallelConfig 报错（约束 6） |
| numPages = 0 | OOM 警告，仿真继续但无法分配页 |
| CUDA Graph replay 路径 | 跳过 PP 通信仿真，使用 graphReplayCostTicks 时间模型 |
| Batch size 超过 CUDA Graph maxBs | 回退到 eager forward 路径 |
| EPLB avg = 0 | 安全返回不重平衡 |

## 风险与注意事项

### 兼容性影响

- 所有新增代码限定在 `server/src/sglang/` 目录，不影响前端和已有 PD-Disagg 仿真模块
- `MockTPGroup` 薄包装确保旧接口兼容
- `SimulatorConfig` 通过默认值保持向后兼容，新增并行参数均有合理默认值（全部 =1）

### 性能影响

- 通信成本纯算术计算（无 I/O），仿真性能不受网络延迟影响
- `computeMoeRanks` / `computeAttnRanks` 为 O(1) 纯函数
- `SimMoeBackend._expertToRank` O(1) 查表替代遍历
- `EPLBSimulator` 贪心重排 O(experts × ranks)，100 步周期下开销可控

### 回滚方案

- 所有新代码位于独立目录 `server/src/sglang/`，可通过 git revert 单次回滚
- `SimulatorConfig` 新增字段均有默认值，回滚不影响现有配置
- 测试文件独立于业务代码，回滚无副作用
