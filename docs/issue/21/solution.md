---
title: "Issue #21 解决方案"
issue_number: 21
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #21 解决方案

## 需求分析

### 问题描述

Issue #21 要求在 `server/src/sglang/parallel/**` 下实现 P0 层并行仿真基础设施，包括三大核心组件：

1. **SimCommGroup**（§3.4.4）：统一通信成本模型，支持 TP all-reduce / EP all-to-all / PP send-recv / CP all-gather / DP-Attention all-gather 五种通信原语
2. **ParallelTopology**（§4.2）：并行拓扑配置，描述 TP×DP×PP 进程网格，支持 rank↔coord 双向映射、MoE/Attention 层级 rank 推导、PP stage 层分割
3. **ParallelMetrics**（§10.9）：并行仿真指标子结构（18+ 字段），嵌入 SimulationMetrics

此外还需：
- SimulatorConfig 新增并行带宽/延迟/效率参数
- MockTPGroup 薄包装（向后兼容旧代码）

### 能力目标

- 提供纯算术通信成本模型：给定通信操作类型和数据量，返回 ticks 开销
- size=1 时所有通信操作退化为 noop（返回 0），不破坏现有单实例行为
- 支持 rank 到 (tp, dp, pp) 三元组的双向映射
- 支持从 TP rank 推导 MoE 层级 (moe_dp, moe_ep, moe_tp) 和 Attention 层级 (attn_cp, attn_tp) 的 rank 映射
- 提供 PP stage 层分割算法
- 收集所有并行维度的通信与负载指标

### 影响范围

- **新增文件**：`server/src/sglang/parallel/comm_group.ts`、`server/src/sglang/parallel/topology.ts`、`server/src/sglang/parallel/metrics.ts`
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出）、`server/src/sglang/types.ts`（SimulatorConfig 新增字段 + SimCommGroup 接口升级）、`server/src/sglang/index.ts`（导出）、`server/src/sglang/metrics/index.ts`（SimulationMetrics 嵌入 ParallelMetrics）
- **不修改**：业务调度逻辑、测试代码、sim/ 引擎代码

### 依赖关系

- **依赖 #10 (S1)**：核心数据结构 Req/Batch/SamplingParams + 工具函数 divEven/divCeil/alignDown/bytesPerElement — **已实现**
- **依赖 #12 (K5)**：内存预算 calculateMemoryBudget — **已实现**（ParallelTopology 初始化可用 K5 估算值）

---

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建三个模块，分别实现 SimCommGroup、ParallelTopology、ParallelMetrics，并通过 `parallel/index.ts` 统一导出。同时扩展 SimulatorConfig 接口和 DEFAULT 配置，并升级 `types.ts` 中的 SimCommGroup 占位接口为完整接口。

### 详细设计

#### 1. SimCommGroup — 统一通信成本模型

**文件**：`server/src/sglang/parallel/comm_group.ts`

**通信组类型**：

```typescript
export type CommGroupType = "tp" | "ep" | "pp" | "cp" | "dp_attn";
```

**成本模型核心公式**（对应 Issue 中的纯算术模型）：

```
bandwidth_bytes_per_us = network_bandwidth_GBps * 1e9 / 1e6 = network_bandwidth_GBps * 1000

all_reduce(bytes) = size=1 ? 0 : 2 * bytes * (size-1) / size / bandwidth_bytes_per_us + latency_us
all_gather(sizes)  = size=1 ? 0 : sum(sizes) * (size-1) / bandwidth_bytes_per_us + latency_us
all_to_all(send_sizes, recv_sizes) = size=1 ? 0 : (sum(send_sizes) + sum(recv_sizes)) * size / bandwidth_bytes_per_us + latency_us * size
send_recv(bytes, peer) = bytes / bandwidth_bytes_per_us + latency_us
```

**效率因子**：TP/EP/CP 各自的 efficiency 参数（0~1）用于缩放结果：

```
actual_ticks = computed_ticks / efficiency
```

**类设计**：

```typescript
export class SimCommGroup {
  readonly groupType: CommGroupType;
  readonly size: number;
  readonly ranks: number[];
  readonly globalRanks: number[];
  readonly networkBandwidthGBps: number;
  readonly latencyUs: number;
  readonly efficiency: number;

  // 内部换算
  private readonly bandwidthBytesPerUs: number;

  constructor(opts: SimCommGroupOpts);
  private _computeCost(opType: CommOpType, bytes: number): number;
  allReduce(tensorBytes: number): number;
  allGather(sizes: number[]): number;
  allToAll(sendSizes: number[], recvSizes: number[]): number;
  sendRecv(bytes: number, peer: number): number;
  barrier(): void;
}
```

**构造选项**：

```typescript
export interface SimCommGroupOpts {
  groupType: CommGroupType;
  size: number;
  ranks?: number[];           // 默认 [0..size-1]
  globalRanks?: number[];     // 默认同 ranks
  networkBandwidthGBps: number;
  latencyUs: number;
  efficiency?: number;        // 默认 1.0
}
```

**关键设计点**：

- size=1 时 allReduce/allGather/allToAll 全部返回 0（noop 退化）
- sendRecv 无 size=1 特殊处理，始终按公式计算
- `_computeCost` 内部统一换算：bytes → 通过 bandwidth 和 latency 计算 ticks
- barrier() 为 noop，不产生成本
- 所有公开方法返回 `comm_ticks`（整数，向上取整 ceil）

#### 2. ParallelTopology — 并行拓扑映射

**文件**：`server/src/sglang/parallel/topology.ts`

**构造选项**：

```typescript
export interface ParallelTopologyOpts {
  tpSize?: number;    // 默认 1
  dpSize?: number;    // 默认 1
  epSize?: number;    // 默认 1（MoE EP，TP rank 重编号，不增加进程数）
  ppSize?: number;    // 默认 1
  cpSize?: number;    // 默认 1（Context Parallel，TP rank 重编号）
  enableDpAttention?: boolean;  // 默认 false
}
```

**类设计**：

```typescript
export class ParallelTopology {
  readonly tpSize: number;
  readonly dpSize: number;
  readonly epSize: number;
  readonly ppSize: number;
  readonly cpSize: number;
  readonly enableDpAttention: boolean;

  constructor(opts?: ParallelTopologyOpts);

  get worldSize(): number;          // tp × dp × pp
  get numDpGroups(): number;        // tp × pp
  get numPpStages(): number;        // pp_size

  rankToCoord(rank: number): [number, number, number];  // → (tp, dp, pp)
  coordToRank(tpIdx: number, dpIdx: number, ppIdx: number): number;

  computeMoeRanks(tpRank: number): [number, number, number]; // → (moe_dp, moe_ep, moe_tp)
  computeAttnRanks(tpRank: number): [number, number];        // → (attn_cp, attn_tp)

  ppStageLayers(numLayers: number): Array<{ start: number; end: number }>;
}
```

**rank↔coord 映射公式**（tp 在最内层，dp 居中，pp 最外层）：

```
rank = pp_idx × (dp_size × tp_size) + dp_idx × tp_size + tp_idx
tp_idx = rank % tp_size
dp_idx = (rank // tp_size) % dp_size
pp_idx = rank // (tp_size × dp_size)
```

**MoE rank 推导**（对应 SGLang `_compute_parallelism_ranks`）：

```
moe_dp_size = dp_size
moe_tp_size = max(1, tp_size // dp_size // ep_size)
moe_dp_rank = tp_rank // (tp_size // moe_dp_size)
moe_ep_rank = (tp_rank % (tp_size // moe_dp_size)) // moe_tp_size
moe_tp_rank = tp_rank % moe_tp_size
```

**Attention rank 推导**：

```
attn_dp_size = dp_size (若 enableDpAttention) 或 1
attn_tp_size = max(1, tp_size // attn_dp_size // cp_size)
attn_cp_rank = (tp_rank // attn_tp_size) % cp_size
attn_tp_rank = tp_rank % attn_tp_size
```

**PP stage 层分割**：num_layers 按 pp_size 均分，余数分配到前几个 stage。

**约束验证**：
- cp_size 必须整除 tp_size
- ep_size 必须整除 tp_size / cp_size

#### 3. ParallelMetrics — 并行指标子结构

**文件**：`server/src/sglang/parallel/metrics.ts`

```typescript
export class ParallelMetrics {
  // TP 指标
  tpCommTicks: number = 0;
  tpAllReduceCount: number = 0;
  tpWeightBytes: number = 0;

  // DP 指标
  dpRankLoad: number[] = [];
  dpAllocatePagesPerRank: number[] = [];
  dpAttnCommTicks: number = 0;

  // EP 指标
  epCommTicks: number = 0;
  epAllToAllCount: number = 0;
  epCrossRankTokens: number = 0;
  epExpertLoad: number[] = [];
  epRebalanceCostTicks: number = 0;

  // PP 指标
  ppBubbleTicks: number = 0;
  ppNumMicroBatches: number = 0;
  ppSendRecvTicks: number = 0;

  // CP 指标
  cpCommTicks: number = 0;
  cpAllGatherCount: number = 0;
  cpSeqLenPerRank: number = 0;

  // 通用维度
  worldSize: number = 1;
  tpSize: number = 1;
  dpSize: number = 1;
  epSize: number = 1;
  ppSize: number = 1;
  cpSize: number = 1;

  // 汇总
  get commTicksTotal(): number;

  summary(): Record<string, unknown>;
  reset(): void;
}
```

**commTicksTotal 计算**：

```
commTicksTotal = tpCommTicks + dpAttnCommTicks + epCommTicks + ppSendRecvTicks + cpCommTicks
```

#### 4. SimulatorConfig 新增字段

在 `types.ts` 的 `SimulatorConfig` 接口中新增：

```typescript
// ===== 并行通信统一参数（P0 新增）=====
networkBandwidthGBps: number;       // 网络带宽 GB/s，默认 100
networkLatencyUs: number;           // 网络延迟 μs，默认 5
tpEfficiency: number;               // TP 通信效率因子 [0,1]，默认 0.95
epEfficiency: number;               // EP 通信效率因子 [0,1]，默认 0.90
cpEfficiency: number;               // CP 通信效率因子 [0,1]，默认 0.90
```

在 `DEFAULT_SIMULATOR_CONFIG` 中补充默认值：

```typescript
networkBandwidthGBps: 100,
networkLatencyUs: 5,
tpEfficiency: 0.95,
epEfficiency: 0.90,
cpEfficiency: 0.90,
```

#### 5. SimCommGroup 接口升级

在 `types.ts` 中升级现有的 `SimCommGroup` 占位接口：

```typescript
/** 通信组（P0 完整实现） */
export interface SimCommGroup {
  readonly groupType: CommGroupType;
  readonly size: number;
  allReduce(tensorBytes: number): number;
  allGather(sizes: number[]): number;
  allToAll(sendSizes: number[], recvSizes: number[]): number;
  sendRecv(bytes: number, peer: number): number;
  barrier(): void;
}
```

新增 `CommGroupType` 类型导出：

```typescript
export type CommGroupType = "tp" | "ep" | "pp" | "cp" | "dp_attn";
```

#### 6. MockTPGroup 向后兼容薄包装

在 `comm_group.ts` 中实现：

```typescript
export class MockTPGroup implements SimCommGroup {
  readonly groupType: CommGroupType = "tp";
  readonly size: number;
  private readonly inner: SimCommGroup;

  constructor(tpSize: number, config: SimulatorConfig);

  allReduce(tensorBytes: number): number;  // 委托 inner.allReduce
  allGather(sizes: number[]): number;
  allToAll(sendSizes: number[], recvSizes: number[]): number;
  sendRecv(bytes: number, peer: number): number;
  barrier(): void;

  /** 旧接口兼容 */
  mockAllReduceSum(dataBytes: number): number;  // = allReduce(dataBytes)
}
```

**关键**：当 tpSize=1 时，inner.size=1，allReduce 直接返回 0，与旧 noop 行为一致。

#### 7. metrics/index.ts 集成

在 `metrics/index.ts` 中新建 `SimulationMetrics` 类，嵌入 `ParallelMetrics`：

```typescript
import { ParallelMetrics } from "../parallel";

export class SimulationMetrics {
  readonly parallel: ParallelMetrics = new ParallelMetrics();

  // 其他已有字段（后续 Issue 补充）

  reset(): void {
    this.parallel.reset();
  }
}
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/comm_group.ts` — SimCommGroup + MockTPGroup 实现
2. **新建** `server/src/sglang/parallel/topology.ts` — ParallelTopology 实现
3. **新建** `server/src/sglang/parallel/metrics.ts` — ParallelMetrics 实现
4. **修改** `server/src/sglang/parallel/index.ts` — 导出上述三个模块
5. **修改** `server/src/sglang/types.ts` — SimulatorConfig 新增 5 个字段 + DEFAULT 配置补充默认值 + SimCommGroup 接口升级 + CommGroupType 类型导出
6. **修改** `server/src/sglang/index.ts` — 导出新增类型和类
7. **修改** `server/src/sglang/metrics/index.ts` — 新建 SimulationMetrics 类嵌入 ParallelMetrics

---

## 测试设计

### 验收测试用例清单

#### SimCommGroup 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | allReduce size=1 返回 0 | tp_size=1 时 all_reduce 为 noop |
| T2 | allReduce size>1 返回正数 | 验证 all_reduce 公式：2×bytes×(size-1)/size / bw + latency |
| T3 | allGather size=1 返回 0 | ep_size=1 时 all_gather 为 noop |
| T4 | allGather size>1 返回正数 | 验证 all_gather 公式 |
| T5 | allToAll size=1 返回 0 | ep_size=1 时 all_to_all 为 noop |
| T6 | allToAll size>1 返回正数 | 验证 all_to_all 公式含 latency×size |
| T7 | sendRecv 正常计算 | 不受 size=1 影响，始终按公式 |
| T8 | barrier 为 noop | 调用无异常且无成本 |
| T9 | efficiency 缩放 | efficiency=0.5 时结果翻倍 |
| T10 | CommGroupType 全类型 | 验证 5 种 group_type 可正常构造 |

#### ParallelTopology 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T11 | worldSize 计算正确 | tp×dp×pp |
| T12 | rankToCoord/coordToRank 互逆 | 验证双向映射一致性 |
| T13 | computeMoeRanks tp=8 dp=2 ep=2 | moe_dp=2, moe_ep=2, moe_tp=2 |
| T14 | computeAttnRanks tp=8 cp=2 | attn_cp_rank 和 attn_tp_rank 正确 |
| T15 | ppStageLayers 32层 pp=4 | 各 stage 8 层 |
| T16 | ppStageLayers 33层 pp=4 | 前 1 stage 9 层，其余 8 层 |
| T17 | cp_size 整除 tp_size 约束 | cp=3 tp=8 抛出 Error |
| T18 | ep_size 整除 tp_size/cp_size 约束 | ep=3 tp=8 抛出 Error |

#### ParallelMetrics 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T19 | 默认值全为 0/空 | 新实例所有计数器为 0 |
| T20 | commTicksTotal 计算 | tp+dp+ep+pp+cp 之和 |
| T21 | reset 清零 | 调用后所有字段回到默认值 |
| T22 | summary 包含全部 22 字段 | 验证输出完整性 |

#### MockTPGroup 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T23 | MockTPGroup(1) allReduce=0 | tp_size=1 时 noop |
| T24 | MockTPGroup(2) allReduce>0 | 委托内部 SimCommGroup |
| T25 | mockAllReduceSum 兼容 | 与 allReduce 结果一致 |

#### SimulatorConfig 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T26 | DEFAULT 含新增字段 | networkBandwidthGBps=100 等 |
| T27 | 新字段类型正确 | 均为 number |

#### 集成测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T28 | SimulationMetrics.parallel 存在 | 嵌入关系正确 |
| T29 | 全并行 size=1 退化为单实例 | ParallelTopology(tp=1,dp=1,pp=1) + SimCommGroup 各类型 size=1 → 全零通信 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | SimCommGroup bytes=0 | 返回 latency_us（仅延迟） |
| B2 | SimCommGroup bandwidth 极大 | ticks 趋近于 latency |
| B3 | SimCommGroup bandwidth=0 | 除零保护，返回 Infinity 或抛错 |
| B4 | ParallelTopology world_size=1 | rankToCoord(0)=(0,0,0) |
| B5 | ParallelTopology ppStageLayers pp=1 | 单 stage 覆盖全部层 |
| B6 | ParallelMetrics commTicksTotal 各项为 0 | 总和为 0 |
| B7 | MockTPGroup tp_size=1 mockAllReduceSum(0) | 返回 0 |
| B8 | efficiency=1.0 | 结果与无效率因子一致 |

---

## 风险与注意事项

### 兼容性影响

- **SimCommGroup 接口变更**：`types.ts` 中的现有占位接口新增 `allGather` 方法，旧的仅实现 `allReduce/allToAll/sendRecv/barrier` 的代码需补充实现。当前无其他代码实现此接口（仅有 `SgSimContext.tpGroup: SimCommGroup | null` 占位），因此无破坏性变更。
- **SimulatorConfig 新增字段**：使用 `...DEFAULT_SIMULATOR_CONFIG` 扩展的方式在现有测试中已验证安全（参见 K5 测试 `makeConfig`），新字段有默认值，不影响现有调用方。
- **MockTPGroup 替换**：`SgSimContext.tpGroup` 当前为 null 赋值，后续 P1a Issue 可替换为 `new MockTPGroup(config.tpSize, config)`，本 Issue 仅提供工具类，不修改 SimSimContext 内部赋值逻辑。

### 性能影响

- 所有通信成本计算均为 O(1) 纯算术，无性能风险
- ParallelTopology 的 rank 推导为 O(1)，ppStageLayers 为 O(pp_size)
- ParallelMetrics.summary() 创建新对象，仅在需要时调用，无热路径影响

### 回滚方案

- 新增文件删除即可回滚
- `types.ts` 和 `index.ts` 的修改通过 git revert 即可恢复
- 无数据库/持久化变更，回滚无数据风险
