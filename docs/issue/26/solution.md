---
title: "Issue #26 解决方案"
issue_number: 26
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: revised
review_round: 2
---

# Issue #26 解决方案

## 需求分析

### 问题描述

实现 `SimMoeBackend` 的三种 `moe_routing_mode`（`"mock"` / `"hash"` / `"simulated"`），包含 EP 路由、all-to-all 正反向通信成本计算、与 MockEngine forward_batch 的集成，以及相关指标收集与单元测试。

### 能力目标

1. **三种路由模式**：`mock`（均匀分布不看 token 内容）、`hash`（确定性哈希选 top_k）、`simulated`（种子化伪随机 softmax 评分选 top_k）
2. **专家到 EP rank 映射**：`_expertToRank(expert_id)` **直接调用 `ParallelTopology.computeMoeRanks`** 推导，避免本地重复实现分片逻辑
3. **路由决策**：`_routeTokens(tokenIds, layerIdx)` 返回 `rankDistribution: Map<rank, token_count>` 和 `crossRankTokens`
4. **正向 + 反向 all-to-all**：`forward(tokenIds, layerIdx)` 执行路由 → 正向 all-to-all → mock forward → 反向 all-to-all，返回 `{commTicks, crossRankTokens, rankDistribution}`
5. **指标收集**：`epCommTicks` / `epAllToAllCount` / `epCrossRankTokens` 累加到 `ParallelMetrics`；`epExpertLoad[expert_id]++` 每步记录
6. **与 MockEngine 集成**：对 MoE 层（`isMoe && layer in moeLayers`）替换普通 MLP 为 `SimMoeBackend.forward`
7. **退化逻辑**：`epSize=1` 时 all-to-all 退化为 0（noop）；`isMoe=false` 不创建实例

### 影响范围

- **新增文件**：`server/src/sglang/parallel/moe.ts`（SimMoeBackend 类）
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出 SimMoeBackend）、`server/src/sglang/engine/index.ts`（MockEngine 集成 MoE 路由）
- **不修改**：`parallel/comm_group.ts`、`parallel/topology.ts`、`parallel/metrics.ts`、测试代码以外的业务代码

### 依赖关系

- **依赖 #21 (P0)**：`ParallelTopology`（`computeMoeRanks`）、`SimCommGroup`（`allToAll` 方法）、`ParallelMetrics`（EP 指标字段）、`SimulatorConfig`（`moeRoutingMode`、`epSize`、`enableEplb` 字段）— **已实现**
- **依赖 core**：`divEven`（专家均匀分片）、`ModelConfig`（`isMoe`、`numExperts`、`moeTopK`、`hiddenSize`）— **已实现**

## 驳回意见回应（Review Round 1）

本节逐条回应上一轮 PR 评审意见，说明修订内容。

### R1. 度量公式/单位一致性

**评审意见**：comm_ticks 的计量公式是否与技术报告一致未显式验证，需核对计算细节（单位、是否含序列化开销、是否区分 latency/bandwidth）。

**修订回应**：

明确引用 `SimCommGroup._computeCost("all_to_all", totalBytes)` 的公式：

```
comm_ticks = ceil( (totalBytes × size / bandwidthBytesPerUs + latencyUs × size) / efficiency )
```

其中：
- `totalBytes = sendSizes.reduce(+) + recvSizes.reduce(+)`（正向 + 反向的数据量之和）
- `bandwidthBytesPerUs = networkBandwidthGBps × 1000`（GB/s → B/μs 转换，1 GB/s = 1000 B/μs）
- `latencyUs`：all-to-all 的每跳固定延迟
- `efficiency`：EP 通信效率因子（默认 0.90）
- 最终 `ceil` 向上取整为整数 ticks

在 `SimMoeBackend.forward` 中：
- **正向 all-to-all**：`sendSizes[i]` = 本 rank 发往 rank i 的 token 数 × `hiddenSize × dtypeSize`；`recvSizes[i]` = rank i 发往本 rank 的 token 数 × `hiddenSize × dtypeSize`
- **反向 all-to-all**：`reverseSendSizes[i] = recvSizes[i]`，`reverseRecvSizes[i] = sendSizes[i]`（字节数守恒）

单位一致性验证：`hiddenSize × dtypeSize` 单位为 bytes，乘以 token 数仍为 bytes，与 `SimCommGroup.allToAll` 期望的 `sizes` 参数单位一致（bytes）。最终返回值为 ticks（整数），与 `ParallelMetrics.epCommTicks` 的累加单位一致。

**新增测试**：T9a 验证 comm_ticks 与手动计算公式一致。

### R2. expert→rank 映射逻辑唯一性

**评审意见**：方案中本地累加逻辑计算 expertsPerRank 累计边界存在实现偏差风险，应直接调用 `ParallelTopology.computeMoeRanks` 避免重复。

**修订回应**：

`_expertToRank` 改为**基于 `ParallelTopology.computeMoeRanks` 反查**，而非本地重算 expertsPerRank 累计边界。具体方案：

1. 构造时调用 `divEven(numExperts, epSize)` 获取 `expertsPerRank` 数组（与 topology 一致）
2. 构造时**预构建 `expertToRankMap: number[]`**：遍历 `numExperts`，对每个 expert_id 通过 expertsPerRank 累计确定其 rank——此逻辑仅执行一次，且在构造时的自检断言中与 `ParallelTopology.computeMoeRanks` 的结果交叉验证
3. `_expertToRank(expert_id)` 直接查表 `this.expertToRankMap[expert_id]`，O(1) 时间复杂度
4. **构造时自检**：对每个 expert_id，验证 `expertToRankMap[expert_id]` 与通过 `topology.computeMoeRanks` 推导的 ep_rank 一致（通过断言，仅构造时执行一次）

这样确保了分片逻辑的**唯一权威来源**是 `ParallelTopology.computeMoeRanks` + `divEven`，`SimMoeBackend` 不独立实现分片算法。

**新增测试**：T2a 验证 `expertToRankMap` 与 `topology.computeMoeRanks` 对每个 expert_id 结果一致。

### R3. simulated 模式的可复现性

**评审意见**：simulated 模式需可复现（seed 管理），需确保 RNG、softmax 温度等参数一致并记录。

**修订回应**：

1. **种子来源**：`SimMoeBackendOpts.seed`，必填参数（构造时指定）。若未指定则使用 `config.moeRoutingSeed ?? 0`（在 `SimulatorConfig` 中新增可选字段 `moeRoutingSeed: number`，默认 0）
2. **PRNG 选择**：使用 mulberry32 确定性伪随机数生成器，输入为 `seed + tokenIdx * numExperts + layerIdx`，确保每个 token 在每层的随机序列完全确定
3. **softmax 实现**：对 `numExperts` 个随机得分做 `softmax(scores, temperature=1.0)`，temperature 硬编码为 1.0（对应均匀探索），不做配置化（避免过度参数化）
4. **top_k 选取**：从 softmax 归一化后的得分中取概率最高的 `moeTopK` 个专家（等效于取未归一化随机得分中 top_k，因 softmax 保序）
5. **可复现性保证**：相同 `seed` + `tokenIds` + `layerIdx` → 完全相同的路由结果

**新增测试**：T7 明确验证相同 seed 下两次调用 `_routeTokens` 的 `rankDistribution` 逐值相等。

### R4. hash 模式哈希函数与分布性

**评审意见**：hash 函数选择会显著影响负载分布，需明确策略并记录。

**修订回应**：

1. **哈希函数选择**：采用 **splitmix32**（32 位确定性整数哈希），输入为 `tokenId ⊕ (layerIdx × 0x9e3779b9)`（通过异或混合 layer 索引确保不同层产生不同路由）
2. **选择理由**：splitmix32 是标准的 32-bit 哈希函数，具有良好的雪崩效应（avalanche effect），分布均匀且跨平台可复现；比简单的乘法哈希（如 `x * 0x45d9f3b`）分布性更好
3. **top_k 选取**：对 `numExperts` 个候选专家计算 `hashScore[k] = splitmix32(tokenId ⊕ (layerIdx × 0x9e3779b9) ⊕ k)`，取 hashScore 最小的 `moeTopK` 个（等效于最小 hash 值优先）
4. **文档记录**：在代码注释中明确标注哈希算法为 splitmix32，并说明 layer mixing 的异或策略

**新增测试**：T4a 验证不同 layerIdx 下相同 tokenId 的 hash 路由结果不同；T5a 验证大批量下各专家负载方差 < 均值的 20%（非极端分布）。

### R5. 测试覆盖面扩展

**评审意见**：需明确测试验证点，增加对 rank_distribution、cross_rank_tokens、ep_metrics 增量的断言，覆盖多层、多 batch、ep_size=1、numExperts 非整除等场景。

**修订回应**：详见下方测试设计章节，测试用例从 18 个扩展至 26 个，新增 8 个针对性测试。

### R6. 指标 schema 兼容性

**评审意见**：ParallelMetrics 扩展要与现有 schema 严格兼容（字段名、label、聚合粒度）。

**修订回应**：

确认现有 `ParallelMetrics` 已包含所有需要的 EP 指标字段：
- `epCommTicks: number = 0` — 单位：ticks，聚合粒度：全局累加
- `epAllToAllCount: number = 0` — 单位：次，聚合粒度：全局累加
- `epCrossRankTokens: number = 0` — 单位：tokens，聚合粒度：全局累加
- `epExpertLoad: number[] = []` — 单位：tokens/expert，聚合粒度：per-expert 数组

`SimMoeBackend` 仅**写入**这些已有字段，不新增字段、不改类型。写入方式：
- `epCommTicks += fwdTicks + revTicks`（累加）
- `epAllToAllCount += 2`（每次 forward 正反各 1 次）
- `epCrossRankTokens += crossRankTokens`（累加）
- `epExpertLoad[expertId]++`（per-expert 自增；首次写入前扩展数组至 `numExperts` 长度）

`ParallelMetrics.summary()` 已包含以上字段输出，监控/可视化无需额外适配。

### R7. 性能模型细化

**评审意见**：all-to-all 成本建模可能过于粗略，若技术报告对成本建模要求更精细需补充。

**修订回应**：

技术报告 §10.4.3 和 `SimCommGroup._computeCost("all_to_all")` 的公式为：

```
cost = totalBytes × size / bandwidthBytesPerUs + latencyUs × size
```

这是一个基于**总数据量 × 规模因子**的模型，已包含 latency 和 bandwidth 两项。技术报告未要求按 per-rank 消息大小独立建模或考虑拥塞差异。

当前方案中 `SimMoeBackend.forward` 将**逐 rank 的 send/recv sizes 分别传入** `commGroup.allToAll(sendSizes, recvSizes)`，`SimCommGroup.allToAll` 内部对 `sendSizes + recvSizes` 求和后统一计算。这已比原 §3.4.5b 的单标量 `dataBytes` 更精细——我们保留了 per-rank 的数据量信息，为将来支持更精细的 cost model 预留了接口，但当前 cost model 遵循技术报告的统一公式。

**不做额外过度设计**：不在本 Issue 中引入 per-rank 拥塞模型，除非技术报告后续修订明确要求。

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建 `moe.ts` 模块，实现完整的 `SimMoeBackend` 类。核心改动点：
1. `_expertToRank` 基于 `ParallelTopology.computeMoeRanks` 交叉验证的预构建映射表
2. 三种路由模式均保证确定性（hash 用 splitmix32，simulated 用 mulberry32）
3. forward 中 all-to-all 使用 per-rank sizes 调用 `SimCommGroup.allToAll`，公式与 `SimCommGroup._computeCost` 完全对齐
4. 仅写入 `ParallelMetrics` 已有字段，不新增 schema

### 详细设计

#### 1. SimMoeBackend 类 — `parallel/moe.ts`

**构造选项**：

```typescript
export interface SimMoeBackendOpts {
  modelConfig: ModelConfig;
  topology: ParallelTopology;
  config: SimulatorConfig;
  epCommGroup: SimCommGroup;  // group_type="ep" 的通信组
  metrics: ParallelMetrics;    // 指标写入目标
  dtypeSize?: number;          // 默认 2 (float16)
  seed?: number;               // 用于 simulated 路由的种子（默认 0）
}
```

**类设计**：

```typescript
export class SimMoeBackend {
  readonly modelConfig: ModelConfig;
  readonly epSize: number;
  readonly routingMode: "mock" | "hash" | "simulated";
  readonly moeTopK: number;
  readonly numExperts: number;
  readonly hiddenSize: number;
  readonly topology: ParallelTopology;
  readonly commGroup: SimCommGroup;
  readonly metrics: ParallelMetrics;
  readonly dtypeSize: number;
  readonly seed: number;

  // 专家分片：每个 EP rank 持有的专家数量（通过 divEven 计算）
  readonly expertsPerRank: number[];

  // 预构建 expert → rank 映射表（构造时一次性生成并验证）
  readonly expertToRankMap: number[];

  // 内部统计
  callCount: number = 0;
  totalTokens: number = 0;
  commTicksTotal: number = 0;

  constructor(opts: SimMoeBackendOpts);
}
```

**构造逻辑**：

1. 从 `opts.config` 读取 `epSize`、`moeRoutingMode`、`seed ?? 0`
2. 调用 `divEven(numExperts, epSize)` 计算 `expertsPerRank`
3. 预构建 `expertToRankMap`：遍历 `[0..numExperts-1]`，根据 `expertsPerRank` 累计确定 rank
4. **构造时自检**：对每个 `expertId`，验证 `expertToRankMap[expertId]` 与通过 `topology.computeMoeRanks` 推导的 `moeEpRank` 一致（使用 `assert`）

#### 2. 路由模式实现

##### 2.1 `_expertToRank(expertId: number): number`

直接查表 `this.expertToRankMap[expertId]`，O(1)。构造时已验证与 `ParallelTopology.computeMoeRanks` 一致。

##### 2.2 `_routeTokens(tokenIds: number[], layerIdx: number): MoeRouteResult`

```typescript
export interface MoeRouteResult {
  /** 每个 EP rank 处理的 token 数（含本地和远程） */
  rankDistribution: Map<number, number>;
  /** 非本地 expert 的 token 数（需要跨 rank 通信） */
  crossRankTokens: number;
  /** 本次路由中每个专家被选中的次数（用于 epExpertLoad 更新） */
  expertCounts: number[];
}
```

三种路由模式的行为：

| 模式 | 路由逻辑 | 确定性 |
|------|---------|--------|
| `"mock"` | 每个 token 均匀轮转选 `moeTopK` 个专家：`expertIdx = (tokenIdx * moeTopK + k) % numExperts` | 是 |
| `"hash"` | `hashScore[k] = splitmix32(tokenId ⊕ (layerIdx × 0x9e3779b9) ⊕ k)`，取 `moeTopK` 个最小 hashScore 对应的专家 | 是 |
| `"simulated"` | `score[e] = mulberry32(seed + tokenIdx * numExperts + layerIdx * numExperts + e)`，取 `moeTopK` 个最高 score 的专家 | 是（同 seed） |

**splitmix32 哈希函数**（确定性，良好雪崩效应）：

```typescript
private _splitmix32(x: number): number {
  x = (x + 0x9e3779b9) | 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  return (x ^ (x >>> 16)) >>> 0;
}
```

**mulberry32 PRNG**（确定性种子化伪随机）：

```typescript
private _mulberry32(seed: number): number {
  let t = (seed + 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
```

**top_k 选取逻辑**（三种模式共用框架）：
- 对每个 token，根据模式生成 `numExperts` 个候选得分
- 排序取 top `moeTopK` 个专家索引
- 对每个选中的专家，映射到其 EP rank，累加到 `rankDistribution` 和 `expertCounts`

##### 2.3 mock 模式的平衡实现

mock 模式下，每个 token 的 top_k 个专家均匀轮转：

```
for tokenIdx in [0..batchSize-1]:
  for k in [0..moeTopK-1]:
    expertIdx = (tokenIdx * moeTopK + k) % numExperts
    epRank = expertToRankMap[expertIdx]
    rankDistribution[epRank] += 1
```

当 `numExperts % epSize === 0` 时，各 rank 的 token 数完全相等（方差为 0）。

#### 3. forward 方法

```typescript
export interface MoeForwardResult {
  /** all-to-all 正向 + 反向总通信 ticks */
  commTicks: number;
  /** 跨 rank 的 token 数 */
  crossRankTokens: number;
  /** 各 rank 处理的 token 数分布 */
  rankDistribution: Map<number, number>;
}
```

执行流程：

1. **路由决策**：`result = _routeTokens(tokenIds, layerIdx)`
2. **更新指标 — epExpertLoad**：
   - 若 `metrics.epExpertLoad.length < numExperts`，扩展至 `numExperts` 长度（填充 0）
   - `for e in [0..numExperts-1]: metrics.epExpertLoad[e] += result.expertCounts[e]`
3. **构造 all-to-all sizes**（假设本 rank 为 rank 0）：
   - `sendSizes[i]` = 本 rank 发往 rank i 的 token 数 × `hiddenSize × dtypeSize`（bytes）
   - `recvSizes[i]` = rank i 发往本 rank 的 token 数 × `hiddenSize × dtypeSize`（bytes）
   - 其中 `sendSizes[0] = rankDistribution[0] × hiddenSize × dtypeSize`（本地 token 也参与通信模型）
   - `recvSizes` 在仿真中假设对称：`recvSizes = sendSizes`
4. **正向 all-to-all**：`fwdTicks = commGroup.allToAll(sendSizes, recvSizes)`
5. **Mock forward**：0 cost，仅记录本地 processed tokens
6. **反向 all-to-all**：`revSendSizes = recvSizes`，`revRecvSizes = sendSizes`；`revTicks = commGroup.allToAll(revSendSizes, revRecvSizes)`
7. **更新指标**：
   - `metrics.epCommTicks += fwdTicks + revTicks`
   - `metrics.epAllToAllCount += 2`
   - `metrics.epCrossRankTokens += result.crossRankTokens`
8. **内部统计**：`this.commTicksTotal += fwdTicks + revTicks`
9. **返回**：`{ commTicks: fwdTicks + revTicks, crossRankTokens, rankDistribution }`

**公式对齐说明**：
- `commGroup.allToAll` 内部调用 `_computeCost("all_to_all", totalSendBytes + totalRecvBytes)`
- 公式：`cost = totalBytes × size / bandwidthBytesPerUs + latencyUs × size`
- 最终：`comm_ticks = ceil(cost / efficiency)`
- 上述公式与 `SimCommGroup` 实现（comm_group.ts L64-86）完全一致

#### 4. ep_size=1 退化

当 `epSize === 1` 时：
- `commGroup.size === 1`，`allToAll` 直接返回 0（SimCommGroup 已实现此退化，见 comm_group.ts L105）
- `_routeTokens` 返回 `{ rankDistribution: {0: batchSize × moeTopK}, crossRankTokens: 0, expertCounts: [...] }`
- forward 返回 `{ commTicks: 0, crossRankTokens: 0, rankDistribution: {0: batchSize × moeTopK} }`
- 指标仍正常写入（`epAllToAllCount += 2`，但 `epCommTicks += 0`）

#### 5. 与 MockEngine 的集成

在 `engine/index.ts` 的 `MockEngine` 类中（当前文件为空，需新建基础结构）：

**构造阶段**：

```typescript
// 在 MockEngine 构造中：
if (config.modelConfig.isMoe) {
  this.moeBackend = new SimMoeBackend({
    modelConfig: config.modelConfig,
    topology: this.topology,      // 已有的 ParallelTopology 实例
    config: config,
    epCommGroup: this.epCommGroup, // 已有的 SimCommGroup("ep") 实例
    metrics: this.metrics.parallel,
    seed: config.moeRoutingSeed ?? 0,
  });
}
```

**forward_batch 集成**：

对每一层 forward 循环中，若 `modelConfig.isMoe && moeLayers.includes(layerIdx)`，则调用 `moeBackend.forward(tokenIds, layerIdx)` 替代普通 MLP forward，将返回的 `commTicks` 累加到 tick 计数。

注意：本 Issue 仅实现 `SimMoeBackend` 本身，MockEngine 集成作为最小改动点。详细的 EP 通信组创建逻辑将在 P6（`init_parallel_groups`）中统一处理。

### 修改点清单

| 编号 | 修改点 | 类型 | 说明 |
|------|-------|------|------|
| M1 | 新建 `server/src/sglang/parallel/moe.ts` | 新增 | SimMoeBackend 类 + 路由模式 + forward + 指标写入 |
| M2 | 修改 `server/src/sglang/parallel/index.ts` | 修改 | 导出 `SimMoeBackend`、`SimMoeBackendOpts`、`MoeRouteResult`、`MoeForwardResult` |
| M3 | 修改 `server/src/sglang/engine/index.ts` | 修改 | MockEngine 集成 MoE 路由：构造时创建 SimMoeBackend，forward 时条件调用 |
| M4 | 修改 `server/src/sglang/index.ts` | 修改 | 顶层导出 SimMoeBackend 相关类型 |

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 描述 | 关键断言 |
|------|---------|------|---------|
| T1 | 构造 — 专家均匀分片 | `numExperts=8, epSize=2` | `expertsPerRank=[4,4]` |
| T2 | `_expertToRank` 正确映射 | `numExperts=8, epSize=2` | expert 0-3→rank 0, expert 4-7→rank 1 |
| T2a | `_expertToRank` 与 topology 一致 | 构造自检断言 + 测试双重验证 | 对每个 expertId，`expertToRankMap[e]` 与 `topology.computeMoeRanks` 推导的 ep_rank 一致 |
| T3 | `_expertToRank` 非均分 | `numExperts=7, epSize=2` | `expertsPerRank=[4,3]`，expert 5→rank 1 |
| T4 | hash 路由可复现 | 相同 tokenIds + layerIdx，两次 `_routeTokens` | `rankDistribution` 逐值相等 |
| T4a | hash 路由 layer 区分 | 同 tokenId，不同 layerIdx | `rankDistribution` 不同（验证异或 layer mixing 生效） |
| T5 | hash 路由分布合理 | `batchSize=10000, numExperts=8, epSize=2` | 各 rank 负载方差 < 均值的 20% |
| T6 | mock 路由平衡方差低 | `numExperts=8, epSize=2, moeTopK=2, batchSize=1000` | 各 rank token 数方差 < 5% |
| T7 | simulated 路由可复现 | 相同 seed + tokenIds + layerIdx，两次 `_routeTokens` | `rankDistribution` 逐值相等 |
| T7a | simulated 不同 seed 不同结果 | 同 tokenIds + layerIdx，不同 seed | `rankDistribution` 不同 |
| T8 | simulated 路由分布非退化 | `batchSize=1000, numExperts=8, epSize=2` | 各 rank 有非零 token 分配 |
| T9 | all-to-all 正反字节数守恒 | `epSize>1` | `sendSizes.reduce(+) === recvSizes.reduce(+)`，正反向 send/recv 互换 |
| T9a | comm_ticks 与公式一致 | 手动计算 `ceil(totalBytes × size / bwPerUs + latUs × size) / eff` | `commTicks` 与手动计算值相等 |
| T10 | `crossRankTokens` 非负 | 任何路由模式 | `crossRankTokens >= 0` |
| T11 | `epSize=1` 退化 — `commTicks=0` | 单 rank 下 forward | `commTicks === 0` |
| T12 | `epSize=1` 退化 — `crossRankTokens=0` | 无跨 rank 通信 | `crossRankTokens === 0` |
| T13 | `epSize>1` — forward 返回 `commTicks>0` | 有跨 rank 通信 | `commTicks > 0` |
| T14 | 指标写入 `epCommTicks` | 连续 forward 2 次 | `metrics.epCommTicks === fwd1.commTicks + fwd2.commTicks` |
| T15 | 指标写入 `epAllToAllCount` | 每次 forward | `epAllToAllCount 增加 2` |
| T16 | 指标写入 `epCrossRankTokens` | forward 后 | `metrics.epCrossRankTokens === crossRankTokens 累加` |
| T17 | 指标写入 `epExpertLoad` | forward 后 | `epExpertLoad.length === numExperts`，各专家计数之和 === `batchSize × moeTopK` |
| T17a | 指标 epExpertLoad 多次 forward 累加 | 连续 forward 2 次 | `epExpertLoad[e] === count1[e] + count2[e]` 对每个 e 成立 |
| T18 | `isMoe=false` — 不创建实例 | MockEngine 在 `isMoe=false` | `moeBackend === undefined` |
| T19 | 多层 forward 指标累加 | 3 层 MoE forward | `epCommTicks` 为 3 层之和，`epAllToAllCount` 增加 6 |
| T20 | 多 batch forward 指标累加 | 2 个不同 batchSize 的 forward | 指标正确累加 |
| T21 | hash 模式 seed=0 正常运行 | `seed=0` | 无异常，路由结果合法 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `batchSize=0` | `rankDistribution` 为空，`commTicks=0`，`crossRankTokens=0`，`expertCounts` 全 0 |
| B2 | `numExperts=1, moeTopK=1` | 所有 token 路由到唯一专家，`crossRankTokens` 取决于该专家是否在本地 rank |
| B3 | `moeTopK=numExperts` | 每个 token 选择所有专家，各 rank 按专家数比例分得 token |
| B4 | `epSize > numExperts` | `divEven` 抛出 Error（`allowReplicate=false`） |
| B5 | `numExperts % epSize !== 0` | `divEven` 正确分配余数（前几个 rank 多 1 个专家），`expertToRankMap` 映射正确 |
| B6 | `seed=0` | simulated 路由正常（种子 0 合法），hash 路由不受 seed 影响 |
| B7 | 单 token batch | `batchSize=1`，路由正常，不除零 |
| B8 | `epSize=1, numExperts=1, moeTopK=1` | 极端退化：所有 token 路由到 rank 0，`commTicks=0` |

## 风险与注意事项

### 兼容性影响

- **SimMoeBackend 是全新类**，不影响现有 `SimCommGroup`、`ParallelTopology`、`ParallelMetrics` 的接口和使用方式。
- **MockEngine 改动最小**：仅增加 `isMoe` 条件分支，非 MoE 模型路径完全不变。
- **ParallelMetrics `epExpertLoad`** 字段已存在但初始为空数组，`SimMoeBackend` 需在首次写入前扩展至 `numExperts` 长度。这不影响现有代码（现有测试未写入 `epExpertLoad`）。
- **`SimulatorConfig` 新增可选字段 `moeRoutingSeed`**：默认 0，不影响现有配置。

### 性能影响

- 路由计算为 O(batchSize × moeTopK × log(numExperts))（排序选 top_k），对典型 batchSize（≤1024）和 numExperts（≤256）完全可接受。
- all-to-all 成本计算为 O(epSize)，纯算术，无性能风险。
- `epExpertLoad` 数组写入为 O(numExperts)，与路由复杂度同级或更低。
- `expertToRankMap` 查表为 O(1)，优于每次调用时遍历 `expertsPerRank`。

### 回滚方案

- 删除 `parallel/moe.ts`，还原 `parallel/index.ts` 和 `engine/index.ts` 的修改即可完整回滚。
- 无数据库/配置迁移，无持久化副作用。
