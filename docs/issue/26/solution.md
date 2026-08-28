---
title: "Issue #26 解决方案"
issue_number: 26
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #26 解决方案

## 需求分析

### 问题描述

实现 `SimMoeBackend` 的三种 `moe_routing_mode`（`"mock"` / `"hash"` / `"simulated"`），包含 EP 路由、all-to-all 正反向通信成本计算、与 MockEngine forward_batch 的集成，以及相关指标收集与单元测试。

### 能力目标

1. **三种路由模式**：`mock`（均匀分布不看 token 内容）、`hash`（`hash(token_id) % num_experts` 选 top_k）、`simulated`（随机 softmax 评分选 top_k）
2. **专家到 EP rank 映射**：`_expert_to_rank(expert_id)` 利用 `ParallelTopology.computeMoeRanks` 的专家分片逻辑将专家索引映射到 EP rank
3. **路由决策**：`_route_tokens(token_ids, layer_idx)` 返回 `rank_distribution: Map<rank, token_count>` 和 `cross_rank_tokens`
4. **正向 + 反向 all-to-all**：`forward(batch, layer_idx)` 执行路由 → 正向 all-to-all → mock forward → 反向 all-to-all，返回 `{comm_ticks, cross_rank_tokens, rank_distribution}`
5. **指标收集**：`ep_comm_ticks` / `ep_all_to_all_count` / `ep_cross_rank_tokens` 累加到 `ParallelMetrics`；`ep_expert_load[expert_id]++` 每步记录
6. **与 MockEngine 集成**：对 MoE 层（`isMoe && layer in moeLayers`）替换普通 MLP 为 `SimMoeBackend.forward`
7. **退化逻辑**：`ep_size=1` 时 all-to-all 退化为 0（noop）；`isMoe=false` 不创建实例

### 影响范围

- **新增文件**：`server/src/sglang/parallel/moe.ts`（SimMoeBackend 类）
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出 SimMoeBackend）、`server/src/sglang/engine/index.ts`（MockEngine 集成 MoE 路由）、`server/src/sglang/metrics/index.ts`（ParallelMetrics EP 指标扩展）
- **不修改**：`parallel/comm_group.ts`、`parallel/topology.ts`、调度逻辑、测试代码以外的业务代码

### 依赖关系

- **依赖 #21 (P0)**：`ParallelTopology`（`computeMoeRanks`）、`SimCommGroup`（`allToAll` 方法）、`ParallelMetrics`（EP 指标字段）、`SimulatorConfig`（`moeRoutingMode`、`epSize`、`enableEplb` 字段）— **已实现**
- **依赖 core**：`divEven`（专家均匀分片）、`ModelConfig`（`isMoe`、`numExperts`、`moeTopK`、`hiddenSize`）— **已实现**

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建 `moe.ts` 模块，实现完整的 `SimMoeBackend` 类，包含三种路由模式、专家分片、all-to-all 正反通信成本计算、指标记录。然后在 `engine/index.ts` 中集成：当 `isMoe=true` 时创建 `SimMoeBackend` 实例，在 forward_batch 中对 MoE 层调用其 `forward` 方法替换普通 MLP 计算。最后扩展 `ParallelMetrics` 增加 `epExpertLoad` 的写入能力。

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
  seed?: number;               // 用于 simulated 路由的可选种子
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

  // 专家分片：每个 EP rank 持有的专家数量
  readonly expertsPerRank: number[];

  // 内部统计
  callCount: number = 0;
  totalTokens: number = 0;
  commTicksTotal: number = 0;

  constructor(opts: SimMoeBackendOpts);
}
```

#### 2. 路由模式实现

##### 2.1 `_expertToRank(expertIdx: number): number`

将专家索引映射到 EP rank。使用 `expertsPerRank` 累计计算：

```
cumulative = 0
for rank in [0..epSize-1]:
  cumulative += expertsPerRank[rank]
  if expertIdx < cumulative:
    return rank
return 0  // fallback（不应到达）
```

此逻辑与 §3.4.5b 中的 Python 实现一致，且可通过 `ParallelTopology.computeMoeRanks` 交叉验证。

##### 2.2 `_routeTokens(tokenIds: number[], layerIdx: number): MoeRouteResult`

```typescript
export interface MoeRouteResult {
  /** 每个 EP rank 处理的 token 数（含本地和远程） */
  rankDistribution: Map<number, number>;
  /** 非本地 expert 的 token 数（需要跨 rank 通信） */
  crossRankTokens: number;
}
```

三种路由模式的行为：

| 模式 | 路由逻辑 | `rankDistribution` |
|------|---------|-------------------|
| `"mock"` | 每个 token 选择 `moeTopK` 个专家，均匀分布到 `numExperts`，等效于 `expertIdx = tokenIdx % numExperts` | 各 rank 约等量 |
| `"hash"` | `expertIdx = hash(tokenId + layerIdx) % numExperts`，选 `moeTopK` 个不同专家（通过 `hash(tokenId + layerIdx * k) % numExperts` 取 top_k 最小 hash） | 由 hash 分布决定 |
| `"simulated"` | 使用种子化的伪随机生成器（`seededRandom(seed + tokenIdx * numExperts + layerIdx)`）模拟 `softmax(random_normal)` 得分，取 top_k | 随机但可复现 |

**hash 函数设计**：使用简单的确定性整数哈希，确保跨平台可复现：

```typescript
private _hash(x: number): number {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x >>> 0; // 无符号
}
```

**simulated 的伪随机生成器**：使用 mulberry32 种子化 PRNG，确保同种子可复现：

```typescript
private _seededRandom(seed: number): number {
  let t = seed + 0x6D2B79F5;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
```

**top_k 选取逻辑**（hash 和 simulated 共用）：
- 生成 `numExperts` 个得分，排序取 top `moeTopK` 个专家索引
- 对每个选中的专家，映射到其 EP rank，累加到 `rankDistribution`

##### 2.3 mock 模式的平衡实现

mock 模式下，每个 token 的 top_k 个专家均匀轮转：

```
for tokenIdx in [0..batchSize-1]:
  for k in [0..moeTopK-1]:
    expertIdx = (tokenIdx * moeTopK + k) % numExperts
    epRank = _expertToRank(expertIdx)
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

forward(tokenIds: number[], layerIdx: number): MoeForwardResult
```

执行流程：

1. **路由决策**：`result = _routeTokens(tokenIds, layerIdx)`
2. **指标记录**：更新 `ep_expert_load[expertIdx]++`（每个被选中的专家）
3. **正向 all-to-all**：
   - `sendSizes[i]` = 本 rank 发往 rank i 的 token 数 × `hiddenSize × dtypeSize`
   - `recvSizes[i]` = rank i 发往本 rank 的 token 数（仿真中假设对称，取 `sendSizes`）
   - `fwdTicks = commGroup.allToAll(sendSizes, recvSizes)`
4. **Mock forward**：0 cost，仅记录本地 processed tokens
5. **反向 all-to-all**：
   - `revTicks = commGroup.allToAll(reverseSizes, reverseRecvSizes)`
   - 反向大小与正向相同（字节数守恒）
6. **更新指标**：
   - `metrics.epCommTicks += fwdTicks + revTicks`
   - `metrics.epAllToAllCount += 2`
   - `metrics.epCrossRankTokens += result.crossRankTokens`
7. **返回**：`{ commTicks: fwdTicks + revTicks, crossRankTokens, rankDistribution }`

#### 4. ep_size=1 退化

当 `epSize === 1` 时：
- `commGroup.size === 1`，`allToAll` 直接返回 0（SimCommGroup 已实现此退化）
- `_routeTokens` 返回 `{ rankDistribution: {0: batchSize}, crossRankTokens: 0 }`
- forward 返回 `{ commTicks: 0, crossRankTokens: 0, rankDistribution: {0: batchSize} }`

#### 5. 与 MockEngine 的集成

在 `engine/index.ts` 的 `MockEngine` 类中：

**构造阶段**：

```typescript
// 在 MockEngine.__init__ 中：
if (config.modelConfig.isMoe) {
  this.moeBackend = new SimMoeBackend({
    modelConfig: config.modelConfig,
    topology: new ParallelTopology({
      tpSize: config.tpSize,
      dpSize: config.dpSize,
      epSize: config.epSize,
      ppSize: config.ppSize,
      cpSize: config.cpSize,
      enableDpAttention: config.enableDpAttention,
    }),
    config: config,
    epCommGroup: new SimCommGroup({
      groupType: "ep",
      size: config.epSize,
      networkBandwidthGBps: config.networkBandwidthGBps,
      latencyUs: config.allToAllLatencyTicks, // 注意：需要从 us 转换或使用专用字段
      efficiency: config.epEfficiency,
    }),
    metrics: this.metrics.parallel,
  });
}
```

**forward_batch 集成**：

对每一层 forward 循环中，若 `modelConfig.isMoe && layerIdx < numMoeLayers`（MoE 层通常在模型后半部分或交替排列），则调用 `moeBackend.forward(tokenIds, layerIdx)` 替代普通 MLP forward，将返回的 `commTicks` 累加到 tick 计数。

注意：本 Issue 仅实现 `SimMoeBackend` 本身，MockEngine 集成作为最小改动点——在 engine forward 循环中增加一个条件分支即可。详细的 EP 通信组创建逻辑将在 P6（`init_parallel_groups`）中统一处理。

#### 6. ParallelMetrics 扩展

当前 `ParallelMetrics` 已有 `epCommTicks`、`epAllToAllCount`、`epCrossRankTokens`、`epExpertLoad` 字段。需要确认 `epExpertLoad` 的写入方式：

- `SimMoeBackend.forward` 中每选一个专家就 `metrics.epExpertLoad[expertIdx]++`
- 如果 `epExpertLoad` 长度不足（初始为空数组），需要自动扩展至 `numExperts` 长度
- 此逻辑在 `SimMoeBackend` 中实现，不需要修改 `ParallelMetrics` 类本身

### 修改点清单

| 编号 | 修改点 | 类型 | 说明 |
|------|-------|------|------|
| M1 | 新建 `server/src/sglang/parallel/moe.ts` | 新增 | SimMoeBackend 类 + 路由模式 + forward + 指标写入 |
| M2 | 修改 `server/src/sglang/parallel/index.ts` | 修改 | 导出 `SimMoeBackend`、`SimMoeBackendOpts`、`MoeRouteResult`、`MoeForwardResult` |
| M3 | 修改 `server/src/sglang/engine/index.ts` | 修改 | MockEngine 集成 MoE 路由：构造时创建 SimMoeBackend，forward 时条件调用 |
| M4 | 修改 `server/src/sglang/index.ts` | 修改 | 顶层导出 SimMoeBackend 相关类型 |

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | `SimMoeBackend` 构造 — 专家均匀分片 | `numExperts=8, epSize=2` → `expertsPerRank=[4,4]` |
| T2 | `_expertToRank` 正确映射 | expert 0-3 → rank 0, expert 4-7 → rank 1 |
| T3 | `_expertToRank` 非均分 | `numExperts=7, epSize=2` → `expertsPerRank=[4,3]`，expert 5 → rank 1 |
| T4 | hash 路由可复现 | 相同 tokenIds + layerIdx，两次调用 `_routeTokens` 结果一致 |
| T5 | hash 路由分布合理 | 大批量 token 下各 rank 分布非极端（不过半集中于一个 rank） |
| T6 | mock 路由平衡方差低 | `numExperts=8, epSize=2, moeTopK=2, batchSize=1000` → 各 rank token 数方差 < 5% |
| T7 | simulated 路由可复现 | 相同 seed 下两次调用结果一致 |
| T8 | simulated 路由分布非退化 | 大批量下各 rank 有非零 token 分配 |
| T9 | all-to-all 正反字节数守恒 | `epSize>1` 时 `sendSizes.reduce` == `recvSizes.reduce`，正反向相同 |
| T10 | `crossRankTokens` 非负 | 任何路由模式下 `crossRankTokens >= 0` |
| T11 | `epSize=1` 退化 — `commTicks=0` | 单 rank 下 forward 返回 `commTicks=0` |
| T12 | `epSize=1` 退化 — `crossRankTokens=0` | 无跨 rank 通信 |
| T13 | `epSize>1` — forward 返回 `commTicks>0` | 有跨 rank 通信时产生正向成本 |
| T14 | 指标写入 `epCommTicks` | 连续 forward 2 次后 `metrics.epCommTicks` 为两次之和 |
| T15 | 指标写入 `epAllToAllCount` | 每次 forward 增加 2（正反各 1） |
| T16 | 指标写入 `epCrossRankTokens` | 与 forward 返回的 `crossRankTokens` 一致 |
| T17 | 指标写入 `epExpertLoad` | forward 后 `epExpertLoad` 长度为 `numExperts`，各专家计数之和 = `batchSize × moeTopK` |
| T18 | `isMoe=false` — 不创建实例 | MockEngine 在 `isMoe=false` 时不创建 `SimMoeBackend` |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `batchSize=0` | `rankDistribution` 为空，`commTicks=0`，`crossRankTokens=0` |
| B2 | `numExperts=1, moeTopK=1` | 所有 token 路由到唯一专家，`crossRankTokens` 取决于该专家是否在本地 |
| B3 | `moeTopK=numExperts` | 每个 token 选择所有专家，各 rank 按专家数比例分得 token |
| B4 | `epSize > numExperts` | `divEven` 抛出 Error（`allowReplicate=false`） |
| B5 | `numExperts % epSize !== 0` | `divEven` 正确分配余数（前几个 rank 多 1 个专家） |
| B6 | `seed=0` | simulated 路由仍可正常运行（种子 0 合法） |
| B7 | 单 token batch | `batchSize=1`，路由正常，不会除零 |

## 风险与注意事项

### 兼容性影响

- **SimMoeBackend 是全新类**，不影响现有 `SimCommGroup`、`ParallelTopology`、`ParallelMetrics` 的接口和使用方式。
- **MockEngine 改动最小**：仅增加 `isMoe` 条件分支，非 MoE 模型路径完全不变。
- **ParallelMetrics `epExpertLoad`** 字段已存在但初始为空数组，`SimMoeBackend` 需在首次写入前扩展至 `numExperts` 长度。这不影响现有代码（现有测试未写入 `epExpertLoad`）。

### 性能影响

- 路由计算为 O(batchSize × moeTopK × log(numExperts))（排序选 top_k），对典型 batchSize（≤1024）和 numExperts（≤256）完全可接受。
- all-to-all 成本计算为 O(epSize)，纯算术，无性能风险。
- `epExpertLoad` 数组写入为 O(batchSize × moeTopK)，与路由复杂度同级。

### 回滚方案

- 删除 `parallel/moe.ts`，还原 `parallel/index.ts` 和 `engine/index.ts` 的修改即可完整回滚。
- 无数据库/配置迁移，无持久化副作用。
