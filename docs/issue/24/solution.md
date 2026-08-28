---
title: "Issue #24 解决方案"
issue_number: 24
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #24 解决方案

## 需求分析

### 问题描述

Issue #24 要求在 `server/src/sglang/parallel/**` 下实现 P2a 层 DataParallelController — 标准 DP（round_robin/shortest_queue 策略 + pages_per_rank 分配），包含以下核心组件：

1. **DPRankState**：单个 DP 副本状态，维护 rank、load（等待中请求数）、pages_allocated、pages_capacity
2. **DataParallelController**（§10.3 标准 DP）：支持 round_robin / shortest_queue 两种请求分发策略，pages_per_rank 按 dp_size 均分 KV 池容量
3. **与 SimScheduler 集成**：`_addRequestFlow` 中调用 `dp_controller.select_rank_for_request` 分配 DP 副本，分配失败时 resp_reject；`_freeRequestResources` 中调用 `dp_controller.free_pages` 释放页
4. **ParallelMetrics.dp_rank_load / dp_allocate_pages_per_rank 字段回填**
5. **dp_size=1 时退化 noop**
6. **单元测试**：round_robin 轮询均匀、shortest_queue 负载均衡、分配失败 reject、free 回写计数、pages_per_rank 越界保护

### 能力目标

- 为仿真器提供标准 DP 请求分发能力，使多 DP 副本各自独立调度
- 通过 round_robin 策略均匀轮询分发，或 shortest_queue 策略选择最小负载副本
- 按 `pages_per_rank = num_pages // dp_size` 为每个 DP 副本划分独立 KV 页池，分配失败时拒绝请求
- 与 SimScheduler 的请求添加/释放流程无缝集成
- dp_size=1 时全部操作退化为 rank 0 的 noop 行为，不破坏现有单实例仿真

### 影响范围

- **新增文件**：`server/src/sglang/parallel/dp_controller.ts`
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出）、`server/src/sglang/types.ts`（SimulatorConfig 已有 dpLoadBalanceStrategy 字段，无需新增）
- **不修改**：业务调度逻辑源码（SimScheduler 集成代码将在后续 P6 init_parallel_groups Issue 中实现）、测试代码

### 依赖关系

- **依赖 #21 (P0)**：SimCommGroup + ParallelTopology + ParallelMetrics — **已实现**，ParallelMetrics 中 `dpRankLoad` / `dpAllocatePagesPerRank` 字段已预留
- **依赖 #10 (S1)**：核心数据结构 Req（已有 dpRank 属性） — **已实现**
- **依赖 #12 (K5)**：calculateMemoryBudget（提供 num_pages 输入给 pages_per_rank 计算） — **已实现**

---

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建 `dp_controller.ts` 模块，实现 DPRankState 和 DataParallelController 两个类。DataParallelController 根据 SimulatorConfig 中已有的 `dpSize` 和 `dpLoadBalanceStrategy` 字段进行初始化，提供 `select_rank_for_request` / `allocate_pages` / `free_pages` 三个核心方法，并通过 `parallel/index.ts` 统一导出。

### 详细设计

#### 1. DPRankState — 单个 DP 副本状态

**文件**：`server/src/sglang/parallel/dp_controller.ts`

```typescript
export class DPRankState {
  readonly rank: number;
  load: number;              // 等待中的请求数
  pages_allocated: number;   // 已分配的页数
  readonly pages_capacity: number;  // 该副本的页容量上限

  constructor(rank: number, pages_capacity: number) {
    this.rank = rank;
    this.load = 0;
    this.pages_allocated = 0;
    this.pages_capacity = pages_capacity;
  }

  /** 当前可用页数 */
  get pages_available(): number {
    return this.pages_capacity - this.pages_allocated;
  }
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| rank | number | DP 副本索引 [0, dp_size) |
| load | number | 当前等待中/运行中的请求数（select 时 +1，free 时 -1） |
| pages_allocated | number | 已分配给请求的页数 |
| pages_capacity | number | 该副本可用的页容量上限（= pages_per_rank） |

#### 2. DataParallelController — 标准 DP 请求分发器

**文件**：`server/src/sglang/parallel/dp_controller.ts`

```typescript
export class DataParallelController {
  readonly strategy: "round_robin" | "shortest_queue";
  readonly dp_size: number;
  private readonly _ranks: DPRankState[];
  private _round_robin_idx: number = 0;

  constructor(dp_size: number, total_num_pages: number,
              strategy: "round_robin" | "shortest_queue") { ... }

  /** 获取所有副本状态（只读） */
  get ranks(): ReadonlyArray<DPRankState>;

  /**
   * 为新请求选择 DP 副本并分配页。
   * 两步操作：先 allocate_pages(needed_pages) 成功 → 设 req.dp_rank = rank；记录 load += 1。
   * 失败返回 null（OOM）。
   */
  select_rank_for_request(needed_pages: number): DPRankState | null;

  /**
   * 为指定副本分配页。
   * 条件：pages_capacity - pages_allocated >= needed_pages。
   * 成功返回 true，失败返回 false。
   */
  allocate_pages(rank_idx: number, needed_pages: number): boolean;

  /**
   * 释放指定副本的页并减少负载。
   * pages_allocated 回写；load -= 1。
   * 包含越界保护：pages_allocated = max(0, pages_allocated - freed_pages)。
   */
  free_pages(rank_idx: number, freed_pages: number): void;
}
```

**构造逻辑**：

```
pages_per_rank = divEven(total_num_pages, dp_size)  // 复用 core/divEven
ranks[i] = new DPRankState(i, pages_per_rank[i])
```

- 使用 `divEven(total_num_pages, dp_size)` 均分页池，余数分配给前几个副本（与 K5 的 `divEven` 语义一致）
- dp_size=1 时：pages_per_rank = [total_num_pages]，只有一个 rank 0

**select_rank_for_request 逻辑**：

```
if dp_size <= 1:
  // 单副本 noop：始终选择 rank 0
  if ranks[0].pages_available >= needed_pages:
    ranks[0].pages_allocated += needed_pages
    ranks[0].load += 1
    return ranks[0]
  else:
    return null

// 多副本：先选出候选 rank
if strategy == "round_robin":
  candidate = _round_robin_idx % dp_size
  _round_robin_idx += 1
elif strategy == "shortest_queue":
  candidate = argmin(ranks[i].load for i in 0..dp_size-1)

// 再尝试 allocate_pages
if ranks[candidate].pages_available >= needed_pages:
  ranks[candidate].pages_allocated += needed_pages
  ranks[candidate].load += 1
  return ranks[candidate]
else:
  return null  // OOM
```

**关键设计点**：

- `select_rank_for_request` 是原子操作：选 rank + 分配页 + 增加 load 三步合一，避免中间状态不一致
- round_robin 策略严格轮询（`_round_robin_idx % dp_size`），不论副本是否有足够页
- shortest_queue 策略选择 `load` 最小的副本（`load = 等待中请求数`，不含 pages 维度）
- 分配失败（页不足）返回 null，由调用方（SimScheduler）决定 reject 逻辑
- free_pages 包含 pages_allocated 下界保护（`max(0, ...)`），防止释放页数超过分配页数

#### 3. 与 SimScheduler 集成接口说明

本 Issue 仅实现 DataParallelController 本身，SimScheduler 集成将在后续 P6 Issue 中完成。以下是预期的集成方式（仅供参考，不在本 Issue 范围内）：

**_addRequestFlow 中**：
```
rank_state = dp_controller.select_rank_for_request(needed_pages)
if rank_state is null:
  resp_reject(rid, "DP OOM")
else:
  req.dp_rank = rank_state.rank
  cache_manager.cache_req(req)
```

**_freeRequestResources 中**：
```
cache_manager.free_cache(rid)
dp_controller.free_pages(req.dp_rank, pages_freed)
```

#### 4. ParallelMetrics 回填说明

ParallelMetrics 中已有 `dpRankLoad: number[]` 和 `dpAllocatePagesPerRank: number[]` 字段（P0 实现）。DataParallelController 的调用方（后续 SimScheduler）在仿真结束时按以下方式回填：

```
parallelMetrics.dpRankLoad = dp_controller.ranks.map(r => r.load)
parallelMetrics.dpAllocatePagesPerRank = dp_controller.ranks.map(r => r.pages_allocated)
```

本 Issue 不修改 ParallelMetrics，仅确保 DPRankState 暴露足够的状态属性供外部回填。

#### 5. parallel/index.ts 导出

在 `parallel/index.ts` 中新增导出：

```typescript
export {
  DPRankState,
  DataParallelController,
} from "./dp_controller";
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/dp_controller.ts` — DPRankState + DataParallelController 实现
2. **修改** `server/src/sglang/parallel/index.ts` — 导出 DPRankState 和 DataParallelController

---

## 测试设计

### 验收测试用例清单

#### DPRankState 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | DPRankState 初始化 | rank=0, pages_capacity=100, load=0, pages_allocated=0 |
| T2 | pages_available 计算 | pages_capacity=100, pages_allocated=30 → pages_available=70 |

#### DataParallelController 基础测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T3 | dp_size=1 退化 noop | select_rank_for_request 始终返回 rank 0，分配成功 |
| T4 | dp_size=1 页不足返回 null | pages_capacity=5, needed_pages=10 → null |
| T5 | pages_per_rank 均分 | total=100, dp_size=3 → [34, 33, 33]（divEven 语义） |
| T6 | allocate_pages 成功 | pages_available >= needed_pages → true, pages_allocated 更新 |
| T7 | allocate_pages 失败 | pages_available < needed_pages → false, pages_allocated 不变 |
| T8 | free_pages 回写 | free_pages 后 pages_allocated 减少, load 减少 |
| T9 | free_pages 越界保护 | free_pages(999) → pages_allocated = max(0, ...) 不为负 |

#### round_robin 策略测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T10 | round_robin 轮询均匀 | dp_size=4, 连续 8 次请求 → [0,1,2,3,0,1,2,3] |
| T11 | round_robin 分配失败不影响轮询索引 | 请求到 rank 2 但页不足返回 null，下次仍从 rank 3 开始 |

#### shortest_queue 策略测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T12 | shortest_queue 选最小负载 | 4 副本，rank 0 load=3, rank 1 load=1, rank 2 load=2, rank 3 load=1 → 选 rank 1（最先出现的最小值） |
| T13 | shortest_queue 负载均衡 | 连续请求均匀分布到各副本 |

#### 分配失败 reject 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T14 | 所有副本页不足返回 null | 各副本 pages_available 均小于 needed_pages → null |
| T15 | 部分副本页不足仍可分配 | round_robin 指向满副本返回 null，但其他副本有空间 |

#### 综合测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T16 | 完整生命周期 | select → allocate → free 一轮后状态回到初始 |
| T17 | 大量请求压力测试 | dp_size=8, 1000 次请求 round_robin 均匀分布 |
| T18 | ParallelMetrics 回填验证 | ranks.map(r => r.load) 和 ranks.map(r => r.pages_allocated) 可正确获取 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | dp_size=0 | 构造函数抛出 Error（dp_size 必须 ≥ 1） |
| B2 | total_num_pages=0 | 所有副本 pages_capacity=0，任何分配均返回 null |
| B3 | needed_pages=0 | select_rank_for_request 成功，load+1，pages_allocated 不变 |
| B4 | freed_pages=0 | free_pages 不改变 pages_allocated，但 load -= 1 |
| B5 | round_robin_idx 溢出 | 使用 `% dp_size` 取模，不受 Number 上限影响 |
| B6 | shortest_queue 多副本 load 相同 | 选最先出现的（index 最小的）副本 |

---

## 风险与注意事项

### 兼容性影响

- **无破坏性变更**：本 Issue 仅新增 `dp_controller.ts` 文件和 `parallel/index.ts` 导出条目，不修改任何已有代码
- **SimulatorConfig 无需变更**：`dpSize` 和 `dpLoadBalanceStrategy` 字段已在 P0 (#21) 中加入 SimulatorConfig
- **Req.dpRank 已存在**：S1 (#10) 中已为 Req 类添加 `dpRank: number = 0` 字段，无需修改
- **ParallelMetrics 已预留**：`dpRankLoad` 和 `dpAllocatePagesPerRank` 字段已在 P0 中定义，无需修改

### 性能影响

- select_rank_for_request 为 O(dp_size)（shortest_queue 需遍历所有副本），dp_size 通常 ≤ 8，无性能风险
- round_robin 为 O(1)，仅取模和索引递增
- allocate_pages / free_pages 为 O(1)，纯算术操作

### 回滚方案

- 删除新增文件 `dp_controller.ts` 即可
- `parallel/index.ts` 移除两行导出即可
- 无数据库/持久化变更，回滚无数据风险
