---
title: "Issue #11 解决方案"
issue_number: 11
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #11 解决方案

## 需求分析

- **问题描述**：Issue #11 要求实现 SGLang 仿真器的 K1 阶段 KVCache 基础抽象层，包括 `BaseKVCachePool`、`BasePrefixCache`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`CacheSizeInfo` 以及 `TableManager`。这些抽象接口是后续 K2（MockKVCachePool + NaivePrefixCache）、K3（CacheManager）和 K4（RadixPrefixCache）实现的基础。

- **能力目标**：
  1. 定义 `BaseKVCachePool` 抽象类，包含 KV cache 存储池的核心属性（`numPages`、`pageSize`、`totalCapacity`、`usedCapacity`）和虚方法（`storeKV`）
  2. 定义 `BasePrefixCache` 抽象类，包含前缀缓存操作接口（`matchPrefix`、`insertPrefix`、`lockHandle`、`evict`、`reset`、`checkIntegrity`）和 `sizeInfo` 属性
  3. 定义 `BaseCacheHandle` 抽象类，作为缓存节点句柄，包含 `cachedLen` 属性和 `getMatchedIndices` 虚方法
  4. 定义 `MatchResult` 和 `InsertResult` 值类型，作为 `matchPrefix`/`insertPrefix` 的返回值
  5. 定义 `CacheSizeInfo` 工具类，提供缓存大小统计（`evictableSize`、`protectedSize`、`totalSize`）
  6. 实现 `TableManager` 类，管理 page_table 行分配与 token_pool
  7. 所有类型使用 TS strict 模式，禁用 `any`

- **影响范围**：仅修改 `server/src/sglang/cache/` 目录（抽象层定义）和 `server/src/sglang/scheduler/` 目录（TableManager 实现），以及 `server/src/sglang/types.ts`（更新占位接口）和 `server/src/sglang/index.ts`（更新 re-export）。

- **Issue 与设计文档差异分析**：

  Issue 描述中的部分字段名与总体设计文档 §9.3 存在差异，本方案以设计文档 §9.3 为权威规格，具体差异及决策如下：

  | Issue 描述 | 设计文档 §9.3 | 决策 |
  |-----------|-------------|------|
  | `BaseKVCachePool.num_cpu_pages/num_gpu_pages/hidden_size/num_attention_heads/v_head_size` | `BaseKVCachePool.numPages/pageSize/totalCapacity/usedCapacity` + `storeKV` | 采用 §9.3 定义；模型相关参数（hidden_size 等）属于 `ModelConfig`，不应在 KVCachePool 上暴露 |
  | `BaseKVCachePool.max_prefill_pages/max_extend_pages/max_append_tokens` | 无此属性 | 不实现；这些属于 `SimulatorConfig` 级别约束，不属于 KVCachePool 抽象 |
  | `BaseKVCachePool.allocate_paged/decode_step_latency/fetch_token_latency` | `storeKV` 虚方法 | `allocatePaged` 属于 `CacheManager` 职责（§3.3.4）；latency 方法属于 `MockKVCachePool` 实现细节（K2） |
  | `BasePrefixCache.match(req)/insert(req)` | `matchPrefix(inputIds)/insertPrefix(inputIds, indices)` | 采用 §9.3 签名；`match(req)` 是 `CacheManager.matchReq` 的签名（§9.11） |
  | `BasePrefixCache._tree_walk/_calc_size_bytes` | `_treeWalk`（内部方法） | 内部方法不属于抽象接口，在具体实现类中定义 |
  | `CacheHandle.value: CacheValueType` + `release` 方法 | `BaseCacheHandle.cachedLen` + `getMatchedIndices()` | 采用 §9.3 定义；`value` 和 `release` 是 RadixPrefixCache 实现的具体细节 |
  | `MatchResult.matched/total/cached_nodes/lengths/indices/same_prefix_computed/reduce_threshold` | `MatchResult(NamedTuple): cudaHandle` | 采用 §9.3 定义；Issue 描述的字段来自更早的草案，§9.3 已简化为只携带 `cudaHandle` |
  | `InsertResult.value/prefix_indices` | `InsertResult(NamedTuple): cachedLen, cudaHandle` | 采用 §9.3 定义 |
  | `CacheSizeInfo.total_bytes/required_bytes/free_bytes/pct` | `CacheSizeInfo.evictableSize/protectedSize/totalSize` | 采用 §9.3 定义；Issue 描述的字段来自另一种缓存视角，§9.3 的 evictable/protected 更契合 RadixTree 语义 |
  | `TableManager.alloc_contiguous/get_cpu_token_to_page` | `allocate()/free()` + `tokenPool` | `allocContiguous` 在仿真中不需要（无真实 GPU 内存分配）；`get_cpu_token_to_page` 属于 PD-Disagg 特性；本阶段实现 §3.3.6 的 `allocate`/`free` + `tokenPool` |

## 改造方案

### 总体思路

K1 是纯抽象层 + 基础设施实现阶段，不涉及业务逻辑。核心工作：

1. **在 `cache/index.ts` 中定义所有 KVCache 抽象类型**：`BaseKVCachePool`、`BasePrefixCache`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`CacheSizeInfo`
2. **在 `scheduler/index.ts` 中实现 `TableManager`**：管理 page_table 行分配与 token_pool
3. **更新 `types.ts` 占位接口**：将 S1 中的 `TableManager` 和 `CacheManager` 接口桩替换为指向新实现的引用
4. **更新 `index.ts` re-export**：导出所有新增类型

### 详细设计

#### 1. CacheSizeInfo 工具类（cache/index.ts）

```typescript
/** 缓存大小统计（§9.3 / §9.11 CacheSizeInfo） */
export class CacheSizeInfo {
  evictableSize: number;  // ref_count=0 的节点 token 数（可被 LRU 驱逐）
  protectedSize: number;  // ref_count>0 的节点 token 数（被锁定）

  constructor(evictableSize: number = 0, protectedSize: number = 0) {
    this.evictableSize = evictableSize;
    this.protectedSize = protectedSize;
  }

  get totalSize(): number {
    return this.evictableSize + this.protectedSize;
  }
}
```

- 可变字段，与 §9.11 的 Python `@dataclass` 定义对齐
- `evictableSize`/`protectedSize` 可由 `RadixPrefixCache.lockHandle` 直接修改
- 提供 `totalSize` getter 便捷计算

#### 2. BaseCacheHandle 抽象类（cache/index.ts）

```typescript
/** 缓存句柄抽象基类，指向缓存树中的节点（§9.3） */
export abstract class BaseCacheHandle {
  abstract get cachedLen(): number;
  abstract getMatchedIndices(): number[];
}
```

- `cachedLen` 为抽象 getter，由子类实现（`RadixCacheHandle` 通过字段实现，`NaiveCacheHandle` 通过字段实现）
- `getMatchedIndices()` 返回从 root 到当前节点路径上所有节点的 value 拼接（页索引列表）
- 不包含 `value`/`release`/`lockHandle` 方法——这些属于具体实现类的职责

#### 3. MatchResult 值类型（cache/index.ts）

```typescript
/** matchPrefix 的返回值（§9.3） */
export class MatchResult {
  readonly cudaHandle: BaseCacheHandle;

  constructor(cudaHandle: BaseCacheHandle) {
    this.cudaHandle = cudaHandle;
  }
}
```

- 使用 class 而非 NamedTuple（TS 无原生 NamedTuple），保持不可变语义（readonly）
- 只携带 `cudaHandle`，通过 `handle.cachedLen` 获取匹配长度，通过 `handle.getMatchedIndices()` 获取页索引

#### 4. InsertResult 值类型（cache/index.ts）

```typescript
/** insertPrefix 的返回值（§9.3） */
export class InsertResult {
  readonly cachedLen: number;       // 插入前已在缓存中的长度
  readonly cudaHandle: BaseCacheHandle;

  constructor(cachedLen: number, cudaHandle: BaseCacheHandle) {
    this.cachedLen = cachedLen;
    this.cudaHandle = cudaHandle;
  }
}
```

- `cachedLen` 表示本次插入操作之前已经在缓存中存在的 token 长度
- `cudaHandle` 与 `MatchResult.cudaHandle` 统一命名（§9.3 要求）

#### 5. BaseKVCachePool 抽象类（cache/index.ts）

```typescript
/** KV cache 存储池抽象基类（§9.3） */
export abstract class BaseKVCachePool {
  abstract get numPages(): number;
  abstract get pageSize(): number;
  abstract get totalCapacity(): number;
  abstract get usedCapacity(): number;

  abstract storeKV(
    k: number[], v: number[], outLoc: number[], layerId: number
  ): void;
}
```

- 四个只读属性：`numPages`/`pageSize`/`totalCapacity`/`usedCapacity`
- `storeKV` 虚方法，仿真中为 noop（具体实现见 K2 `MockKVCachePool`）
- 不包含 `allocatePaged`——该方法属于 `CacheManager`（§3.3.4）
- 不包含模型相关属性（`hiddenSize` 等）——这些属于 `ModelConfig`

#### 6. BasePrefixCache 抽象类（cache/index.ts）

```typescript
/** 前缀缓存抽象基类（§9.3） */
export abstract class BasePrefixCache {
  abstract get sizeInfo(): CacheSizeInfo;

  abstract matchPrefix(inputIds: number[]): MatchResult;
  abstract insertPrefix(inputIds: number[], indices: number[]): InsertResult;
  abstract lockHandle(handle: BaseCacheHandle, unlock?: boolean): void;
  abstract evict(size: number): number[];
  abstract reset(): void;
  abstract checkIntegrity(): void;
}
```

- `matchPrefix(inputIds)` 对应 §9.3 签名，返回 `MatchResult`
- `insertPrefix(inputIds, indices)` 对应 §9.3 签名，返回 `InsertResult`
- `lockHandle(handle, unlock?)` 统一 lock/unlock 接口，`unlock` 默认 `false`
- `evict(size)` 返回被驱逐的页索引列表
- `reset()` 重置缓存
- `checkIntegrity()` 完整性校验（仿真中简化为空实现）
- `sizeInfo` 为 getter，返回 `CacheSizeInfo` 实例

#### 7. TableManager 实现（scheduler/index.ts）

```typescript
/** 管理 page_table 行分配（§3.3.6 / §9.11 完整实现） */
export class TableManager {
  readonly maxRunningReq: number;
  pageTable: number[][];
  tokenPool: number[][];
  freeTableIndices: number[];

  constructor(maxRunningReq: number, pageTable: number[][]) {
    this.maxRunningReq = maxRunningReq;
    this.pageTable = pageTable;
    // 最后一行预留给 dummy req
    this.freeTableIndices = Array.from(
      { length: maxRunningReq }, (_, i) => i
    );
    this.tokenPool = Array.from(
      { length: maxRunningReq + 1 },
      () => new Array(pageTable[0].length).fill(0)
    );
  }

  get availableSize(): number {
    return this.freeTableIndices.length;
  }

  allocate(): number {
    if (this.freeTableIndices.length === 0) {
      throw new Error("No available table indices");
    }
    return this.freeTableIndices.pop()!;
  }

  free(tableIdx: number): void {
    this.freeTableIndices.push(tableIdx);
  }
}
```

- 严格遵循 §3.3.6 / §9.11 规格
- `pageTable` 为外部传入的引用（由 `MockEngine` 创建）
- `tokenPool` 在内部创建，与 `pageTable` 相同 shape
- `freeTableIndices` 使用栈（后进先出），`allocate` 从末尾 pop，`free` 追加到末尾
- `availableSize` 为 getter，返回可用行数
- 不实现 `allocContiguous` 和 `get_cpu_token_to_page`——前者在仿真中不需要真实连续内存，后者属于 PD-Disagg 特性

#### 8. types.ts 更新

替换 S1 中的占位接口，使其引用 K1 的实际实现：

- `TableManager` 接口 → 导入 `scheduler/index.ts` 中的 `TableManager` 类
- `CacheManager` 接口保持占位（K3 实现），但 `matchReq` 返回值类型改为 `MatchResult`

```typescript
// 替换 TableManager 占位接口
export { TableManager } from "./scheduler";

// CacheManager 占位接口更新
export interface CacheManager {
  readonly availableSize: number;
  matchReq(req: unknown): MatchResult;
  lockReq(handle: BaseCacheHandle): void;
  unlockReq(handle: BaseCacheHandle): void;
}
```

### 修改点清单

1. **`server/src/sglang/cache/index.ts`**：从占位注释升级为完整抽象层实现，包含 `CacheSizeInfo`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`BaseKVCachePool`、`BasePrefixCache` 六个类型定义
2. **`server/src/sglang/scheduler/index.ts`**：从占位注释升级为实现 `TableManager` 类
3. **`server/src/sglang/types.ts`**：更新 `TableManager` 导出（从占位接口改为引用 scheduler 中的类）、更新 `CacheManager` 接口返回类型
4. **`server/src/sglang/index.ts`**：新增 re-export `CacheSizeInfo`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`BaseKVCachePool`、`BasePrefixCache`；更新 `TableManager` 的导出来源
5. **`server/src/test/sglang-k1.test.ts`**：新增 K1 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | CacheSizeInfo 构造 | 默认值 0,0；自定义值正确赋值 |
| T2 | CacheSizeInfo.totalSize | evictableSize + protectedSize |
| T3 | MatchResult 构造 | cudaHandle 正确持有 |
| T4 | MatchResult 不可变 | readonly 字段赋值抛错 |
| T5 | InsertResult 构造 | cachedLen + cudaHandle 正确持有 |
| T6 | InsertResult 不可变 | readonly 字段赋值抛错 |
| T7 | BaseCacheHandle 子类化 | 子类可实现 cachedLen getter 和 getMatchedIndices |
| T8 | BaseKVCachePool 子类化 | 子类可实现所有抽象属性和方法 |
| T9 | BasePrefixCache 子类化 | 子类可实现所有抽象方法和 sizeInfo |
| T10 | TableManager 构造 | freeTableIndices 初始化为 0..maxRunningReq-1 |
| T11 | TableManager.allocate | 分配并返回末尾 index |
| T12 | TableManager.free | 释放 index 回收到栈 |
| T13 | TableManager.availableSize | 反映当前可用行数 |
| T14 | TableManager 分配耗尽 | 无可用行时抛出 Error |
| T15 | TableManager.tokenPool | 正确创建 maxRunningReq+1 行 |
| T16 | TableManager 循环分配释放 | allocate→free→allocate 正确工作 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `CacheSizeInfo(0, 0)` | totalSize = 0 |
| B2 | `TableManager(maxRunningReq=1)` | 只有一行可用，最后一行给 dummy |
| B3 | `TableManager` 连续 allocate 直到耗尽 | 抛出 Error |
| B4 | `TableManager` free 后重新 allocate | 返回刚释放的 index |
| B5 | `MatchResult` handle.cachedLen = 0 | 空匹配场景 |
| B6 | `InsertResult` cachedLen = 0 | 完全未缓存场景 |
| B7 | `BasePrefixCache.lockHandle` 默认 unlock=false | lock 语义正确 |

## 风险与注意事项

- **兼容性影响**：S1 的 `types.ts` 中 `TableManager` 从 `interface` 变为引用 `class`，需要确保所有使用处兼容。由于 S1 中 `SgSimContext.tableMgr` 类型为 `TableManager | null`，切换到 class 类型后兼容性良好。
- **性能影响**：`TableManager.freeTableIndices` 使用数组栈（pop/push），O(1) 操作，无性能问题。
- **回滚方案**：所有改动在 `issue-11` 分支，合并前可安全回滚。
- **依赖关系**：Issue #10 (S1) 必须已完成并合并。K1 依赖 S1 的 `Req`、`Batch`、`PendingReq` 类型定义。
- **阻塞关系**：本 Issue 完成后，K2（MockKVCachePool + NaivePrefixCache）、K3（CacheManager）、K4（RadixPrefixCache）才能启动，因为它们都需要继承/实现 K1 定义的抽象类型。
- **抽象层设计原则**：本阶段只定义抽象接口，不做任何具体实现。`MockKVCachePool`（K2）和 `RadixPrefixCache`（K4）将在各自 Issue 中实现。`TableManager` 是唯一在本阶段有具体实现的类，因为它不涉及缓存策略，纯粹是资源管理基础设施。
