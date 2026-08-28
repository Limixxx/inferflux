---
title: "Issue #14 解决方案"
issue_number: 14
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #14 解决方案

## 需求分析

- **问题描述**：Issue #14 要求实现 SGLang 仿真器的 K3 阶段核心组件 `CacheManager`（naive backend）。CacheManager 是调度器中最复杂的组件，负责 KV cache 页分配、前缀缓存管理、eviction 触发以及页数守恒校验。当前代码库中 `CacheManager` 仅存在于 `types.ts` 中的占位接口声明，需要升级为完整的 class 实现。

- **能力目标**：
  1. `CacheManager` 类持有 `kvPool: BaseKVCachePool` 与 `prefixCache: BasePrefixCache`；对外 API 包括 `cacheReq(req, finished)` + `freeCache(req)` + `availableSize(prefillTokenBudget)`
  2. `cacheReq` 核心逻辑实现 **5 区域精细划分**（严格按 §9.11 伪码）：
     - 区域 1（前部保留区）：`lastComputedStartIdx` 之前，且 ≤ `computedPrefixLen` 的区域 → prefix_matched（不释放）
     - 区域 2（前部已释放区）：`lastComputedStartIdx` 之前，但 > `matchedPrefixLen` 的区域 → lazy_free_region 释放
     - 区域 3（新写入区）：`[matchedPrefixLen, computedPrefixLen)` → allocate_paged 分配，insert_backend
     - 区域 4（尾部保留区）：已有的 `nextStartPos` 之前已分配的区域 → prefix_unmatched（保留但不命中）
     - 区域 5（尾部已释放区）：其他旧分配 → lazy_free_region 释放
  3. `lazyFreeRegion()` 上下文管理器：在上下文内收集所有 `_free` 调用到 `lazyFreeList`，退出时一次性合并到 `freeSlots`；页对齐切片确保 `pageSize>1` 时只取每页起始位置去重
  4. `availableSize(prefillTokenBudget)` 返回 `{ pages, tokens }`（`numPages - allocatedPages × pageSize`）
  5. **页数守恒不变式**：`allocatedPages + freeSlots.length === numPages`（每次 `cacheReq` 后校验）
  6. 单元测试：短/中/长 prompt 各一组；`matched < computed`；prefix 增长；`lazyFreeRegion` 正确计数；`availableSize` 单调/不越界

- **影响范围**：仅修改 `server/src/sglang/cache/` 目录（新增 `cache_manager.ts` 文件），以及 `server/src/sglang/cache/index.ts`（新增 re-export）、`server/src/sglang/types.ts`（将 `CacheManager` 从占位 interface 替换为 class 引用）、`server/src/sglang/index.ts`（更新 re-export）、`server/src/test/sglang-k3.test.ts`（新增测试文件）。

## 改造方案

### 总体思路

按照 §9.11 中 `CacheManager` 的完整伪码，将其从 `types.ts` 中的占位接口升级为独立的 class 实现。核心设计决策如下：

1. **新建 `cache/cache_manager.ts`**：CacheManager 作为独立文件，依赖 K1 抽象层（`BaseKVCachePool`、`BasePrefixCache`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`CacheSizeInfo`）和 K2 实现（`MockKVCachePool`、`NaivePrefixCache`），不引入新的外部依赖
2. **free_slots 与 MockKVCachePool.freePagesPool 的关系**：CacheManager 持有自己的 `freeSlots` 数组（初始化为 `[0, pageSize, 2*pageSize, ...]`），与 `MockKVCachePool.freePagesPool` 是**同一份页资源的两个视角** — CacheManager 的 `freeSlots` 管理空闲页的物理位置，MockKVCachePool 的 `freePagesPool` 由 K2 独立管理。在本阶段（naive backend），CacheManager **直接使用自己的 `freeSlots`** 进行页分配与回收，不与 MockKVCachePool 交互（MockKVCachePool 仅用于内存记账）
3. **TypeScript 适配**：Python 的 `@contextmanager` 适配为 TS 的 `[Symbol.dispose]` 或显式 `beginLazyFree()/endLazyFree()` 方法对；`div_ceil` 复用 `core/divCeil`；`alignDown` 复用 `core/alignDown`
4. **5 区域 cacheReq 实现**：严格映射 §9.11 的 `cache_req` 伪码，5 区域通过 `alignedLen`、`oldHandle.cachedLen`、`cachedLen`（insert 返回值）三个关键长度边界来划分

### 详细设计

#### 1. CacheManager 类（cache/cache_manager.ts）

```typescript
import { BasePrefixCache, BaseCacheHandle, MatchResult, InsertResult, CacheSizeInfo } from "./index";
import { divCeil, alignDown } from "../core";

export class CacheManager {
  readonly numPages: number;
  readonly pageSize: number;
  readonly pageTable: number[][];

  /** 空闲页物理位置列表（初始 [0, pageSize, 2*pageSize, ...]） */
  freeSlots: number[];

  /** lazy free 收集列表 */
  lazyFreeList: number[];

  /** 是否在 lazy_free_region 上下文内 */
  private _inLazyFree: boolean;

  /** 前缀缓存实例（naive 或 radix） */
  readonly prefixCache: BasePrefixCache;

  constructor(numPages: number, pageSize: number, pageTable: number[][], cacheType: "radix" | "naive" = "naive") {
    this.numPages = numPages;
    this.pageSize = pageSize;
    this.pageTable = pageTable;
    this.freeSlots = Array.from({ length: numPages }, (_, i) => i * pageSize);
    this.lazyFreeList = [];
    this._inLazyFree = false;

    // 根据 cacheType 选择前缀缓存策略
    if (cacheType === "radix") {
      // K4 实现，K3 阶段暂不使用
      throw new Error("RadixPrefixCache not implemented yet (K4)");
    } else {
      // K2 的 NaivePrefixCache
      this.prefixCache = new NaivePrefixCache(numPages, pageSize);
    }
  }
}
```

#### 2. availableSize 属性

```typescript
get availableSize(): number {
  return this.prefixCache.sizeInfo.evictableSize + this.freeSlots.length * this.pageSize;
}
```

对齐 §9.11：`evictable_size + len(free_slots) * page_size`。

#### 3. matchReq 方法

```typescript
matchReq(req: { inputIds: number[]; inputLen: number }): MatchResult {
  const inputLen = req.inputLen;
  if (inputLen <= 0) throw new Error("matchReq: inputLen must be > 0");
  // 排除最后一个 token（语义对齐 SGLang：最后一个 token 无 KV cache）
  return this.prefixCache.matchPrefix(req.inputIds.slice(0, inputLen - 1));
}
```

#### 4. lock / unlock 方法

```typescript
lock(handle: BaseCacheHandle): void {
  this.prefixCache.lockHandle(handle);
}

unlock(handle: BaseCacheHandle): void {
  this.prefixCache.lockHandle(handle, true);
}
```

#### 5. allocatePaged 方法

```typescript
allocatePaged(req: { deviceLen: number; cachedLen: number; tableIdx: number }): void {
  const { deviceLen, cachedLen, tableIdx } = req;
  const lastPage = divCeil(deviceLen, this.pageSize) - divCeil(cachedLen, this.pageSize);
  const neededPages = Math.max(0, lastPage);

  if (neededPages > this.freeSlots.length) {
    // 触发 eviction
    const evictSize = (neededPages - this.freeSlots.length) * this.pageSize;
    const evicted = this.prefixCache.evict(evictSize);
    this.freeSlots.push(...evicted);
  }

  for (let i = 0; i < neededPages; i++) {
    if (this.freeSlots.length === 0) break;
    const pageIdx = this.freeSlots.pop()!;
    const startPos = (divCeil(cachedLen, this.pageSize) + i) * this.pageSize;
    for (let j = 0; j < this.pageSize; j++) {
      const pos = startPos + j;
      if (pos < this.pageTable[tableIdx].length) {
        this.pageTable[tableIdx][pos] = pageIdx;
      }
    }
  }
}
```

#### 6. cacheReq 方法（核心：5 区域逻辑）

```typescript
cacheReq(req: { inputIds: number[]; cachedLen: number; tableIdx: number; cacheHandle: BaseCacheHandle | null }, finished: boolean): void {
  // page-aligned 切片：只处理完整页
  const alignedLen = alignDown(req.cachedLen, this.pageSize);
  const insertIds = req.inputIds.slice(0, alignedLen);
  const pageIndices = this.pageTable[req.tableIdx].slice(0, alignedLen);
  const oldHandle = req.cacheHandle;

  // 插入前缀到前缀缓存
  const insertResult = this.prefixCache.insertPrefix(insertIds, pageIndices);
  const cachedLen = insertResult.cachedLen;
  const newHandle = insertResult.cudaHandle;

  // 解锁旧 handle
  if (oldHandle !== null) {
    this.unlock(oldHandle);
  }

  // 释放已存在于缓存中的重复部分
  // 区域 2（前部已释放区）：oldHandle.cachedLen ~ cachedLen 之间的页
  if (oldHandle !== null) {
    this._free(pageIndices.slice(oldHandle.cachedLen, cachedLen));
  }

  if (finished) {
    // 区域 5（尾部已释放区）：释放 newHandle.cachedLen 之后的所有页
    this._free(pageIndices.slice(newHandle.cachedLen));
  } else {
    // 区域 4（尾部保留区）：保留，更新 handle
    (req as { cacheHandle: BaseCacheHandle | null }).cacheHandle = newHandle;
    this.lock(newHandle);
  }
}
```

**5 区域映射说明**（以 naive backend 为基准）：

| 区域 | 名称 | 边界 | 操作 |
|------|------|------|------|
| 1 | 前部保留区 | `[0, oldHandle.cachedLen)` | prefix_matched — 不释放 |
| 2 | 前部已释放区 | `[oldHandle.cachedLen, cachedLen)` | `_free` 释放 |
| 3 | 新写入区 | `[cachedLen, alignedLen)` | 已由 insertPrefix 注册到缓存 |
| 4 | 尾部保留区 | `[alignedLen, deviceLen)` 未 finished 时 | 保留，更新 cacheHandle |
| 5 | 尾部已释放区 | `[newHandle.cachedLen, alignedLen)` finished 时 | `_free` 释放 |

在 naive backend 下（`NaivePrefixCache` 总是 miss，`cachedLen` 始终为 0），区域 2 退化为空，区域 1 退化为空，简化为：finished 时释放所有页，未 finished 时保留所有页并锁定 handle。

#### 7. _free 方法

```typescript
private _free(indices: number[]): void {
  if (this.pageSize > 1) {
    // 页对齐：只取每页起始位置（去重）
    indices = indices.filter((_, i) => i % this.pageSize === 0);
  }
  if (this._inLazyFree) {
    this.lazyFreeList.push(...indices);
  } else {
    this.freeSlots.push(...indices);
  }
}
```

#### 8. lazyFreeRegion 上下文管理

TypeScript 无原生 `@contextmanager`，采用显式 begin/end 方法对：

```typescript
beginLazyFree(): void {
  this._inLazyFree = true;
  this.lazyFreeList = [];
}

endLazyFree(): void {
  this._inLazyFree = false;
  this.freeSlots.push(...this.lazyFreeList);
  this.lazyFreeList = [];
}
```

调用方（`SimScheduler._processLastData`）使用模式：
```typescript
this.cacheManager.beginLazyFree();
try {
  // ... cache_req / free_cache calls
} finally {
  this.cacheManager.endLazyFree();
}
```

#### 9. checkIntegrity 方法

```typescript
checkIntegrity(): void {
  this.prefixCache.checkIntegrity();
  const cachePages = Math.floor(this.prefixCache.sizeInfo.totalSize / this.pageSize);
  if (this.freeSlots.length + cachePages !== this.numPages) {
    throw new Error(
      `CacheManager integrity check failed: free_pages(${this.freeSlots.length}) + ` +
      `cache_pages(${cachePages}) != num_pages(${this.numPages})`
    );
  }
}
```

#### 10. freeCache 方法

```typescript
freeCache(req: { inputIds: number[]; cachedLen: number; tableIdx: number; cacheHandle: BaseCacheHandle | null }): void {
  this.cacheReq(req, true);
}
```

语义等同于 `cache_req(req, finished=True)`，提供更直观的命名。

### 接口变更

1. **`types.ts`**：删除 `CacheManager` 占位 interface，替换为从 `cache/cache_manager.ts` 导入的 class 引用
2. **`cache/index.ts`**：新增 `export { CacheManager } from "./cache_manager"`
3. **`index.ts`**：新增 `CacheManager` 的 re-export

### 数据结构改动

无新增数据结构。CacheManager 内部使用 `freeSlots: number[]` 和 `lazyFreeList: number[]` 两个数组，与 §9.11 伪码完全对齐。

### 修改点清单

1. **新建 `server/src/sglang/cache/cache_manager.ts`**：CacheManager class 完整实现，包含 `constructor`、`availableSize`、`matchReq`、`lock`、`unlock`、`allocatePaged`、`cacheReq`（5 区域）、`_free`、`beginLazyFree`/`endLazyFree`、`checkIntegrity`、`freeCache`
2. **修改 `server/src/sglang/cache/index.ts`**：新增 `export { CacheManager } from "./cache_manager"`
3. **修改 `server/src/sglang/types.ts`**：删除 `CacheManager` 占位 interface，改为 `export { CacheManager } from "./cache"`（class 引用）
4. **修改 `server/src/sglang/index.ts`**：新增 `CacheManager` 的 re-export
5. **新建 `server/src/test/sglang-k3.test.ts`**：K3 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | CacheManager 构造（naive backend） | numPages/pageSize/freeSlots/prefixCache 正确初始化 |
| T2 | CacheManager.availableSize | 初始 = numPages × pageSize（naive 无 evictable） |
| T3 | CacheManager.matchReq | naive backend 总是返回 cachedLen=0 |
| T4 | CacheManager.lock/unlock | 调用不抛错 |
| T5 | CacheManager.allocatePaged 基本分配 | 1 页写入 pageTable 正确 |
| T6 | CacheManager.allocatePaged 多页分配 | 3 页，pageTable 写入正确 |
| T7 | CacheManager.allocatePaged 不足触发 eviction | naive eviction 返回空，不足时 break |
| T8 | CacheManager.cacheReq（finished=true） | 释放所有页，freeSlots 恢复 |
| T9 | CacheManager.cacheReq（finished=false） | 保留页，handle 更新并 lock |
| T10 | CacheManager._free 页对齐切片 | pageSize=4 时只释放每页起始位置 |
| T11 | CacheManager.beginLazyFree/endLazyFree | lazy 模式下 free 收集到 lazyFreeList，end 时合并到 freeSlots |
| T12 | CacheManager.checkIntegrity | naive backend 初始状态通过校验 |
| T13 | CacheManager.checkIntegrity 失败 | 手动修改 freeSlots 后抛错 |
| T14 | CacheManager.freeCache | 语义等同 cacheReq(finished=true) |
| T15 | 短 prompt 测试 | inputIds.length < pageSize |
| T16 | 中 prompt 测试 | inputIds.length = 3 × pageSize |
| T17 | 长 prompt 测试 | inputIds.length = 10 × pageSize |
| T18 | matched < computed 场景 | 模拟 oldHandle.cachedLen < insertResult.cachedLen |
| T19 | prefix 增长场景 | 连续 cacheReq 后 availableSize 单调递减 |
| T20 | lazyFreeRegion 正确计数 | lazy 模式下多次 _free 后 freeSlots 增量正确 |
| T21 | availableSize 不越界 | availableSize ∈ [0, numPages × pageSize] |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `CacheManager(numPages=0)` | freeSlots 为空，availableSize=0 |
| B2 | `CacheManager(numPages=1, pageSize=1)` | 最小配置，单页分配/回收 |
| B3 | `cacheReq` 时 `alignedLen=0` | 无页操作，不抛错 |
| B4 | `allocatePaged` 时 `neededPages=0` | 无分配，不抛错 |
| B5 | `endLazyFree` 在无 `beginLazyFree` 时调用 | `_inLazyFree=false`，直接合并空列表（防御性） |
| B6 | `pageSize=16` 的页对齐 `_free` | 只取 `[::pageSize]` 切片 |
| B7 | `matchReq` 时 `inputLen=1` | 匹配空前缀（slice(0, 0)） |
| B8 | 连续 allocatePaged 全部页耗尽 | freeSlots=[], 再分配 break 不抛错 |

## 风险与注意事项

- **兼容性影响**：`types.ts` 中 `CacheManager` 从 `interface` 变为 class 引用，需确保所有使用处兼容。当前仅有 `SgSimContext.cacheMgr` 使用此类型（类型为 `CacheManager | null`），class 引用完全兼容。
- **性能影响**：`freeSlots.pop()` / `freeSlots.push()` 是 O(1) 操作；`lazyFreeList` 收集后一次性 `extend` 替代多次 `extend`，减少合并开销。页数守恒检查在 `checkIntegrity` 中显式调用，不影响热路径性能。
- **回滚方案**：所有改动在 `issue-14` 分支，合并前可安全回滚。
- **依赖关系**：Issue #13 (K2: MockKVCachePool + NaivePrefixCache) 必须已完成并合并。K3 依赖 K1 的抽象类型定义和 K2 的具体实现。
- **阻塞关系**：本 Issue 完成后，S2 (PrefillManager + DecodeManager) 和 S3 (MockEngine/SimScheduler) 才能集成 CacheManager 的完整 API。
- **naive vs radix 差异**：本阶段仅实现 naive backend（NaivePrefixCache），radix backend（RadixPrefixCache）将在 K4 Issue 中实现。CacheManager 构造函数预留了 `cacheType` 参数，K4 时扩展即可。
