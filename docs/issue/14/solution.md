---
title: "Issue #14 解决方案"
issue_number: 14
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: revised
review_round: 2
---

# Issue #14 解决方案

## 需求分析

- **问题描述**：Issue #14 要求实现 SGLang 仿真器的 K3 阶段核心组件 `CacheManager`（naive backend）。CacheManager 是调度器中最复杂的组件，负责 KV cache 页分配、前缀缓存管理、eviction 触发以及页数守恒校验。当前代码库中 `CacheManager` 仅存在于 `types.ts` 中的占位接口声明，需要升级为完整的 class 实现。

- **能力目标**：
  1. `CacheManager` 类持有 `kvPool: BaseKVCachePool` 与 `prefixCache: BasePrefixCache`；对外 API 包括 `cacheReq(req, finished)` + `freeCache(req)` + `availableSize`
  2. `cacheReq` 核心逻辑实现 **5 区域精细划分**（严格按 §9.11 伪码）：
     - 区域 1（前部保留区）：`oldHandle.cachedLen` 之前且已存在于前缀缓存中的区域 → prefix_matched（不释放）
     - 区域 2（前部已释放区）：`[oldHandle.cachedLen, cachedLen)` → 被其他请求抢先缓存，需 `_free` 释放重复页
     - 区域 3（新写入区）：`[cachedLen, newHandle.cachedLen)` → 本次新插入 prefix cache，无需额外操作
     - 区域 4（尾部保留区）：`[newHandle.cachedLen, alignedLen)` 未 finished 时 → 保留并更新 cacheHandle
     - 区域 5（尾部已释放区）：finished 时释放 `[newHandle.cachedLen, alignedLen)` → `_free` 释放
  3. `lazyFreeRegion()` 上下文管理器：在上下文内收集所有 `_free` 调用到 `lazyFreeList`，退出时一次性合并到 `freeSlots`
  4. `availableSize` 属性返回 `evictableSize + len(freeSlots) × pageSize`
  5. **页数守恒不变式**：`allocatedPages + freeSlots.length === numPages`（每次 `cacheReq` 后校验）
  6. 单元测试：短/中/长 prompt 各一组；`matched < computed`；prefix 增长；`lazyFreeRegion` 正确计数；`availableSize` 单调/不越界

- **影响范围**：仅修改 `server/src/sglang/cache/` 目录（新增 `cache_manager.ts` 文件），以及 `server/src/sglang/cache/index.ts`（新增 re-export）、`server/src/sglang/types.ts`（将 `CacheManager` 从占位 interface 替换为 class 引用）、`server/src/sglang/index.ts`（更新 re-export）、`server/src/test/sglang-k3.test.ts`（新增测试文件）。

## Round 1 驳回意见回应

### 🔴 HIGH：`_free` 中页对齐切片去重算法修正

**原始问题**：Round 1 方案中 `_free` 使用 `indices.filter((_, i) => i % pageSize === 0)` 按数组索引步长过滤，仅在传入切片从页边界起始时正确。当 `_free` 接收非页对齐起始的子切片（如 `pageIndices[oldHandle.cachedLen:cachedLen]`）时，步长过滤会导致错位，可能释放错误的页或重复释放同一页，破坏页数守恒不变式。

**修正方案**：`_free` 应当对传入的 indices 值（这些值是 `page_table` 中的物理页位置）进行去重，确保同一物理页只释放一次。正确做法：

```typescript
private _free(indices: number[]): void {
  if (indices.length === 0) return;
  if (this.pageSize > 1) {
    // 去重：indices 中同一页的 page_size 个连续位置存储相同的物理页起始位置，
    // 用 Set 去重确保每个物理页只释放一次
    indices = [...new Set(indices)];
  }
  if (this._inLazyFree) {
    this.lazyFreeList.push(...indices);
  } else {
    this.freeSlots.push(...indices);
  }
}
```

**设计原理**：`page_table[table_idx]` 中，同一页的 `page_size` 个连续位置存储的都是该页的起始物理位置值。例如 `pageSize=4` 时，`pageTable[0] = [0,0,0,0, 4,4,4,4, 8,8,8,8]`（第 0 页物理位置 0，第 1 页物理位置 4，第 2 页物理位置 8）。当子切片不从页边界开始（如 `pageIndices[2:6] = [0,0,4,4]`），步长过滤 `[::4]` 只取第一个元素 `0`，遗漏 `4`；而 `Set` 去重得到 `[0, 4]`，正确释放两个页。

当 `pageSize=1` 时，`pageTable` 每个位置存唯一值，无需去重，直接释放。

### 🟡 MEDIUM：5 区域边界在 naive backend 下的退化行为明确注释

**原始问题**：Round 1 方案的 5 区域映射说明不够精确，特别是 naive backend 下各区域退化为空的情形未显式注释。

**修正方案**：在 `cacheReq` 方法的伪码中添加详细的区域划分注释，特别标注 naive backend（`NaivePrefixCache` 总是 miss，`insertPrefix` 返回 `cachedLen=0`）下的退化行为：

- 区域 1 退化：`oldHandle.cachedLen = 0`（首次请求无旧 handle）或 `oldHandle.cachedLen = 0`（naive 总 miss），前部保留区为空
- 区域 2 退化：`cachedLen = 0` 且 `oldHandle.cachedLen = 0`，前部已释放区为空（`_free(pageIndices[0:0])` 为空操作）
- 区域 3 退化：`newHandle.cachedLen = 0`（naive insert 不注册任何页到缓存），新写入区为空
- 区域 4：finished=false 时，保留 `[0, alignedLen)` 范围内的页，更新 handle 为 `newHandle`（`cachedLen=0`），并 lock
- 区域 5：finished=true 时，`_free(pageIndices[0:])` 释放 `[0, alignedLen)` 范围内的所有页

### 🟡 MEDIUM：matchReq 与 lock 的交互确认

**原始问题**：评审者担心 `matchReq` 返回的 `MatchResult` 是否包含可被 `lock` 调用的 `cudaHandle`。

**确认结果**：K1/K2 的 `MatchResult` 类已包含 `cudaHandle: BaseCacheHandle` 字段（见 `cache/index.ts` 第 31-37 行），`lock/unlock` 方法接收 `BaseCacheHandle` 类型参数并委托给 `prefixCache.lockHandle()`。在 naive backend 下，`NaivePrefixCache.matchPrefix()` 返回的 `MatchResult` 包含 `NaiveCacheHandle(0)`（`cachedLen=0`），`lockHandle()` 为 noop 操作（`NaivePrefixCache` 无引用计数）。整个链路完整兼容，无需额外修改。

### 测试补充：页对齐去重 + matchReq+lock 互动

**原始问题**：Round 1 测试缺少页对齐切片去重验证和 matchReq+lock 互动测试。

**修正方案**：新增以下测试用例：
- `_free` 非页对齐子切片去重测试（验证 `Set` 去重逻辑正确性）
- `matchReq` → `lock(result.cudaHandle)` → `unlock(result.cudaHandle)` 完整链路测试
- 多页重复释放测试（验证页数守恒不变式不被破坏）

## 改造方案

### 总体思路

按照 §9.11 中 `CacheManager` 的完整伪码，将其从 `types.ts` 中的占位接口升级为独立的 class 实现。核心设计决策如下：

1. **新建 `cache/cache_manager.ts`**：CacheManager 作为独立文件，依赖 K1 抽象层（`BaseKVCachePool`、`BasePrefixCache`、`BaseCacheHandle`、`MatchResult`、`InsertResult`、`CacheSizeInfo`）和 K2 实现（`MockKVCachePool`、`NaivePrefixCache`），不引入新的外部依赖
2. **free_slots 与 MockKVCachePool.freePagesPool 的关系**：CacheManager 持有自己的 `freeSlots` 数组（初始化为 `[0, pageSize, 2*pageSize, ...]`），与 `MockKVCachePool.freePagesPool` 是**同一份页资源的两个视角** — CacheManager 的 `freeSlots` 管理空闲页的物理位置，MockKVCachePool 的 `freePagesPool` 由 K2 独立管理。在本阶段（naive backend），CacheManager **直接使用自己的 `freeSlots`** 进行页分配与回收，不与 MockKVCachePool 交互（MockKVCachePool 仅用于内存记账）
3. **TypeScript 适配**：Python 的 `@contextmanager` 适配为 TS 的显式 `beginLazyFree()/endLazyFree()` 方法对；`div_ceil` 复用 `core/divCeil`；`alignDown` 复用 `core/alignDown`
4. **5 区域 cacheReq 实现**：严格映射 §9.11 的 `cache_req` 伪码，5 区域通过 `oldHandle.cachedLen`、`insertResult.cachedLen`、`newHandle.cachedLen` 三个关键长度边界来划分，并对 naive backend 退化行为添加显式注释

### 详细设计

#### 1. CacheManager 类（cache/cache_manager.ts）

```typescript
import { BasePrefixCache, BaseCacheHandle, MatchResult, InsertResult, CacheSizeInfo, NaivePrefixCache } from "./index";
import { divCeil } from "../core";

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

**交互说明**：返回的 `MatchResult` 包含 `cudaHandle: BaseCacheHandle`，调用方可通过 `cacheManager.lock(result.cudaHandle)` 锁定前缀缓存结果。naive backend 下 `lockHandle()` 为 noop。

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
    // 触发 eviction：从 RadixCache 的 evictable 节点中回收页
    const evictSize = (neededPages - this.freeSlots.length) * this.pageSize;
    const evicted = this.prefixCache.evict(evictSize);
    this.freeSlots.push(...evicted);
  }

  // 分配页并写入 page_table
  // 边界情况：如果 eviction 后仍不足，循环 break，未分配位置保持为 0
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
  // page-aligned 切片：只处理完整页（aligned_len = floor(cachedLen / pageSize) * pageSize）
  const alignedLen = Math.floor(req.cachedLen / this.pageSize) * this.pageSize;
  const insertIds = req.inputIds.slice(0, alignedLen);
  const pageIndices = this.pageTable[req.tableIdx].slice(0, alignedLen);
  const oldHandle = req.cacheHandle;

  // 插入前缀到前缀缓存
  const insertResult = this.prefixCache.insertPrefix(insertIds, pageIndices);
  const cachedLen = insertResult.cachedLen;       // 插入前已在缓存中的长度
  const newHandle = insertResult.cudaHandle;       // 新的缓存句柄

  // 解锁旧 handle
  if (oldHandle !== null) {
    this.unlock(oldHandle);
  }

  // ---- 5 区域划分（§9.11 / §5.3.3）----
  //
  // 区域 1 [0, oldHandle.cachedLen) — 前部保留区：已在 prefix cache，无需操作
  //   naive 退化：oldHandle.cachedLen=0（总是 miss），区域 1 为空
  //
  // 区域 2 [oldHandle.cachedLen, cachedLen) — 前部已释放区：被其他请求抢先缓存，需释放重复页
  //   naive 退化：cachedLen=0 且 oldHandle.cachedLen=0，区域 2 为空（空操作）
  if (oldHandle !== null) {
    this._free(pageIndices.slice(oldHandle.cachedLen, cachedLen));
  }

  // 区域 3 [cachedLen, newHandle.cachedLen) — 新写入区：本次 insertPrefix 新注册到缓存
  //   无需额外操作（insertPrefix 已处理）
  //   naive 退化：newHandle.cachedLen=0，区域 3 为空

  if (finished) {
    // 区域 4 + 5 合并释放（finished 时全部释放）
    // 区域 4 [newHandle.cachedLen, alignedLen) — 尾部保留区：finished 时释放
    // 区域 5 [alignedLen, req.cachedLen) 之后 — 超出 cached_len 的 forward 部分（非页对齐碎片，不存于 pageIndices）
    //   naive 退化：newHandle.cachedLen=0，释放 pageIndices[0:] 即全部已分配页
    this._free(pageIndices.slice(newHandle.cachedLen));
  } else {
    // 区域 4（未 finished）：保留，更新 handle
    (req as { cacheHandle: BaseCacheHandle | null }).cacheHandle = newHandle;
    this.lock(newHandle);
  }
}
```

**5 区域映射对照表**：

| 区域 | 名称 | 边界 | 操作 | naive 退化 |
|------|------|------|------|-----------|
| 1 | 前部保留区 | `[0, oldHandle.cachedLen)` | 无操作 | 空（oldHandle.cachedLen=0） |
| 2 | 前部已释放区 | `[oldHandle.cachedLen, cachedLen)` | `_free` | 空操作（cachedLen=0） |
| 3 | 新写入区 | `[cachedLen, newHandle.cachedLen)` | insertPrefix 已处理 | 空（newHandle.cachedLen=0） |
| 4 | 尾部保留区 | `[newHandle.cachedLen, alignedLen)` | finished=false: 保留+lock；finished=true: 释放 | finished=false: 保留全部页；finished=true: 释放全部页 |
| 5 | 尾部已释放区 | `[alignedLen, ...)` | 非页对齐碎片，不在 pageIndices 中 | 同左 |

#### 7. _free 方法（修正版：Set 去重）

```typescript
private _free(indices: number[]): void {
  if (indices.length === 0) return;
  if (this.pageSize > 1) {
    // 关键修正（Round 1 驳回）：使用 Set 去重而非步长切片
    // page_table 中同一页的 page_size 个连续位置存储相同的物理页起始位置值，
    // 例如 pageSize=4: pageTable[0] = [0,0,0,0, 4,4,4,4, 8,8,8,8]
    // 当子切片不从页边界开始（如 pageIndices[2:6] = [0,0,4,4]），
    // 步长切片 [::pageSize] 只取第一个元素 0，遗漏 4；
    // Set 去重得到 [0, 4]，正确释放两个页。
    indices = [...new Set(indices)];
  }
  if (this._inLazyFree) {
    this.lazyFreeList.push(...indices);
  } else {
    this.freeSlots.push(...indices);
  }
}
```

**与 §9.11 Python 实现的差异说明**：技术报告 `_free` 使用 `indices[::self.page_size]`（步长切片），在 Python 中同样存在非页对齐子切片时的错位问题。但 SGLang 实际运行时，`cache_req` 的调用场景保证子切片起始索引 `oldHandle.cachedLen` 始终是页对齐的（因为 `cachedLen` 是前缀匹配返回值，而前缀匹配在树节点级别返回，树节点以页为粒度）。因此 `[::page_size]` 在 SGLang 生产代码中不会出错。

然而，仿真器的测试场景可能构造非页对齐的子切片（如直接测试 `_free` 逻辑），为确保鲁棒性和防御性编程，本方案采用 `Set` 去重替代步长切片。性能影响可忽略（`_free` 调用频率低，`Set` 创建开销极小）。

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

1. **`types.ts`**：删除 `CacheManager` 占位 interface（第 224-230 行），替换为从 `cache/cache_manager.ts` 导入的 class 引用
2. **`cache/index.ts`**：新增 `export { CacheManager } from "./cache_manager"`
3. **`index.ts`**：新增 `CacheManager` 的 re-export（移除从 types 的 re-export，改为从 cache 的 re-export）

### 数据结构改动

无新增数据结构。CacheManager 内部使用 `freeSlots: number[]` 和 `lazyFreeList: number[]` 两个数组，与 §9.11 伪码完全对齐。

### 修改点清单

1. **新建 `server/src/sglang/cache/cache_manager.ts`**：CacheManager class 完整实现，包含 `constructor`、`availableSize`、`matchReq`、`lock`、`unlock`、`allocatePaged`、`cacheReq`（5 区域，含 naive 退化注释）、`_free`（Set 去重版）、`beginLazyFree`/`endLazyFree`、`checkIntegrity`、`freeCache`
2. **修改 `server/src/sglang/cache/index.ts`**：新增 `export { CacheManager } from "./cache_manager"`
3. **修改 `server/src/sglang/types.ts`**：删除 `CacheManager` 占位 interface（第 224-230 行），改为 `export { CacheManager } from "./cache"`（class 引用）
4. **修改 `server/src/sglang/index.ts`**：将 `CacheManager` 从 `./types` 的 re-export 移到 `./cache` 的 re-export
5. **新建 `server/src/test/sglang-k3.test.ts`**：K3 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | CacheManager 构造（naive backend） | numPages/pageSize/freeSlots/prefixCache 正确初始化 |
| T2 | CacheManager.availableSize | 初始 = numPages × pageSize（naive 无 evictable） |
| T3 | CacheManager.matchReq | naive backend 总是返回 MatchResult(cudaHandle: NaiveCacheHandle(0)) |
| T4 | CacheManager.matchReq → lock → unlock 完整链路 | matchReq 结果可直接传递给 lock/unlock，不抛错 |
| T5 | CacheManager.lock/unlock | 调用不抛错（naive 为 noop） |
| T6 | CacheManager.allocatePaged 基本分配 | 1 页写入 pageTable 正确 |
| T7 | CacheManager.allocatePaged 多页分配 | 3 页，pageTable 写入正确 |
| T8 | CacheManager.allocatePaged 不足触发 eviction | naive eviction 返回空，不足时 break |
| T9 | CacheManager.cacheReq（finished=true） | 释放所有页，freeSlots 恢复 |
| T10 | CacheManager.cacheReq（finished=false） | 保留页，handle 更新并 lock |
| T11 | CacheManager._free 页对齐 Set 去重（修正版） | pageSize=4 时非页对齐子切片正确去重释放 |
| T12 | CacheManager._free 多页重复释放验证 | 同一物理页在子切片中出现多次时只释放一次 |
| T13 | CacheManager.beginLazyFree/endLazyFree | lazy 模式下 free 收集到 lazyFreeList，end 时合并到 freeSlots |
| T14 | CacheManager.checkIntegrity | naive backend 初始状态通过校验 |
| T15 | CacheManager.checkIntegrity 失败 | 手动修改 freeSlots 后抛错 |
| T16 | CacheManager.freeCache | 语义等同 cacheReq(finished=true) |
| T17 | 短 prompt 测试 | inputIds.length < pageSize |
| T18 | 中 prompt 测试 | inputIds.length = 3 × pageSize |
| T19 | 长 prompt 测试 | inputIds.length = 10 × pageSize |
| T20 | matched < computed 场景 | 模拟 oldHandle.cachedLen < insertResult.cachedLen |
| T21 | prefix 增长场景 | 连续 cacheReq 后 availableSize 单调递减 |
| T22 | lazyFreeRegion 正确计数 | lazy 模式下多次 _free 后 freeSlots 增量正确 |
| T23 | availableSize 不越界 | availableSize ∈ [0, numPages × pageSize] |
| T24 | _free Set 去重后的页数守恒 | 多次非对齐 _free 后 checkIntegrity 仍通过 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `CacheManager(numPages=0)` | freeSlots 为空，availableSize=0 |
| B2 | `CacheManager(numPages=1, pageSize=1)` | 最小配置，单页分配/回收 |
| B3 | `cacheReq` 时 `alignedLen=0` | 无页操作，不抛错 |
| B4 | `allocatePaged` 时 `neededPages=0` | 无分配，不抛错 |
| B5 | `endLazyFree` 在无 `beginLazyFree` 时调用 | `_inLazyFree=false`，直接合并空列表（防御性） |
| B6 | `pageSize=16` 的页对齐 `_free` Set 去重 | 正确去重释放 |
| B7 | `matchReq` 时 `inputLen=1` | 匹配空前缀（slice(0, 0)），返回 NaiveCacheHandle(0) |
| B8 | 连续 allocatePaged 全部页耗尽 | freeSlots=[], 再分配 break 不抛错 |
| B9 | `_free` 传入空数组 | 直接 return，不修改 freeSlots |

## 风险与注意事项

- **兼容性影响**：`types.ts` 中 `CacheManager` 从 `interface` 变为 class 引用，需确保所有使用处兼容。当前仅有 `SgSimContext.cacheMgr` 使用此类型（类型为 `CacheManager | null`），class 引用完全兼容。`index.ts` 的 re-export 来源需要从 `./types` 迁移到 `./cache`。
- **性能影响**：`freeSlots.pop()` / `freeSlots.push()` 是 O(1) 操作；`_free` 中 `Set` 去重在典型场景下元素极少（≤ 数十页），创建开销可忽略。`lazyFreeList` 收集后一次性合并替代多次合并，减少开销。页数守恒检查在 `checkIntegrity` 中显式调用，不影响热路径性能。
- **回滚方案**：所有改动在 `issue-14` 分支，合并前可安全回滚。
- **依赖关系**：Issue #13 (K2: MockKVCachePool + NaivePrefixCache) 必须已完成并合并。K3 依赖 K1 的抽象类型定义和 K2 的具体实现。
- **阻塞关系**：本 Issue 完成后，S2 (PrefillManager + DecodeManager) 和 S3 (MockEngine/SimScheduler) 才能集成 CacheManager 的完整 API。
- **naive vs radix 差异**：本阶段仅实现 naive backend（NaivePrefixCache），radix backend（RadixPrefixCache）将在 K4 Issue 中实现。CacheManager 构造函数预留了 `cacheType` 参数，K4 时扩展即可。
- **`_free` 算法差异**：本方案采用 `Set` 去重替代 §9.11 的 `[::page_size]` 步长切片，在功能上更鲁棒（防御非页对齐子切片），且在生产场景下行为一致（因为 `oldHandle.cachedLen` 和 `cachedLen` 始终页对齐）。如后续需严格对齐 Python 行为，可改为步长切片 + 前置断言保证页对齐。
