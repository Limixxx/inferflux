---
title: "Issue #13 解决方案"
issue_number: 13
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #13 解决方案

## 需求分析

- **问题描述**：Issue #13 要求实现 SGLang 仿真器的 K2 阶段，包括 `MockKVCachePool`（继承 `BaseKVCachePool`）和 `NaivePrefixCache`（继承 `BasePrefixCache`）两个具体实现类。前者是 KV cache 存储池的模拟实现（只做内存记账，不存真实 tensor），后者是 Phase 1 无前缀匹配的基线缓存实现。

- **能力目标**：
  1. `MockKVCachePool` 实现完整页管理：`cache_per_page` 常量公式、`_calcTotalPages` 初始化 free_pages_pool、`allocatePaged`/`deallocatePageAllocation` 分配回收、`decodeStepLatency` 延迟模型
  2. `NaivePrefixCache` 实现 `BasePrefixCache` 的所有抽象方法：`matchPrefix` 总是 miss（cachedLen=0）、`insertPrefix` 返回空 handle、`lockHandle`/`evict`/`reset`/`checkIntegrity` 为 noop 或空实现
  3. `NaiveCacheHandle` 实现 `BaseCacheHandle`：`cachedLen` 固定为 0，`getMatchedIndices` 返回空数组
  4. `PageAllocation` 数据结构：封装分配结果（pages + slots + slotCount）
  5. 单元测试覆盖：allocate/deallocate 循环、页数守恒不变式、naive 全 miss 不变式

- **影响范围**：仅修改 `server/src/sglang/cache/` 目录（新增 `mha_pool.ts` 和 `naive_cache.ts`，更新 `index.ts` re-export），以及 `server/src/sglang/index.ts`（更新顶层导出）和 `server/src/test/sglang-k2.test.ts`（新增测试文件）。

- **Issue 与设计文档差异分析**：

  | Issue 描述 | 设计文档 §9.3b / §3.4.2 | 决策 |
  |-----------|----------------------|------|
  | `MockKVCachePool extends BaseKVCachePool` | §3.4.2 `MockKVCachePool(BaseKVCachePool)` | 采用 §3.4.2 规格，增加 Issue 要求的 `cache_per_page`/`allocatePaged`/`deallocatePageAllocation`/`decodeStepLatency` |
  | `cache_per_page = kv_heads × v_head_size × 2 × page_size × bytes_per_element(dtype)` | §3.3.9 `cache_per_page = 2 × head_dim × kvHeadsPerGpu × page_size × dtypeSize × numLayers` | Issue 公式遗漏 `numLayers` 因子；采用 §3.3.9 完整公式（`v_head_size` 即 `headDim`，`kv_heads` 即 `kvHeadsPerGpu`） |
  | `_calc_total_pages`：按 num_pages 初始化 free_pages_pool | §3.4.2 构造函数仅存储 `_num_pages`，无 free_pages_pool | Issue 要求的 `free_pages_pool` 实际属于 `CacheManager`（§3.3.4）而非 KVCachePool；但为满足 Issue 需求，在 `MockKVCachePool` 中实现 `freePagesPool` 用于页级别记账（与 `CacheManager.free_slots` 互补，后者是 token 级别） |
  | `max_prefill_pages/max_extend_pages/max_append_tokens` 都用 num_pages × page_size | 无此属性 | 这些属于 `SimulatorConfig` 级别约束，不应在 KVCachePool 上暴露；不实现 |
  | `allocate_paged(req, needed_pages)`：分配 PageAllocation | §3.3.4 CacheManager.allocate_paged | Issue 将 `allocatePaged` 放在 `MockKVCachePool` 上；本方案在 `MockKVCachePool` 上实现简化版（仅页分配/回收记账），完整的 `allocate_paged`（写入 page_table）由 K3 的 `CacheManager` 实现 |
  | `decode_step_latency(batch)`：num_decode_tokens × token_decode_cost + CUDA graph overhead | 无此方法 | Issue 要求的延迟模型方法；在 `MockKVCachePool` 上实现，返回 decode 阶段的 tick 开销 |

## 改造方案

### 总体思路

K2 是 K1 抽象层的第一个具体实现阶段，核心工作：

1. **新建 `cache/mha_pool.ts`**：实现 `MockKVCachePool` 类和 `PageAllocation` 数据结构
2. **新建 `cache/naive_cache.ts`**：实现 `NaivePrefixCache` 和 `NaiveCacheHandle` 类
3. **更新 `cache/index.ts`**：re-export 新增类型
4. **更新 `index.ts`**：顶层导出新增类型
5. **新建测试文件**：`server/src/test/sglang-k2.test.ts`

### 详细设计

#### 1. PageAllocation 数据结构（cache/mha_pool.ts）

```typescript
/** 页分配结果，封装分配的页索引、槽位和总数 */
export class PageAllocation {
  /** 分配的页索引列表（每页起始 token 位置） */
  readonly pages: number[];
  /** 分配的槽位列表（展开后的 token 级别位置） */
  readonly slots: number[];
  /** 分配的槽位总数 */
  readonly slotCount: number;

  constructor(pages: number[], slots: number[], slotCount: number) {
    this.pages = pages;
    this.slots = slots;
    this.slotCount = slotCount;
  }
}
```

- `pages`：页索引列表，元素为每页起始位置（如 `[0, 16, 32]` 表示 3 个 pageSize=16 的页）
- `slots`：展开后的 token 级别位置列表（如 pageSize=16 时 `[0,1,...,15, 16,17,...,31, 32,33,...,47]`）
- `slotCount`：等于 `pages.length × pageSize`
- 使用 class + readonly 保持不可变语义

#### 2. MockKVCachePool 类（cache/mha_pool.ts）

```typescript
import type { ModelConfig, SimulatorConfig } from "../types";
import { BaseKVCachePool } from "./index";
import { PageAllocation } from "./mha_pool";
import { divEven } from "../core";

export class MockKVCachePool extends BaseKVCachePool {
  private readonly _numPages: number;
  private readonly _pageSize: number;
  private readonly _numLayers: number;
  private readonly _headDim: number;
  private readonly _numKvHeads: number;
  private readonly _cachePerPage: number;
  private _usedPages: number = 0;

  /** 空闲页池：存储每页的起始 token 位置 */
  freePagesPool: number[];

  constructor(modelConfig: ModelConfig, numPages: number, pageSize: number,
              config?: SimulatorConfig) {
    super();
    this._numPages = numPages;
    this._pageSize = pageSize;
    this._numLayers = modelConfig.numLayers;
    this._headDim = modelConfig.headDim;
    this._numKvHeads = this._calcKvHeadsPerGpu(modelConfig, config);
    this._cachePerPage = this._calcCachePerPage(modelConfig, config);

    // 初始化 free_pages_pool：每页起始位置 = page_index * pageSize
    this.freePagesPool = Array.from(
      { length: numPages }, (_, i) => i * pageSize
    );
  }

  // ===== BaseKVCachePool 抽象属性实现 =====
  get numPages(): number { return this._numPages; }
  get pageSize(): number { return this._pageSize; }
  get totalCapacity(): number { return this._numPages * this._pageSize; }
  get usedCapacity(): number { return this._usedPages * this._pageSize; }

  // ===== 额外属性 =====
  get cachePerPage(): number { return this._cachePerPage; }
  get usedPages(): number { return this._usedPages; }
  get freePages(): number { return this.freePagesPool.length; }

  // ===== BaseKVCachePool 抽象方法实现 =====
  storeKV(_k: number[], _v: number[], _outLoc: number[], _layerId: number): void {
    // 仿真中为 noop，不存储真实数据
  }

  // ===== 页分配/回收 =====

  /** 计算 GPU 上的 KV head 数（含 TP 分布） */
  private _calcKvHeadsPerGpu(modelConfig: ModelConfig,
                              config?: SimulatorConfig): number {
    const tpSize = config?.tpSize ?? 1;
    return divEven(modelConfig.numKvHeads, tpSize, true)
      .reduce((sum, v) => sum + v, 0);
  }

  /** 计算 cache_per_page 常量（§3.3.9） */
  private _calcCachePerPage(modelConfig: ModelConfig,
                             config?: SimulatorConfig): number {
    const dtypeSize = config?.dtypeSize ?? 2;
    return 2 *                    // key + value
           modelConfig.headDim *
           this._numKvHeads *
           this._pageSize *
           dtypeSize *
           modelConfig.numLayers;
  }

  /** 分配指定数量的页，返回 PageAllocation */
  allocatePaged(neededPages: number): PageAllocation {
    if (neededPages > this.freePagesPool.length) {
      throw new Error(
        `MockKVCachePool: allocatePaged failed, needed=${neededPages}, ` +
        `available=${this.freePagesPool.length}`
      );
    }

    const pages: number[] = [];
    const slots: number[] = [];
    for (let i = 0; i < neededPages; i++) {
      const pageStart = this.freePagesPool.pop()!;
      pages.push(pageStart);
      for (let j = 0; j < this._pageSize; j++) {
        slots.push(pageStart + j);
      }
    }
    this._usedPages += neededPages;

    return new PageAllocation(pages, slots, neededPages * this._pageSize);
  }

  /** 回收已分配的 PageAllocation */
  deallocatePageAllocation(pageAlloc: PageAllocation): void {
    for (const pageStart of pageAlloc.pages) {
      this.freePagesPool.push(pageStart);
    }
    this._usedPages -= pageAlloc.pages.length;
  }

  // ===== 延迟模型 =====

  /** 计算 decode 步骤的延迟（ticks） */
  decodeStepLatency(
    numDecodeTokens: number,
    tokenDecodeCost: number = 1,
    cudaGraphOverhead: number = 0
  ): number {
    return numDecodeTokens * tokenDecodeCost + cudaGraphOverhead;
  }
}
```

**设计要点**：
- `cachePerPage` 使用 §3.3.9 完整公式（包含 `numLayers` 因子），而非 Issue 中的简化公式
- `freePagesPool` 存储 token 级别起始位置（与 `CacheManager.free_slots` 一致），初始化为 `[0, pageSize, 2*pageSize, ...]`
- `allocatePaged` 使用栈式分配（pop），与 `TableManager.allocate` 风格一致
- `deallocatePageAllocation` 回收页到 freePagesPool 末尾
- `decodeStepLatency` 为纯计算方法，接收 `numDecodeTokens` 而非 `Batch` 对象，避免对调度器数据结构的耦合
- `config` 参数为可选，支持无 config 场景下使用默认值（tpSize=1, dtypeSize=2）

#### 3. NaiveCacheHandle 类（cache/naive_cache.ts）

```typescript
import { BaseCacheHandle } from "./index";

/** 无前缀匹配的缓存句柄（§9.3b） */
export class NaiveCacheHandle extends BaseCacheHandle {
  readonly cachedLenValue: number;

  constructor(cachedLen: number = 0) {
    super();
    this.cachedLenValue = cachedLen;
  }

  get cachedLen(): number { return this.cachedLenValue; }
  getMatchedIndices(): number[] { return []; }
}
```

- `cachedLenValue` 为构造时固定的值（Naive 场景下始终为 0）
- `getMatchedIndices` 返回空数组——无前缀匹配
- 使用独立字段 `cachedLenValue` 而非将 `cachedLen` 同时作为字段和 getter，避免 TypeScript 抽象 getter 与字段名冲突

#### 4. NaivePrefixCache 类（cache/naive_cache.ts）

```typescript
import { BasePrefixCache } from "./index";
import { CacheSizeInfo } from "./index";
import { MatchResult } from "./index";
import { InsertResult } from "./index";
import { BaseCacheHandle } from "./index";
import { NaiveCacheHandle } from "./naive_cache";

export class NaivePrefixCache extends BasePrefixCache {
  private readonly _numPages: number;
  private readonly _pageSize: number;
  private readonly _sizeInfo: CacheSizeInfo;

  constructor(numPages: number, pageSize: number) {
    super();
    this._numPages = numPages;
    this._pageSize = pageSize;
    this._sizeInfo = new CacheSizeInfo(0, 0);
  }

  get sizeInfo(): CacheSizeInfo { return this._sizeInfo; }

  /** 总是 miss：返回 cachedLen=0 的空 handle */
  matchPrefix(_inputIds: number[]): MatchResult {
    return new MatchResult(new NaiveCacheHandle(0));
  }

  /** 不做树操作，返回空 handle（cachedLen=0） */
  insertPrefix(_inputIds: number[], _indices: number[]): InsertResult {
    return new InsertResult(0, new NaiveCacheHandle(0));
  }

  /** noop：NaiveCache 无引用计数 */
  lockHandle(_handle: BaseCacheHandle, _unlock?: boolean): void {}

  /** 无 evictable 内容 */
  evict(_size: number): number[] { return []; }

  /** 重置 size info */
  reset(): void {
    this._sizeInfo.evictableSize = 0;
    this._sizeInfo.protectedSize = 0;
  }

  /** 空实现 */
  checkIntegrity(): void {}
}
```

**设计要点**：
- 严格遵循 §9.3b 规格
- `matchPrefix` 始终返回 `NaiveCacheHandle(0)`——无前缀命中
- `insertPrefix` 返回 `InsertResult(0, NaiveCacheHandle(0))`——所有页在 `finished=True` 时通过 `CacheManager._free(pageIndices[0:])` 全部回收
- `evict` 返回空数组——NaiveCache 无 evictable 内容
- `sizeInfo` 始终为 `CacheSizeInfo(0, 0)`

#### 5. cache/index.ts 更新

在现有导出基础上新增：

```typescript
// K2: MockKVCachePool + NaivePrefixCache (§3.4.2 / §9.3b)
export { MockKVCachePool, PageAllocation } from "./mha_pool";
export { NaivePrefixCache, NaiveCacheHandle } from "./naive_cache";
```

#### 6. index.ts 更新

在现有导出基础上新增：

```typescript
// K2: MockKVCachePool + NaivePrefixCache
export { MockKVCachePool, PageAllocation } from "./cache";
export { NaivePrefixCache, NaiveCacheHandle } from "./cache";
```

### 修改点清单

1. **`server/src/sglang/cache/mha_pool.ts`**（新建）：实现 `PageAllocation` 和 `MockKVCachePool` 类
2. **`server/src/sglang/cache/naive_cache.ts`**（新建）：实现 `NaiveCacheHandle` 和 `NaivePrefixCache` 类
3. **`server/src/sglang/cache/index.ts`**（修改）：新增 `MockKVCachePool`、`PageAllocation`、`NaivePrefixCache`、`NaiveCacheHandle` 的 re-export
4. **`server/src/sglang/index.ts`**（修改）：新增顶层导出
5. **`server/src/test/sglang-k2.test.ts`**（新建）：K2 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | MockKVCachePool 构造 | numPages/pageSize/cachePerPage/freePagesPool 正确初始化 |
| T2 | MockKVCachePool.cachePerPage 公式 | 与 `calculateMemoryBudget` 中的 cachePerPage 计算结果一致 |
| T3 | MockKVCachePool.storeKV noop | 调用不抛错，不改变任何状态 |
| T4 | MockKVCachePool.allocatePaged 基本分配 | 分配 1 页，返回正确 PageAllocation（pages/slots/slotCount） |
| T5 | MockKVCachePool.allocatePaged 多页分配 | 分配 3 页，slotCount = 3 × pageSize |
| T6 | MockKVCachePool.allocatePaged 不足抛错 | needed > free 时抛出 Error |
| T7 | MockKVCachePool.deallocatePageAllocation | 回收后 freePages 增加，usedPages 减少 |
| T8 | MockKVCachePool allocate-deallocate 循环 | 完整分配→回收→再分配循环正常工作 |
| T9 | MockKVCachePool 页数守恒 | usedPages + freePages === numPages 在任意操作后成立 |
| T10 | MockKVCachePool.decodeStepLatency | numDecodeTokens × cost + graphOverhead 结果正确 |
| T11 | MockKVCachePool.usedCapacity/totalCapacity | usedCapacity = usedPages × pageSize，totalCapacity = numPages × pageSize |
| T12 | NaiveCacheHandle 构造 | cachedLen=0，getMatchedIndices=[] |
| T13 | NaivePrefixCache.matchPrefix 全 miss | 返回 MatchResult 的 handle.cachedLen=0 |
| T14 | NaivePrefixCache.insertPrefix | 返回 InsertResult(cachedLen=0, NaiveCacheHandle) |
| T15 | NaivePrefixCache.lockHandle noop | 调用不抛错 |
| T16 | NaivePrefixCache.evict 空返回 | 返回空数组 |
| T17 | NaivePrefixCache.sizeInfo | 始终 CacheSizeInfo(0, 0) |
| T18 | NaivePrefixCache.reset | 调用不抛错，sizeInfo 保持 (0,0) |
| T19 | NaivePrefixCache.checkIntegrity | 调用不抛错 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `MockKVCachePool` 分配全部页后耗尽 | freePages=0，再分配抛错 |
| B2 | `MockKVCachePool` deallocate 后重新 allocate | 回收的页可被重新分配 |
| B3 | `MockKVCachePool(numPages=0)` | freePagesPool 为空，totalCapacity=0 |
| B4 | `MockKVCachePool(numPages=1)` | 只有 1 页，分配 1 页后耗尽 |
| B5 | `MockKVCachePool.allocatePaged(0)` | 返回空的 PageAllocation |
| B6 | `NaiveCacheHandle` 非 0 构造 | 允许但 NaivePrefixCache 总是传 0 |
| B7 | `NaivePrefixCache` 多次 match/insert | 每次都 miss，状态无变化 |
| B8 | `MockKVCachePool` pageSize=16 的 slot 展开 | slots 包含每个页内所有 token 位置 |
| B9 | `decodeStepLatency` numDecodeTokens=0 | 返回 cudaGraphOverhead |

## 风险与注意事项

- **兼容性影响**：新增文件不影响现有代码。K1 的抽象类型定义需要已经存在（Issue #11），否则 `MockKVCachePool` 和 `NaivePrefixCache` 无法继承。当前 `cache/index.ts` 缺少 K1 的抽象类导出，K2 实现时需同步补充 K1 的类型定义（此为 K1 Issue 的遗留问题，非 K2 范围内新增改动，但为保证编译通过必须一并修复）。
- **性能影响**：`freePagesPool` 使用数组栈（pop/push），O(1) 操作，无性能问题。`allocatePaged` 中 slots 展开为 O(neededPages × pageSize)，仿真场景下可接受。
- **回滚方案**：所有改动在 `issue-13` 分支，合并前可安全回滚。
- **依赖关系**：
  - Issue #11（K1 抽象层）和 Issue #12（K5 内存预算公式）必须已完成。K2 继承 K1 的抽象类，使用 K5 的 `calculateMemoryBudget` 验证 `cachePerPage` 计算。
  - K2 的 `MockKVCachePool.allocatePaged` 是简化版（仅页分配记账），完整的 `CacheManager.allocate_paged`（写入 page_table）将在 K3（Issue #14）实现。
- **阻塞关系**：本 Issue 完成后，K3（CacheManager naive 5 区域 cache_req）才能启动，因为 `CacheManager` 依赖 `MockKVCachePool` 和 `NaivePrefixCache`。
- **cachePerPage 公式差异**：Issue 描述中 `cache_per_page = kv_heads × v_head_size × 2 × page_size × bytes_per_element(dtype)` 遗漏了 `numLayers` 因子。SGLang 实际实现中每层都有独立的 K/V cache，因此每页大小必须乘以 `numLayers`。本方案采用 §3.3.9 的完整公式。
