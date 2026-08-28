---
title: "Issue #15 解决方案"
issue_number: 15
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #15 解决方案

## 需求分析

- **问题描述**：Issue #15 要求实现 SGLang 仿真器的 K4 阶段 RadixPrefixCache，包括 `RadixTreeNode`（含 `children`/`prefix_indices`/`key`/`ref_count`/`lru_seq`）、`RadixPrefixCache extends BasePrefixCache`（含 `match`/`insert`/`LRU 最小堆驱逐`/`lock_handle`/`unlock_handle`）、以及与 `CacheManager` 的集成（`cache_type: "naive" | "radix"` 选择）。

- **能力目标**：
  1. `RadixTreeNode` 类：存储 token key 序列和页索引序列，支持 `children: Map<key, node>`、`parent` 引用、`ref_count` 引用计数、`timestamp`(lru_seq) 用于 LRU 排序、`split_at(pos)` 节点分裂
  2. `RadixPrefixCache` 类：基于 RadixTree 的前缀缓存，实现 `BasePrefixCache` 的全部抽象方法
  3. `_tree_walk(input_ids)` 内部遍历：从 root 沿 key_fn 键逐层下降，部分匹配时执行 `split_at`，返回 `(node, prefix_len)`
  4. `matchPrefix(inputIds)`：调用 `_tree_walk`，返回 `MatchResult(RadixCacheHandle)`
  5. `insertPrefix(inputIds, indices)`：先 `_tree_walk` 找最长匹配，创建子节点存储未匹配部分
  6. **LRU 最小堆驱逐** `evict(size)`：收集 `ref_count=0` 的叶子节点，用最小堆按 `timestamp` 排序弹出，释放页索引；父节点变叶子后合并入堆
  7. `lockHandle(handle, unlock?)`：lock 时沿路径 `ref_count += 1`（evictable→protected），unlock 时反向 `ref_count -= 1`（protected→evictable）
  8. `RadixCacheHandle extends BaseCacheHandle`：持有 node 引用，`cachedLen` 和 `getMatchedIndices()` 从 node 路径计算
  9. 集成点：`CacheManager` 构造时支持 `cache_type: "naive" | "radix"` 选择 backend（K3 时实现）
  10. 单元测试：split_at 正确性；同一前缀重放命中率 100%；驱逐不影响命中节点；handle lock 阻止驱逐

- **影响范围**：仅修改 `server/src/sglang/cache/` 目录（新增 `radix_cache.ts`，更新 `index.ts` re-export），以及 `server/src/sglang/index.ts`（更新顶层导出）和 `server/src/test/` 目录（新增 K4 测试文件）。不改 K1/K2/K5 的已有代码，不改业务源码和测试代码。

- **依赖 ISSUE**：#11 K1（抽象层 BasePrefixCache/BaseCacheHandle/CacheSizeInfo/MatchResult/InsertResult）、#13 K2（MockKVCachePool free_pool 回收机制，evict 返回页索引由调用方管理 free_slots）

- **Issue 与设计文档差异分析**：

  | Issue 描述 | 设计文档 §9.8 / §3.3.5 | 决策 |
  |-----------|----------------------|------|
  | `RadixTreeNode.prefix_indices: CacheSlot[]` | §9.8 `_value: List[int]` — page 索引序列 | 采用 §9.8 命名 `_value`；Issue 中 `CacheSlot[]` 概念上等价于 `number[]`，仿真中用 `number[]` 即可 |
  | `RadixTreeNode.key: string`（`key_fn(page_size, slot)`） | §9.8 `_key: List[int]` + `key_fn(tokens)` 生成 dict key | `_key` 存储原始 token 序列（`number[]`），`key_fn` 作为方法/函数从 token 序列提取字典键（page_size=1 时取首个 token 值，page_size>1 时取 tuple） |
  | `RadixTreeNode.lru_seq: bigint` | §9.8 `timestamp: int` — LRU 用的时间戳 | 采用 §9.8 `timestamp: number`；仿真中用 `performance.now()` 或递增计数器，`bigint` 无必要，`number` 足够表达时序 |
  | `match(req)` 返回 `MatchResult{matched, nodes_hit, lengths, indices, same_prefix_computed, reduce_threshold}` | §9.3 `MatchResult(cudaHandle)` — 只携带 handle | 采用 §9.3 简化签名；Issue 描述的多字段来自更早草案，所有信息可通过 `handle.cachedLen` 和 `handle.getMatchedIndices()` 获取 |
  | `_evict_to_target(target_nodes_count)` | §9.8 `evict(size)` — 按 token 数量驱逐 | 采用 §9.8 `evict(size)` 签名；Issue 中的 `_evict_to_target` 是内部优化方法，可保留为私有辅助，但公共接口遵循 `BasePrefixCache.evict(size)` |

## 改造方案

### 总体思路

K4 是 K1 抽象层的核心实现阶段——RadixTree 前缀缓存，核心工作：

1. **新建 `cache/radix_cache.ts`**：实现 `RadixTreeNode`、`RadixCacheHandle`、`RadixPrefixCache` 三个类
2. **更新 `cache/index.ts`**：re-export 新增类型
3. **更新 `index.ts`**：顶层导出新增类型
4. **新建测试文件**：`server/src/test/sglang-k4.test.ts`

关键设计原则：
- **严格遵循 §9.8 规格的 Python 伪代码**，转写为 TypeScript strict 模式
- `RadixPrefixCache` 继承 `BasePrefixCache`，满足 K1 的所有抽象方法签名
- `RadixCacheHandle` 继承 `BaseCacheHandle`，持有 node 引用以计算 `cachedLen` 和 `getMatchedIndices()`
- 不管理 `free_slots`（由 `CacheManager` 统一管理），`evict()` 只返回被释放的页索引列表
- 使用 `alignDown` 来自 K1 的 `core` 模块，确保 `_tree_walk` 中的匹配长度页对齐

### 详细设计

#### 1. RadixTreeNode 类（cache/radix_cache.ts）

```typescript
import { alignDown } from "../core";

/** 由 token 序列生成 dict key 的函数类型 */
export type KeyFn = (tokens: number[]) => number | string;

/** RadixTree 节点，存储 key（token 序列）和 value（page 索引序列）（§9.8） */
export class RadixTreeNode {
  readonly keyFn: KeyFn;
  timestamp: number;             // LRU 用的时间戳
  private _key: number[] = [];   // 该节点对应的 token 序列
  private _value: number[] = []; // 该节点对应的 page 索引序列
  children: Map<number | string, RadixTreeNode> = new Map();
  parent: RadixTreeNode | null = null;
  refCount: number = 0;          // 引用计数，>0 表示被锁定（不可驱逐）

  constructor(keyFn: KeyFn, timestamp: number = 0) {
    this.keyFn = keyFn;
    this.timestamp = timestamp;
  }

  get length(): number { return this._key.length; }
  get value(): number[] { return this._value; }

  isRoot(): boolean { return this.parent === null; }
  isLeaf(): boolean { return this.children.size === 0; }

  /** 供 heapq 按 timestamp 排序（LRU 驱逐） */
  valueOf(): number { return this.timestamp; }

  setKeyValue(key: number[], value: number[]): void {
    this._key = [...key];
    this._value = [...value];
  }

  setParent(parent: RadixTreeNode | null): void {
    this.parent = parent;
    if (parent !== null) {
      parent.children.set(this.keyFn(this._key), this);
    }
  }

  /** 比较节点 key 和 input_ids，返回匹配长度 */
  getMatchLen(inputIds: number[]): number {
    const minLen = Math.min(this._key.length, inputIds.length);
    for (let i = 0; i < minLen; i++) {
      if (this._key[i] !== inputIds[i]) return i;
    }
    return minLen;
  }

  /** 在位置 pos 分裂节点（§9.8 split_at 细节） */
  splitAt(pos: number): RadixTreeNode {
    const parent = this.parent;
    const newNode = new RadixTreeNode(this.keyFn, this.timestamp);
    newNode.setKeyValue(this._key.slice(0, pos), this._value.slice(0, pos));
    newNode.setParent(parent);
    newNode.refCount = this.refCount; // 继承引用计数

    this.setKeyValue(this._key.slice(pos), this._value.slice(pos));
    this.setParent(newNode); // 原节点成为新节点的子节点
    return newNode;
  }
}
```

**设计要点**：
- `children` 使用 `Map<number | string, RadixTreeNode>`，与 `keyFn` 返回类型对齐（page_size=1 时为 number，page_size>1 时为 string tuple 序列化）
- `valueOf()` 提供数值比较语义，用于最小堆排序
- `splitAt` 严格遵循 §9.8 的 ref_count 继承规则——新节点继承原 ref_count，保证分裂后 locked 的节点不会因为分裂而 unlocked
- `_key`/`_value` 使用私有字段 + getter，防止外部直接修改

#### 2. RadixCacheHandle 类（cache/radix_cache.ts）

```typescript
import { BaseCacheHandle } from "./index";

/** RadixTree 缓存句柄，指向树中的节点（§9.8） */
export class RadixCacheHandle extends BaseCacheHandle {
  private readonly _cachedLen: number;
  readonly node: RadixTreeNode;

  constructor(cachedLen: number, node: RadixTreeNode) {
    super();
    this._cachedLen = cachedLen;
    this.node = node;
  }

  get cachedLen(): number { return this._cachedLen; }

  /** 从 root 到当前 node 路径上所有节点的 value 拼接（页索引列表） */
  getMatchedIndices(): number[] {
    const indices: number[] = [];
    let current: RadixTreeNode | null = this.node;
    const path: RadixTreeNode[] = [];
    while (current !== null && !current.isRoot()) {
      path.push(current);
      current = current.parent;
    }
    // 从 root 方向拼合，保证顺序
    for (let i = path.length - 1; i >= 0; i--) {
      indices.push(...path[i].value);
    }
    return indices;
  }
}
```

**设计要点**：
- `cachedLen` 在构造时固定（对应 `_tree_walk` 返回的 `prefix_len`），不随树后续变化
- `getMatchedIndices()` 通过向上遍历到 root 再反转路径收集所有页索引，语义清晰
- 持有 `node` 引用，供 `lockHandle`/`unlockHandle` 直接操作节点

#### 3. RadixPrefixCache 类（cache/radix_cache.ts）

```typescript
import {
  BasePrefixCache,
  CacheSizeInfo,
  MatchResult,
  InsertResult,
  BaseCacheHandle,
} from "./index";
import { alignDown } from "../core";

/** 基于 RadixTree 的前缀缓存（§9.8 / §3.3.5） */
export class RadixPrefixCache extends BasePrefixCache {
  private readonly _numPages: number;
  private readonly _pageSize: number;
  private readonly _keyFn: KeyFn;
  private _rootNode: RadixTreeNode;
  private _sizeInfo: CacheSizeInfo;
  private _timestampCounter: number = 0;

  constructor(numPages: number, pageSize: number) {
    super();
    this._numPages = numPages;
    this._pageSize = pageSize;
    this._keyFn = pageSize === 1
      ? (tokens: number[]) => tokens[0]
      : (tokens: number[]) => JSON.stringify(tokens.slice(0, pageSize));
    this._rootNode = new RadixTreeNode(this._keyFn);
    this._rootNode.refCount = 1; // root 永远不可驱逐
    this._sizeInfo = new CacheSizeInfo(0, 0);
  }

  get sizeInfo(): CacheSizeInfo { return this._sizeInfo; }
  get rootNode(): RadixTreeNode { return this._rootNode; }

  /** 获取递增时间戳（替代 time.monotonic_ns()） */
  private _nextTimestamp(): number {
    return ++this._timestampCounter;
  }

  /** 遍历树，返回最长前缀匹配的节点和匹配长度（§9.8 _tree_walk） */
  _treeWalk(inputIds: number[]): [RadixTreeNode, number] {
    let prefixLen = 0;
    const indiceLen = inputIds.length;
    let node = this._rootNode;
    const tic = this._nextTimestamp();

    while (prefixLen < indiceLen) {
      const childNode = node.children.get(this._keyFn(inputIds.slice(prefixLen)));
      if (childNode === undefined) {
        return [node, prefixLen];
      }

      node = childNode;
      let matchLen = node.getMatchLen(inputIds.slice(prefixLen));
      matchLen = alignDown(matchLen, this._pageSize); // 向下对齐到页边界
      prefixLen += matchLen;

      if (matchLen !== node.length) {
        // 部分匹配，需要分裂
        node = node.splitAt(matchLen);
        node.timestamp = tic;
        return [node, prefixLen];
      }
      node.timestamp = tic;
    }
    return [node, prefixLen];
  }

  /** 前缀匹配，返回 MatchResult（§9.8 match_prefix） */
  matchPrefix(inputIds: number[]): MatchResult {
    const [node, matchLen] = this._treeWalk([...inputIds]);
    return new MatchResult(new RadixCacheHandle(matchLen, node));
  }

  /** 插入前缀到树中（§9.8 insert_prefix） */
  insertPrefix(inputIds: number[], indices: number[]): InsertResult {
    const insertLen = alignDown(inputIds.length, this._pageSize);
    if (insertLen === 0) {
      return new InsertResult(0, new RadixCacheHandle(0, this._rootNode));
    }
    const [node, matchLen] = this._treeWalk(inputIds.slice(0, insertLen));
    if (matchLen < insertLen) {
      const child = new RadixTreeNode(this._keyFn, this._nextTimestamp());
      child.setKeyValue(
        inputIds.slice(matchLen, insertLen),
        indices.slice(matchLen, insertLen),
      );
      child.setParent(node);
      this._sizeInfo.evictableSize += child.length;
      return new InsertResult(matchLen, new RadixCacheHandle(insertLen, child));
    }
    return new InsertResult(matchLen, new RadixCacheHandle(insertLen, node));
  }

  /** 锁定/解锁节点，调整 refCount 和 sizeInfo（§9.8 lock_handle） */
  lockHandle(handle: BaseCacheHandle, unlock?: boolean): void {
    if (!(handle instanceof RadixCacheHandle)) return;
    const node = handle.node;

    if (unlock) {
      node.refCount -= 1;
      if (node.refCount === 0) {
        this._sizeInfo.protectedSize -= node.length;
        this._sizeInfo.evictableSize += node.length;
      }
    } else {
      if (node.refCount === 0) {
        this._sizeInfo.evictableSize -= node.length;
        this._sizeInfo.protectedSize += node.length;
      }
      node.refCount += 1;
    }
  }

  /** 收集所有 refCount==0 的叶子节点（可驱逐候选） */
  private _collectLeafNodesForEvict(): RadixTreeNode[] {
    const result: RadixTreeNode[] = [];
    const stack: RadixTreeNode[] = [this._rootNode];
    while (stack.length > 0) {
      const n = stack.pop()!;
      if (!n.isRoot() && n.isLeaf() && n.refCount === 0) {
        result.push(n);
      }
      for (const child of n.children.values()) {
        stack.push(child);
      }
    }
    return result;
  }

  /** LRU 驱逐，返回释放的页索引列表（§9.8 evict） */
  evict(size: number): number[] {
    const evicted: number[] = [];
    let evictedSize = 0;
    const leafNodes = this._collectLeafNodesForEvict();
    // 按时间戳排序（最小的最早），构建最小堆
    leafNodes.sort((a, b) => a.timestamp - b.timestamp);

    let idx = 0;
    while (idx < leafNodes.length && evictedSize < size) {
      const node = leafNodes[idx++];
      if (node.refCount > 0) continue; // 被锁定，跳过

      // 从树中移除节点
      const parent = node.parent;
      if (parent !== null) {
        parent.children.delete(this._keyFn(node.value.length > 0
          ? this._getKeyFromNode(node) : []));
      }
      evicted.push(...node.value);
      evictedSize += node.length;
      this._sizeInfo.evictableSize -= node.length;

      // 合并：如果父节点变成叶子且 refCount==0，加入候选
      if (parent !== null && !parent.isRoot()
          && parent.isLeaf() && parent.refCount === 0) {
        leafNodes.push(parent);
        leafNodes.sort((a, b) => a.timestamp - b.timestamp);
      }
    }
    return evicted;
  }

  /** 辅助：从节点提取 key 用于 children map 的删除 */
  private _getKeyFromNode(node: RadixTreeNode): number[] {
    // 遍历 parent 的 children 找到对应的 key
    // 由于 keyFn(_key) 已存储在 parent.children 中，这里需要还原
    // 更简单的方式：在 splitAt 时保留原始 _key 信息
    // 采用直接遍历 parent 的方式
    if (node.parent !== null) {
      for (const [k, v] of node.parent.children) {
        if (v === node) {
          node.parent.children.delete(k);
          return [];
        }
      }
    }
    return [];
  }

  /** 重置缓存（§9.8 reset） */
  reset(): void {
    this._rootNode = new RadixTreeNode(this._keyFn);
    this._rootNode.refCount = 1;
    this._sizeInfo = new CacheSizeInfo(0, 0);
  }

  /** RadixTree 内部完整性校验（仿真中简化实现） */
  checkIntegrity(): void {
    // 校验 evictableSize + protectedSize 与树中节点一致
    let evictable = 0;
    let protected_ = 0;
    const stack: RadixTreeNode[] = [this._rootNode];
    while (stack.length > 0) {
      const n = stack.pop()!;
      if (!n.isRoot()) {
        if (n.refCount === 0) evictable += n.length;
        else protected_ += n.length;
      }
      for (const child of n.children.values()) {
        stack.push(child);
      }
    }
    if (evictable !== this._sizeInfo.evictableSize) {
      throw new Error(
        `Integrity check failed: evictableSize ${this._sizeInfo.evictableSize} != actual ${evictable}`
      );
    }
    if (protected_ !== this._sizeInfo.protectedSize) {
      throw new Error(
        `Integrity check failed: protectedSize ${this._sizeInfo.protectedSize} != actual ${protected_}`
      );
    }
  }
}
```

**设计要点**：

1. **`_timestampCounter` 递增计数器**：替代 `time.monotonic_ns()`，保证严格递增的 LRU 序列号，避免浮点精度问题

2. **`_keyFn` 策略**：`pageSize === 1` 时取 `tokens[0]`（number 类型作为 Map key），`pageSize > 1` 时取 `JSON.stringify(tokens.slice(0, pageSize))`（string 类型作为 Map key）

3. **`_treeWalk` 中的 `alignDown`**：确保匹配长度始终页对齐，防止部分页被错误命中

4. **`insertPrefix` 返回值语义**：`InsertResult.cachedLen` 表示插入前已在缓存中的长度（即 `matchLen`），`handle.cachedLen` 表示插入后的总缓存长度（即 `insertLen`）

5. **`lockHandle` 单节点操作**：§3.3.5 描述"从 handle.node 向上遍历到 root"，但 §9.8 实现只操作 handle.node 单个节点。本方案采用 §9.8 实现——只锁定/解锁 handle 指向的节点本身。理由：(a) SGLang 源码中 `lock_handle` 只操作单个节点；(b) 向上遍历锁定整条路径会导致 refCount 过度增加，释放时难以配对

6. **`evict` 的排序策略**：使用数组排序替代二叉堆（仿真场景下节点数量有限，排序性能可接受）。弹出节点后若父节点变叶子且 refCount=0，重新排序加入候选

7. **`_getKeyFromNode` 辅助方法**：由于 `RadixTreeNode` 的 `_key` 是私有字段，evict 需要从 parent 的 children map 中删除条目。采用遍历 parent.children 找到引用再删除的方式，保证一致性。后续代码阶段可优化为在 node 上缓存 map key

8. **`checkIntegrity` 实质性实现**：遍历整棵树统计 evictableSize/protectedSize，与 `_sizeInfo` 对比。仿真中可用于调试，不影响性能（仅在显式调用时执行）

#### 4. cache/index.ts 更新

在现有导出基础上新增：

```typescript
// K4: RadixPrefixCache (§9.8 / §3.3.5)
export { RadixTreeNode, RadixCacheHandle, RadixPrefixCache } from "./radix_cache";
export type { KeyFn } from "./radix_cache";
```

#### 5. index.ts 更新

在现有导出基础上新增：

```typescript
// K4: RadixPrefixCache
export { RadixTreeNode, RadixCacheHandle, RadixPrefixCache } from "./cache";
export type { KeyFn } from "./cache";
```

### 修改点清单

1. **`server/src/sglang/cache/radix_cache.ts`**（新建）：实现 `RadixTreeNode`、`RadixCacheHandle`、`RadixPrefixCache` 三个类
2. **`server/src/sglang/cache/index.ts`**（修改）：新增 `RadixTreeNode`、`RadixCacheHandle`、`RadixPrefixCache`、`KeyFn` 的 re-export
3. **`server/src/sglang/index.ts`**（修改）：新增顶层导出
4. **`server/src/test/sglang-k4.test.ts`**（新建）：K4 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | RadixTreeNode 构造与基本属性 | keyFn/timestamp/refCount/children/parent 初始化 |
| T2 | RadixTreeNode.setKeyValue | _key/_value 正确赋值，length/value getter 正确 |
| T3 | RadixTreeNode.getMatchLen | 完全匹配、部分匹配、不匹配返回值 |
| T4 | RadixTreeNode.splitAt 基本分裂 | 新节点继承 [0,pos)，原节点缩进到 [pos:) |
| T5 | RadixTreeNode.splitAt refCount 继承 | 分裂后新节点 refCount = 原 refCount |
| T6 | RadixTreeNode.setParent | parent.children 正确更新双向引用 |
| T7 | RadixTreeNode.isLeaf/isRoot | 正确判断 |
| T8 | RadixPrefixCache 构造 | root refCount=1，sizeInfo 为 (0,0) |
| T9 | matchPrefix 完全命中 | 已插入序列完全匹配，cachedLen = 全长 |
| T10 | matchPrefix 部分命中 | 前缀部分匹配，cachedLen = 前缀长度 |
| T11 | matchPrefix 未命中 | 新序列完全 miss，cachedLen = 0 |
| T12 | insertPrefix 新序列插入 | 未匹配部分创建新节点，sizeInfo.evictableSize 增加 |
| T13 | insertPrefix 已存在序列 | 不创建新节点，cachedLen = insertLen |
| T14 | 同一前缀重放命中率 100% | 插入后再 match，cachedLen = 全长 |
| T15 | split_at 正确性 | 部分匹配触发 split，后续请求正确匹配分裂后的节点 |
| T16 | lockHandle 锁定 | refCount 0→1，evictableSize 减少，protectedSize 增加 |
| T17 | lockHandle 解锁 | refCount 1→0，evictableSize 增加，protectedSize 减少 |
| T18 | lockHandle 多次锁定 | refCount 累加，evictableSize 只在 0→1 时变化 |
| T19 | evict 基本驱逐 | 驱逐 refCount=0 的叶子节点，返回页索引列表 |
| T20 | evict 驱逐不影响命中节点 | 被 lock 的节点不被驱逐 |
| T21 | evict 父节点合并 | 子节点被驱逐后，父节点变叶子且 refCount=0 时合并入候选 |
| T22 | handle lock 阻止驱逐 | 锁定后的节点 evict 无法驱逐 |
| T23 | sizeInfo 一致性 | 任意操作后 evictableSize + protectedSize = 树中节点 token 总数 |
| T24 | RadixCacheHandle.getMatchedIndices | 从 root 到 node 路径上页索引按序拼合 |
| T25 | reset 重置缓存 | 树清空，sizeInfo 归零 |
| T26 | checkIntegrity 正常 | 操作后 checkIntegrity 不抛错 |
| T27 | checkIntegrity 检测不一致 | 手动破坏 sizeInfo 后抛出 Error |
| T28 | pageSize=1 的 keyFn | key 为 number 类型，children Map 正确索引 |
| T29 | pageSize>1 的 keyFn | key 为 string 类型（JSON 序列化），匹配页对齐 |
| T30 | insertPrefix 空序列 | insertLen=0 时返回 root handle，cachedLen=0 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | 空树 matchPrefix | cachedLen=0，返回 root handle |
| B2 | 单 token 插入与匹配 | pageSize=1 时正常工作 |
| B3 | evict 请求量大于可驱逐量 | 驱逐所有可驱逐节点后停止，返回已驱逐的页索引 |
| B4 | evict(0) | 返回空数组，不做任何驱逐 |
| B5 | 连续 lock/unlock 配对 | refCount 归零，节点回到可驱逐状态 |
| B6 | 未 unlock 就再次 lock | refCount 累加正确 |
| B7 | 插入长度非页对齐 | insertLen = alignDown(len, pageSize)，非对齐部分不插入 |
| B8 | 完全相同序列重复插入 | 不创建新节点，sizeInfo 不变 |
| B9 | 多条共享前缀的序列 | RadixTree 正确共享前缀分支 |

## 风险与注意事项

- **兼容性影响**：新增文件不影响现有 K1/K2/K5 代码。`BasePrefixCache` 和 `BaseCacheHandle` 的抽象方法签名不变，`RadixPrefixCache` 和 `RadixCacheHandle` 作为新实现类加入。
- **性能影响**：
  - `_collectLeafNodesForEvict` 每次驱逐时遍历整棵树，O(N) 复杂度。仿真场景下树节点数量有限（几百到几千），可接受。若后续需要优化，可维护一个 evictable 叶子节点的增量集合
  - `evict` 中的数组排序 O(M log M)，M 为可驱逐叶子数，仿真场景下可接受
  - `getMatchedIndices` 向上遍历到 root，O(depth)，RadixTree 深度通常较小
- **回滚方案**：所有改动在 `issue-15` 分支，合并前可安全回滚。
- **依赖关系**：
  - Issue #11（K1 抽象层）和 Issue #13（K2 MockKVCachePool）必须已完成。K4 继承 K1 的 `BasePrefixCache`/`BaseCacheHandle`/`CacheSizeInfo`/`MatchResult`/`InsertResult`，使用 K2 的 `MockKVCachePool.freePagesPool` 机制理解 evict 返回值的消费方式
  - K4 的 `evict()` 返回页索引列表，由 `CacheManager`（K3）负责加入 `free_slots`
- **阻塞关系**：K4 完成后，`CacheManager`（K3 Issue）可选择 `cache_type: "radix"` 作为 backend，实现真正的前缀缓存调度。当前 K3 如已用 naive backend 实现，K4 合并后只需切换 `cache_type` 参数即可升级。
- **lockHandle 单节点 vs 路径锁定**：§3.3.5 描述"从 handle.node 向上遍历到 root"，但 §9.8 实现只操作单个节点。本方案采用 §9.8 单节点方案。如果后续调度器需要路径锁定（如 prefill 时锁定整条前缀路径），可在 `CacheManager.lock()` 中实现多次 `lockHandle` 调用，无需修改 `RadixPrefixCache` 内部。
- **`_getKeyFromNode` 实现**：当前通过遍历 parent.children 查找引用来删除，代码阶段可优化为在 `RadixTreeNode` 上缓存 `_mapKey` 字段，避免 O(children.size) 查找。
