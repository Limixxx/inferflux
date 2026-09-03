// radix_cache — K4: RadixPrefixCache + RadixTreeNode + RadixCacheHandle (§9.8 / §3.3.5)

import {
  BaseCacheHandle,
  BasePrefixCache,
  CacheSizeInfo,
  MatchResult,
  InsertResult,
} from "./base";
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
  /** 缓存在 parent.children Map 中的 key，用于高效删除 */
  private _mapKey: number | string | null = null;

  constructor(keyFn: KeyFn, timestamp: number = 0) {
    this.keyFn = keyFn;
    this.timestamp = timestamp;
  }

  get length(): number { return this._key.length; }
  get value(): number[] { return this._value; }

  isRoot(): boolean { return this.parent === null; }
  isLeaf(): boolean { return this.children.size === 0; }

  /** 供排序按 timestamp 比较（LRU 驱逐） */
  valueOf(): number { return this.timestamp; }

  setKeyValue(key: number[], value: number[]): void {
    this._key = [...key];
    this._value = [...value];
    // Invalidate cached mapKey since _key changed; setParent will recompute it
    this._mapKey = null;
  }

  setParent(parent: RadixTreeNode | null): void {
    if (this.parent !== null && this._mapKey !== null) {
      this.parent.children.delete(this._mapKey);
    }
    this.parent = parent;
    if (parent !== null) {
      this._mapKey = this.keyFn(this._key);
      parent.children.set(this._mapKey, this);
    } else {
      this._mapKey = null;
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

  /** 获取缓存的 mapKey（用于 evict 删除） */
  get mapKey(): number | string | null { return this._mapKey; }
}

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
    if (size <= 0) return [];

    const evicted: number[] = [];
    let evictedSize = 0;
    const leafNodes = this._collectLeafNodesForEvict();
    // 按时间戳排序（最小的最早），构建最小堆效果
    leafNodes.sort((a, b) => a.timestamp - b.timestamp);

    let idx = 0;
    while (idx < leafNodes.length && evictedSize < size) {
      const node = leafNodes[idx++];
      if (node.refCount > 0) continue; // 被锁定，跳过
      if (node.parent === null) continue; // root，跳过

      // 从树中移除节点
      const parent = node.parent;
      const mapKey = node.mapKey;
      if (mapKey !== null) {
        parent.children.delete(mapKey);
      }
      node.parent = null;
      evicted.push(...node.value);
      evictedSize += node.length;
      this._sizeInfo.evictableSize -= node.length;

      // 合并：如果父节点变成叶子且 refCount==0，加入候选
      if (!parent.isRoot() && parent.isLeaf() && parent.refCount === 0) {
        leafNodes.push(parent);
        leafNodes.sort((a, b) => a.timestamp - b.timestamp);
      }
    }
    return evicted;
  }

  /** 重置缓存（§9.8 reset） */
  reset(): void {
    this._rootNode = new RadixTreeNode(this._keyFn);
    this._rootNode.refCount = 1;
    this._sizeInfo = new CacheSizeInfo(0, 0);
    this._timestampCounter = 0;
  }

  /** RadixTree 内部完整性校验 */
  checkIntegrity(): void {
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
