// cache_manager — K3: CacheManager（naive backend）§9.11

import {
  BasePrefixCache,
  BaseCacheHandle,
  MatchResult,
  InsertResult,
} from "./base";
import { NaivePrefixCache } from "./naive_cache";
import { divCeil } from "../core";

/**
 * CacheManager — KV cache 页分配与前缀缓存管理（§9.11）
 *
 * 持有 prefixCache: BasePrefixCache，对外提供：
 * - cacheReq(req, finished): 5 区域精细划分的缓存请求处理
 * - freeCache(req): 语义等同 cacheReq(req, true)
 * - matchReq(req): 前缀匹配
 * - availableSize: 可用缓存大小
 * - allocatePaged(req): 页分配
 * - lock/unlock: 缓存句柄锁定
 * - beginLazyFree/endLazyFree: 延迟释放上下文管理
 * - checkIntegrity: 页数守恒不变式校验
 */
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

  /** 已分配页数（含缓存跟踪页 + 请求持有页） */
  private _allocatedPages: number;

  /** 前缀缓存实例（naive 或 radix） */
  readonly prefixCache: BasePrefixCache;

  constructor(
    numPages: number,
    pageSize: number,
    pageTable: number[][],
    cacheType: "radix" | "naive" = "naive",
  ) {
    this.numPages = numPages;
    this.pageSize = pageSize;
    this.pageTable = pageTable;
    this.freeSlots = Array.from({ length: numPages }, (_, i) => i * pageSize);
    this.lazyFreeList = [];
    this._inLazyFree = false;
    this._allocatedPages = 0;

    if (cacheType === "radix") {
      throw new Error("RadixPrefixCache not implemented yet (K4)");
    } else {
      this.prefixCache = new NaivePrefixCache(numPages, pageSize);
    }
  }

  // ===== availableSize =====

  /** 可用缓存大小 = evictableSize + freeSlots.length * pageSize（§9.11） */
  get availableSize(): number {
    return this.prefixCache.sizeInfo.evictableSize + this.freeSlots.length * this.pageSize;
  }

  // ===== matchReq =====

  /**
   * 前缀匹配请求（§9.11）
   * 排除最后一个 token（SGLang 语义：最后一个 token 无 KV cache）
   * 返回 MatchResult，调用方可通过 lock(result.cudaHandle) 锁定缓存结果
   */
  matchReq(req: { inputIds: number[]; inputLen: number }): MatchResult {
    const inputLen = req.inputLen;
    if (inputLen <= 0) throw new Error("matchReq: inputLen must be > 0");
    return this.prefixCache.matchPrefix(req.inputIds.slice(0, inputLen - 1));
  }

  // ===== lock / unlock =====

  /** 锁定缓存句柄（naive backend 为 noop） */
  lock(handle: BaseCacheHandle): void {
    this.prefixCache.lockHandle(handle);
  }

  /** 解锁缓存句柄（naive backend 为 noop） */
  unlock(handle: BaseCacheHandle): void {
    this.prefixCache.lockHandle(handle, true);
  }

  // ===== allocatePaged =====

  /**
   * 按页分配 KV cache 空间（§9.11）
   * 将分配的页物理位置写入 pageTable[tableIdx]
   */
  allocatePaged(req: {
    deviceLen: number;
    cachedLen: number;
    tableIdx: number;
  }): void {
    const { deviceLen, cachedLen, tableIdx } = req;
    const lastPage = divCeil(deviceLen, this.pageSize) - divCeil(cachedLen, this.pageSize);
    const neededPages = Math.max(0, lastPage);

    if (neededPages > this.freeSlots.length) {
      // 触发 eviction：从 prefix cache 的 evictable 节点中回收页
      const evictSize = (neededPages - this.freeSlots.length) * this.pageSize;
      const evicted = this.prefixCache.evict(evictSize);
      this.freeSlots.push(...evicted);
    }

    // 分配页并写入 page_table
    // 边界情况：如果 eviction 后仍不足，循环 break，未分配位置保持原值
    for (let i = 0; i < neededPages; i++) {
      if (this.freeSlots.length === 0) break;
      const pageIdx = this.freeSlots.pop()!;
      this._allocatedPages++;
      const startPos = (divCeil(cachedLen, this.pageSize) + i) * this.pageSize;
      for (let j = 0; j < this.pageSize; j++) {
        const pos = startPos + j;
        if (pos < this.pageTable[tableIdx].length) {
          this.pageTable[tableIdx][pos] = pageIdx;
        }
      }
    }
  }

  // ===== cacheReq（核心：5 区域逻辑）=====

  /**
   * 缓存请求处理 — 5 区域精细划分（§9.11 / §5.3.3）
   *
   * 区域划分基于三个关键长度边界：
   *   oldHandle.cachedLen — 旧缓存句柄的已缓存长度
   *   insertResult.cachedLen — insertPrefix 前已在缓存中的长度
   *   newHandle.cachedLen — insertPrefix 后新的已缓存长度
   *
   * | 区域 | 名称                | 边界                             | 操作        | naive 退化                           |
   * |------|---------------------|----------------------------------|-------------|--------------------------------------|
   * | 1    | 前部保留区          | [0, oldHandle.cachedLen)         | 无操作      | 空（oldHandle.cachedLen=0）           |
   * | 2    | 前部已释放区        | [oldHandle.cachedLen, cachedLen) | _free       | 空操作（cachedLen=0）                 |
   * | 3    | 新写入区            | [cachedLen, newHandle.cachedLen) | insert 已处理 | 空（newHandle.cachedLen=0）           |
   * | 4    | 尾部保留区          | [newHandle.cachedLen, alignedLen)| 保留/释放   | finished=false:保留全部; true:释放全部 |
   * | 5    | 尾部已释放区        | [alignedLen, ...)                | 非页对齐碎片 | 同左                                 |
   */
  cacheReq(
    req: {
      inputIds: number[];
      cachedLen: number;
      tableIdx: number;
      cacheHandle: BaseCacheHandle | null;
    },
    finished: boolean,
  ): void {
    // page-aligned 切片：只处理完整页
    const alignedLen = Math.floor(req.cachedLen / this.pageSize) * this.pageSize;
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

    // 区域 1 [0, oldHandle.cachedLen) — 前部保留区：已在 prefix cache，无需操作
    //   naive 退化：oldHandle.cachedLen=0（总是 miss），区域 1 为空

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
      // 区域 5 [alignedLen, req.cachedLen) — 超出 aligned_len 的 forward 部分（非页对齐碎片，不存于 pageIndices）
      //   naive 退化：newHandle.cachedLen=0，释放 pageIndices[0:] 即全部已分配页
      this._free(pageIndices.slice(newHandle.cachedLen));
    } else {
      // 区域 4（未 finished）：保留，更新 handle
      (req as { cacheHandle: BaseCacheHandle | null }).cacheHandle = newHandle;
      this.lock(newHandle);
    }
  }

  // ===== _free（Set 去重版）=====

  /**
   * 释放页索引到 freeSlots（或 lazyFreeList）
   *
   * 使用 Set 去重替代步长切片 [::page_size]：
   * page_table 中同一页的 page_size 个连续位置存储相同的物理页起始位置值，
   * 例如 pageSize=4: pageTable[0] = [0,0,0,0, 4,4,4,4, 8,8,8,8]
   * 当子切片不从页边界开始（如 pageIndices[2:6] = [0,0,4,4]），
   * 步长切片 [::pageSize] 只取第一个元素 0，遗漏 4；
   * Set 去重得到 [0, 4]，正确释放两个页。
   */
  private _free(indices: number[]): void {
    if (indices.length === 0) return;
    if (this.pageSize > 1) {
      indices = Array.from(new Set(indices));
    }
    this._allocatedPages -= indices.length;
    if (this._inLazyFree) {
      this.lazyFreeList.push(...indices);
    } else {
      this.freeSlots.push(...indices);
    }
  }

  // ===== lazyFreeRegion 上下文管理 =====

  /** 进入延迟释放模式，后续 _free 调用收集到 lazyFreeList */
  beginLazyFree(): void {
    this._inLazyFree = true;
    this.lazyFreeList = [];
  }

  /** 退出延迟释放模式，将收集的页索引合并到 freeSlots */
  endLazyFree(): void {
    this._inLazyFree = false;
    this.freeSlots.push(...this.lazyFreeList);
    this.lazyFreeList = [];
  }

  // ===== checkIntegrity =====

  /** 页数守恒不变式校验：allocatedPages + freeSlots.length === numPages */
  checkIntegrity(): void {
    this.prefixCache.checkIntegrity();
    if (this._allocatedPages + this.freeSlots.length !== this.numPages) {
      throw new Error(
        `CacheManager integrity check failed: allocated_pages(${this._allocatedPages}) + ` +
        `free_pages(${this.freeSlots.length}) != num_pages(${this.numPages})`,
      );
    }
  }

  // ===== freeCache =====

  /** 释放请求缓存 — 语义等同 cacheReq(req, true) */
  freeCache(req: {
    inputIds: number[];
    cachedLen: number;
    tableIdx: number;
    cacheHandle: BaseCacheHandle | null;
  }): void {
    this.cacheReq(req, true);
  }
}
