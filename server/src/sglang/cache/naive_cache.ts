// naive_cache — K2: NaivePrefixCache + NaiveCacheHandle (§9.3b)

import {
  BaseCacheHandle,
  BasePrefixCache,
  CacheSizeInfo,
  MatchResult,
  InsertResult,
} from "./index";

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

/** Phase 1 基线前缀缓存（§9.3b） — 总是 miss，不命中不占用 */
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
