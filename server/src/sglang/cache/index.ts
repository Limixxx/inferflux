// cache — K1: KVCache 基础抽象层

// ===== CacheSizeInfo 工具类（§9.3 / §9.11） =====

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

// ===== BaseCacheHandle 抽象类（§9.3） =====

/** 缓存句柄抽象基类，指向缓存树中的节点（§9.3） */
export abstract class BaseCacheHandle {
  abstract get cachedLen(): number;
  abstract getMatchedIndices(): number[];
}

// ===== MatchResult 值类型（§9.3） =====

/** matchPrefix 的返回值（§9.3） */
export class MatchResult {
  readonly cudaHandle: BaseCacheHandle;

  constructor(cudaHandle: BaseCacheHandle) {
    this.cudaHandle = cudaHandle;
  }
}

// ===== InsertResult 值类型（§9.3） =====

/** insertPrefix 的返回值（§9.3） */
export class InsertResult {
  readonly cachedLen: number;       // 插入前已在缓存中的长度
  readonly cudaHandle: BaseCacheHandle;

  constructor(cachedLen: number, cudaHandle: BaseCacheHandle) {
    this.cachedLen = cachedLen;
    this.cudaHandle = cudaHandle;
  }
}

// ===== BaseKVCachePool 抽象类（§9.3） =====

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

// ===== BasePrefixCache 抽象类（§9.3） =====

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

// K5: 内存预算基础公式 (§3.3.9)
export {
  MemoryBudgetResult,
  estimateModelMemory,
  estimateGraphBuffer,
  calculateMemoryBudget,
} from "./budget";

// K2: MockKVCachePool + NaivePrefixCache (§3.4.2 / §9.3b)
export { MockKVCachePool, PageAllocation } from "./mha_pool";
export { NaivePrefixCache, NaiveCacheHandle } from "./naive_cache";

// K3: CacheManager (§9.11)
export { CacheManager } from "./cache_manager";
