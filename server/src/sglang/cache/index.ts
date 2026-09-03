// cache — K1: KVCache 基础抽象层

// 基础抽象类和值类型（提取到 base.ts 避免循环依赖）
export {
  CacheSizeInfo,
  BaseCacheHandle,
  MatchResult,
  InsertResult,
  BaseKVCachePool,
  BasePrefixCache,
} from "./base";

// K5: 内存预算基础公式 (§3.3.9)
export {
  MemoryBudgetResult,
  estimateModelMemory,
  estimateGraphBuffer,
  calculateMemoryBudget,
} from "./budget";

// P1b: ParallelMemoryCorrections 从 parallel/budget re-export
export type { ParallelMemoryCorrections } from "../parallel/budget";

// K2: MockKVCachePool + NaivePrefixCache (§3.4.2 / §9.3b)
export { MockKVCachePool, PageAllocation } from "./mha_pool";
export { NaivePrefixCache, NaiveCacheHandle } from "./naive_cache";

// K3: CacheManager (§9.11)
export { CacheManager } from "./cache_manager";
// K4: RadixPrefixCache (§9.8 / §3.3.5)
export { RadixTreeNode, RadixCacheHandle, RadixPrefixCache } from "./radix_cache";
export type { KeyFn } from "./radix_cache";
