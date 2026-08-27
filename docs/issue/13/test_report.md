---
issue_number: 13
issue_type: Feature
test_date: 2026-08-27
test_result: pass
---

# Issue #13 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | MockKVCachePool 构造（numPages/pageSize/freePagesPool 正确初始化） | ✅ pass |
| T2 | MockKVCachePool.cachePerPage 公式（与 calculateMemoryBudget 一致） | ✅ pass |
| T3 | MockKVCachePool.storeKV noop（调用不抛错，不改变状态） | ✅ pass |
| T4 | MockKVCachePool.allocatePaged 基本分配（1 页 PageAllocation） | ✅ pass |
| T5 | MockKVCachePool.allocatePaged 多页分配（3 页，slotCount=48） | ✅ pass |
| T6 | MockKVCachePool.allocatePaged 不足抛错 | ✅ pass |
| T7 | MockKVCachePool.deallocatePageAllocation（回收后 freePages 增加） | ✅ pass |
| T8 | MockKVCachePool allocate-deallocate 循环 | ✅ pass |
| T9 | MockKVCachePool 页数守恒不变式 | ✅ pass |
| T10 | MockKVCachePool.decodeStepLatency | ✅ pass |
| T11 | MockKVCachePool.usedCapacity/totalCapacity | ✅ pass |
| T12 | NaiveCacheHandle 构造（cachedLen=0, getMatchedIndices=[]） | ✅ pass |
| T13 | NaivePrefixCache.matchPrefix 全 miss | ✅ pass |
| T14 | NaivePrefixCache.insertPrefix（返回 InsertResult(0, NaiveCacheHandle)） | ✅ pass |
| T15 | NaivePrefixCache.lockHandle noop | ✅ pass |
| T16 | NaivePrefixCache.evict 空返回 | ✅ pass |
| T17 | NaivePrefixCache.sizeInfo（始终 CacheSizeInfo(0, 0)） | ✅ pass |
| T18 | NaivePrefixCache.reset noop | ✅ pass |
| T19 | NaivePrefixCache.checkIntegrity noop | ✅ pass |
| B1 | 分配全部页后耗尽（freePages=0，再分配抛错） | ✅ pass |
| B2 | deallocate 后重新 allocate | ✅ pass |
| B3 | MockKVCachePool(numPages=0)（freePagesPool 为空，totalCapacity=0） | ✅ pass |
| B4 | MockKVCachePool(numPages=1)（只有 1 页，分配后耗尽） | ✅ pass |
| B5 | MockKVCachePool.allocatePaged(0)（返回空 PageAllocation） | ✅ pass |
| B6 | NaiveCacheHandle 非 0 构造（允许但 NaivePrefixCache 总是传 0） | ✅ pass |
| B7 | NaivePrefixCache 多次 match/insert（每次都 miss，状态无变化） | ✅ pass |
| B8 | pageSize=16 的 slot 展开（slots 包含每个页内所有 token 位置） | ✅ pass |
| B9 | decodeStepLatency numDecodeTokens=0（返回 cudaGraphOverhead） | ✅ pass |
| T_extra | MockKVCachePool extends BaseKVCachePool | ✅ pass |
| T_extra | NaivePrefixCache extends BasePrefixCache | ✅ pass |
| T_extra | NaiveCacheHandle extends BaseCacheHandle | ✅ pass |

## 类型检查
- 结果: pass

## 回归测试
- K1 验收测试: 23/23 pass
- K5 验收测试: 20/20 pass

## 失败用例详情（如有）
无

## 边界条件覆盖
- numPages=0: ✅ 空池构造正常
- numPages=1: ✅ 单页分配/耗尽/回收正常
- allocatePaged(0): ✅ 返回空分配
- 页数守恒: ✅ usedPages + freePages === numPages 在任意操作后成立
- 全页分配后耗尽: ✅ 抛出 Error
- deallocate 后重新 allocate: ✅ 回收的页可被重新分配
- NaiveCacheHandle 非 0 构造: ✅ 允许但不影响 NaivePrefixCache 行为
- 多次 match/insert: ✅ 始终 miss
- pageSize=16 slot 展开: ✅ 每个 slot 正确映射
- decodeStepLatency(0): ✅ 返回 graph overhead
