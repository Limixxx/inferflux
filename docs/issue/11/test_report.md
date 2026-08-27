---
issue_number: 11
issue_type: Feature
test_date: 2026-08-27
test_result: pass
---

# Issue #11 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | CacheSizeInfo 构造（默认值 0,0；自定义值正确赋值） | ✅ pass |
| T2 | CacheSizeInfo.totalSize（evictableSize + protectedSize） | ✅ pass |
| T3 | MatchResult 构造（cudaHandle 正确持有） | ✅ pass |
| T4 | MatchResult readonly 字段验证 | ✅ pass |
| T5 | InsertResult 构造（cachedLen + cudaHandle 正确持有） | ✅ pass |
| T6 | InsertResult readonly 字段验证 | ✅ pass |
| T7 | BaseCacheHandle 子类化（cachedLen getter + getMatchedIndices） | ✅ pass |
| T8 | BaseKVCachePool 子类化（所有抽象属性和方法） | ✅ pass |
| T9 | BasePrefixCache 子类化（所有抽象方法和 sizeInfo） | ✅ pass |
| T10 | TableManager 构造（freeTableIndices 初始化为 0..maxRunningReq-1） | ✅ pass |
| T11 | TableManager.allocate（分配并返回末尾 index） | ✅ pass |
| T12 | TableManager.free（释放 index 回收到栈） | ✅ pass |
| T13 | TableManager.availableSize（反映当前可用行数） | ✅ pass |
| T14 | TableManager 分配耗尽（无可用行时抛出 Error） | ✅ pass |
| T15 | TableManager.tokenPool（正确创建 maxRunningReq+1 行） | ✅ pass |
| T16 | TableManager 循环分配释放（allocate→free→allocate） | ✅ pass |
| B1 | CacheSizeInfo(0, 0) totalSize = 0 | ✅ pass |
| B2 | TableManager(maxRunningReq=1) 只有一行可用 | ✅ pass |
| B3 | TableManager 连续 allocate 直到耗尽抛出 Error | ✅ pass |
| B4 | TableManager free 后重新 allocate 返回刚释放的 index | ✅ pass |
| B5 | MatchResult handle.cachedLen = 0（空匹配场景） | ✅ pass |
| B6 | InsertResult cachedLen = 0（完全未缓存场景） | ✅ pass |
| B7 | BasePrefixCache.lockHandle 默认 unlock=false | ✅ pass |

## 类型检查
- 结果: pass

## 失败用例详情（如有）
无

## 边界条件覆盖
- CacheSizeInfo(0, 0) → totalSize = 0
- TableManager(maxRunningReq=1) → 仅一行可用，tokenPool 有 2 行（含 dummy）
- TableManager 分配耗尽 → 抛出 Error("No available table indices")
- TableManager free 后重新 allocate → 返回刚释放的 index（LIFO 栈行为）
- MatchResult 空匹配 → cudaHandle.cachedLen = 0
- InsertResult 完全未缓存 → cachedLen = 0
- BasePrefixCache.lockHandle 默认 unlock 参数 → undefined（等效 false）
