---
issue_number: 14
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #14 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | CacheManager construction (naive backend) | ? pass |
| T2 | CacheManager.availableSize | ? pass |
| T3 | CacheManager.matchReq naive always miss | ? pass |
| T4 | CacheManager.matchReq lock unlock chain | ? pass |
| T5 | CacheManager.lock/unlock naive noop | ? pass |
| T6 | CacheManager.allocatePaged basic allocation | ? pass |
| T7 | CacheManager.allocatePaged multi-page allocation | ? pass |
| T8 | CacheManager.allocatePaged insufficient eviction break | ? pass |
| T9 | CacheManager.cacheReq finished=true releases all pages | ? pass |
| T10 | CacheManager.cacheReq finished=false retains pages | ? pass |
| T11 | CacheManager._free Set dedup non-page-aligned | ? pass |
| T12 | CacheManager._free multi-page duplicate release | ? pass |
| T13 | CacheManager.beginLazyFree/endLazyFree | ? pass |
| T14 | CacheManager.checkIntegrity naive initial state | ? pass |
| T15 | CacheManager.checkIntegrity failure on corrupted | ? pass |
| T16 | CacheManager.freeCache equals cacheReq finished=true | ? pass |
| T17 | Short prompt inputIds less than pageSize | ? pass |
| T18 | Medium prompt 3*pageSize | ? pass |
| T19 | Long prompt 10*pageSize | ? pass |
| T20 | matched less than computed | ? pass |
| T21 | Prefix growth availableSize decreases | ? pass |
| T22 | lazyFreeRegion correct count | ? pass |
| T23 | availableSize within bounds | ? pass |
| T24 | _free Set dedup page count invariant | ? pass |
| B1 | CacheManager numPages=0 | ? pass |
| B2 | CacheManager numPages=1 pageSize=1 | ? pass |
| B3 | cacheReq alignedLen=0 | ? pass |
| B4 | allocatePaged neededPages=0 | ? pass |
| B5 | endLazyFree without beginLazyFree | ? pass |
| B6 | pageSize=16 _free Set dedup | ? pass |
| B7 | matchReq inputLen=1 | ? pass |
| B8 | allocatePaged exhaust all pages break | ? pass |
| B9 | _free empty array noop | ? pass |
| T_extra | CacheManager exported as class | ? pass |
| T_extra | CacheManager rejects radix cacheType | ? pass |

通过: 35 / 失败: 0

## 类型检查
- 结果: pass（K3 相关文件无编译错误）

## 回归测试
- K2 测试: 31/31 全部通过，无回归

## 修复记录
1. T24 页数守恒失败 — 新增 _allocatedPages 私有计数器
2. TS2802 编译错误 — Array.from(new Set(indices)) 替代展开运算符

## 边界条件覆盖
- numPages=0, numPages=1/pageSize=1, alignedLen=0, neededPages=0
- endLazyFree未配对, pageSize=16去重, inputLen=1
- 全页耗尽break, _free空数组
