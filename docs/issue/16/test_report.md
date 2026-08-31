---
issue_number: 16
issue_type: Feature
test_date: 2026-08-31
test_result: pass
---

# Issue #16 测试报告

## 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | PrefillAdder construction | ✅ pass |
| T2 | PrefillAdder.tryAddOne - short prompt one-shot prefill | ✅ pass |
| T3 | PrefillAdder.tryAddOne - token budget insufficient -> ChunkedReq | ✅ pass |
| T4 | PrefillAdder.tryAddOne - first available_size check fail | ✅ pass |
| T5 | PrefillAdder.tryAddOne - second available_size check fail (lock reduces available) | ✅ pass |
| T6 | PrefillAdder._tryAddOneChunked - continuation | ✅ pass |
| T7 | PrefillAdder - two available_size checks consistency | ✅ pass |
| T8 | PrefillManager.addOneReq | ✅ pass |
| T9 | PrefillManager.addBatch | ✅ pass |
| T10 | PrefillManager.scheduleNextBatch - empty queue | ✅ pass |
| T11 | PrefillManager.scheduleNextBatch - short prompt one-shot | ✅ pass |
| T12 | PrefillManager.scheduleNextBatch - long prompt chunked then continuation | ✅ pass |
| T13 | PrefillManager.abortReq - existing uid | ✅ pass |
| T14 | PrefillManager.abortReq - non-existing uid | ✅ pass |
| T15 | DecodeManager.addReq/removeReq | ✅ pass |
| T16 | DecodeManager.filterReqs | ✅ pass |
| T17 | DecodeManager.inflightTokens | ✅ pass |
| T18 | DecodeManager.scheduleNextBatch - empty set | ✅ pass |
| T19 | DecodeManager.scheduleNextBatch - non-empty | ✅ pass |
| T20 | DecodeManager.abortReq | ✅ pass |
| T21 | PrefillManager + DecodeManager integration - short prompt full flow | ✅ pass |
| T22 | PrefillManager + DecodeManager integration - long prompt two ticks | ✅ pass |
| T23 | PrefillAdder - unlock rollback after second check fail | ✅ pass |
| T24 | PrefillAdder - tableManager allocate failure | ✅ pass |
| T25 | PrefillAdder - consecutive tryAddOne consistency | ✅ pass |
| T26 | PrefillManager - multiple chunked requests continuation priority | ✅ pass |
| T27 | DecodeManager.filterReqs - empty newReqs | ✅ pass |
| T28 | DecodeManager - pageSize=1 tokens_reserved=0 | ✅ pass |
| T29 | PrefillAdder._tryAddOneChunked - resource insufficient on continuation | ✅ pass |
| T30 | PrefillAdder - full cache hit (extendLen=0) | ✅ pass |
| B1 | PrefillAdder.tokenBudget = 0 -> all tryAddOne return null | ✅ pass |
| B2 | PrefillAdder - extendLen = 0 (full cache hit) | ✅ pass |
| B3 | PrefillAdder - tableManager.availableSize = 0 -> first check fail | ✅ pass |
| B4 | PrefillManager - multiple chunked requests continuation order | ✅ pass |
| B5 | DecodeManager - pageSize=1 tokens_reserved=0 | ✅ pass |
| B6 | DecodeManager.filterReqs - empty newReqs filters only existing | ✅ pass |
| B7 | PrefillAdder._tryAddOneChunked - resource insufficient on continuation | ✅ pass |
| B8 | PrefillAdder - retry after second available_size check fail | ✅ pass |
| B9 | tableManager allocate exhausted | ✅ pass |
| B10 | DecodeManager.inflightTokens - no running requests | ✅ pass |
| R1 | estimatedLen includes outputLen - first check in tryAddOne | ✅ pass |
| R2 | estimatedLen includes outputLen - tryAddOne passes when fits | ✅ pass |
| R3 | estimatedLen includes outputLen - _tryAddOneChunked resource check | ✅ pass |
| R4 | maxDeviceLen explicitly set - ChunkedReq from tryAddOne | ✅ pass |
| R5 | maxDeviceLen explicitly set - Req from tryAddOne | ✅ pass |
| R6 | maxDeviceLen explicitly set - Req from _tryAddOneChunked | ✅ pass |
| R7 | maxDeviceLen explicitly set - ChunkedReq from _tryAddOneChunked | ✅ pass |
| R8 | estimatedLen with outputLen=0 equals extendLen | ✅ pass |

## 类型检查
- 结果: pass（scheduler/index.ts 和 sglang-s2.test.ts 无类型错误）
- 注: 预先存在的 barrel file 语法错误（sglang/index.ts, engine/index.ts, parallel/index.ts）不属于本 Issue 范围

## 失败用例详情（如有）
无

## 边界条件覆盖
- tokenBudget = 0：所有 tryAddOne 返回 null（B1）
- extendLen = 0（全缓存命中）：consumedTokens 不增加，返回完整 Req（T30, B2）
- tableManager.availableSize = 0：第一次 available_size 检查失败（B3）
- tableManager 分配耗尽：allocate 抛异常，unlock 回滚（T24, B9）
- 第二次 available_size 检查失败：lock 后资源不足，unlock 回滚（T5, T23, B8）
- 续接时资源不足：返回 null（T29, B7）
- pageSize=1：tokens_reserved = 0（T28, B5）
- 无 running 请求：inflightTokens = 0（B10）
- filterReqs 空集合：仅过滤已有请求（T27, B6）
- chunked 请求续接优先级：放回队列头部优先调度（T26, B4）
- estimatedLen 包含 outputLen：extend+output 超出 availableSize 时拒绝（R1, R3）
- estimatedLen 包含 outputLen：extend+output 在 availableSize 内时通过（R2）
- outputLen=0 时 estimatedLen 退化为 extendLen（R8）
- ChunkedReq.maxDeviceLen = cachedLen + chunkSize（R4, R7）
- Req.maxDeviceLen = cachedLen + extendLen + outputLen（R5）
- 续接 Req.maxDeviceLen = inputLen + outputLen（R6）
