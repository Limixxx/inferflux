---
issue_number: 18
issue_type: Feature
test_date: 2026-09-02
test_result: pass
---

# Issue #18 测试报告

## 驳回修复

| 驳回意见 | 修复方式 |
|----------|----------|
| determineCudaGraphBs 未实现 totalGpuMemory 自动计算：cudaGraphMaxBs 为 null 时缺少 `totalGpuMemory > 80GiB → maxBs=256` 自动推断逻辑 | ✅ 已修复：determineCudaGraphBs 在 cudaGraphMaxBs 为 null 时，根据 totalGpuMemory 自动推断 maxBs（>80GiB→256，≤80GiB→160）。同步更新 T4/T5 测试用例 |

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | SimGraphRunner 构造 | ✅ pass |
| T2 | determineCudaGraphBs - 用户指定 | ✅ pass |
| T3 | determineCudaGraphBs - 自动计算 | ✅ pass |
| T4 | determineCudaGraphBs - 大显存自动推断 | ✅ pass |
| T5 | determineCudaGraphBs - 小显存自动推断 | ✅ pass |
| T6 | determineCudaGraphBs - 禁用 | ✅ pass |
| T7 | canUseCudaGraph - 禁用 | ✅ pass |
| T8 | canUseCudaGraph - decode batch | ✅ pass |
| T9 | canUseCudaGraph - prefill batch | ✅ pass |
| T10 | canUseCudaGraph - bs 超限 | ✅ pass |
| T11 | canUseCudaGraph - invalidate 后 | ✅ pass |
| T12 | padBatch - decode batch pad 到分桶 | ✅ pass |
| T13 | padBatch - prefill batch 不 pad | ✅ pass |
| T14 | padBatchToBs - 显式指定目标 | ✅ pass |
| T15 | padBatchToBs - 使用 dummyReq | ✅ pass |
| T16 | padBatchToBs - 不干扰 KV 分配计数 | ✅ pass |
| T17 | graphReplayCostTicks - 基本值 | ✅ pass |
| T18 | graphReplayCostTicks - 随 bs 增长 | ✅ pass |
| T19 | graphReplayCostTicks - 大 bs | ✅ pass |
| T20 | eagerForwardCostTicks - prefill | ✅ pass |
| T21 | eagerForwardCostTicks - decode | ✅ pass |
| T22 | estimateGraphBuffer - 空 graphBsList | ✅ pass |
| T23 | estimateGraphBuffer - 正常计算 | ✅ pass |
| T24 | estimateGraphBuffer - 与 budget.ts 一致 | ✅ pass |
| T25 | invalidate - 标记失效 | ✅ pass |
| T26 | invalidate - destroyCudaGraphs 恢复 | ✅ pass |
| T27 | replay - 返回正确行数 | ✅ pass |
| T28 | replay - 返回正确列数 | ✅ pass |
| T29 | MockEngine.simGraphRunner 属性 | ✅ pass |
| T30 | MockEngine.forward_batch - graph replay 时间 | ✅ pass |
| T31 | MockEngine.forward_batch - eager 时间 | ✅ pass |
| T32 | MockEngine.forward_batch - isGraphCapture 标识 | ✅ pass |
| T33 | SimScheduler._prepareBatch 使用 simGraphRunner | ✅ pass |
| T34 | 分桶边界 bs=31→32 | ✅ pass |
| T35 | eager 与 graph 切换一致 | ✅ pass |
| T36 | destroyCudaGraphs 为 noop | ✅ pass |

## 类型检查
- 结果: pass（S4 相关文件无类型错误；其他 Issue 的测试文件存在既有类型错误，与本次修改无关）

## 边界条件覆盖
| 编号 | 边界条件 | 结果 |
|------|---------|------|
| B1 | cudaGraphBs 为空列表 | ✅ pass |
| B2 | bs=0 的空 batch | ✅ pass |
| B3 | bs 恰好等于分桶值 | ✅ pass |
| B4 | bs=1 的 decode batch | ✅ pass |
| B5 | chunked prefill batch | ✅ pass |
| B6 | graphReplayCostTicks=0 | ✅ pass |
| B7 | eagerForwardCostTicks=0 | ✅ pass |
| B8 | 多次 padBatch 调用 | ✅ pass |
| B9 | 连续多次 invalidate | ✅ pass |
| B10 | invalidate 后 forward_batch 走 eager | ✅ pass |

## 回归测试
- S3 测试全部通过（52 passed, 0 failed）
