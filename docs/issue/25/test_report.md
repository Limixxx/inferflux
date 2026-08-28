---
issue_number: 25
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #25 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | 启用条件 — useMla=true && enableDpAttention=true && dpSize=2 | ✅ pass |
| T2 | 不启用 — useMla=false | ✅ pass |
| T3 | 不启用 — enableDpAttention=false | ✅ pass |
| T4 | 不启用 — dpSize=1 | ✅ pass |
| T5 | simulateMlpForward dpSize=2 batch=2 分块 1+1 | ✅ pass |
| T6 | all_gather_ticks 随 dpSize 线性增长 | ✅ pass |
| T7 | simulateMlpForward 未启用返回 0 (mla=false) | ✅ pass |
| T8 | totalAllGatherBytesPerStep 启用时返回正值 | ✅ pass |
| T9 | totalAllGatherBytesPerStep 未启用返回 0 | ✅ pass |
| T10 | commGroup.groupType === dp_attn | ✅ pass |
| T11 | commGroup.size equals dpSize | ✅ pass |
| T12 | allGatherBytes 计算正确 | ✅ pass |
| B1 | simulateMlpForward with empty localBatchSizes | ✅ pass |
| B2 | simulateMlpForward with all-zero localBatchSizes | ✅ pass |
| B3 | totalAllGatherBytesPerStep with batch=0 | ✅ pass |
| B5 | disabled when useMla is undefined (falsy) | ✅ pass |

## 类型检查
- 结果: pass（P2b/parallel 相关模块无类型错误）

## 失败用例详情
无

## 边界条件覆盖
- B1: localBatchSizes 为空数组 → allGatherBytes=0, commTicks 包含 latency
- B2: localBatchSizes 全为 0 → allGatherBytes=0, commTicks 仅含 latency
- B3: batch=0 传入 totalAllGatherBytesPerStep → 返回 0
- B5: useMla=undefined (falsy) → enabled=false, 退化为 noop

## 修复记录
- T6: 使用更大 batch sizes (64/rank) 和更小 latency (1μs) 使带宽成本主导计算，确保 dpSize=4 的 commTicks > dpSize=2
- B5: 源码 `dp_attn.ts` 中 `enabled` 赋值改用 `!!()` 包裹，确保始终返回 boolean 类型
