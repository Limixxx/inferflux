---
issue_number: 27
issue_type: Feature
test_date: 2026-08-31
test_result: pass
---

# Issue #27 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | enabled=false 时 maybe_rebalance 返回 shouldRebalance=false | ✅ pass |
| T2 | epSize<=1 时 maybe_rebalance 返回 shouldRebalance=false | ✅ pass |
| T3 | step=50 非 100 倍数 → shouldRebalance=false | ✅ pass |
| T4 | step=100 进入方差判定（均匀负载 → 跳过） | ✅ pass |
| T5 | 方差低跳过 — 负载均匀 variance_ratio=0 < 0.1 | ✅ pass |
| T6 | 方差高触发重平衡 | ✅ pass |
| T7 | movedExperts 非负 | ✅ pass |
| T8 | 重排后新 max rank 负载不超过旧 max | ✅ pass |
| T9 | 触发重平衡时 rebalanceTicks 等于 rebalanceCostFixedTicks | ✅ pass |
| T10 | 多次重平衡后 metrics.epRebalanceCostTicks 累加 | ✅ pass |
| T11 | expertToRankMap 更新后路由反映新映射 | ✅ pass |
| T12 | avg=0 时不重平衡（不除零） | ✅ pass |
| T13 | step=100, 200, 300 各触发一次检查 | ✅ pass |
| B1 | enabled=false → maybe_rebalance 始终返回 shouldRebalance=false | ✅ pass |
| B2 | epSize=1 → 退化返回不重平衡 | ✅ pass |
| B3 | 所有 expertLoadCounts=0 → avg=0 → 安全返回 | ✅ pass |
| B4 | 每 rank 仅 1 expert → movedExperts=0 | ✅ pass |
| B6 | 极度不均（某 rank 全部负载，另一 rank 为 0）→ 触发重平衡 | ✅ pass |
| B7 | 非检查周期返回 false，不重复累加成本 | ✅ pass |
| B8 | 重排后每 rank 至少保留 1 个 expert | ✅ pass |
| — | expertLoadCounts getter 返回正确快照 | ✅ pass |
| — | expertLoadCounts 返回浅拷贝（不影响原始指标） | ✅ pass |
| — | 自定义 rebalanceIntervalSteps=50 在 step=50 触发 | ✅ pass |
| — | 自定义 loadVarianceThreshold=0.5 方差低阈值高→跳过 | ✅ pass |
| — | 自定义 rebalanceCostFixedTicks=100 成本累加正确 | ✅ pass |

## 类型检查
- 结果: pass（P3b 相关代码无类型错误）

## 回归测试
- P3a 验收测试: 37/37 pass ✅

## 失败用例详情（如有）
无
