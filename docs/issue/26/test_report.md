---
issue_number: 26
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #26 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | 构造 expertsPerRank numExperts=8 epSize=2 | pass |
| T2 | _expertToRank 正确映射 | pass |
| T2a | _expertToRank 与 topology.computeMoeRanks 一致 | pass |
| T3 | _expertToRank 非均分 numExperts=7 epSize=2 | pass |
| T4 | hash 路由可复现 | pass |
| T4a | hash 路由不同 layerIdx 结果不同 | pass |
| T5 | hash 路由分布合理 | pass |
| T6 | mock 路由平衡方差低 | pass |
| T7 | simulated 路由可复现 | pass |
| T7a | simulated 不同 seed 不同结果 | pass |
| T8 | simulated 路由分布非退化 | pass |
| T9 | all-to-all 正反字节数守恒 | pass |
| T9a | comm_ticks 与公式一致 | pass |
| T10 | crossRankTokens 非负 | pass |
| T11 | epSize=1 退化 commTicks=0 | pass |
| T12 | epSize=1 退化 crossRankTokens=0 | pass |
| T13 | epSize>1 forward 返回 commTicks>0 | pass |
| T14 | 指标 epCommTicks 累加 | pass |
| T15 | 指标 epAllToAllCount 每次 forward 增加 2 | pass |
| T16 | 指标 epCrossRankTokens 累加 | pass |
| T17 | 指标 epExpertLoad | pass |
| T17a | 指标 epExpertLoad 多次 forward 累加 | pass |
| T18 | isMoe=false 不创建 moeBackend | pass |
| T19 | 多层 forward 指标累加 | pass |
| T20 | 多 batch forward 指标累加 | pass |
| T21 | hash 模式 seed=0 正常运行 | pass |
| B1 | batchSize=0 | pass |
| B2 | numExperts=1 moeTopK=1 | pass |
| B3 | moeTopK=numExperts | pass |
| B4 | epSize > numExperts throws | pass |
| B5 | numExperts 非整除 epSize | pass |
| B6 | seed=0 simulated 正常 | pass |
| B7 | 单 token batch | pass |
| B8 | 极端退化 | pass |
| 集成1 | MockEngine isMoe=true 创建 moeBackend | pass |
| 集成2 | MockEngine forwardBatch MoE 层返回 commTicks | pass |
| 集成3 | MockEngine forwardBatch 非 MoE 层返回 0 | pass |

## 类型检查
- 结果: pass

## 失败用例详情
无

## 边界条件覆盖
- B1-B8 全部覆盖通过
