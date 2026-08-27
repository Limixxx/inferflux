---
issue_number: 21
issue_type: Feature
test_date: 2026-08-27
test_result: pass
---

# Issue #21 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | allReduce size=1 返回 0 | ✅ pass |
| T2 | allReduce size>1 返回正数 | ✅ pass |
| T3 | allGather size=1 返回 0 | ✅ pass |
| T4 | allGather size>1 返回正数 | ✅ pass |
| T5 | allToAll size=1 返回 0 | ✅ pass |
| T6 | allToAll size>1 返回正数含 latency×size | ✅ pass |
| T7 | sendRecv 正常计算（不受 size=1 影响） | ✅ pass |
| T8 | barrier 为 noop | ✅ pass |
| T9 | efficiency 缩放 | ✅ pass |
| T10 | CommGroupType 全类型 | ✅ pass |
| T11 | worldSize 计算正确 | ✅ pass |
| T12 | rankToCoord/coordToRank 互逆 | ✅ pass |
| T13 | computeMoeRanks tp=8 dp=2 ep=2 | ✅ pass |
| T14 | computeAttnRanks tp=8 cp=2 | ✅ pass |
| T15 | ppStageLayers 32层 pp=4 | ✅ pass |
| T16 | ppStageLayers 33层 pp=4（余数分配） | ✅ pass |
| T17 | cp_size 整除 tp_size 约束 | ✅ pass |
| T18 | ep_size 整除 tp_size/cp_size 约束 | ✅ pass |
| T19 | ParallelMetrics 默认值全为 0/空 | ✅ pass |
| T20 | commTicksTotal 计算 | ✅ pass |
| T21 | reset 清零 | ✅ pass |
| T22 | summary 包含全部 23 字段 | ✅ pass |
| T23 | MockTPGroup(1) allReduce=0 | ✅ pass |
| T24 | MockTPGroup(2) allReduce>0 | ✅ pass |
| T25 | mockAllReduceSum 兼容 | ✅ pass |
| T26 | DEFAULT 含新增 P0 字段 | ✅ pass |
| T27 | 新字段类型正确 | ✅ pass |
| T28 | SimulationMetrics.parallel 存在 | ✅ pass |
| T29 | 全并行 size=1 退化为单实例 | ✅ pass |

## 边界条件覆盖
| 编号 | 边界条件 | 结果 |
|------|---------|------|
| B1 | SimCommGroup bytes=0 → 返回 latency | ✅ pass |
| B2 | SimCommGroup bandwidth 极大 → ticks 趋近 latency | ✅ pass |
| B3 | SimCommGroup bandwidth=0 → Infinity | ✅ pass |
| B4 | ParallelTopology world_size=1 → (0,0,0) | ✅ pass |
| B5 | ppStageLayers pp=1 覆盖全部层 | ✅ pass |
| B6 | commTicksTotal 各项为 0 → 总和为 0 | ✅ pass |
| B7 | MockTPGroup(1) mockAllReduceSum(0) 返回 0 | ✅ pass |
| B8 | efficiency=1.0 与无效率因子一致 | ✅ pass |

## 类型检查
- 结果: pass（新增代码无 TS 错误；已有错误 BaseCacheHandle/MatchResult/InsertResult 缺失导出与本次修改无关）

## 失败用例详情
无
