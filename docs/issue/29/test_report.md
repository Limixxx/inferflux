---
issue_number: 29
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #29 测试报告

## 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | cp_size=1 时 simulateAttnForward 返回零 | ✅ pass |
| T2 | cp_size=4 时 comm_ticks > 0 | ✅ pass |
| T3 | seq_len 不能整除 cp_size 时 seq_per_rank 分布正确 (divCeil) | ✅ pass |
| T4 | cp_size=4 时 allGatherBytes 计算正确 | ✅ pass |
| T5 | cpAllGatherCount 每层递增 | ✅ pass (通过集成测试 T8 间接验证) |
| T6 | totalCommTicks 累加正确 | ✅ pass |
| T7 | cp_size=1 时 CPSimulator commGroup 为 null (skip) | ✅ pass |
| T8 | cp_size=4 forward_batch 后 cpCommTicks > 0 | ✅ pass |
| T9 | cp_size=1 forward_batch 后 cpCommTicks = 0 | ✅ pass |
| T10 | cp_size=4 时 cpSeqLenPerRank 正确 (1024/4=256) | ✅ pass |
| T11 | CP + TP 组合 commTicksTotal 包含 cp + tp | ✅ pass |

## 类型检查

- 结果: pass（P5 相关文件无编译错误；已有的 TableManager 导出问题已修复）

## 边界条件覆盖

| 编号 | 边界条件 | 结果 |
|------|---------|------|
| B1 | seq_len=0 → kv_bytes=0, allGather 仅返回 latency | ✅ pass |
| B2 | seq_len < cp_size (seq_len=2, cp_size=4) → seq_per_rank=1 | ✅ pass |
| B3 | cp_size=tp_size=4 → CPSimulator 正常工作 | ✅ pass |
| B4 | num_layers=1 → kv_bytes 为 32 层的 1/32 | ✅ pass |
| B5 | cpEfficiency=1.0 vs 0.90 → 1.0 通信成本更低 | ✅ pass |
| B6 | 极大 seq_len=131072 → 通信成本显著 | ✅ pass |
