---
issue_number: 22
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #22 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | tpSize=1 全 noop — allReduceAfterAttn/Mlp 返回 0，localNumHeads=原值 | ✅ pass |
| T2 | tpSize=2 内存修正 — localNumHeads/NumKvHeads/Intermediate 正确切分 | ✅ pass |
| T3 | allReduceAfterAttn 正值 — tpSize=2 时返回正值，公式匹配 SimCommGroup.allReduce | ✅ pass |
| T4 | allReduceAfterMlp 正值 — 与 allReduceAfterAttn 相同数据量，返回相同值 | ✅ pass |
| T5 | totalCommTicksPerStep 累加 — 多次调用后 total 等于各次之和 | ✅ pass |
| T6 | resetStepComm 清零 — 调用后 totalCommTicksPerStep=0 | ✅ pass |
| T7 | divEven GQA kv_heads 复制 — numKvHeads=2, tpSize=4 时 localNumKvHeads=1 | ✅ pass |
| T8 | SimCommGroup 效率因子 — TPSimulator 使用 config.tpEfficiency | ✅ pass |
| T9 | tpSize=1 TPCommInfraSimulator 全 noop | ✅ pass |
| T10 | zmqBroadcast 正值 — tpSize=2 时返回 Math.ceil(msgSize/bandwidth) | ✅ pass |
| T11 | cpuBarrier 固定 1 tick — tpSize=2 时始终返回 1 | ✅ pass |
| T12 | gpuAllReduce 委托 — 结果与 SimCommGroup("tp").allReduce 相同 | ✅ pass |
| T13 | broadcastAll 批量 — 多个 token_ids_list 正确汇总 bytes | ✅ pass |
| T14 | zmqBroadcastTicks 累加 — 多次调用后等于各次返回值之和 | ✅ pass |
| T15 | barrierTicks 累加 — 多次调用后等于各次返回值之和 | ✅ pass |
| T16 | cpuGroupType/gpuGroupType 读取 — 正确存储 config 中的值 | ✅ pass |
| T17 | TPSimulator + TPCommInfraSimulator 独立实例互不干扰 | ✅ pass |
| T18 | tpSize=1 退化单实例 — 全部返回 0 | ✅ pass |
| T19 | 与 ParallelMetrics 字段对应 — comm ticks 可正确写入 ParallelMetrics.tpCommTicks | ✅ pass |

## 类型检查
- 结果: pass
- 注: 项目既有测试文件 (k1, s0) 有类型错误，与本 Issue 无关；P1a 新增代码无类型错误

## 失败用例详情（如有）
无

## 边界条件覆盖
| 编号 | 边界条件 | 预期行为 | 结果 |
|------|---------|---------|------|
| B1 | batchSize=0 | allReduceAfterAttn(0) 数据量为 0，SimCommGroup.allReduce(0) 返回 latency | ✅ pass |
| B2 | msgSize=0 | zmqBroadcast(0) 返回 0 | ✅ pass |
| B3 | tpSize 极大 | allReduce 成本随 size 增大趋近于 2×bytes/bw + latency | ✅ pass |
| B4 | numKvHeads=0 | divEven(0, tpSize) 返回全 0，localNumKvHeads=0 | ✅ pass |
| B5 | commBandwidthBytesPerTick=0 | zmqBroadcast 中 Math.ceil(msgSize/max(1,0)) = msgSize | ✅ pass |
| B6 | tpEfficiency=1.0 | 结果与无效率因子一致 | ✅ pass |
