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
| T1 | cp_size=1 simulateAttnForward 返回零 | ✅ pass |
| T2 | cp_size=4 comm_ticks > 0 | ✅ pass |
| T3 | divCeil seq_per_rank when seq_len not divisible | ✅ pass |
| T4 | allGatherBytes formula correct (single layer, per-rank) | ✅ pass |
| T5 | cpAllGatherCount increments per layer | ✅ pass |
| T6 | totalCommTicks accumulates correctly | ✅ pass |
| T7 | cp_size=1 commGroup is null | ✅ pass |
| P1 | precise total cpCommTicks = numLayers * singleLayerCommTicks | ✅ pass |
| P2 | kvBytesPerRank does NOT include numLayers factor | ✅ pass |
| P3 | allGather bytes based on seqLenPerRank not full seqLen | ✅ pass |
| T8 | cp_size=4 forwardBatch cpCommTicks > 0 | ✅ pass |
| T9 | cp_size=1 forwardBatch cpCommTicks = 0 | ✅ pass |
| T10 | cp_size=4 seqLen=1024 cpSeqLenPerRank=256 | ✅ pass |
| T11 | CP+TP combined commTicksTotal includes cp | ✅ pass |
| B1 | seqLen=0 kv_bytes=0, commTicks latency-based | ✅ pass |
| B2 | seqLen=2 cpSize=4 seqLenPerRank=1 | ✅ pass |
| B3 | cp_size=tp_size=4 CPSimulator works | ✅ pass |
| B4 | numLayers=1 precise total commTicks | ✅ pass |
| B4b | numLayers=1 vs numLayers=32 allGatherBytes identical per-layer | ✅ pass |
| B5 | cpEfficiency=1.0 vs 0.90 | ✅ pass |
| B6 | large seqLen=131072 significant comm cost | ✅ pass |
| T_export | CPSimulator exported from sglang index | ✅ pass |
| T_export | CPAttnResult interface fields exist | ✅ pass |

## 类型检查
- 结果: pass (`npx tsc --noEmit` 无错误)

## 回归测试
- P0 测试: 37/37 通过
- S1 测试: 26/26 通过

## 本轮修复内容（Review Round 2）
1. **修复双重计数**：`simulateAttnForward` 移除 `numLayers` 因子，改为计算单层 KV all-gather 数据量，在 MockEngine 层循环中逐层调用并累加
2. **修复 per-rank bytes 歧义**：`allGather` 传入基于 `seqLenPerRank` 计算的每 rank 字节数，而非完整序列字节数
3. **类型紧化**：CPSimulator 构造函数改用已有的 `SimulatorConfig` 和 `ModelConfig` 类型
4. **增加精确性断言**：新增 P1/P2/P3 测试用例，精确验证总通信量公式和各因子语义

## 边界条件覆盖
- seqLen=0: kv_bytes=0, commTicks 仅含 latency ✅
- seqLen < cpSize: divCeil 正确向上取整 ✅
- cpSize=tpSize: CP 与 TP 共存 ✅
- numLayers=1: 总 cpCommTicks = 单层 commTicks ✅
- 不同 numLayers 下 per-layer allGatherBytes 一致 ✅
- cpEfficiency 影响: efficiency 越高通信成本越低 ✅
- 极大 seqLen (128K): 通信量正确 ✅
