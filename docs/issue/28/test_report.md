---
issue_number: 28
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #28 测试报告

## 验收测试结果

### P4 (PPPipelineSimulator) 测试 — 39/39 通过

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | gpipe bubble formula | ✅ pass |
| T2 | gpipe bubble quadratic growth | ✅ pass |
| T3 | 1f1b bubble optimal | ✅ pass |
| T4 | interleaved bubble | ✅ pass |
| T5 | pp_size=1 degeneration | ✅ pass |
| T6 | send/recv communication cost | ✅ pass |
| T7 | isPpLastStage | ✅ pass |
| T8 | intermediate stage does not sample | ✅ pass |
| T9 | last stage samples normally | ✅ pass |
| T10 | sampling_counter unchanged at intermediate stage | ✅ pass |
| T11 | ParallelMetrics backfill | ✅ pass |
| T12 | pp_size=1 metrics all zero | ✅ pass |
| T13 | micro-batch split even | ✅ pass |
| T14 | micro-batch split uneven | ✅ pass |
| T15 | numMicroBatches=1 | ✅ pass |
| T16 | TP×PP correction (R2-5) | ✅ pass |
| T17 | tp=1 degeneration | ✅ pass |
| T18 | CUDA Graph skips PP (R2-3) | ✅ pass |
| T19 | pp_stage_layers | ✅ pass |
| T20 | perStageTicks calculation | ✅ pass |
| T21 | comm overlap mode (R2-2) full overlap | ✅ pass |
| T21b | comm non-overlap mode | ✅ pass |
| T22 | comm partial overlap (R2-2) | ✅ pass |
| T23 | interleaved numChunks configurable (R2-4) | ✅ pass |
| T24 | extreme micro-batch batchSize<numMB (R2-6) | ✅ pass |
| T25 | many micro-batches (R2-6) | ✅ pass |
| T26 | TP hiddenSize non-divisible (R2-5) | ✅ pass |
| B1 | pp_size=0 throws | ✅ pass |
| B2 | numMicroBatches=0 returns all zero | ✅ pass |
| B3 | batchSize=0 sendRecvTicks=0 | ✅ pass |
| B4 | unknown schedule throws | ✅ pass |
| B5 | commGroup=null returns 0 | ✅ pass |
| B6 | bandwidth=0 returns Infinity | ✅ pass |
| B7 | interleaved numChunks=0 bubble=0 | ✅ pass |
| B8 | interleaved numChunks=1 same as 1f1b | ✅ pass |
| E2E-1 | gpipe full flow | ✅ pass |
| E2E-2 | pp_size=1 full flow | ✅ pass |
| E2E-3 | CUDA Graph + PP | ✅ pass |
| E2E-4 | overlap mode comparison | ✅ pass |

### P0 回归测试 — 37/37 通过

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1-T27 | P0 基础功能（通信、拓扑、指标） | ✅ pass |
| B1-B8 | P0 边界条件 | ✅ pass |

## 类型检查

- 结果: pass（仅存在预先的 k1/s0 测试文件 TableManager 引用错误，非本 Issue 引入）

## 失败用例详情

无

## 边界条件覆盖

- pp_size=0 抛异常
- pp_size=1 全零退化
- numMicroBatches=0 返回全零
- batchSize=0 通信成本为 0
- unknown schedule 抛异常
- commGroup=null 返回 0
- bandwidth=0 返回 Infinity
- interleaved numChunks=0 bubble=0
- interleaved numChunks=1 等同 1f1b
