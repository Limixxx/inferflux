---
issue_number: 30
issue_type: Feature
test_date: 2026-09-02
test_result: pass
---

# Issue #30 测试报告

## PR #83 驳回修复

本轮修复 PR #83 驳回意见，核心问题是两个 SimScheduler 类并存，需合并为单一类。

| 偏离项 | 修复描述 | 验证 |
|--------|---------|------|
| (1) SimSchedulerImpl._processOneMsg 仅 req_in/req_resume | 合并后 _processOneMsg 支持 4 种消息类型 (batch/exit/req_in/abort) | ✅ S3 T24-T28, P6 C6-T5~T7 |
| (2) SimSchedulerImpl._forward 不读写 tokenPool | 合并后 _forward 从 tokenPool 读取 input_ids 并写回 next_tokens | ✅ S3 T37 端到端, P6 C6-T6 |
| (3) SimSchedulerImpl._processLastData 无 copyDoneEvent | 合并后 _processLastData 含 copyDoneEvent.synchronize() | ✅ S3 T31, B11 |
| (4) forward_batch 无并行层循环 | forward_batch 改为调用 forwardBatch（含完整并行层循环） | ✅ P6 C1-T2, C2-T2 |
| (5) 两个 SimScheduler 类并存 | 合并为单一 SimScheduler 类 + SimSchedulerImpl 类型别名 | ✅ 类型检查通过, P6 C6-T4~T7 |

## P6 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| C1-T1 | initParallelGroups with all size=1 creates components | ✅ pass |
| C1-T2 | all size=1 forwardBatch total comm ticks = 0 | ✅ pass |
| C1-T3 | topology worldSize=1 for all size=1 | ✅ pass |
| C2-T1 | initParallelGroups creates all 9 components for full parallel config | ✅ pass |
| C2-T2 | MockEngine forwardBatch with full parallel config produces comm ticks | ✅ pass |
| C2-T3 | throughput with parallel config > 1x baseline | ✅ pass |
| C3-T1 | dpAttnSim created when enableDpAttention && useMla | ✅ pass |
| C3-T2 | dpAttnSim is null when enableDpAttention=false | ✅ pass |
| C3-T3 | dpAttnSim creation rejected when useMla=false | ✅ pass |
| C3-T4 | forward with DP-Attn produces dpAttnCommTicks | ✅ pass |
| C4-T1 | 1f1b bubble = (ppSize-1) × microBatchTicks | ✅ pass |
| C4-T2 | gpipe bubble = (ppSize-1) × microBatchTicks × numMicroBatches | ✅ pass |
| C4-T3 | 1f1b bubble ≈ gpipe bubble / numMicroBatches | ✅ pass |
| C5-T1 | Constraint 1: world_size must equal tp*dp*pp | ✅ pass |
| C5-T2 | Constraint 2: ep_size>1 requires isMoe | ✅ pass |
| C5-T3 | Constraint 3: tp_size % cp_size must be 0 | ✅ pass |
| C5-T4 | Constraint 4: (tp/cp) % ep_size must be 0 | ✅ pass |
| C5-T5 | Constraint 5: pp_size must be >= 1 | ✅ pass |
| C5-T6 | Constraint 6: enableDpAttention requires useMla | ✅ pass |
| C5-T7 | Constraint 7: memoryRatio must be in (0,1] | ✅ pass |
| C5-T8 | memoryRatio > 1 violates Constraint 7 | ✅ pass |
| C5-T9 | valid config passes all constraints | ✅ pass |
| C5-T10 | initParallelGroups throws on invalid config | ✅ pass |
| C6-T1 | initParallelGroups exported from sglang index | ✅ pass |
| C6-T2 | ParallelGroups type is usable | ✅ pass |
| C6-T3 | SimulationMetrics.toJSON returns valid object | ✅ pass |
| C6-T4 | SimSchedulerImpl exported | ✅ pass |
| C6-T5 | SimSchedulerImpl can be constructed | ✅ pass |
| C6-T6 | SimSchedulerImpl runTick increments globalStep | ✅ pass |
| C6-T7 | SimSchedulerImpl with ParallelGroups has groups reference | ✅ pass |
| B1 | dpSize=1 dpController.select_rank returns rank 0 | ✅ pass |
| B2 | cpSize=1 cpSim is null | ✅ pass |
| B3 | enableEplb=false eplbSim is null | ✅ pass |
| B4 | isMoe=false moeBackend is null | ✅ pass |
| B5 | numPages=0 DP allocate returns null (OOM) | ✅ pass |
| B6 | large world_size (32) works | ✅ pass |
| B7 | EPLB called at tick end not in forwardBatch | ✅ pass |
| B8 | MockEngine intermediate PP stage returns isIntermediate=true | ✅ pass |

P6 总计: 38 通过, 0 失败

## S3 回归测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1-T12 | MockEvent/MockSampler/MockAttnBackend/MockEngine 基础 | ✅ pass |
| T13-T18 | MockEngine.forward_batch 各场景 | ✅ pass |
| T19-T20 | SchedulerIOMixin | ✅ pass |
| T21-T23 | SimScheduler 构造与基本 tick | ✅ pass |
| T24-T28 | _processOneMsg 4种消息类型 | ✅ pass |
| T29-T30 | _scheduleNextBatch | ✅ pass |
| T31-T36 | _processLastData 各场景 | ✅ pass |
| T37 | 端到端完整流程 | ✅ pass |
| T38-T40 | _freeReqResources / GraphRunner / dummyReq | ✅ pass |
| B1-B12 | 边界条件 | ✅ pass |

S3 总计: 52 通过, 0 失败

## 全量回归测试

| 测试套件 | 用例数 | 结果 |
|----------|--------|------|
| P6 | 38 | ✅ pass |
| S3 | 52 | ✅ pass |
| P5 | 23 | ✅ pass |
| P0 | 37 | ✅ pass |
| K1 | 23 | ✅ pass |

总计: 5 套件, 173 用例, 全部通过

## 类型检查

- 结果: pass
- `npx tsc --noEmit` 零错误

## 代码变更清单

| 文件 | 变更描述 |
|------|---------|
| server/src/sglang/scheduler/index.ts | 合并 SimScheduler(S3) 和 SimSchedulerImpl(P6) 为单一 SimScheduler 类，保留 SimSchedulerImpl const 别名 |
| server/src/sglang/engine/index.ts | forward_batch 改为调用 forwardBatch（含完整并行层循环），补充 S3 时间模型字段 |
| server/src/test/sglang-p6.test.ts | 更新构造器调用适配新签名，添加 cacheType:"naive" 避免 RadixPrefixCache 未实现问题 |

## 边界条件覆盖

- dpSize=1 时 dpController.select_rank 始终返回 rank 0 (B1)
- cpSize=1 时 cpSim 为 null (B2)
- enableEplb=false 时 eplbSim 为 null (B3)
- isMoe=false 时 moeBackend 为 null (B4)
- numPages=0 (OOM) 时 DP 分配返回 null (B5)
- 极大 world_size (32) 正常创建 (B6)
- EPLB 在 tick 末尾而非 forwardBatch 内调用，globalStep 递增验证 (B7)
- 中间 PP stage 返回 isIntermediate=true (B8)
