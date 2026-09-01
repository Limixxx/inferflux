---
issue_number: 30
issue_type: Feature
test_date: 2026-09-01
test_result: pass
---

# Issue #30 测试报告

## 验收测试结果

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

总计: 38 通过, 0 失败

## 全量回归测试

| 测试套件 | 用例数 | 结果 |
|----------|--------|------|
| P6 | 38 | ✅ pass |
| P3a | 37 | ✅ pass |
| P4 | 39 | ✅ pass |
| S1 | 26 | ✅ pass |
| S2 | 48 | ✅ pass |
| P5 | 23 | ✅ pass |
| P0 | 37 | ✅ pass |
| K3 | 35 | ✅ pass |
| P1a | 25 | ✅ pass |
| P1b | 32 | ✅ pass |
| P2a | 24 | ✅ pass |
| P2b | 16 | ✅ pass |
| P3b | 25 | ✅ pass |
| S0 | 22 | ✅ pass |
| K1 | 23 | ✅ pass |
| K2 | 31 | ✅ pass |
| K4 | 41 | ✅ pass |
| K5 | 20 | ✅ pass |

总计: 18 套件, 537 用例, 全部通过

## 类型检查

- 结果: pass
- `npx tsc --noEmit` 零错误

## PR #79 驳回修复确认

| 偏离项 | 修复描述 | 验证 |
|--------|---------|------|
| (1) SimSchedulerImpl 核心调度为桩 | 实现完整调度循环: _processOneMsg → _scheduleNextBatch → _forward → _processLastData | ✅ C6-T5~T7, B7 |
| (2) 旧方法删除破坏兼容 | MockEngine 保留 forwardBatchReq/forwardBatchSeqLen 向后兼容方法 | ✅ B8 |
| (3) 缺少 S3 组件 | SamplingParams.ignoreEos 属性; _overlap_tick 参数类型修正 | ✅ 类型检查通过 |
| (4) GraphRunner 缺 dummyReq/padBatch + includes→some | canUseCudaGraph 使用 some(cbs => cbs >= bs); 添加 dummyReq 字段和 padBatch 方法 | ✅ 类型检查通过 |

## 边界条件覆盖

- dpSize=1 时 dpController.select_rank 始终返回 rank 0 (B1)
- cpSize=1 时 cpSim 为 null (B2)
- enableEplb=false 时 eplbSim 为 null (B3)
- isMoe=false 时 moeBackend 为 null (B4)
- numPages=0 (OOM) 时 DP 分配返回 null (B5)
- 极大 world_size (32) 正常创建 (B6)
- EPLB 在 tick 末尾而非 forwardBatch 内调用，globalStep 递增验证 (B7)
- 中间 PP stage 返回 isIntermediate=true (B8)
- TPCommInfra ZMQ 广播在层循环前调用（tpSize>1 时 zmqBroadcastTicks>0）
- TPCommInfra CPU barrier 在层循环后调用（tpSize>1 时 barrierTicks>0）
