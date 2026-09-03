---
title: "Issue #20 S6 测试报告（PR #93 驳回修复后）"
issue_number: 20
date: 2026-09-03
---

# Issue #20 S6 测试报告

## 测试环境

- **TypeScript**: 5.9.3
- **Node.js**: v20.x
- **运行命令**: `npx ts-node src/test/sglang-s6.test.ts`
- **类型检查**: `npx tsc --noEmit` 通过

## PR #93 驳回修复项

| # | 偏离项 | 修复内容 |
|---|--------|----------|
| 1 | gpuBusy 粗略估算（恒为 1） | 改为基于 ForwardOutput 精确判断：每 tick 开始重置 lastForwardOutput，根据本 tick 是否有 forward 判断 gpuBusy=0/1 |
| 2 | recordCudaGraphReplay/recordEagerForward 无调用点 | 在 scheduler._forward 中根据 forwardOutput 同步调用 _simMetrics 计数方法 |
| 3 | _sampleArrival uniform 分支与 Poisson 相同 | 改为 `return index`（每 tick 1 个请求按序到达） |
| 4 | SGHttpApi not_bound 时 throw | 改为返回 `{ error: { message, type, code: 503 } }` 错误对象 |

## 测试结果总览

| 指标 | 值 |
|------|------|
| 总测试数 | 39 |
| 通过 | 39 |
| 失败 | 0 |
| 通过率 | 100% |

## 验收测试用例详情

### WorkloadGenerator 测试 (9 项)

| # | 测试名称 | 状态 | 说明 |
|---|----------|------|------|
| 1 | test_workload_generator_poisson | ✅ | Poisson 到达分布生成 10 个请求，arrivalTick 单调递增 |
| 1v | test_workload_generator_zero_arrival_rate | ✅ | arrivalRate=0 时所有请求 arrivalTick=0 |
| 1v | test_workload_generator_zero_requests | ✅ | numRequests=0 返回空数组 |
| 2 | test_workload_generator_uniform | ✅ | 均匀分布到达，arrivalTick === index（修复偏离 #3） |
| 3 | test_workload_generator_trace_replay | ✅ | trace 模式直接返回预定义序列 |
| 4 | test_workload_generator_shared_prefix | ✅ | sharedPrefixRatio=0.3 时 uid%3===0 请求共享前缀 |
| 4v | test_workload_generator_no_shared_prefix | ✅ | sharedPrefixRatio=0 无共享前缀 |
| 5 | test_workload_generator_normal_distribution | ✅ | normal 分布下长度值在 [min, max] 范围内 |
| 6 | test_workload_generator_sampling_params | ✅ | samplingParams.maxNewTokens 等于 outputLen（对齐 §4.4） |

### SimulationMetrics 测试 (14 项)

| # | 测试名称 | 状态 | 说明 |
|---|----------|------|------|
| 7 | test_simulation_metrics_record_reply | ✅ | recordReply 正确递增 completedRequests 和 totalTokensGenerated |
| 7v | test_simulation_metrics_record_reply_empty | ✅ | 空 replies 列表 → recordReply 为 noop |
| 8 | test_simulation_metrics_record_batch | ✅ | recordBatch 正确更新 avgPrefillBatchSize |
| 9 | test_simulation_metrics_record_tick | ✅ | recordTick 正确计算 gpuUtilization、gpuIdleTicks |
| 9v | test_simulation_metrics_record_tick_zero | ✅ | totalTicks=0 → gpuUtilization=0 |
| 9v | test_simulation_metrics_record_tick_no_gpu_work | ✅ | gpuBusy=0 → gpuIdleTicks=totalTicks |
| 10 | test_simulation_metrics_record_request_latency | ✅ | recordRequestLatency 正确记录 TTFT/TBT/E2E |
| 11 | test_simulation_metrics_record_cache_snapshot | ✅ | recordCacheSnapshot 正确更新 cache 指标 |
| 12 | test_simulation_metrics_to_json | ✅ | toJSON 返回包含所有 §4.5 指标字段 + parallel，无 pagesAllocated/pagesFree |
| 13 | test_simulation_metrics_reset | ✅ | reset 清零所有指标字段和 parallel 子结构 |
| 14 | test_simulation_metrics_tick_clock_integration | ✅ | SimulationClock.onTick 回调正确触发 |
| v1 | test_simulation_metrics_cuda_graph_counters | ✅ | CUDA Graph / Eager 计数器正常工作 |
| v2 | test_simulation_metrics_record_request_latency_zero_decode_steps | ✅ | decodeSteps=0 时 TBT 分母为 max(1, 0) |
| 新 | test_cuda_graph_eager_counters_in_engine | ✅ | 通过 scheduler 同步后 CUDA Graph/Eager 计数器更新（修复偏离 #2） |

### SGHttpApi 测试 (6 项)

| # | 测试名称 | 状态 | 说明 |
|---|----------|------|------|
| 15 | test_sg_http_api_chat_completions | ✅ | handleChatCompletions 接受请求并注入调度器 |
| 15v | test_sg_http_api_not_bound | ✅ | SGHttpApi 未 bind 时 → 返回 { error: { code: 503 } }（修复偏离 #4） |
| 15v | test_sg_http_api_default_max_tokens | ✅ | max_tokens 未指定 → 使用默认 128 |
| 16 | test_sg_http_api_internal_metrics | ✅ | handleInternalMetrics 返回 metrics.toJSON() + 调度器快照 |
| 17 | test_sg_http_api_internal_state | ✅ | handleInternalState 返回调度器状态快照 |
| 17v | test_sg_http_api_internal_state_not_bound | ✅ | 未 bind 时 internal state 返回 { error: { code: 503 } } |

### createSimulator / SgSimInstance 测试 (6 项)

| # | 测试名称 | 状态 | 说明 |
|---|----------|------|------|
| 18 | test_create_simulator_online | ✅ | 在线模式 start/shutdown 正常工作 |
| 18v | test_create_simulator_shutdown_enqueue_safe | ✅ | shutdown 后 enqueue 不崩溃 |
| 18v | test_create_simulator_zero_tick_interval | ✅ | tickIntervalMs=0 使用默认 10ms |
| 19 | test_create_simulator_offline | ✅ | 离线模式 start 同步运行所有 tick |
| 19v | test_create_simulator_offline_no_workload | ✅ | 离线模式无 workload → 立即完成 |
| 20 | test_create_simulator_enqueue | ✅ | enqueue 注入请求后 runTick 产出响应 |
| 21 | test_create_simulator_get_metrics | ✅ | getMetrics 返回完整指标快照 |

### 集成与修复验证测试 (3 项)

| # | 测试名称 | 状态 | 说明 |
|---|----------|------|------|
| 22 | test_http_service_v1_routes | ✅ | HttpService /v1/* 路由正确设置 |
| 23 | test_e2e_workload_through_scheduler | ✅ | WorkloadGenerator 生成的请求经完整调度循环后产出正确响应 |
| 新 | test_gpu_busy_uses_forward_output_time | ✅ | gpuUtilization 在 [0,1] 范围内（修复偏离 #1） |

## 边界条件覆盖

| 边界条件 | 测试覆盖 | 结果 |
|----------|----------|------|
| arrivalRate = 0 → 所有请求 arrivalTick = 0 | 测试 1 变体 | ✅ |
| sharedPrefixRatio = 0 → 无共享前缀 | 测试 4 变体 | ✅ |
| numRequests = 0 → 返回空数组 | 测试 1 变体 | ✅ |
| totalTicks = 0 → gpuUtilization = 0 | 测试 9 变体 | ✅ |
| gpuBusy = 0 → gpuIdleTicks = totalTicks | 测试 9 变体 | ✅ |
| 空 replies 列表 → recordReply 为 noop | 测试 7 变体 | ✅ |
| SGHttpApi 未 bind 时 → 返回 503 error 对象 | 测试 15/17 变体 | ✅ |
| max_tokens 未指定 → 使用默认 128 | 测试 15 变体 | ✅ |
| createSimulator shutdown 后 enqueue 不崩溃 | 测试 18 变体 | ✅ |
| tickIntervalMs = 0 → 使用默认 10ms | 测试 18 变体 | ✅ |
| 离线模式无 workload → 立即完成 | 测试 19 变体 | ✅ |
| decodeSteps=0 → TBT 分母为 max(1,0) | 测试 v2 | ✅ |
| CUDA Graph/Eager 计数器在 engine forward 后更新 | 新增测试 | ✅ |
| gpuUtilization ∈ [0,1]（精确时间模型） | 新增测试 | ✅ |

## 回归测试

| 测试套件 | 通过数 | 失败数 |
|----------|--------|--------|
| S3 测试 | 52 | 0 |
| S4 测试 | 46 | 0 |
| S5 测试 | 23 | 0 |
