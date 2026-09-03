---
title: "Issue #20 S6 服务验证日志（PR #93 驳回修复后）"
issue_number: 20
date: 2026-09-03
---

# Issue #20 S6 服务验证日志

## 类型检查

**命令**: `npx tsc --noEmit`
**结果**: ✅ 通过（0 错误）

## S6 验收测试

**命令**: `npx ts-node src/test/sglang-s6.test.ts`
**结果**: ✅ 39/39 通过

```
  ✓ test_workload_generator_poisson
  ✓ test_workload_generator_zero_arrival_rate
  ✓ test_workload_generator_zero_requests
  ✓ test_workload_generator_uniform
  ✓ test_workload_generator_trace_replay
  ✓ test_workload_generator_shared_prefix
  ✓ test_workload_generator_no_shared_prefix
  ✓ test_workload_generator_normal_distribution
  ✓ test_workload_generator_sampling_params
  ✓ test_simulation_metrics_record_reply
  ✓ test_simulation_metrics_record_reply_empty
  ✓ test_simulation_metrics_record_batch
  ✓ test_simulation_metrics_record_tick
  ✓ test_simulation_metrics_record_tick_zero
  ✓ test_simulation_metrics_record_tick_no_gpu_work
  ✓ test_simulation_metrics_record_request_latency
  ✓ test_simulation_metrics_record_cache_snapshot
  ✓ test_simulation_metrics_to_json
  ✓ test_simulation_metrics_reset
  ✓ test_simulation_metrics_tick_clock_integration
  ✓ test_sg_http_api_chat_completions
  ✓ test_sg_http_api_not_bound
  ✓ test_sg_http_api_default_max_tokens
  ✓ test_sg_http_api_internal_metrics
  ✓ test_sg_http_api_internal_state
  ✓ test_sg_http_api_internal_state_not_bound
  ✓ test_create_simulator_online
  ✓ test_create_simulator_shutdown_enqueue_safe
  ✓ test_create_simulator_zero_tick_interval
  ✓ test_create_simulator_offline
  ✓ test_create_simulator_offline_no_workload
  ✓ test_create_simulator_enqueue
  ✓ test_create_simulator_get_metrics
  ✓ test_http_service_v1_routes
  ✓ test_e2e_workload_through_scheduler
  ✓ test_simulation_metrics_cuda_graph_counters
  ✓ test_cuda_graph_eager_counters_in_engine
  ✓ test_gpu_busy_uses_forward_output_time
  ✓ test_simulation_metrics_record_request_latency_zero_decode_steps

=== S6 Test Results: 39 passed, 0 failed ===
```

## 回归测试

### S5 测试

**命令**: `npx ts-node src/test/sglang-s5.test.ts`
**结果**: ✅ 23/23 通过

### S4 测试

**命令**: `npx ts-node src/test/sglang-s4.test.ts`
**结果**: ✅ 46/46 通过

### S3 测试

**命令**: `npx ts-node src/test/sglang-s3.test.ts`
**结果**: ✅ 52/52 通过

## HTTP 服务启动验证

**命令**: `npx ts-node src/index.ts --http-port=9876 --sim-port=9877`
**结果**: ✅ 服务启动成功

```
╔══════════════════════════════════════════════════════════════╗
║  PD-Disaggregation Simulator — Server Running                 ║
║                                                                ║
║  Frontend:  http://localhost:9876                          ║
║  Sim API:   http://localhost:9877                          ║
║                                                                ║
║  Press Ctrl+C to stop.                                         ║
╚══════════════════════════════════════════════════════════════╝
```

### 端点验证

| 端点 | 方法 | 状态 | 响应 |
|------|------|------|------|
| /health (SimService :9877) | GET | 200 | `{"ok":true}` |
| /state (SimService :9877) | GET | 200 | 完整仿真状态 JSON |
| /v1/internal/metrics (HttpService :9876) | GET | 503 | `{"error":"SGHttpApi not available"}`（未绑定 simulator，预期行为） |
| /api/internal/metrics (HttpService :9876) | GET | 503 | `{"error":"SimulationMetrics not available"}`（未注入 metrics，预期行为） |

注：/v1/chat/completions 和 /v1/internal/state 代理到 SimService 返回 404，这是因为 SimService 当前版本未实现 /v1/* 路由（这些路由通过 HttpService 直接处理或需要绑定 SGHttpApi 后才有意义）。此为已有行为，非本次修复引入。

## 修改文件清单（PR #93 驳回修复）

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `server/src/sglang/Simulator.ts` | 修改 | 偏离 #1：gpuBusy 改为精确判断（_extractGpuBusy 返回 0/1），每 tick 重置 lastForwardOutput |
| `server/src/sglang/scheduler/index.ts` | 修改 | 偏离 #1：添加 lastForwardOutput 属性；偏离 #2：在 _forward 中同步 CUDA Graph/Eager 计数到 _simMetrics |
| `server/src/sglang/engine/index.ts` | 修改 | 偏离 #2：在 forward_batch 中调用 recordEagerForward/recordCudaGraphReplay |
| `server/src/sglang/workload/index.ts` | 修改 | 偏离 #3：uniform 分支改为 `return index` |
| `server/src/sglang/api/index.ts` | 修改 | 偏离 #4：not_bound 时返回 503 错误对象而非 throw |
| `server/src/test/sglang-s6.test.ts` | 修改 | 更新测试覆盖 4 项修复：uniform 精确断言、503 错误对象、CUDA Graph 计数器、gpuBusy 精确判断 |

## 验证结论

所有验证项目均通过：
- ✅ TypeScript 类型检查无错误
- ✅ S6 验收测试 39/39 通过（含 2 项新增修复验证测试）
- ✅ S3/S4/S5 回归测试全部通过
- ✅ HTTP 服务启动正常
- ✅ /health 端点返回正常
- ✅ 未绑定 simulator 时 /v1/internal/metrics 和 /api/internal/metrics 正确返回 503
