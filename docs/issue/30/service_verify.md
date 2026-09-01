---
issue_number: 30
verify_date: 2026-09-01
service_status: ok
---

# Issue #30 服务验证日志

## 服务启动

- HTTP 服务: ✅ ok
  - HttpService 在端口 19877 启动成功
  - 静态文件服务正常
  - API 代理配置正常 (→ localhost:3001)

## API 端点验证

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /api/internal/metrics | GET | ✅ 503 | 未注入 metrics 时返回 { error: "SimulationMetrics not available" } |
| /api/internal/metrics (注入 metrics) | GET | ✅ 200 | parallel 指标 JSON（含 tpCommTicks=386, ppBubbleTicks=10, ppSendRecvTicks=2, tpSize=2, ppSize=2, worldSize=4） |

## 验证详情

### T1: 未注入 metrics 时返回 503
- ✅ 新建 HttpService 实例（未调用 setSimulationMetrics）
- ✅ GET /api/internal/metrics 返回 503 状态码
- ✅ 响应体: { error: "SimulationMetrics not available" }

### T2: 注入 metrics 后返回 200
- ✅ MockEngine 创建 (tpSize=2, ppSize=2) 并执行 forwardBatchSeqLen(128)
- ✅ HttpService.setSimulationMetrics(engine.metrics) 注入成功
- ✅ GET /api/internal/metrics 返回 200 状态码
- ✅ 响应体含 parallel 对象

### T3: parallel 对象指标验证
- ✅ parallel.tpCommTicks = 386 (number)
- ✅ parallel.ppBubbleTicks = 10 (number)
- ✅ parallel.ppSendRecvTicks = 2 (number)
- ✅ parallel.tpSize = 2 (number)
- ✅ parallel.ppSize = 2 (number)
- ✅ parallel.worldSize = 4 (number)

### T4: 全部并行维度指标字段存在
- ✅ tpCommTicks, dpAttnCommTicks, epCommTicks 存在
- ✅ ppBubbleTicks, cpCommTicks, ppSendRecvTicks 存在
- ✅ worldSize, tpSize, dpSize, epSize, ppSize, cpSize 存在

## 异常信息

无异常
