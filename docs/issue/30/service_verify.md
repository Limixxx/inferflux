# Issue #30 — 服务验证日志

**日期**: 2026-09-02
**测试命令**: `npx ts-node src/test/verify-metrics-http.test.ts`
**测试结果**: ALL PASS ✓

## 验证步骤

### T1: 未注入 metrics 时返回 503
- 启动未设置 metrics 的 HttpService
- 请求 `/api/internal/metrics` → 返回 503 ✓

### T2: 注入 metrics 后返回 200
- 创建 MockEngine(tp=2, pp=2)，执行 forwardBatchSeqLen(128)
- 注入 simulationMetrics 到 HttpService
- 请求 `/api/internal/metrics` → 返回 200 ✓

### T3: parallel 对象包含并行指标
| 字段 | 期望类型 | 实际值 |
|------|----------|--------|
| tpCommTicks | number | 386 |
| ppBubbleTicks | number | 10 |
| ppSendRecvTicks | number | 2 |
| tpSize | number | 2 |
| ppSize | number | 2 |
| worldSize | number | 4 |

全部匹配 ✓

### T4: parallel 对象包含全部并行维度指标字段
检查字段: tpCommTicks, dpAttnCommTicks, epCommTicks, ppBubbleTicks, cpCommTicks, worldSize, tpSize, dpSize, epSize, ppSize, cpSize — 全部存在 ✓
