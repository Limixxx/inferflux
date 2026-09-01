---
issue_number: 17
verify_date: 2026-09-01
service_status: ok
---

# Issue #17 服务验证日志

## 服务启动
- HTTP 服务: ✅ ok（http://localhost:8888，静态文件 + API 代理）
- SimService: ✅ ok（http://localhost:3001，仿真引擎 REST API）
- 启动命令：`npx ts-node src/index.ts`（无报错，无异常日志）

## API 端点验证

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /health | GET | ✅ 200 | `{"ok":true}` |
| /state | GET | ✅ 200 | 完整仿真状态 JSON（params/gauges/snapshot/series 均正常返回） |

## 验证要点
- SimService 与 HttpService 均成功监听对应端口，无冲突
- `/health` 健康检查返回 `{"ok":true}`，证明服务进程正常
- `/state` 返回完整仿真状态：仿真时钟 now=23088，params 完整（qps/inputLenMean/outputLenMean/cacheHitRate/mode 等），gauges 正常统计（running=27、inflight=27），snapshot 指标完整（ttft/tpot/e2e 均有数据），series 时间序列正常
- API 代理链路（HttpService → SimService :3001）运行正常

## 异常信息
无。服务启动与端点访问全程无报错、无异常。

## 结论
服务可正常启动，核心 API 端点返回正常数据，端到端服务链路验证通过。