---
issue_number: 7
verify_date: 2026-09-04
service_status: ok
---

# Issue #7 服务验证日志

## 服务启动

- HTTP 服务: ✅ ok（HttpService 在 http://localhost:19878 启动）
- SgSimService: ✅ ok（在 http://localhost:3097 启动）
- 启动命令: `npx ts-node src/index.ts --mode=sglang --sglang-port=3097 --http-port=19878`
- 启动无报错

## API 端点验证

### SgSimService 端点（:3097）

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /health | GET | ✅ 200 | `{"ok": true}` |
| /state | GET | ✅ 200 | 含 scheduler + parallel + metrics 字段，数据结构完整 |
| /command | POST | ✅ 200 | step(dt=3) 后 tickCounter=3 |
| /preset | POST | ✅ 200 | fullCombo 预设加载成功，config 含 tpSize=4,dpSize=2,epSize=2,ppSize=2,cpSize=2 |

### HttpService 端点（:19878）

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /sglang.html | GET | ✅ 200 | 前端页面正常返回 |
| /v1/internal/metrics | GET | ✅ 200 | 含 SimulationMetrics JSON（totalRequests, parallel 等字段） |
| / | GET | ✅ 200 | 默认重定向到 /sglang.html |

## 异常信息

无异常。
