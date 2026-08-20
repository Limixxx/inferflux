---
issue_number: 1
verify_date: 2026-08-20
service_status: ok
---

# Issue #1 服务验证日志

## 服务启动

- HTTP 服务 (HttpService :8888): ✅ ok
- Sim 服务 (SimService :3001): ✅ ok

启动命令：
```bash
cd server
npx ts-node src/index.ts --sim-port=3001 --http-port=8888
```

启动日志（无报错）：
```
[SimService] listening on http://localhost:3001
[HttpService] serving D:\agents\inferflux\.worktree\issue-1\server\public on http://localhost:8888
[HttpService] API proxy → http://localhost:3001
```

## API 端点验证

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /health | GET | ✅ 200 | `{"ok":true}` |
| /state (pd-disagg) | GET | ✅ 200 | mode=pd-disagg, breakdown=7列, series含16个key |
| /params {mode:agg} | POST | ✅ 200 | 模式切换成功，mode=agg |
| /render (agg) | GET | ✅ 200 | mode=agg, wList=2, pList=0, dList=0, wList[0]含id/kvUsed/waitingQ/running等字段 |
| /state (agg) | GET | ✅ 200 | mode=agg, breakdown=4列, wQueue/kvW gauge 可用 |
| /preset {aggChunkedPrefill} | POST | ✅ 200 | mode=agg, chunkedPrefill=true, chunkSize=8192 |
| /pd-disagg.html | GET | ✅ 200 | 69125 bytes, 含 drawAgg/drawWorker/aggBalanced |

## 关键验证点

### 1. 默认 pd-disagg 模式正常
- `/state` 返回 `mode: "pd-disagg"`，breakdown 为 7 列
- gauges 中 wQueue/kvW 为 0（agg 专用 gauge 在 pd-disagg 模式置零）
- series 包含全部 16 个 key（含新增 wQueue、kvW）

### 2. 切换到 agg 模式正常
- POST `/params` 传入 `{mode:"agg", numWorkers:2, kvGb:99, chunkedPrefill:false}`
- 切换后 `/render` 返回：
  - `mode: "agg"`
  - `wList` 含 2 个 worker 实例
  - `pList` / `dList` 为空数组
  - 每个 worker 含字段：`id, kvUsed, draining, maxTokens, ntr, retractGlow, nextStepAt, waitingQ, running`
- `/state` breakdown 切换为 4 列

### 3. agg 预设加载正常
- POST `/preset {preset:"aggChunkedPrefill"}` 后：
  - `mode: "agg"`
  - `chunkedPrefill: true`
  - `chunkSize: 8192`
  - `numWorkers: 2`

### 4. 前端 HTML 正常服务
- HttpService 返回 `pd-disagg.html`（200, 69125 bytes）
- HTML 内容含新增的 agg 模式代码：`drawAgg`、`drawWorker`、`aggBalanced` 等

## 异常信息

无。服务启动、模式切换、API 响应均正常，未出现任何错误或异常。
