---
issue_number: 7
verify_date: 2026-09-04
service_status: ok
---

# Issue #7 服务验证日志

## 服务启动
- HTTP 服务: ✅ ok (HttpService :8889)
- SGLang Sim API: ✅ ok (SgSimService :3003)
- 启动模式: `--mode=sglang`
- 启动命令: `npx ts-node src/index.ts --mode=sglang --http-port=8889 --sglang-port=3003`

## API 端点验证

| 端点 | 方法 | 状态 | 响应摘要 |
|------|------|------|---------|
| /health | GET | ✅ 200 | `{ ok: true }` |
| /state | GET | ✅ 200 | 含 scheduler/parallel/metrics 字段 |
| /command (step dt=5) | POST | ✅ 200 | tickCounter=5 |
| /command (pause) | POST | ✅ 200 | paused=true |
| /command (reset) | POST | ✅ 200 | tickCounter=0 |
| /preset (fullCombo) | POST | ✅ 200 | tp=4 dp=2 ep=2 pp=2 cp=2 |
| /preset (single) | POST | ✅ 200 | tp=1 dp=1 ep=1 pp=1 cp=1 |
| /params (tpSize=4) | POST | ✅ 200 | config.tpSize=4 |
| /v1/internal/metrics | GET | ✅ 200 | 含 parallel.tpSize 等字段 |
| /sglang.html | GET | ✅ 200 | HTML 含 SGLang/canvas 关键字 |

## 控制端点功能验证

- **step**: 发送 `{action:"step", dt:5}` → tickCounter 正确递增到 5 ✅
- **pause**: 发送 `{action:"pause"}` → paused=true ✅
- **reset**: 发送 `{action:"reset"}` → tickCounter 重置为 0 ✅
- **preset fullCombo**: 正确设置 tp=4,dp=2,ep=2,pp=2,cp=2 ✅
- **params tpSize=4**: 从 single 状态设置 tpSize=4 成功 ✅
- **params 非法配置**: epSize=2 + isMoe=false → 返回 400 ✅

## 前端验证

- `/sglang.html` 页面返回 200，包含完整 HTML/CSS/JS ✅
- 页面包含 canvas 元素（rank 拓扑 + 时间分解图） ✅
- 页面包含 i18n 字典（zh/en） ✅
- 页面包含 6 个场景预设按钮 ✅

## 异常信息
无
