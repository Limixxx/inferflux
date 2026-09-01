---
issue_number: 17
issue_type: Feature
test_date: 2026-09-01
test_result: pass
---

# Issue #17 测试报告

## 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | MockEvent 构造与 synchronize | ✅ pass |
| T2 | MockSampler 构造 | ✅ pass |
| T3 | MockSampler.prepare - greedy batch | ✅ pass |
| T4 | MockSampler.prepare - mixed batch | ✅ pass |
| T5 | MockSampler.sample - greedy 模式 | ✅ pass |
| T6 | MockSampler.sample - random 模式 | ✅ pass |
| T7 | MockSampler.sample - fixed 模式 | ✅ pass |
| T8 | MockSampler.apply_temperature | ✅ pass |
| T9 | MockSampler.apply_top_p_top_k | ✅ pass |
| T10 | MockSampler.apply_logits_penalty | ✅ pass |
| T11 | MockAttnBackend.prepare_metadata | ✅ pass |
| T12 | MockAttnBackend.simulate_kv_recycle | ✅ pass |
| T13 | MockEngine.forward_batch - prefill batch | ✅ pass |
| T14 | MockEngine.forward_batch - decode batch | ✅ pass |
| T15 | MockEngine.forward_batch - ChunkedReq 跳过 completeOne | ✅ pass |
| T16 | MockEngine.forward_batch - CUDA Graph isGraphCapture | ✅ pass |
| T17 | MockEngine.forward_batch - copyDoneEvent | ✅ pass |
| T18 | MockEngine.forward_batch - isChunkPrefill | ✅ pass |
| T19 | SchedulerIOMixin - offline 模式 | ✅ pass |
| T20 | SchedulerIOMixin - online 模式 | ✅ pass |
| T21 | SimScheduler 构造 | ✅ pass |
| T22 | SimScheduler._normalTick - 空 tick | ✅ pass |
| T23 | SimScheduler end-to-end - 短 prompt | ✅ pass |
| T24 | SimScheduler._processOneMsg - req_in | ✅ pass |
| T25 | SimScheduler._processOneMsg - maxTokens 调整 | ✅ pass |
| T26 | SimScheduler._processOneMsg - ExitMsg | ✅ pass |
| T27 | SimScheduler._processOneMsg - BatchMsg | ✅ pass |
| T28 | SimScheduler._processOneMsg - AbortMsg | ✅ pass |
| T29 | SimScheduler._scheduleNextBatch - prefill 优先 | ✅ pass |
| T30 | SimScheduler._scheduleNextBatch - 仅 decode | ✅ pass |
| T31 | SimScheduler._processLastData - copyDoneEvent | ✅ pass |
| T32 | SimScheduler._processLastData - prefill 完成 | ✅ pass |
| T33 | SimScheduler._processLastData - 请求完成 | ✅ pass |
| T34 | SimScheduler._processLastData - EOS 终止 | ✅ pass |
| T35 | SimScheduler._processLastData - ChunkedReq 跳过 | ✅ pass |
| T36 | SimScheduler._processLastData - finishedReqs 更新 | ✅ pass |
| T37 | SimScheduler end-to-end - 完整流程 | ✅ pass |
| T38 | SimScheduler._freeReqResources | ✅ pass |
| T39 | GraphRunner.padBatch - 使用 dummyReq | ✅ pass |
| T40 | MockEngine dummyReq 初始化 | ✅ pass |
| B1 | 空 incoming + 空 pending + 空 decode | ✅ pass |
| B2 | maxNewTokens 被截断为 0 | ✅ pass |
| B3 | 单 token 输入请求 | ✅ pass |
| B4 | ChunkedReq 在 _processLastData 中 | ✅ pass |
| B5 | decode batch 空 | ✅ pass |
| B6 | prefill batch 含混合 ChunkedReq 和 Req | ✅ pass |
| B7 | greedy 采样 + temperature=0 | ✅ pass |
| B8 | offline 模式 + 正常 runTick 调用 | ✅ pass |
| B9 | ExitMsg 在 BatchMsg 内 | ✅ pass |
| B10 | AbortMsg 目标请求不存在 | ✅ pass |
| B11 | copyDoneEvent.synchronize 多次调用 | ✅ pass |
| B12 | ignoreEos=true 时输出 eos token 不终止 | ✅ pass |

## 类型检查
- 结果: pass
- `npx tsc --noEmit` 对 S3 相关模块（core、engine、scheduler、types、cache、entities、sglang-s3.test）零错误
- TS strict zero-any 合规
- 注：`src/test/sglang-p1a/p2a/p5/pp` 存在 TS1361 类型错误（type export 被当作值使用），属于 Issue #17 之前的历史遗留问题，不在本次 S3 范围内

## 失败用例详情（如有）
无。S3 全部 52 个用例通过。

## 边界条件覆盖
- 空输入/空队列场景（B1、B5）
- token 限制边界：maxNewTokens 截断为 0（B2）、单 token 输入（B3）
- ChunkedReq 在调度与结果处理中的跳过逻辑（B4、B6、T15、T35）
- 采样边界：greedy + temperature=0（B7）、fixed 输出模式（T7）
- 消息边界：ExitMsg 在 BatchMsg 内（B9）、AbortMsg 目标不存在（B10）
- 事件边界：copyDoneEvent 多次 synchronize（B11）
- EOS 语义边界：ignoreEos=false 时 EOS 终止（T34）、ignoreEos=true 时输出 eos token 不终止（B12）

## 本轮修复记录（对应上一轮 Code PR #77 驳回意见）

1. **cacheType 硬编码修复**：`SimScheduler` 构造函数中 `new CacheManager(..., "naive")` 改为 `new CacheManager(..., config.cacheType)`，与 `SimulatorConfig.cacheType` 配置对齐（PR #77 评审意见第 1 条）。
2. **EOS 判断字段修复**：`_processLastData` 中 EOS 判断由 `!req.samplingParams.skipSpecialTokens` 改为 `!req.samplingParams.ignoreEos`（PR #77 评审意见第 2 条）。`skipSpecialTokens` 默认值为 `true`，原先逻辑导致默认情况下 `!skipSpecialTokens === false`，EOS 永远无法触发；现新增 `SamplingParams.ignoreEos` 字段（默认 `false`，与 §9.11 L2969 对齐），EOS 在默认配置下正确触发。
3. **测试同步更新**：T34/T36 移除 `skipSpecialTokens: false` 的旧写法（默认 ignoreEos=false 即检测 EOS）；新增 B12 验证 `ignoreEos=true` 时固定输出 eos token 也不提前终止。

## 回归测试结果

| 测试套件 | 用例数 | 通过 | 失败 |
|----------|--------|------|------|
| S1 (core/entities) | 26 | 26 | 0 |
| S2 (prefill/decode/scheduler) | 48 | 48 | 0 |
| S3 (本次新增) | 52 | 52 | 0 |