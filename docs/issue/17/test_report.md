# Issue #17 — S3 Test Report

## 测试环境

- **日期**: 2026-08-31
- **Node**: v20+ (Windows PowerShell)
- **TypeScript**: 5.9.3 (strict mode)
- **运行方式**: `npx ts-node src/test/sglang-s3.test.ts`

## S3 验收测试结果: 51/51 PASSED

| 类别 | 用例数 | 通过 | 失败 |
|------|--------|------|------|
| T1-T18: MockEngine/MockSampler/MockAttnBackend/MockEvent | 18 | 18 | 0 |
| T19-T20: SchedulerIOMixin | 2 | 2 | 0 |
| T21-T40: SimScheduler | 20 | 20 | 0 |
| B1-B11: 边界条件 | 11 | 11 | 0 |
| **合计** | **51** | **51** | **0** |

### 详细结果

```
  ✓ T1: MockEvent 构造与 synchronize
  ✓ T2: MockSampler 构造
  ✓ T3: MockSampler.prepare - greedy batch
  ✓ T4: MockSampler.prepare - mixed batch
  ✓ T5: MockSampler.sample - greedy 模式
  ✓ T6: MockSampler.sample - random 模式
  ✓ T7: MockSampler.sample - fixed 模式
  ✓ T8: MockSampler.apply_temperature
  ✓ T9: MockSampler.apply_top_p_top_k
  ✓ T10: MockSampler.apply_logits_penalty
  ✓ T11: MockAttnBackend.prepare_metadata
  ✓ T12: MockAttnBackend.simulate_kv_recycle
  ✓ T13: MockEngine.forward_batch - prefill batch
  ✓ T14: MockEngine.forward_batch - decode batch
  ✓ T15: MockEngine.forward_batch - ChunkedReq 跳过 completeOne
  ✓ T16: MockEngine.forward_batch - CUDA Graph isGraphCapture
  ✓ T17: MockEngine.forward_batch - copyDoneEvent
  ✓ T18: MockEngine.forward_batch - isChunkPrefill
  ✓ T19: SchedulerIOMixin - offline 模式
  ✓ T20: SchedulerIOMixin - online 模式
  ✓ T21: SimScheduler 构造
  ✓ T22: SimScheduler._normalTick - 空 tick
  ✓ T23: SimScheduler end-to-end - 短 prompt
  ✓ T24: SimScheduler._processOneMsg - req_in
  ✓ T25: SimScheduler._processOneMsg - maxTokens 调整
  ✓ T26: SimScheduler._processOneMsg - ExitMsg
  ✓ T27: SimScheduler._processOneMsg - BatchMsg
  ✓ T28: SimScheduler._processOneMsg - AbortMsg
  ✓ T29: SimScheduler._scheduleNextBatch - prefill 优先
  ✓ T30: SimScheduler._scheduleNextBatch - 仅 decode
  ✓ T31: SimScheduler._processLastData - copyDoneEvent
  ✓ T32: SimScheduler._processLastData - prefill 完成
  ✓ T33: SimScheduler._processLastData - 请求完成
  ✓ T34: SimScheduler._processLastData - EOS 终止
  ✓ T35: SimScheduler._processLastData - ChunkedReq 跳过
  ✓ T36: SimScheduler._processLastData - finishedReqs 更新
  ✓ T37: SimScheduler end-to-end - 完整流程
  ✓ T38: SimScheduler._freeReqResources
  ✓ T39: GraphRunner.padBatch - 使用 dummyReq
  ✓ T40: MockEngine dummyReq 初始化
  ✓ B1: 空 incoming + 空 pending + 空 decode
  ✓ B2: maxNewTokens 被截断为 0
  ✓ B3: 单 token 输入请求
  ✓ B4: ChunkedReq 在 _processLastData 中
  ✓ B5: decode batch 空
  ✓ B6: prefill batch 含混合 ChunkedReq 和 Req
  ✓ B7: greedy 采样 + temperature=0
  ✓ B8: offline 模式 + 正常 runTick 调用
  ✓ B9: ExitMsg 在 BatchMsg 内
  ✓ B10: AbortMsg 目标请求不存在
  ✓ B11: copyDoneEvent.synchronize 多次调用
```

## 回归测试结果

| 测试套件 | 用例数 | 通过 | 失败 |
|----------|--------|------|------|
| S1 (core/entities) | 26 | 26 | 0 |
| S2 (prefill/decode/scheduler) | 48 | 48 | 0 |
| S3 (本次新增) | 51 | 51 | 0 |

**结论**: 无回归，所有已有测试继续通过。

## TypeScript 类型检查

- `npx tsc --noEmit` 对 S3 相关模块（core、engine、scheduler、types、cache、entities、index、sglang-s3.test）无类型错误
- 已有 P1a/P2a/P5/PP 测试文件存在 TS1361 类型错误（type export 被当作值使用），属于历史遗留问题，不在本次 S3 范围内

## 测试修复记录

初次运行有 5 个失败用例，修复如下：

1. **T24/T25/T28**: 测试通过 `runTick(msgs)` 后检查 `prefillManager.pendingList`，但 `runTick` 执行完整 tick（消息处理→调度→forward→结果处理），请求已不在 pendingList 中。修复：改用 `_processOneMsg` 直接调用以隔离测试消息处理逻辑。
2. **T29**: 同理，`runTick(msgs)` 后请求已被调度完毕。修复：通过 `_processOneMsg` 添加请求后直接调用 `_scheduleNextBatch`。
3. **T39**: `GraphRunner.canUseCudaGraph` 使用 `includes(bs)` 仅支持精确匹配，无法支持 padding 对齐。修复：改为 `some(cbs => cbs >= bs)` 以支持 batch size 小于 captured graph size 时的 padding 场景。
