---
title: "Issue #19 解决方案"
issue_number: 19
issue_type: Feature
created: 2026-09-02
updated: 2026-09-02
status: revised
review_round: 2
---

# Issue #19 解决方案

## 需求分析

- **问题描述**：Issue #19 要求实现 S5 阶段的 Overlap Scheduling 完整仿真逻辑与 `SimulationClock` 组件，涵盖 `last_data` 延迟机制、空 tick 刷新策略、`messages` deque 高水位控制，以及与 `SimGraphRunner.isGraphCapture` 的集成。

- **能力目标**：
  1. **SimulationClock（§4.1 / §4.3）**：`current_tick: int` 单调递增时钟；`advance(delta_ticks)` 方法；GPU 任务调度与重叠检测；tick 回调队列（供 `MetricsCollector.tick` 使用）
  2. **SimScheduler._overlap_tick()（§9.4 Overlap Scheduling）**完整升级：
     - **last_data 到达延迟**：上一次 tick 的 resp_token 还没被下游 last_data ack 时，本 tick 不立即新 forward；引入 `overlap_wait_ticks = token_recv_delay_us ÷ tick_us` 松弛窗口
     - **空 tick 刷新**：连续 `idle_count` 个 tick 无消息时，调度 `scheduler_prefill_schedule` 以 flush 等待续接的 chunked prefill
     - **messages deque 长度阈值**：`len > high_watermark` 时暂停 engine forward（仅 push 出 resp），实现背压
  3. **与 SchedulerIOMixin/forward_batch 的集成**：`forward_batch.isGraphCapture` 影响 last_data 延迟（graph_replay 更快，延迟低）
  4. **finished_reqs 防重复释放**（§9.4 L2150-2158）：overlap 模式下跨 tick 重复释放的防御性去重
  5. **单元测试**：短 prompt 不受 overlap 影响；长 chunked prefill 被空 tick 续接；last_data ack 延迟导致 forward 延后 2-3 tick

- **影响范围**：修改 `server/src/sglang/scheduler/index.ts`（SimScheduler._overlapTick 升级 + SimulationClock 类）、`server/src/sglang/types.ts`（新增 Overlap 相关配置字段），以及新增测试文件 `server/src/test/sglang-s5.test.ts`。不修改已有测试代码。

- **依赖 Issue**：
  - #17 S3: SimScheduler normal_tick（已完成）
  - #18 S4: SimGraphRunner graph_replay_cost 提供 isGraphCapture（已完成）

- **阻塞 Issue**：
  - S6: WorkloadGenerator + SimulationMetrics + HTTP

## 改造方案

### 总体思路

在现有 `_overlapTick` 基础上（已实现 Phase 1-5 的基本流程），按照 §9.4 完整规格升级为具备 last_data 延迟、空 tick 刷新、高水位控制的完整 Overlap Scheduling 仿真器，并新增 `SimulationClock` 类（§4.1 / §4.3）用于 GPU 时序追踪。

核心变更分四部分：

1. **SimulationClock 类**：新建独立的时钟类，提供 tick 计数、advance、GPU 任务调度、重叠检测及 tick 回调队列功能
2. **SimScheduler._overlapTick 升级**：增加 last_data 延迟窗口、空闲计数驱动的空 tick 刷新、messages deque 高水位背压机制、finished_reqs 防重复释放
3. **配置扩展**：在 `SimulatorConfig` 中新增 Overlap 调度所需的参数字段
4. **测试覆盖**：新增 `sglang-s5.test.ts` 覆盖所有核心机制

### 评审意见回应（Review Round 1 → Round 2 修订）

| # | 评审意见 | 修订措施 |
|---|----------|----------|
| 1 | **`finished_reqs` 防重复释放缺失**：§9.4 L2150-2158 明确要求集合去重，方案未提及 | ✅ 本次方案新增 `finishedReqs` 防重复释放机制（详见 §2.5），与现有 `SimScheduler.finishedReqs` 字段对齐 |
| 2 | **高水位背压缺乏合理性论证**：超出报告规格，需补充与源码的对应关系及不丢消息保证 | ✅ 补充源码对应说明及消息安全保证（详见 §2.4） |
| 3 | **`currentTick` 来源不明**：`_lastDataAckTick = currentTick + X` 中 currentTick 取自何处 | ✅ 明确 currentTick 由 `_tickCounter` 实例字段维护，每个 tick 末尾递增，与 SimulationClock 解耦（详见 §2.1） |
| 4 | **`_forcePrefillSchedule()` 调度范围**：方案称"仅 prefill 不 decode"，但续接后可能立即转 decode | ✅ 澄清 `_forcePrefillSchedule()` 仅触发 `prefillManager.scheduleNextBatch()`，续接完成的请求通过 `PrefillAdder` 自动加入 `decodeManager`，下一 tick 的 decode 阶段自然调度之（详见 §2.3） |
| 5 | **类型差异 bigint vs int**：报告中 tick 类型为 int | ✅ 修正为 `int`（与 §4.3 L1236-1241 / L1253 一致），避免 bigint JSON 序列化问题（详见 §1） |
| 6 | **tick 回调机制超出报告规格** | ✅ 保留为合理增强，明确标注为方案扩展项（详见 §1） |

### 详细设计

#### 1. SimulationClock 类（§4.1 / §4.3）

**位置**：`server/src/sglang/scheduler/index.ts`（与 SimScheduler 同文件）

**设计**：

```typescript
interface SimEvent {
  tick: number
  eventType: "gpu_start" | "gpu_end" | "cpu_schedule" | "cpu_process"
  duration: number
}

class SimulationClock {
  currentTick: number = 0          // 当前 tick 计数（int 类型，对齐 §4.3）
  gpuBusyUntil: number = 0         // GPU 何时空闲（tick 号）
  events: SimEvent[] = []          // 事件记录列表
  private _tickCallbacks: Array<(tick: number) => void> = []

  advance(deltaTicks: number = 1): void
    // 单调递增：assert deltaTicks > 0, currentTick += deltaTicks
    // 触发所有 tick 回调

  scheduleGpu(durationTicks: number): number
    // 安排 GPU 任务：start = max(currentTick, gpuBusyUntil)
    // gpuBusyUntil = start + durationTicks
    // 记录 SimEvent，返回完成时间

  canOverlap(): boolean
    // 返回 currentTick < gpuBusyUntil（GPU 正忙，可以重叠）

  onTick(callback: (tick: number) => void): () => void
    // 注册 tick 回调，返回取消注册函数
}
```

**关键设计决策**：

- **`currentTick` 使用 `number`（int）类型**（而非 `bigint`），严格对齐技术报告 §4.3 L1236-1241 / L1253 中 `SimEvent.tick` 和 `current_tick` 的 `int` 类型定义。同时避免 `bigint` 在 JSON 序列化时的兼容性问题
- `gpuBusyUntil` 同样为 `number`，与 `currentTick` 类型一致
- `advance` 单调不回退：若 `deltaTicks <= 0` 抛出 `Error`
- tick 回调在 `advance` 内同步触发，供 `MetricsCollector.tick` 注册 — **此为方案扩展项**（报告 §4.3 未定义 `advance` 触发回调），标注为合理增强，为 S6 MetricsCollector 预留接口，不影响核心正确性
- `SimulationClock` 是可选工具，默认不在 SimScheduler 中实例化；需要时通过配置或构造器选项启用 — 严格符合 §4.3 "默认不实例化，不影响仿真正确性"

#### 2. SimScheduler._overlapTick 升级

##### 2.1 新增实例字段

| 字段 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `_tickCounter` | `number` | `0` | 当前 tick 计数器，每个 tick 末尾递增，作为 `currentTick` 的来源 |
| `_overlapWaitTicks` | `number` | `config.tokenRecvDelayTicks` | graph_replay 路径下 last_data 延迟等待窗口 |
| `_eagerExtraDelayTicks` | `number` | `config.eagerForwardExtraDelayTicks` | eager forward 路径额外延迟 |
| `_idleCounter` | `number` | `0` | 连续空闲 tick 计数 |
| `_lastDataPending` | `boolean` | `false` | last_data 是否挂起等待 ack |
| `_lastDataAckTick` | `number` | `0` | last_data 可被处理的最早 tick |
| `_clock` | `SimulationClock \| null` | `null` | 可选时钟实例 |
| `_highWatermark` | `number` | `config.messagesHighWatermark` | messages deque 高水位阈值 |

**`currentTick` 的来源说明**（回应评审意见 #3）：

- `_tickCounter` 是 SimScheduler 内部维护的独立字段，不依赖 `SimulationClock`
- 每个 tick 的末尾（Phase 5 之前）执行 `_tickCounter += 1`
- `_lastDataAckTick = _tickCounter + overlapWaitTicks` 直接读取 `_tickCounter`
- `SimulationClock.currentTick` 仅用于 GPU 时序追踪，与 `_tickCounter` 解耦
- 这样设计的原因：`SimulationClock` 是可选的（默认 null），核心调度逻辑不应依赖可选组件

##### 2.2 last_data 到达延迟

**问题**：当前 `_overlapTick` 在 Phase 3 无条件处理 `_lastOverlapData`，但实际上游 ack 可能还未到达。在真实 SGLang 中，`resp_token` 从 GPU 拷贝到 CPU 需要 `token_recv_delay_us` 时间，仿真中用 `overlap_wait_ticks` 表示。

**设计**：

- 当 `_lastOverlapData` 被保存时（Phase 3.5），计算 `_lastDataAckTick = _tickCounter + overlapWaitTicks`
- `overlapWaitTicks` 受 `isGraphCapture` 影响：
  - `isGraphCapture = true`（CUDA Graph replay）→ `overlapWaitTicks = config.tokenRecvDelayTicks`（默认 0，graph_replay 更快）
  - `isGraphCapture = false`（eager forward）→ `overlapWaitTicks = config.tokenRecvDelayTicks + config.eagerForwardExtraDelayTicks`（eager 路径额外延迟）
- Phase 3 中仅当 `_tickCounter >= _lastDataAckTick` 时才处理 `_lastOverlapData`
- 当 `tokenRecvDelayTicks = 0` 且 `isGraphCapture = true` 时，`_lastDataAckTick = _tickCounter`，即同 tick 处理，行为与修订前一致

**流程变更**：

```
_overlapTick(incoming):
  Phase 1: 消息处理 + 调度
    for msg in incoming: _processOneMsg(msg)
    if _incomingQueue.length > _highWatermark:
      // 背压：跳过本次调度，仅处理 last_data
      forwardInput = null
    else:
      forwardInput = _scheduleNextBatch()

  Phase 2: forward 当前批（若调度成功）
    forwardOutput = forwardInput ? _forward(forwardInput) : null

  Phase 3: 处理上一批结果（带延迟检查）
    if _lastOverlapData !== null AND _tickCounter >= _lastDataAckTick:
      replies = _processLastData(_lastOverlapData)
      _lastOverlapData = null
      _lastDataPending = false
    else:
      replies = []

  Phase 3.5: 保存当前批数据（带延迟计算）
    if forwardOutput !== null AND forwardInput !== null:
      _lastOverlapData = { forwardInput, forwardOutput }
      isGraphCapture = forwardOutput.isGraphCapture ?? false
      if isGraphCapture:
        _lastDataAckTick = _tickCounter + config.tokenRecvDelayTicks
      else:
        _lastDataAckTick = _tickCounter + config.tokenRecvDelayTicks + config.eagerForwardExtraDelayTicks
      _lastDataPending = true

  Phase 4: 空闲 tick 刷新检查
    if incoming.length === 0 AND forwardInput === null:
      _idleCounter += 1
      if _idleCounter >= config.idleCountForFlush:
        _forcePrefillSchedule()
        _idleCounter = 0
    else:
      _idleCounter = 0

  Phase 4.5: 递增 tickCounter
    _tickCounter += 1

  Phase 5: EPLB + globalStep
```

##### 2.3 空 tick 刷新

**问题**：chunked prefill 请求在上一 tick 消耗了 token budget 后剩余部分被放回 `prefillManager.pendingList` 头部。如果后续若干 tick 都没有新请求进来，调度器不会主动触发 prefill 调度，导致 chunked 请求"饿死"。

**设计**：

- 引入 `_idleCounter`：连续空闲 tick 计数器
- 当 `_idleCounter >= idleCountForFlush`（默认值 2）时，调用 `_forcePrefillSchedule()`
- `_forcePrefillSchedule()` 仅调用 `this.prefillManager.scheduleNextBatch(this.prefillBudget)` 并对结果执行 `_prepareBatch` + `_forward`，**不单独调度 decode batch**
- 每次有新消息或成功调度后重置 `_idleCounter = 0`

**`_forcePrefillSchedule()` 的调度范围澄清**（回应评审意见 #4）：

- `_forcePrefillSchedule()` 的目的是续接 pending 的 chunked 请求，因此仅触发 prefill 调度路径
- 当一个 chunked 请求的最后一段在 `_forcePrefillSchedule()` 中被调度时，`PrefillAdder.tryAddOne()` 会将其转为完整 `Req` 并调用 `decodeManager.addReq(req)` — 这是 `PrefillAdder` 内部的标准行为
- 该请求将在**下一个 tick** 的 `_scheduleNextBatch()` 中通过 `decodeManager.scheduleNextBatch()` 被调度 decode，而不是在当前 `_forcePrefillSchedule()` 中立即 decode
- 这样设计保证了调度流程的一致性：prefill 和 decode 始终在不同阶段处理，避免在 flush 路径中混合调度逻辑

##### 2.4 messages deque 高水位控制

**问题**：当外部请求速率远超调度器处理能力时，`_incomingQueue` 无限增长可能导致内存问题。

**源码对应说明**（回应评审意见 #2a）：

真实 SGLang 中虽然没有直接的 `_incomingQueue` 高水位背压机制（ZMQ 队列由系统管理），但存在等效的流控策略：

1. **TokenizerManager 端限制**：`max_running_requests` 限制并发请求数，超出的请求在 API Server 层排队等待（HTTP 连接保持），相当于隐式背压
2. **Scheduler 端拒绝**：`PrefillAdder` 中 `available_size` 检查失败时返回 `null`，请求留在 `pending_list` 中等待，等效于暂停入队
3. **GPU 显存约束**：`CacheManager.availableSize` 不足时自然阻止新请求进入

仿真中的 `_highWatermark` 机制是对上述多个隐式约束的简化建模：当 `_incomingQueue` 积压超过阈值时，表明调度器处理速度跟不上输入速率，此时跳过新 forward 以让已有结果先完成，等效于 SGLang 中 GPU 显存/调度资源受限时的自然背压。

**消息安全保证**（回应评审意见 #2b）：

- **不丢消息**：背压仅影响 `_scheduleNextBatch()` 的调用，消息入队不受影响 — `_processOneMsg()` 照常将请求加入 `prefillManager.pendingList`，不会丢弃
- **不阻塞入队**：`_incomingQueue.push()` 在 `_processOneMsg()` 内执行，不受高水位检查影响（高水位检查发生在消息处理之后）
- **进度保证**：即使在高水位期间，Phase 3 的 `_processLastData` 仍然执行，finished 请求会被释放，`_incomingQueue` 中的请求会在水位回落后被调度
- **无死锁风险**：`_incomingQueue.length` 只在 `_processOneMsg` 后增加、在 `_scheduleNextBatch` 成功后减少（请求从 pending 移入 running），背压期间请求停留在 pending 中等待

##### 2.5 finished_reqs 防重复释放（§9.4 L2150-2158）

**问题**（回应评审意见 #1）：在 overlap 模式中，`_processLastData` 延迟到下一 tick 执行。如果同一请求在上一 tick 的 `_processLastData` 中已被标记 finished 并释放资源，理论上不应再被处理。但作为防御性编程，§9.4 L2150-2158 要求使用 `finished_reqs` 集合去重，防止跨 tick 重复释放。

**现有代码状态**：当前 `SimScheduler` 已有 `finishedReqs: Set<Req>` 字段，在 `_processLastData` 中使用：

```typescript
// 现有代码 (scheduler/index.ts L1003-1008)
if (isFinished && !this.finishedReqs.has(req)) {
  req.finished = true;
  req.finishReason = isEos ? "eos" : "length";
  this.decodeManager.removeReq(req);
  this._freeReqResources(req);
  newFinishedReqs.add(req);
}
// ...
this.finishedReqs = newFinishedReqs;
```

**修订方案**：无需修改 `finishedReqs` 的核心逻辑，现有实现已与 §9.4 L2150-2158 对齐。但需确保在新增的 `_forcePrefillSchedule()` 路径中，`_processLastData` 同样受 `finishedReqs` 保护。具体措施：

1. `_forcePrefillSchedule()` 执行的 forward 结果保存到 `_lastOverlapData`，由下一 tick 的 Phase 3 处理，天然受 `finishedReqs` 保护
2. `_forcePrefillSchedule()` 不直接调用 `_processLastData`，避免绕过防护
3. 在测试中新增 `finished_reqs` 防重复释放的专项测试用例（测试用例 #11）

##### 2.6 SimulationClock 与 SimScheduler 的集成

**方式**：`SimScheduler` 构造器新增可选 `clock` 选项：

```typescript
constructor(config: SimulatorConfig, opts?: {
  ...,
  clock?: SimulationClock
})
```

- 若传入 `clock`，则 `this._clock = clock`
- 若 `config.enableOverlap && config.enableMetrics` 且未传入 `clock`，自动创建 `SimulationClock` 实例
- 其他情况 `this._clock = null`

**集成点**：
- `_overlapTick` Phase 2 后调用 `this._clock?.scheduleGpu(gpuDuration)` 记录 GPU 占用
- `_overlapTick` Phase 5 末尾调用 `this._clock?.advance(1)` 推进时钟并触发回调
- `_normalTick` 末尾同样调用 `this._clock?.advance(1)`
- `MetricsCollector`（S6 实现）可通过 `clock.onTick(callback)` 注册收集逻辑

#### 3. 配置扩展（SimulatorConfig）

在 `SimulatorConfig` interface 和 `DEFAULT_SIMULATOR_CONFIG` 中新增以下字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokenRecvDelayTicks` | `number` | `0` | graph_replay 路径下 last_data 延迟（默认 0 表示 graph_replay 极快，同 tick 处理） |
| `eagerForwardExtraDelayTicks` | `number` | `2` | eager forward 路径相比 graph_replay 的额外延迟 ticks |
| `idleCountForFlush` | `number` | `2` | 连续空闲 tick 数达到此值时触发 prefill flush |
| `messagesHighWatermark` | `number` | `1024` | 消息队列高水位阈值（0 表示禁用背压） |

### 修改点清单

1. **`server/src/sglang/scheduler/index.ts`**
   - 新增 `SimulationClock` 类（§4.1 / §4.3）：`currentTick`, `gpuBusyUntil`, `events`, `advance()`, `scheduleGpu()`, `canOverlap()`, `onTick()`
   - 新增 `SimEvent` interface
   - `SimScheduler` 新增字段：`_tickCounter`, `_overlapWaitTicks`, `_eagerExtraDelayTicks`, `_idleCounter`, `_lastDataPending`, `_lastDataAckTick`, `_clock`, `_highWatermark`
   - `SimScheduler` 新增构造器选项：`clock`
   - 重写 `_overlapTick()`：增加 last_data 延迟检查、空闲 tick 刷新、高水位背压、tickCounter 递增
   - 新增 `_forcePrefillSchedule()` 私有方法
   - `_normalTick()` 末尾增加 `this._clock?.advance(1)`
   - 导出 `SimulationClock` 和 `SimEvent`

2. **`server/src/sglang/types.ts`**
   - `SimulatorConfig` interface 新增 4 个字段：`tokenRecvDelayTicks`, `eagerForwardExtraDelayTicks`, `idleCountForFlush`, `messagesHighWatermark`
   - `DEFAULT_SIMULATOR_CONFIG` 新增对应默认值

3. **`server/src/sglang/index.ts`**
   - Re-export `SimulationClock` 和 `SimEvent`

4. **`server/src/test/sglang-s5.test.ts`**（新增）
   - 测试 SimulationClock 的基本功能
   - 测试短 prompt 在 overlap 模式下正常完成
   - 测试长 chunked prefill 被空 tick 续接
   - 测试 last_data ack 延迟导致 forward 延后
   - 测试高水位背压机制
   - 测试 finished_reqs 防重复释放

## 测试设计

### 验收测试用例清单

| # | 测试名称 | 验证内容 |
|---|----------|----------|
| 1 | `test_simulation_clock_advance` | `advance(1)` 递增 currentTick；`advance(0)` 抛出 Error；单调不回退 |
| 2 | `test_simulation_clock_schedule_gpu` | `scheduleGpu(5)` 返回 `currentTick + 5`；连续调度正确更新 `gpuBusyUntil` |
| 3 | `test_simulation_clock_can_overlap` | GPU 繁忙时 `canOverlap()=true`；GPU 空闲时 `canOverlap()=false` |
| 4 | `test_simulation_clock_on_tick_callback` | 注册回调后，`advance()` 触发回调并传入当前 tick；取消注册后不再触发 |
| 5 | `test_overlap_short_prompt_normal` | 短 prompt（input_len < prefill_budget）在 overlap 模式下正常 prefill→decode→finish，不受延迟窗口影响 |
| 6 | `test_overlap_chunked_prefill_idle_flush` | 长 prompt 触发 chunked prefill，空闲 2 tick 后自动续接，最终完成 |
| 7 | `test_overlap_last_data_delay_graph_capture` | graph_replay 路径下 `tokenRecvDelayTicks=1`，last_data 延迟 1 tick 后才被处理 |
| 8 | `test_overlap_last_data_delay_eager` | eager 路径下 `eagerForwardExtraDelayTicks=2` 且 `tokenRecvDelayTicks=1`，last_data 延迟 1+2=3 tick 后才被处理 |
| 9 | `test_overlap_empty_tick_flush_last_data` | overlap 模式结束后，调用空 tick `runTick([])` 刷新残留 last_data |
| 10 | `test_overlap_high_watermark_backpressure` | 大量请求超过 highWatermark 时，跳过 forward 仅处理 last_data |
| 11 | `test_overlap_finished_reqs_dedup` | 验证 `finishedReqs` 集合在 overlap 模式下防止同一请求被重复释放资源 |

### 边界条件覆盖

| 边界条件 | 测试覆盖 |
|----------|----------|
| `tokenRecvDelayTicks = 0` 且 `isGraphCapture = true` → last_data 无延迟，同 tick 处理 | 测试 5 |
| `idleCountForFlush = 0` → 每个 tick 都尝试 flush | 测试 6（变体） |
| 无 last_data 时空 tick 不报错 | 测试 9 |
| `highWatermark = 0` → 始终背压 | 测试 10（变体） |
| SimulationClock `advance(0)` → 抛出 Error | 测试 1 |
| SimulationClock `scheduleGpu(0)` → start == finish，无实际 GPU 占用 | 测试 2（变体） |
| `finishedReqs` 中已存在的请求不再被 `_freeReqResources` 释放 | 测试 11 |
| 背压期间 `_processLastData` 仍正常执行，finished 请求被释放使水位回落 | 测试 10 |

## 风险与注意事项

- **兼容性影响**：`_overlapTick` 的行为变更可能影响已有测试中 overlap 模式的时序预期。由于当前项目尚无 `_overlapTick` 的专项单元测试，风险可控。需要确保 `_normalTick` 路径不受任何影响。
- **性能影响**：`SimulationClock` 使用 `number` 运算（非 bigint），无额外性能开销。SimulationClock 是可选的（默认不实例化），且仿真器本身不是高性能路径，影响可忽略。
- **回滚方案**：所有变更通过配置字段控制（`tokenRecvDelayTicks=0`, `eagerForwardExtraDelayTicks=0`, `idleCountForFlush=Infinity`, `messagesHighWatermark=0`），恢复默认值即可回退到当前行为。`SimulationClock` 为可选组件，不影响正确性。
- **finished_reqs 防御性编程**：现有 `finishedReqs` 机制已在 `_processLastData` 中实现，S5 新增的 `_forcePrefillSchedule()` 路径通过 `_lastOverlapData` 间接调用 `_processLastData`，天然受保护。
