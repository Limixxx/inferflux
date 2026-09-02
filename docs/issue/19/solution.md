---
title: "Issue #19 解决方案"
issue_number: 19
issue_type: Feature
created: 2026-09-02
updated: 2026-09-02
status: draft
review_round: 1
---

# Issue #19 解决方案

## 需求分析

- **问题描述**：Issue #19 要求实现 S5 阶段的 Overlap Scheduling 完整仿真逻辑与 `SimulationClock` 组件，涵盖 `last_data` 延迟机制、空 tick 刷新策略、`messages` deque 高水位控制，以及与 `SimGraphRunner.isGraphCapture` 的集成。

- **能力目标**：
  1. **SimulationClock（§4.1）**：`current_ticks: bigint` 单调递增时钟；`advance(delta_ticks)` 方法；tick 回调队列（供 `MetricsCollector.tick` 使用）
  2. **SimScheduler._overlap_tick()（§9.4 Overlap Scheduling）**完整升级：
     - **last_data 到达延迟**：上一次 tick 的 resp_token 还没被下游 last_data ack 时，本 tick 不立即新 forward；引入 `overlap_wait_ticks = token_recv_delay_us ÷ tick_us` 松弛窗口
     - **空 tick 刷新**：连续 `idle_count` 个 tick 无消息时，调度 `scheduler_prefill_schedule` 以 flush 等待续接的 chunked prefill
     - **messages deque 长度阈值**：`len > high_watermark` 时暂停 engine forward（仅 push 出 resp），实现背压
  3. **与 SchedulerIOMixin/forward_batch 的集成**：`forward_batch.isGraphCapture` 影响 last_data 延迟（graph_replay 更快，延迟低）
  4. **单元测试**：短 prompt 不受 overlap 影响；长 chunked prefill 被空 tick 续接；last_data ack 延迟导致 forward 延后 2-3 tick

- **影响范围**：修改 `server/src/sglang/scheduler/index.ts`（SimScheduler._overlapTick 升级 + SimulationClock 类）、`server/src/sglang/types.ts`（新增 Overlap 相关配置字段），以及新增测试文件 `server/src/test/sglang-s5.test.ts`。不修改已有测试代码。

- **依赖 Issue**：
  - #17 S3: SimScheduler normal_tick（已完成）
  - #18 S4: SimGraphRunner graph_replay_cost 提供 isGraphCapture（已完成）

- **阻塞 Issue**：
  - S6: WorkloadGenerator + SimulationMetrics + HTTP

## 改造方案

### 总体思路

在现有 `_overlapTick` 基础上（已实现 Phase 1-5 的基本流程），按照 §9.4 完整规格升级为具备 last_data 延迟、空 tick 刷新、高水位控制的完整 Overlap Scheduling 仿真器，并新增 `SimulationClock` 类（§4.1）用于 GPU 时序追踪。

核心变更分三部分：

1. **SimulationClock 类**：新建独立的时钟类，提供 tick 计数、advance、GPU 任务调度、重叠检测及 tick 回调队列功能
2. **SimScheduler._overlapTick 升级**：增加 last_data 延迟窗口、空闲计数驱动的空 tick 刷新、messages deque 高水位背压机制
3. **配置扩展**：在 `SimulatorConfig` 中新增 Overlap 调度所需的参数字段

### 详细设计

#### 1. SimulationClock 类（§4.1）

**位置**：`server/src/sglang/scheduler/index.ts`（与 SimScheduler 同文件），或独立文件 `server/src/sglang/clock.ts` 后 re-export

**设计**：

```
class SimulationClock {
  currentTicks: bigint          // 当前 tick 计数，从 0 开始
  gpuBusyUntil: bigint          // GPU 何时空闲（tick 号）
  events: SimEvent[]            // 事件记录列表
  private _tickCallbacks: Array<(tick: bigint) => void>  // tick 回调队列

  advance(deltaTicks: bigint = 1n): void
    // 单调递增：assert deltaTicks > 0, currentTicks += deltaTicks
    // 触发所有 tick 回调

  scheduleGpu(durationTicks: number): bigint
    // 安排 GPU 任务：start = max(currentTicks, gpuBusyUntil)
    // gpuBusyUntil = start + BigInt(durationTicks)
    // 记录 SimEvent，返回完成时间

  canOverlap(): boolean
    // 返回 currentTicks < gpuBusyUntil（GPU 正忙，可以重叠）

  onTick(callback: (tick: bigint) => void): () => void
    // 注册 tick 回调，返回取消注册函数
}
```

**关键设计决策**：
- `currentTicks` 使用 `bigint` 类型（而非 `number`），对齐 Issue 描述中的 `current_ticks: bigint`，避免长时间仿真中溢出
- `gpuBusyUntil` 同样为 `bigint`，与 `currentTicks` 类型一致
- `advance` 单调不回退：若 `deltaTicks <= 0` 抛出 `Error`
- tick 回调在 `advance` 内同步触发，供 `MetricsCollector.tick` 注册
- `SimulationClock` 是可选工具，默认不在 SimScheduler 中实例化；需要时通过配置或构造器选项启用

**SimEvent 数据结构**：

```
interface SimEvent {
  tick: bigint
  eventType: string     // "gpu_start" | "gpu_end" | "cpu_schedule" | "cpu_process"
  duration: number      // 持续 ticks 数
}
```

#### 2. SimScheduler._overlapTick 升级

##### 2.1 新增实例字段

| 字段 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `_overlapWaitTicks` | `number` | `config.tokenRecvDelayTicks` | last_data 延迟等待窗口 |
| `_overlapWaitCounter` | `number` | `0` | 当前等待计数器 |
| `_idleCounter` | `number` | `0` | 连续空闲 tick 计数 |
| `_lastDataPending` | `boolean` | `false` | last_data 是否挂起等待 ack |
| `_lastDataAckTick` | `bigint` | `0n` | last_data 可被处理的最早 tick |
| `_clock` | `SimulationClock \| null` | `null` | 可选时钟实例 |
| `_highWatermark` | `number` | `config.messagesHighWatermark` | messages deque 高水位阈值 |

##### 2.2 last_data 到达延迟

**问题**：当前 `_overlapTick` 在 Phase 3 无条件处理 `_lastOverlapData`，但实际上游 ack 可能还未到达。在真实 SGLang 中，`resp_token` 从 GPU 拷贝到 CPU 需要 `token_recv_delay_us` 时间，仿真中用 `overlap_wait_ticks` 表示。

**设计**：

- 当 `_lastOverlapData` 被保存时，计算 `_lastDataAckTick = currentTick + overlapWaitTicks`
- `overlapWaitTicks` 受 `isGraphCapture` 影响：
  - `isGraphCapture = true`（CUDA Graph replay）→ `overlapWaitTicks = config.tokenRecvDelayTicks`（默认 0，graph_replay 更快）
  - `isGraphCapture = false`（eager forward）→ `overlapWaitTicks = config.tokenRecvDelayTicks + config.eagerForwardExtraDelayTicks`（eager 路径额外延迟）
- Phase 3 中仅当 `currentTick >= _lastDataAckTick` 时才处理 `_lastOverlapData`

**流程变更**：

```
_overlapTick(incoming):
  Phase 1: 消息处理 + 调度
    for msg in incoming: _processOneMsg(msg)
    if messages deque length > highWatermark:
      // 背压：跳过本次调度，仅处理 last_data
      forwardInput = null
    else:
      forwardInput = _scheduleNextBatch()

  Phase 2: forward 当前批（若调度成功）
    forwardOutput = forwardInput ? _forward(forwardInput) : null

  Phase 3: 处理上一批结果（带延迟检查）
    if _lastOverlapData !== null AND currentTick >= _lastDataAckTick:
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
        _lastDataAckTick = currentTick + config.tokenRecvDelayTicks
      else:
        _lastDataAckTick = currentTick + config.tokenRecvDelayTicks + config.eagerForwardExtraDelayTicks
      _lastDataPending = true

  Phase 4: 空闲 tick 刷新检查
    if incoming.length === 0 AND forwardInput === null:
      _idleCounter += 1
      if _idleCounter >= config.idleCountForFlush:
        // 强制调度 prefill 以续接 chunked 请求
        _forcePrefillSchedule()
        _idleCounter = 0
    else:
      _idleCounter = 0

  Phase 5: EPLB + globalStep
```

##### 2.3 空 tick 刷新

**问题**：chunked prefill 请求在上一 tick 消耗了 token budget 后剩余部分被放回 `prefillManager.pendingList` 头部。如果后续若干 tick 都没有新请求进来，调度器不会主动触发 prefill 调度，导致 chunked 请求"饿死"。

**设计**：

- 引入 `_idleCounter`：连续空闲 tick 计数器
- 当 `_idleCounter >= idleCountForFlush`（默认值 2）时，调用 `_forcePrefillSchedule()`
- `_forcePrefillSchedule()` 仅调度 prefill batch（不调度 decode），目的为续接 pending 的 chunked 请求
- 每次有新消息或成功调度后重置 `_idleCounter = 0`

##### 2.4 messages deque 高水位控制

**问题**：当外部请求速率远超调度器处理能力时，`_incomingQueue` 无限增长可能导致内存问题。在真实 SGLang 中也存在类似的背压机制。

**设计**：

- 在 Phase 1 消息入队后，检查 `_incomingQueue.length > highWatermark`
- 若超过高水位，跳过本轮 `_scheduleNextBatch()`（`forwardInput = null`），仅执行 Phase 3 处理已有 last_data
- 高水位阈值通过 `config.messagesHighWatermark` 配置（默认 1024）
- 注意：此处 `_incomingQueue` 即 `SchedulerIOMixin._incomingQueue`，消息入队后检查长度

#### 3. 配置扩展（SimulatorConfig）

在 `SimulatorConfig` interface 和 `DEFAULT_SIMULATOR_CONFIG` 中新增以下字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokenRecvDelayTicks` | `number` | `0` | graph_replay 路径下 last_data 延迟（默认 0 表示 graph_replay 极快） |
| `eagerForwardExtraDelayTicks` | `number` | `2` | eager forward 路径相比 graph_replay 的额外延迟 ticks |
| `idleCountForFlush` | `number` | `2` | 连续空闲 tick 数达到此值时触发 prefill flush |
| `messagesHighWatermark` | `number` | `1024` | 消息队列高水位阈值 |

#### 4. SimulationClock 与 SimScheduler 的集成

**方式**：`SimScheduler` 构造器新增可选 `clock` 选项：

```
constructor(config, opts?: { ..., clock?: SimulationClock })
```

- 若传入 `clock`，则 `this._clock = clock`
- 若 `config.enableOverlap && config.enableMetrics` 且未传入 `clock`，自动创建 `SimulationClock` 实例
- 其他情况 `this._clock = null`

**集成点**：
- `_overlapTick` Phase 2 后调用 `this._clock?.scheduleGpu(gpuDuration)` 记录 GPU 占用
- `_overlapTick` Phase 5 末尾调用 `this._clock?.advance(1n)` 推进时钟并触发回调
- `_normalTick` 末尾同样调用 `this._clock?.advance(1n)`
- `MetricsCollector`（S6 实现）可通过 `clock.onTick(callback)` 注册收集逻辑

### 修改点清单

1. **`server/src/sglang/scheduler/index.ts`**
   - 新增 `SimulationClock` 类（§4.1）：`currentTicks`, `gpuBusyUntil`, `events`, `advance()`, `scheduleGpu()`, `canOverlap()`, `onTick()`
   - 新增 `SimEvent` interface
   - `SimScheduler` 新增字段：`_overlapWaitCounter`, `_idleCounter`, `_lastDataPending`, `_lastDataAckTick`, `_clock`, `_highWatermark`
   - `SimScheduler` 新增构造器选项：`clock`
   - 重写 `_overlapTick()`：增加 last_data 延迟检查、空闲 tick 刷新、高水位背压
   - 新增 `_forcePrefillSchedule()` 私有方法
   - `_normalTick()` 末尾增加 `this._clock?.advance(1n)`
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

## 测试设计

### 验收测试用例清单

| # | 测试名称 | 验证内容 |
|---|----------|----------|
| 1 | `test_simulation_clock_advance` | `advance(1n)` 递增 currentTicks；`advance(0n)` 抛出 Error；单调不回退 |
| 2 | `test_simulation_clock_schedule_gpu` | `scheduleGpu(5)` 返回 `currentTicks + 5n`；连续调度正确更新 `gpuBusyUntil` |
| 3 | `test_simulation_clock_can_overlap` | GPU 繁忙时 `canOverlap()=true`；GPU 空闲时 `canOverlap()=false` |
| 4 | `test_simulation_clock_on_tick_callback` | 注册回调后，`advance()` 触发回调并传入当前 tick；取消注册后不再触发 |
| 5 | `test_overlap_short_prompt_normal` | 短 prompt（input_len < prefill_budget）在 overlap 模式下正常 prefill→decode→finish，不受延迟窗口影响 |
| 6 | `test_overlap_chunked_prefill_idle_flush` | 长 prompt 触发 chunked prefill，空闲 2 tick 后自动续接，最终完成 |
| 7 | `test_overlap_last_data_delay_graph_capture` | graph_replay 路径下 `tokenRecvDelayTicks=1`，last_data 延迟 1 tick 后才被处理 |
| 8 | `test_overlap_last_data_delay_eager` | eager 路径下 `eagerForwardExtraDelayTicks=2`，last_data 延迟 2+1=3 tick 后才被处理 |
| 9 | `test_overlap_empty_tick_flush_last_data` | overlap 模式结束后，调用空 tick `runTick([])` 刷新残留 last_data |
| 10 | `test_overlap_high_watermark_backpressure` | 大量请求超过 highWatermark 时，跳过 forward 仅处理 last_data |

### 边界条件覆盖

| 边界条件 | 测试覆盖 |
|----------|----------|
| `tokenRecvDelayTicks = 0` 且 `isGraphCapture = true` → last_data 无延迟，同 tick 处理 | 测试 5 |
| `idleCountForFlush = 0` → 每个 tick 都尝试 flush | 测试 6（变体） |
| 无 last_data 时空 tick 不报错 | 测试 9 |
| `highWatermark = 0` → 始终背压 | 测试 10（变体） |
| SimulationClock `advance(0n)` → 抛出 Error | 测试 1 |
| SimulationClock `scheduleGpu(0)` → start == finish，无实际 GPU 占用 | 测试 2（变体） |

## 风险与注意事项

- **兼容性影响**：`_overlapTick` 的行为变更可能影响已有测试中 overlap 模式的时序预期。由于当前项目尚无 `_overlapTick` 的单元测试，风险可控。需要确保 `_normalTick` 路径不受任何影响。
- **性能影响**：`SimulationClock` 使用 `bigint` 运算，在高频 tick 循环中可能带来微小开销。但由于 SimulationClock 是可选的（默认不实例化），且仿真器本身不是高性能路径，影响可忽略。
- **bigint 兼容性**：TypeScript `bigint` 类型在 JSON 序列化时需要特殊处理（`BigInt.toJSON` 或手动转换）。`SimulationClock.toJSON()` 需将 `bigint` 转为 `string` 输出。
- **回滚方案**：所有变更通过配置字段控制（`tokenRecvDelayTicks=0`, `eagerForwardExtraDelayTicks=0`, `idleCountForFlush=Infinity`, `messagesHighWatermark=Infinity`），恢复默认值即可回退到当前行为。`SimulationClock` 为可选组件，不影响正确性。
