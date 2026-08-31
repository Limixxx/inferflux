---
title: "Issue #16 解决方案"
issue_number: 16
issue_type: Feature
created: 2026-08-31
updated: 2026-08-31
status: revised
review_round: 2
---

# Issue #16 解决方案

## 需求分析

- **问题描述**：Issue #16 要求实现 SGLang 仿真器 S2 阶段的三个调度核心组件：`PrefillAdder`、`PrefillManager`、`DecodeManager`。这三个组件是 `SimScheduler` 调度循环的核心，负责请求从前缀匹配到 prefill 调度再到 decode 执行的全生命周期管理。当前代码库中 `scheduler/index.ts` 仅包含 `TableManager` 实现，三个目标组件尚不存在。

- **能力目标**：
  1. **PrefillAdder**（§9.5/§9.7/§9.11）：`tryAddOne(pendingReq)` 方法实现两次 `available_size` 检查（lock 前宽松、lock 后严格）、token budget 预算管理、chunked prefill 分块逻辑、以及 `_tryAddOneChunked` 续接路径
  2. **PrefillManager**（§9.2/§9.5/§9.11）：`addOneReq(msg)` + `addBatch(reqs)` + `abortReq(uid)` + `scheduleNextBatch(tokenBudget)` 方法，管理 pending 队列，chunked 请求续接放回队列头部
  3. **DecodeManager**（§9.2/§9.11）：`addReq(req)` + `removeReq(req)` + `filterReqs(newReqs)` + `abortReq(uid)` + `scheduleNextBatch()` + `inflightTokens` 属性，管理 runningReqs 集合
  4. 单元测试覆盖：短 prompt 一次性 prefill、长 prompt 两次 tick 分块、两次 available_size 检查行为一致性

- **影响范围**：仅修改 `server/src/sglang/scheduler/` 目录（在现有 `index.ts` 中新增三个 class），以及 `server/src/sglang/index.ts`（新增 re-export）、`server/src/test/` 目录（新增 `sglang-s2.test.ts`）。不修改任何已有业务源码或测试代码。

- **依赖 Issue**：#14（K3: CacheManager naive）已完成，`CacheManager` 的 `availableSize`、`matchReq`、`lock`、`unlock`、`allocatePaged`、`cacheReq`、`beginLazyFree`/`endLazyFree` API 已就绪。

- **阻塞 Issue**：S3（MockEngine + SimScheduler normal_tick + SchedulerIOMixin）依赖本 Issue 的 PrefillManager/DecodeManager 实例化。

## 改造方案

### 总体思路

按照 §9.11 完整实现代码集，将 `PrefillAdder`、`PrefillManager`、`DecodeManager` 三个 class 实现于 `scheduler/index.ts`（与现有 `TableManager` 同文件）。核心设计决策如下：

1. **同文件放置**：三个 class 与 `TableManager` 共存于 `scheduler/index.ts`，因它们体量均不大且高度内聚（PrefillAdder 仅在 PrefillManager.scheduleNextBatch 中实例化，DecodeManager 仅被前两者引用），拆分文件增加不必要的导入复杂度。当前三个类合计约 200 行，与 TableManager 的 37 行处于同一量级，单文件保持可读性。若 S3 阶段新增 `SimScheduler` 类导致文件膨胀超过 400 行，届时再拆分为独立模块。
2. **TypeScript 适配**：
   - Python 的 `Req | ChunkedReq | None` 联合返回类型 -> TS 的 `Req | ChunkedReq | null`
   - Python 的 `Set[Req]` -> TS 的 `Set<Req>`（利用 Req 基于 rid 的引用唯一性）
   - Python 列表切片赋值 -> TS 的逐元素赋值或 `Array.prototype.slice` + 循环
   - `cache_manager.lock_handle(handle, unlock=True)` -> 已有的 `cacheManager.unlock(handle)` 封装
3. **与 CacheManager 的耦合点**：PrefillAdder 调用 `cacheManager.matchReq`、`cacheManager.lock/unlock`、`cacheManager.availableSize`；PrefillManager 通过 PrefillAdder 间接使用；DecodeManager 通过 `inflightTokens` 属性为 PrefillAdder 提供 `reservedSize`
4. **chunked prefill 续接**：当 `pendingReq.chunkedReq !== null` 时走 `_tryAddOneChunked` 路径，复用已有 `tableIdx` 和 `cacheHandle`，仅处理剩余 token

### 详细设计

#### 1. PrefillAdder 类

核心职责：逐个尝试将请求加入 prefill batch，实现两次 available_size 检查、token budget 管理、chunked prefill 分块与续接。

**tryAddOne 核心流程（对齐 §9.11 伪码步骤 0-10）**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 0 | `if pendingReq.chunkedReq !== null` -> `_tryAddOneChunked` | chunked 续接路径 |
| 1 | `cacheManager.matchReq(pendingReq)` | 前缀匹配 |
| 2 | **第一次 available_size 检查**（宽松）：`estimatedLen + reservedSize > cacheManager.availableSize` -> return null | lock 前，evictable 部分尚未因 lock 移入 protected |
| 3 | token budget 检查：`remainingBudget <= 0` -> return null | |
| 4 | `cacheManager.lock(handle)` | 改变 evictable_size（ref_count 0->1 的节点移入 protected） |
| 5 | **第二次 available_size 检查**（严格）：`estimatedLen + reservedSize > cacheManager.availableSize` -> unlock + return null | lock 后 available_size 可能减小 |
| 6 | `tableManager.allocate()` | 分配 table_idx |
| 7 | 复制 cached 部分的 token 和 page entry 到 token_pool/page_table | |
| 7b | 复制 extend 部分的 token 到 token_pool | 供 _forward 读取 batch.input_ids |
| 8 | 决定 chunk_size：`min(extendLen, remainingBudget)`，判断 `is_chunked` | |
| 9 | 更新 consumedTokens | |
| 10 | 非 chunked 的请求加入 decodeManager | |

**两次 available_size 检查的设计原理（§9.7）**：

- 第一次检查（lock 前）：`available_size = evictable_size + free_pages * page_size`，此时 evictable 部分包含将被 lock 保护的前缀节点，所以 available_size 偏大（宽松）
- lock 操作：将匹配的前缀节点从 evictable 移入 protected，`evictable_size` 减小
- 第二次检查（lock 后）：`available_size` 可能减小，若此时仍不足则需 unlock 并放弃

**锁与异常安全保证（回应评审意见 §1）**：

`tryAddOne` 的 lock-unlock 生命周期严格遵循以下规则，确保任何异常/失败路径下均不会遗留锁：

| 失败点 | lock 状态 | 回滚动作 | 代码路径 |
|--------|-----------|----------|----------|
| 步骤 2 第一次 available_size 检查失败 | **未 lock** | 无需回滚，直接 return null | `return null` |
| 步骤 3 token budget 不足 | **未 lock** | 无需回滚，直接 return null | `return null` |
| 步骤 5 第二次 available_size 检查失败 | **已 lock** | `cacheManager.unlock(handle)` 后 return null | `unlock → return null` |
| 步骤 6 tableManager.allocate() 抛出异常 | **已 lock** | 在 try-catch 中捕获异常，执行 `unlock(handle)` + `free(tableIdx)`（若已分配），然后 re-throw 或 return null | 见下方异常处理伪码 |
| 步骤 7/7b token 复制失败（理论上不会发生，仅数组赋值） | **已 lock** | 同上，unlock + free | |

异常处理封装伪码（TypeScript）：

```typescript
tryAddOne(pendingReq: PendingReq): Req | ChunkedReq | null {
  // ... 步骤 0-4 ...
  // 步骤 4: lock
  this.cacheManager.lock(handle);

  try {
    // 步骤 5: 第二次检查
    if (estimatedLen + this.reservedSize > this.cacheManager.availableSize) {
      this.cacheManager.unlock(handle);
      return null;
    }
    // 步骤 6: 分配 table_idx
    const tableIdx = this.tableManager.allocate();
    try {
      // 步骤 7/7b/8/9: 复制 token、决定 chunk_size
      // ...（正常逻辑）
    } catch (err) {
      // token 复制或后续逻辑异常（防御性）
      this.tableManager.free(tableIdx);
      this.cacheManager.unlock(handle);
      throw err;  // 向上传播，由调度器决定处理策略
    }
  } catch (err) {
    // 步骤 5 的 unlock 已在内部处理；
    // 此处仅捕获 allocate() 抛出的异常
    this.cacheManager.unlock(handle);
    return null;  // 分配失败视为资源不足，静默返回 null
  }
}
```

关键保证：
- lock 后的所有失败路径均有对应的 unlock 调用
- allocate 成功后的异常路径同时释放 tableIdx 和 unlock
- 不存在"已 lock 但未 unlock"的返回路径

**并发与竞态说明（回应评审意见 §2）**：

仿真器采用 **单线程 tick 驱动模型**：`SimScheduler.run_tick()` 由外部仿真循环串行调用，每个 tick 内依次执行 `_process_one_msg` → `_schedule_next_batch` → `_forward` → `_process_last_data`。因此：

- **不存在多线程并发**：PrefillAdder.tryAddOne、PrefillManager.scheduleNextBatch、DecodeManager.filterReqs 均在同一 tick 的同步执行上下文中调用，不存在跨线程的 lock 竞争
- **不存在跨 tick 竞态**：每个 tick 结束后，所有 lock 已被后续流程（cache_req/unlock）妥善处理，下一 tick 开始时不存在残留的锁状态
- **不需要加锁机制**：所有状态变更（runningReqs、pendingList、consumedTokens 等）均在单线程内完成，TypeScript 单线程事件循环保证可见性

真实 SGLang 的多进程 TP 架构中确实存在调度竞态，但仿真器故意简化了这一层——技术报告 §1 明确指出仿真器用于研究调度策略与内存管理，不需要真实的多进程通信。

**原子性与一致性保证（回应评审意见 §3）**：

在 `tryAddOne` 中，资源分配与写入分为三个阶段，每个阶段失败时回滚前序阶段的资源：

| 阶段 | 操作 | 失败回滚 |
|------|------|----------|
| A: Cache 锁定 | `lock(handle)` | unlock(handle) |
| B: Table 分配 | `allocate()` -> tableIdx | free(tableIdx) + unlock(handle) |
| C: Token/Page 写入 | 写入 token_pool、page_table | free(tableIdx) + unlock(handle)（写入是数组赋值，覆盖即可，无需显式清除） |

关键设计点：
- **allocate_paged 与 token_pool 写入的顺序保证**：根据 §9.11 的设计，`allocate_paged`（写 page_table 的 extend 部分）在 `_prepare_batch` 中统一执行，不在 `tryAddOne` 内执行。因此 `tryAddOne` 仅负责 token_pool 的写入和 page_table cached 部分的写入，这两者均为简单的数组赋值，不存在部分写入风险
- **page_table 的 cached 部分写入**：`handle.getMatchedIndices()` 返回只读数组，直接赋值给 `page_table[tableIdx]` 的前 `cachedLen` 个位置，TypeScript 数组赋值是原子操作
- **token_pool 的写入**：同样为简单的数组切片赋值，不存在中间状态暴露的风险
- **若 allocate_paged 在 _prepare_batch 中失败**：此时所有 req 的 tableIdx 已分配，需要整体回滚当前 batch（释放所有 tableIdx + unlock 所有 handle），由 SimScheduler 层面的错误处理负责（S3 阶段实现）

**_tryAddOneChunked 流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 取 `prevReq = pendingReq.chunkedReq` | 上一 tick 的 ChunkedReq |
| 2 | `cachedLen = prevReq.deviceLen` | 上次已处理到的位置 |
| 3 | 资源检查 + token budget 检查 | 与 tryAddOne 步骤 2-3 相同 |
| 4 | 复用 `prevReq.tableIdx`（不重新分配） | 关键：续接不占新表行 |
| 5 | 决定 chunk_size，复制本 chunk 的 token 到 token_pool | |
| 6 | 若 `!is_chunked` 则加入 decodeManager | 最后一个 chunk 转为完整 Req |

续接中不调用 lock 的原因：chunked 请求的 `cacheHandle` 在首次 `tryAddOne` 时已经 lock，续接路径复用该 handle，无需再次 lock。

**_tryAddOneChunked 的异常安全**：续接路径无需 lock/unlock 操作（handle 已锁定），也无需 allocate/free（复用已有 tableIdx），因此不存在锁泄漏或资源泄漏风险。唯一的失败点是步骤 3 的资源/budget 检查，此时直接 return null，无需任何回滚。

#### 2. PrefillManager 类

核心职责：管理待 prefill 的请求队列，通过 PrefillAdder 逐个调度，chunked 请求剩余部分放回队列头部优先续接。

**scheduleNextBatch 核心流程**：

1. 若 `pendingList` 为空 -> return null
2. 创建 `PrefillAdder`（`reservedSize = decodeManager.inflightTokens`）
3. 遍历 `pendingList`，逐个调用 `adder.tryAddOne(pendingReq)`
4. 若返回 null -> break（资源/budget 不足，保留未处理请求）
5. 若返回 `ChunkedReq` -> 构造新的 `PendingReq`（携带 `chunkedReq`），加入 `chunkedList`
6. 遍历结束后：`self.pendingList = chunkedList + self.pendingList[i:]`
   - `chunkedList` 优先放回队列头部，确保下 tick 优先续接
   - `pendingList[i:]` 保留因 break 未调度的请求
7. 若 `reqs` 非空 -> 返回 `Batch`

**Batch 构造说明**：当前 `Batch` class 使用 `Map<number, Req>` 按 `rid` 索引，`scheduleNextBatch` 需将 `Req[]` 转换为 `Map` 并设置 `readyIds`。

#### 3. DecodeManager 类

核心职责：管理可 decode 的请求集合，生成 decode batch，计算 inflightTokens。

**inflightTokens 计算公式（§9.11，回应评审意见 §4）**：

```
tokens_reserved = (pageSize - 1) * len(runningReqs)
inflightTokens = sum(req.remainLen for req in runningReqs) + tokens_reserved
```

- `tokens_reserved`：每个 running req 因页对齐最多浪费 `pageSize - 1` 个 token 位置
- `remainLen`：`maxDeviceLen - deviceLen`，即请求剩余可解码长度

**reservedSize 的语义与范围**（回应评审意见 §4 详细说明）：

| 问题 | 回答 |
|------|------|
| 是否包含即将进入 decode 的 tokens？ | **不包含**。`inflightTokens` 仅统计当前已在 `runningReqs` 中的请求。即将进入 decode 的请求在 `tryAddOne` 步骤 10 被 `addReq` 加入 `runningReqs` 后，将在下一个 tick 的 `scheduleNextBatch` 中反映到 `inflightTokens` |
| 是否计入 chunked 剩余量？ | **不计入**。ChunkedReq 不加入 `runningReqs`（§9.11 中 `ChunkedReq.canDecode = false`），其剩余 token 仅在下一个 tick 的续接路径中处理。这符合 §9.11 的设计：chunked prefill 的资源需求由 `available_size` 检查覆盖，而非 `reservedSize` |
| 与技术报告公式一致性 | 完全一致。`inflightTokens` = `sum(remainLen) + (pageSize-1) * len(runningReqs)`，其中 `remainLen = maxDeviceLen - deviceLen`，对应 §9.11 `DecodeManager.inflight_tokens` 属性 |

**filterReqs 语义**：每次 forward 后调用，更新 `runningReqs`：

```
runningReqs = (runningReqs | newReqs) 中 canDecode 为 true 的子集
```

**scheduleNextBatch**：按 `rid` 排序 `runningReqs`，构造 `Batch`。

### 接口变更

1. **`scheduler/index.ts`**：新增 `PrefillAdder`、`PrefillManager`、`DecodeManager` 三个 class 导出
2. **`sglang/index.ts`**：新增三个 class 的 re-export

### 数据结构改动

无新增数据结构。三个 class 使用已有的 `PendingReq`、`Req`、`ChunkedReq`、`Batch` 类型。

### 修改点清单

1. **修改 `server/src/sglang/scheduler/index.ts`**：新增 `PrefillAdder`、`PrefillManager`、`DecodeManager` 三个 class 实现
2. **修改 `server/src/sglang/index.ts`**：新增三个 class 的 re-export
3. **新建 `server/src/test/sglang-s2.test.ts`**：S2 验收测试文件

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | PrefillAdder 构造 | tokenBudget/reservedSize/consumedTokens 正确初始化 |
| T2 | PrefillAdder.tryAddOne - 短 prompt 一次性 prefill | extendLen <= tokenBudget，返回 Req，consumedTokens = extendLen |
| T3 | PrefillAdder.tryAddOne - token budget 不足 | remainingBudget < extendLen，返回 ChunkedReq，consumedTokens = remainingBudget |
| T4 | PrefillAdder.tryAddOne - 资源不足（第一次 available_size 检查） | estimatedLen + reservedSize > availableSize -> return null |
| T5 | PrefillAdder.tryAddOne - 第二次 available_size 检查失败 | lock 后 availableSize 减小导致检查失败，unlock 后 return null |
| T6 | PrefillAdder._tryAddOneChunked 续接 | 第二次 tick 续接剩余 token，最终返回完整 Req |
| T7 | PrefillAdder - 两次 available_size 检查行为一致性 | 宽松检查通过但严格检查失败时正确 unlock 放弃；两者都通过时正常分配 |
| T8 | PrefillManager.addOneReq | 请求加入 pendingList |
| T9 | PrefillManager.addBatch | 批量请求加入 pendingList |
| T10 | PrefillManager.scheduleNextBatch - 空队列 | pendingList 为空 -> return null |
| T11 | PrefillManager.scheduleNextBatch - 短 prompt 一次性 | 返回包含 Req 的 Batch |
| T12 | PrefillManager.scheduleNextBatch - 长 prompt 分块 | 第一次返回 ChunkedReq，chunked 续接放回队列头部；第二次续接完成 |
| T13 | PrefillManager.abortReq - 存在的 uid | 从 pendingList 移除并返回 chunkedReq（如有） |
| T14 | PrefillManager.abortReq - 不存在的 uid | return null |
| T15 | DecodeManager.addReq/removeReq | 请求加入/移出 runningReqs |
| T16 | DecodeManager.filterReqs | forward 后过滤掉 canDecode=false 的请求，加入新请求 |
| T17 | DecodeManager.inflightTokens | 计算公式正确：sum(remainLen) + (pageSize-1)*len(runningReqs) |
| T18 | DecodeManager.scheduleNextBatch - 空集 | runningReqs 为空 -> return null |
| T19 | DecodeManager.scheduleNextBatch - 非空 | 按 rid 排序返回 Batch |
| T20 | DecodeManager.abortReq | 按 uid 查找并移除请求 |
| T21 | PrefillManager + DecodeManager 集成 - 短 prompt 全流程 | addOneReq -> scheduleNextBatch -> Req 加入 decodeManager -> decodeManager.scheduleNextBatch |
| T22 | PrefillManager + DecodeManager 集成 - 长 prompt 两次 tick | 第一次 chunked -> 第二次续接完成 -> 加入 decodeManager |

### 扩展测试用例（回应评审意见 §6）

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T23 | PrefillAdder - lock 后 unlock 回滚 | 第二次 available_size 检查失败后，验证 cache_handle 已被正确 unlock（后续相同请求可重新 lock） |
| T24 | PrefillAdder - tableManager 分配失败 | 模拟 tableManager.availableSize=0 导致 allocate() 抛异常，验证 lock 已被释放、无资源泄漏 |
| T25 | PrefillAdder - 连续多次 tryAddOne 的一致性 | 同一个 PrefillAdder 实例连续添加多个请求，验证 consumedTokens 累积正确、各请求状态独立 |
| T26 | PrefillManager - 多个 chunked 请求续接优先级 | 两个请求均产生 ChunkedReq，验证下一 tick 两个续接请求均优先于新请求 |
| T27 | DecodeManager.filterReqs - 空 newReqs | 仅过滤现有 runningReqs 中 canDecode=false 的请求 |
| T28 | DecodeManager - pageSize=1 时 tokens_reserved=0 | inflightTokens = sum(remainLen)，无页对齐浪费 |
| T29 | PrefillAdder._tryAddOneChunked - 续接时资源不足 | 第二次 tick 续接时 available_size 不足，return null，保留 pendingReq 在队列中等待下次 tick |
| T30 | PrefillAdder - 全缓存命中（extendLen=0） | chunk_size=0，is_chunked=false，返回 Req，consumedTokens 不增加 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | PrefillAdder.tokenBudget = 0 | remainingBudget=0，所有 tryAddOne 返回 null |
| B2 | PrefillAdder - extendLen = 0（全缓存命中） | chunk_size=0，is_chunked=false，返回 Req，consumedTokens 不增加 |
| B3 | PrefillAdder - tableManager.availableSize = 0 | 第一次检查 estimatedLen > 0 时 return null |
| B4 | PrefillManager - 多个 chunked 请求续接 | chunkedList 优先于 pendingList[i:] 放回队列头部 |
| B5 | DecodeManager - pageSize=1 时 tokens_reserved=0 | inflightTokens = sum(remainLen)，无页对齐浪费 |
| B6 | DecodeManager.filterReqs - 空 newReqs | 仅过滤现有 runningReqs |
| B7 | PrefillAdder._tryAddOneChunked - 续接时资源不足 | return null，保留 pendingReq 在队列中等待下次 tick |
| B8 | PrefillAdder - 第二次 available_size 检查失败后重试 | unlock 后同一请求在下次 tick 可重新成功 lock（验证无残留锁） |
| B9 | tableManager 分配耗尽 | allocate() 抛出异常，lock 被释放，tryAddOne 返回 null |
| B10 | DecodeManager.inflightTokens - 无 running 请求 | 返回 0 |

## 风险与注意事项

- **兼容性影响**：新增三个 class 不影响现有 API。`Batch` 构造函数当前不接受 `reqs` 参数，`scheduleNextBatch` 需手动构建 `Map<number, Req>` 并设置 `readyIds`。若后续 S3 需要 Batch 接受构造参数，可在彼时扩展。
- **性能影响**：`Set<Req>` 基于 ES 标准 Set，使用引用相等性（Req 实例唯一），查找/删除 O(1)。PrefillAdder 每次 schedule 创建新实例，对象创建开销可忽略。
- **回滚方案**：所有改动在 `issue-16` 分支，合并前可安全回滚。
- **依赖关系**：Issue #14（K3: CacheManager）必须已完成并合并。本 Issue 依赖 K1 抽象类型（`BaseCacheHandle`、`MatchResult`）、K2 实现（`NaivePrefixCache`、`NaiveCacheHandle`）、K3 实现（`CacheManager`）、S1 实现（`Req`、`ChunkedReq`、`PendingReq`、`Batch`、`SamplingParams`、`TableManager`）。
- **chunked 续接中 tableIdx 复用**：续接路径不复用 `ChunkedReq` 实例本身，而是从中提取 `cacheHandle` 和 `tableIdx`，创建新的 `Req` 或 `ChunkedReq`。这确保每个 batch 中的 Req 实例独立，避免跨 batch 的状态耦合。
- **两次 available_size 检查在 naive backend 下的退化**：NaivePrefixCache 的 `lockHandle` 为 noop，不会改变 `evictableSize`。因此在 naive backend 下，两次检查的 `availableSize` 值相同，第二次检查不会额外拒绝请求。但这不影响代码正确性——两次检查的逻辑仍然执行，只是效果在 naive 下退化为冗余。当 K4（RadixPrefixCache）实现后，lock 会真正改变 `evictableSize`，两次检查的差异将体现。
- **Batch phase 标识**：当前 `Batch` class 无 `phase` 属性。`scheduleNextBatch` 通过设置 `readyIds` 来区分 prefill/decode batch。若后续需要 `phase` 字段用于统计或指标，可在 S3 中扩展。
- **锁与异常安全的实现承诺**：tryAddOne 中 lock 后的所有失败路径均有对应的 unlock 调用，通过 try-catch-finally 结构保证。tableManager.allocate() 的异常会被捕获并执行 unlock 后返回 null。续接路径 `_tryAddOneChunked` 因无需 lock 操作，不存在锁泄漏风险。
- **单线程模型下的竞态豁免**：仿真器采用单线程 tick 驱动模型，不存在多线程并发。因此不需要并发控制机制（mutex、CAS 等），两次 available_size 检查在单线程上下文中是安全的。真实 SGLang 的多进程竞态不在仿真范围内。
- **原子性保证**：tryAddOne 中的资源分配与写入按 A(Cache锁定) → B(Table分配) → C(写入) 三阶段进行，每个阶段失败时回滚前序阶段的资源。page_table 的 extend 部分写入延迟到 `_prepare_batch` 中的 `allocate_paged`，避免重复分配和不一致风险。
