---
title: "Issue #10 解决方案"
issue_number: 10
issue_type: Feature
created: 2026-08-27
updated: 2026-08-27
status: draft
review_round: 1
---

# Issue #10 解决方案

## 需求分析

- **问题描述**：Issue #10 要求实现 SGLang 仿真器的 S1 阶段核心数据结构，包括 `SamplingParams`、`Req`、`ChunkedReq`、`Batch`、`PendingReq` 以及工具函数 `div_even`、`div_ceil`、`align_down`、`bytes_per_element`。这些数据结构是整个仿真器调度逻辑的基础，后续所有调度器、缓存管理器、prefill/decode 管理器都依赖它们。

- **能力目标**：
  1. 将 S0 中定义的 `SamplingParams` 接口升级为完整的类实现，增加 `frequencyPenalty`、`repetitionPenalty`、`minP`、`stopTokenIds`、`skipSpecialTokens`、`dtype` 字段，并提供 `isGreedy` 计算属性
  2. 实现 `Req` 类，包含请求状态管理（`rid`、`originInputLen`、`inputIds`、`outputIds`、`promptLogprobStartPos`、`samplingParams`、`finished`、`finishReason`、`samplingCounter`、`maxNewTokens`、`dpRank`），以及 `deviceLen`/`maxDeviceLen` 可变属性和 `completeOne`/`appendHost` 方法
  3. 实现 `ChunkedReq` 继承 `Req`，覆盖 `appendHost`（抛出异常）和 `canDecode`（返回 false）
  4. 实现 `Batch` 类，包含 `reqs` 字典、`initLen`、`promptTokens`、`extendInputTokens`、`extendOutputTokens`、`numDecodeTokens`、`hasIdleReqs`、`readyIds`、`nextId`、`schedulerThinkingBatch`，以及 `nextReadyReq`/`nextBatchReq` 纯函数
  5. 实现 `PendingReq` 数据结构，包含 `rid`、`priority`、`nextScheduledTime`、`inputIds`、`samplingParams`、`chunkedReq` 引用
  6. 实现工具函数 `divEven`、`divCeil`、`alignDown`、`bytesPerElement`

- **影响范围**：仅修改 `server/src/sglang/` 目录下的文件。S0 已有的 `types.ts` 中 `SamplingParams` 接口需要扩展，`core/index.ts` 和 `entities/index.ts` 需要从占位注释升级为完整实现。

## 改造方案

### 总体思路

S0 已建立了模块骨架和顶层配置/消息类型。S1 在此基础上：

1. **扩展 `SamplingParams`**：从 S0 的简单接口升级为完整的 `class`，增加 Issue #10 要求的字段，保持向后兼容
2. **在 `core/index.ts` 中实现 `Req`、`Batch`**：作为核心数据结构，与 §9.2 规格对齐
3. **在 `entities/index.ts` 中实现 `ChunkedReq`、`PendingReq`**：作为 `Req` 的扩展和调度辅助结构
4. **在 `core/index.ts` 中实现工具函数**：`divEven`、`divCeil`、`alignDown`、`bytesPerElement`，严格遵循 §3.4.1 规格
5. **从 `types.ts` 导出新类型**：更新顶层 `index.ts` 的 re-export

### 详细设计

#### 1. SamplingParams 类（core/index.ts）

将 S0 的 `interface SamplingParams` 升级为 `class SamplingParams`：

```typescript
export class SamplingParams {
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly topP: number;
  readonly topK: number;
  readonly frequencyPenalty: number;
  readonly repetitionPenalty: number;
  readonly minP: number;
  readonly stopTokenIds: number[];
  readonly skipSpecialTokens: boolean;
  readonly dtype: "float32" | "float16" | "bfloat16";

  constructor(opts?: Partial<SamplingParams>) { /* 带默认值 */ }

  get isGreedy(): boolean {
    return (this.temperature <= 0 || this.topK === 1) && this.topP === 1.0;
  }
}
```

- S0 的 `interface SamplingParams` 中 `maxTokens` 改名为 `maxNewTokens`，与 §9.2 规格对齐
- S0 的 `ignoreEos` 保留为兼容字段（通过 `stopTokenIds` 间接表达）
- `dtype` 用于 `bytesPerElement` 计算，对应 `SimulatorConfig.dtypeSize`
- 构造函数接受 `Partial<SamplingParams>` 以提供默认值

#### 2. Req 类（core/index.ts）

```typescript
export class Req {
  readonly rid: number;
  readonly originInputLen: number;
  inputIds: number[];
  outputIds: number[];
  promptLogprobStartPos: number;
  samplingParams: SamplingParams;
  finished: boolean;
  finishReason: string | null;
  samplingCounter: number;
  maxNewTokens: number;
  dpRank: number;

  // 可变属性（非 getter）
  deviceLen: number;
  maxDeviceLen: number;

  constructor(rid: number, inputIds: number[], samplingParams: SamplingParams, ...) { ... }

  get remainLen(): number { return this.maxDeviceLen - this.deviceLen; }
  get extendLen(): number { return this.deviceLen - this.cachedLen; }  // 需要 cachedLen 外部管理
  get canDecode(): boolean { return this.remainLen > 0; }

  completeOne(): void {
    this.deviceLen += 1;
    this.samplingCounter += 1;
  }

  appendHost(nextToken: number): void {
    this.inputIds = [...this.inputIds, nextToken];
    this.outputIds.push(nextToken);
  }
}
```

- `deviceLen` 和 `maxDeviceLen` 是普通可变属性，非 `@property`，与 §2.2.2 一致
- `cachedLen` 由外部（CacheManager/TableManager）管理，Req 不持有此状态
- `originInputLen` 记录初始输入长度，用于统计

#### 3. ChunkedReq 类（entities/index.ts）

```typescript
export class ChunkedReq extends Req {
  get canDecode(): boolean { return false; }

  appendHost(_nextToken: number): never {
    throw new Error("ChunkedReq should not be sampled");
  }
}
```

- 继承 `Req`，覆盖 `canDecode` 返回 false，`appendHost` 抛出错误
- 与 §9.11 ChunkedReq 定义一致

#### 4. Batch 类（core/index.ts）

```typescript
export class Batch {
  reqs: Map<number, Req>;
  initLen: number;
  promptTokens: number;
  extendInputTokens: number;
  extendOutputTokens: number;
  numDecodeTokens: number;
  hasIdleReqs: boolean;
  readyIds: number[];
  nextId: number;
  schedulerThinkingBatch: boolean;

  constructor() { ... }

  nextReadyReq(): Req | undefined { ... }
  nextBatchReq(): Req | undefined { ... }
}
```

- `reqs` 使用 `Map<number, Req>` 按 `rid` 索引，与 §9.2 `reqs[id]` 对齐
- `nextReadyReq`/`nextBatchReq` 为纯函数，从 `readyIds` 取下一个请求

#### 5. PendingReq 数据结构（entities/index.ts）

```typescript
export class PendingReq {
  readonly rid: number;
  readonly priority: number;
  nextScheduledTime: number;
  inputIds: number[];
  samplingParams: SamplingParams;
  chunkedReq: ChunkedReq | null;

  constructor(rid: number, inputIds: number[], samplingParams: SamplingParams, opts?: { ... }) { ... }
}
```

- `chunkedReq` 非 null 时表示上一 tick chunked 的请求，需续接
- `priority` 用于调度排序（默认 0）
- `nextScheduledTime` 用于延迟调度

#### 6. 工具函数（core/index.ts）

```typescript
export function alignDown(n: number, alignment: number): number {
  return n - (n % alignment);
}

export function divCeil(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

export function divEven(a: number, b: number, allowReplicate: boolean = false): number[] {
  if (!allowReplicate && a < b) {
    throw new Error(`divEven(${a}, ${b}) with allowReplicate=false requires a >= b`);
  }
  if (a === 0) return new Array(b).fill(0);
  const base = Math.floor(a / b);
  const remainder = a % b;
  return [...Array(remainder).fill(base + 1), ...Array(b - remainder).fill(base)];
}

export function bytesPerElement(dtype: "float32" | "float16" | "bfloat16"): number {
  switch (dtype) {
    case "float32": return 4;
    case "float16": return 2;
    case "bfloat16": return 2;
  }
}
```

- 严格遵循 §3.4.1 / §9.2 规格中的语义
- TS strict 模式，零 `any`
- 纯函数，便于单元测试

### 修改点清单

1. **`server/src/sglang/types.ts`**：将 `SamplingParams` 从 `interface` 改为引用 `core/index.ts` 中的 `class`，或保留接口兼容层；更新 `SimRequestMsg` 等依赖类型
2. **`server/src/sglang/core/index.ts`**：实现 `SamplingParams` 类、`Req` 类、`Batch` 类、工具函数 `divEven`/`divCeil`/`alignDown`/`bytesPerElement`
3. **`server/src/sglang/entities/index.ts`**：实现 `ChunkedReq` 类、`PendingReq` 类
4. **`server/src/sglang/index.ts`**：更新 re-export 列表，导出所有新增类型
5. **`server/src/test/sglang-s1.test.ts`**：新增 S1 验收测试文件，覆盖所有数据结构和工具函数

### types.ts 兼容性处理

S0 的 `types.ts` 中 `SamplingParams` 是 `interface`，S1 需要升级为 `class`。处理策略：

- 在 `core/index.ts` 中定义 `SamplingParams` 类
- `types.ts` 中保留 `SamplingParams` 作为类型别名指向类实例类型：`export type { SamplingParams } from "./core"`
- 从 `core/index.ts` 导出 `SamplingParams` 类本身，确保 `new SamplingParams()` 和类型标注都能正常工作
- S0 的 `SimRequestMsg.samplingParams` 字段类型自然兼容

## 测试设计

### 验收测试用例清单

| 编号 | 测试名称 | 验证内容 |
|------|---------|---------|
| T1 | SamplingParams 默认值 | 所有字段均有合理默认值 |
| T2 | SamplingParams.isGreedy | temperature≤0 或 topK=1 且 topP=1.0 时返回 true |
| T3 | SamplingParams.isGreedy 非 greedy | temperature>0 且 topK≠1 时返回 false |
| T4 | Req 构造 | rid/inputIds/samplingParams 正确赋值，finished=false |
| T5 | Req.completeOne | deviceLen 递增，samplingCounter 递增 |
| T6 | Req.appendHost | inputIds 追加 token，outputIds 追加 token |
| T7 | Req.canDecode | remainLen>0 时返回 true，=0 时返回 false |
| T8 | ChunkedReq.canDecode | 始终返回 false |
| T9 | ChunkedReq.appendHost | 抛出错误 |
| T10 | Batch 构造与 reqs 管理 | reqs 为空 Map，nextId 正确递增 |
| T11 | Batch.nextReadyReq/nextBatchReq | 从 readyIds 正确取出请求 |
| T12 | PendingReq 构造 | rid/inputIds/samplingParams/chunkedReq 正确赋值 |
| T13 | PendingReq.chunkedReq 续接 | chunkedReq 非 null 时表示续接状态 |
| T14 | alignDown | alignDown(10,3)=9, alignDown(0,5)=0, alignDown(7,1)=7 |
| T15 | divCeil | divCeil(7,3)=3, divCeil(6,3)=2, divCeil(0,5)=0 |
| T16 | divEven 均分 | divEven(8,3)=[3,3,2], divEven(6,3)=[2,2,2] |
| T17 | divEven allowReplicate | divEven(2,4,true)=[1,1,0,0] |
| T18 | divEven 禁止复制时抛错 | divEven(2,4,false) 抛出 Error |
| T19 | bytesPerElement | float32=4, float16=2, bfloat16=2 |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | `alignDown(0, n)` | 返回 0 |
| B2 | `divCeil(0, n)` | 返回 0 |
| B3 | `divEven(0, n)` | 返回 `[0]*n` |
| B4 | `SamplingParams` 全部默认构造 | 无异常，isGreedy=true |
| B5 | `Req` 空 inputIds | deviceLen=0, maxDeviceLen=maxNewTokens |
| B6 | `PendingReq` 无 chunkedReq | chunkedReq=null，首次调度 |
| B7 | `Batch` 无 readyIds | nextReadyReq 返回 undefined |

## 风险与注意事项

- **兼容性影响**：S0 的 `SamplingParams` 从 `interface` 升级为 `class`，需要确保 `SimRequestMsg` 等使用 `SamplingParams` 的类型仍然兼容。通过类型别名方案确保向后兼容。
- **性能影响**：`Req.inputIds` 使用 `[...spread]` 复制而非原地 push，在高频 decode 场景可能产生 GC 压力，但仿真器不需要极端性能，可接受。若后续成为瓶颈，可改为原地 push + 长度追踪。
- **回滚方案**：所有改动在 `issue-10` 分支，合并前可安全回滚。S0 的 `types.ts` 保留接口兼容层，即使回滚 S1 也不影响 S0 功能。
- **依赖关系**：Issue #9 (S0) 必须已完成并合并。S1 的 `Req`/`Batch` 等类将替代 S0 中的占位接口，但不删除 S0 的类型定义，仅做类型别名转发。
- **阻塞关系**：本 Issue 完成后，K1 (KVCache 基础抽象层)、K5 (内存预算公式)、P0 (通信组基础设施) 才能启动，因为它们依赖 `Req`/`Batch` 等核心数据结构。
