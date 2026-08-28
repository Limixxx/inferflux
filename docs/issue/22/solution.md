---
title: "Issue #22 解决方案"
issue_number: 22
issue_type: Feature
created: 2026-08-28
updated: 2026-08-28
status: draft
review_round: 1
---

# Issue #22 解决方案

## 需求分析

### 问题描述

Issue #22 要求在 `server/src/sglang/parallel/**` 下实现 P1a 层 TP 张量并行仿真组件，包括两大核心模块：

1. **TPSimulator**（§10.2）：TP 张量并行仿真器，封装通信组与内存修正逻辑，在 forward 路径中注入 all-reduce 通信成本
2. **TPCommInfraSimulator**（§10.6）：TP 通信基础设施仿真器，仿真 ZMQ 广播 + gloo barrier + nccl all-reduce 的三层通信成本

同时需要将 TPSimulator 与 MockEngine 进行集成，在 forward_batch 开始/结束时调用 tp_sim.all_reduce_after_attn/mlp，把 total_comm_ticks 累加到 batch 耗时，并回填 ParallelMetrics.tp_* 字段。

### 能力目标

- **TPSimulator**：
  - 权重按 TP 切分：`num_attention_heads / tp_size`、`intermediate_size / tp_size`、`kv_heads = div_even(orig_kv_heads, tp_size)`
  - 逐层通信：`all_reduce_after_attn(batch_size)` 和 `all_reduce_after_mlp(batch_size)`，返回 comm_ticks
  - `total_comm_ticks_per_step` = 每层 2 次 all_reduce × layers
  - tp_size=1 时退化 noop（所有方法返回 0）

- **TPCommInfraSimulator**：
  - ZMQ 广播：`broadcast_all(token_ids_list)` 成本 = list×bytes/bandwidth
  - gloo barrier：`barrier_all_workers` 成本 = latency_us + 2×size×sync_bytes_bw
  - NCCL all-reduce delegate：包装 `SimCommGroup("tp").all_reduce`，加上 `tp_efficiency` 校正系数
  - tp_size=1 时退化 noop

- **MockEngine 集成**：
  - forward_batch 中注入 TP 通信成本
  - ParallelMetrics.tp_* 字段回填

### 影响范围

- **新增文件**：`server/src/sglang/parallel/tp_simulator.ts`、`server/src/sglang/parallel/tp_comm_infra.ts`
- **修改文件**：`server/src/sglang/parallel/index.ts`（导出新增模块）、`server/src/sglang/index.ts`（导出新增类型和类）
- **不修改**：已有 `comm_group.ts`、`topology.ts`、`metrics.ts`、业务调度逻辑、测试代码

### 依赖关系

- **依赖 #21 (P0)**：SimCommGroup + ParallelTopology + ParallelMetrics — **已实现**
- **阻塞 P6**：init_parallel_groups（最终组合）

---

## 改造方案

### 总体思路

在 `server/src/sglang/parallel/` 下新建两个模块文件，分别实现 TPSimulator 和 TPCommInfraSimulator。两个类均依赖 P0 已实现的 SimCommGroup 进行通信成本计算，并利用 ParallelMetrics 进行指标回填。通过 `parallel/index.ts` 和 `sglang/index.ts` 统一导出。

### 详细设计

#### 1. TPSimulator — TP 张量并行仿真器

**文件**：`server/src/sglang/parallel/tp_simulator.ts`

**类设计**：

```typescript
import { SimCommGroup } from "./comm_group";
import type { SimulatorConfig, ModelConfig } from "../types";
import { divEven } from "../core";

export class TPSimulator {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly tpSize: number;
  readonly commGroup: SimCommGroup;

  // 内存修正：本地（单 rank）可见的参数量
  readonly localNumHeads: number;
  readonly localNumKvHeads: number;
  readonly localIntermediate: number;

  // 通信记录
  private _commTicksLog: number[] = [];

  constructor(config: SimulatorConfig, modelConfig: ModelConfig);
}
```

**核心方法**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `allReduceAfterAttn` | `(batchSize: number) => number` | attention 后 all-reduce，数据量 = batch × hidden × dtype |
| `allReduceAfterMlp` | `(batchSize: number) => number` | MLP 后 all-reduce，数据量 = batch × hidden × dtype |
| `totalCommTicksPerStep` | `() => number` | 返回当前 step 累计的 TP 通信 ticks |
| `resetStepComm` | `() => void` | 重置当前 step 的通信记录 |

**内存修正逻辑**：

```
if tpSize > 1:
  localNumHeads = divEven(numAttentionHeads, tpSize)[0]
  localNumKvHeads = divEven(numKvHeads, tpSize)[0]
  localIntermediate = intermediateSize // tpSize
else:
  localNumHeads = numAttentionHeads
  localNumKvHeads = numKvHeads
  localIntermediate = intermediateSize
```

**通信成本计算**：

```
allReduceAfterAttn(batchSize):
  if tpSize <= 1: return 0
  dataBytes = batchSize × hiddenSize × dtypeSize
  ticks = commGroup.allReduce(dataBytes)
  _commTicksLog.push(ticks)
  return ticks

allReduceAfterMlp(batchSize):
  if tpSize <= 1: return 0
  dataBytes = batchSize × hiddenSize × dtypeSize
  ticks = commGroup.allReduce(dataBytes)
  _commTicksLog.push(ticks)
  return ticks

totalCommTicksPerStep():
  return _commTicksLog.reduce((sum, v) => sum + v, 0)
```

**构造函数中创建 SimCommGroup**：

```typescript
this.commGroup = new SimCommGroup({
  groupType: "tp",
  size: config.tpSize,
  networkBandwidthGBps: config.networkBandwidthGBps,
  latencyUs: config.networkLatencyUs,
  efficiency: config.tpEfficiency,
});
```

#### 2. TPCommInfraSimulator — TP 通信基础设施仿真器

**文件**：`server/src/sglang/parallel/tp_comm_infra.ts`

**类设计**：

```typescript
import { SimCommGroup } from "./comm_group";
import type { SimulatorConfig, ModelConfig } from "../types";

export class TPCommInfraSimulator {
  readonly config: SimulatorConfig;
  readonly modelConfig: ModelConfig;
  readonly tpSize: number;

  readonly commGroup: SimCommGroup;

  readonly cpuGroupType: string;  // "gloo"
  readonly gpuGroupType: string;  // "nccl" | "pynccl"

  zmqBroadcastTicks: number = 0;
  barrierTicks: number = 0;

  constructor(config: SimulatorConfig, modelConfig: ModelConfig);
}
```

**核心方法**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `zmqBroadcast` | `(msgSize: number) => number` | ZMQ 广播成本（primary rank → 其他 TP rank） |
| `cpuBarrier` | `() => number` | gloo barrier 成本（固定延迟 1 tick） |
| `gpuAllReduce` | `(dataBytes: number) => number` | nccl all-reduce 成本（委托 SimCommGroup） |
| `broadcastAll` | `(tokenIdsList: number[][]) => number` | 批量 ZMQ 广播 |

**通信成本计算**：

```
zmqBroadcast(msgSize):
  if tpSize <= 1: return 0
  cost = Math.ceil(msgSize / max(1, config.commBandwidthBytesPerTick))
  zmqBroadcastTicks += cost
  return cost

cpuBarrier():
  if tpSize <= 1: return 0
  cost = 1
  barrierTicks += cost
  return cost

gpuAllReduce(dataBytes):
  return commGroup.allReduce(dataBytes)

broadcastAll(tokenIdsList):
  if tpSize <= 1: return 0
  totalBytes = tokenIdsList.reduce((sum, list) => sum + list.length * 4, 0)
  return zmqBroadcast(totalBytes)
```

**构造函数中创建 SimCommGroup**：

```typescript
this.commGroup = new SimCommGroup({
  groupType: "tp",
  size: config.tpSize,
  networkBandwidthGBps: config.networkBandwidthGBps,
  latencyUs: config.networkLatencyUs,
  efficiency: config.tpEfficiency,
});
```

#### 3. MockEngine 集成说明

本 Issue 实现 TPSimulator 和 TPCommInfraSimulator 的工具类，为 MockEngine 的 forward 路径集成做好准备。集成将在后续 Issue（P6: init_parallel_groups）中完成，届时：

- `MockEngine.__init__` 中创建 `this.tpSim = new TPSimulator(config, modelConfig)`
- `forward_batch` 中调用 `tpSim.allReduceAfterAttn(batch.size)` 和 `tpSim.allReduceAfterMlp(batch.size)`
- 通信成本累加到 `metrics.parallel.tpCommTicks` 和 `metrics.parallel.tpAllReduceCount`

**集成伪代码**（供后续 Issue 参考）：

```typescript
// MockEngine.forward_batch 中：
const totalComm = this.mockModelForwardWithTp(batch);
if (totalComm > 0) {
  this.metrics.parallel.tpCommTicks += totalComm;
  this.metrics.parallel.tpAllReduceCount += this.modelConfig.numLayers * 2;
}

mockModelForwardWithTp(batch: Batch): number {
  let totalComm = 0;
  for (let i = 0; i < this.modelConfig.numLayers; i++) {
    totalComm += this.tpSim.allReduceAfterAttn(batch.size);
    totalComm += this.tpSim.allReduceAfterMlp(batch.size);
  }
  return totalComm;
}
```

#### 4. ParallelMetrics 新增字段说明

ParallelMetrics 已在 P0 中定义了 `tpCommTicks`、`tpAllReduceCount`、`tpWeightBytes` 字段，本 Issue 无需修改 metrics.ts。TPSimulator 和 TPCommInfraSimulator 的指标写入将在集成阶段完成。

#### 5. 导出更新

**`parallel/index.ts` 新增导出**：

```typescript
export { TPSimulator } from "./tp_simulator";
export { TPCommInfraSimulator } from "./tp_comm_infra";
```

**`sglang/index.ts` 新增导出**：

```typescript
export {
  TPSimulator,
  TPCommInfraSimulator,
} from "./parallel";
```

### 修改点清单

1. **新建** `server/src/sglang/parallel/tp_simulator.ts` — TPSimulator 实现
2. **新建** `server/src/sglang/parallel/tp_comm_infra.ts` — TPCommInfraSimulator 实现
3. **修改** `server/src/sglang/parallel/index.ts` — 导出 TPSimulator 和 TPCommInfraSimulator
4. **修改** `server/src/sglang/index.ts` — 导出新增类

---

## 测试设计

### 验收测试用例清单

#### TPSimulator 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T1 | tpSize=1 全 noop | tpSize=1 时 allReduceAfterAttn/allReduceAfterMlp 返回 0，localNumHeads=原值 |
| T2 | tpSize=2 内存修正 | localNumHeads=numAttentionHeads/2, localNumKvHeads=divEven(kvHeads,2)[0], localIntermediate=intermediateSize/2 |
| T3 | allReduceAfterAttn 正值 | tpSize=2 时返回正值，公式 = batch×hidden×dtype → SimCommGroup.allReduce |
| T4 | allReduceAfterMlp 正值 | 与 T3 相同数据量，返回相同值 |
| T5 | totalCommTicksPerStep 累加 | 多次调用后 total 等于各次之和 |
| T6 | resetStepComm 清零 | 调用后 totalCommTicksPerStep=0 |
| T7 | divEven GQA kv_heads 复制 | numKvHeads=2, tpSize=4 时 divEven 允许复制，localNumKvHeads=1 |
| T8 | SimCommGroup 效率因子 | TPSimulator 使用 config.tpEfficiency 传递给 SimCommGroup |

#### TPCommInfraSimulator 测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T9 | tpSize=1 全 noop | zmqBroadcast/cpuBarrier/gpuAllReduce 均返回 0 |
| T10 | zmqBroadcast 正值 | tpSize=2 时返回 Math.ceil(msgSize/bandwidth) |
| T11 | cpuBarrier 固定 1 tick | tpSize=2 时始终返回 1 |
| T12 | gpuAllReduce 委托 | 结果与 SimCommGroup("tp").allReduce 相同 |
| T13 | broadcastAll 批量 | 多个 token_ids_list 正确汇总 bytes 后调用 zmqBroadcast |
| T14 | zmqBroadcastTicks 累加 | 多次调用后 zmqBroadcastTicks 等于各次返回值之和 |
| T15 | barrierTicks 累加 | 多次调用后 barrierTicks 等于各次返回值之和 |
| T16 | cpuGroupType/gpuGroupType 读取 | 正确存储 config 中的 tpCpuGroupType/tpGpuGroupType |

#### 组合测试

| 编号 | 测试名称 | 描述 |
|------|---------|------|
| T17 | TPSimulator + TPCommInfraSimulator 独立 | 两者可独立构造，互不干扰 |
| T18 | tpSize=1 退化单实例 | 全部返回 0，不影响现有行为 |
| T19 | 与 ParallelMetrics 字段对应 | TPSimulator 的通信 ticks 可正确写入 ParallelMetrics.tpCommTicks |

### 边界条件覆盖

| 编号 | 边界条件 | 预期行为 |
|------|---------|---------|
| B1 | batchSize=0 | allReduceAfterAttn(0) 数据量为 0，SimCommGroup.allReduce(0) 返回 latency（非 0 当 size>1） |
| B2 | msgSize=0 | zmqBroadcast(0) 返回 0 |
| B3 | tpSize 极大 | allReduce 成本随 size 增大趋近于 2×bytes/bw + latency |
| B4 | numKvHeads=0 | divEven(0, tpSize) 返回全 0，localNumKvHeads=0 |
| B5 | commBandwidthBytesPerTick=0 | zmqBroadcast 中 Math.ceil(msgSize/max(1,0)) = msgSize，即每 byte 1 tick |
| B6 | tpEfficiency=1.0 | 结果与无效率因子一致 |

---

## 风险与注意事项

### 兼容性影响

- **纯新增**：TPSimulator 和 TPCommInfraSimulator 是全新的类，不修改任何已有代码的行为
- **导出变更**：`parallel/index.ts` 和 `sglang/index.ts` 仅新增导出，不影响现有导出项
- **ParallelMetrics 字段**：已在 P0 中定义，本 Issue 仅作为写入方使用，不修改字段定义
- **MockEngine 集成**：本 Issue 不修改 MockEngine 代码，集成留给 P6 Issue

### 性能影响

- TPSimulator 所有方法为 O(1) 纯算术，无性能风险
- TPCommInfraSimulator 的 zmqBroadcast/cpuBarrier 为 O(1)，broadcastAll 为 O(list.length)
- 两个类的构造开销极低，仅创建一个 SimCommGroup 实例

### 回滚方案

- 新增文件删除即可回滚
- `index.ts` 的导出修改通过 git revert 即可恢复
- 无数据库/持久化变更，回滚无数据风险
