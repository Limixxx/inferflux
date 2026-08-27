# SGLang 框架仿真模拟器技术报告

## 1. 项目目标

本报告旨在指导开发一个 **SGLang 框架仿真模拟器**，该模拟器能够：

1. **忠实地仿真** SGLang 推理框架的调度逻辑、KV cache 管理、请求生命周期、内存分配策略等系统层行为
2. **模拟实现** GPU 计算相关的组件（模型前向传播、attention kernel、CUDA Graph 等），用轻量级 mock 替代真实 GPU 计算
3. **不包含** 模型本身的实现（即不需要实现 Llama/Qwen 等模型的 Transformer 层计算），而是通过配置参数描述模型特征

模拟器的核心价值：在不依赖 GPU 的环境下，研究 SGLang 的调度策略、内存管理、批处理优化等系统层问题。

---

## 2. SGLang 架构总览

### 2.1 进程拓扑

SGLang 采用多进程架构，通过 ZMQ 进行控制面通信，通过 NCCL/torch.distributed 进行数据面通信：

```
User Request
    │
    ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  API Server   │────▶│  TokenizerManager │────▶│  Scheduler (TP Rank 0) │
│  (FastAPI)    │◀────│  (tokenize/        │◀────│  - PrefillManager      │
│  /v1/chat     │     │   detokenize)     │     │  - DecodeManager       │
└──────────────┘     └──────────────────┘     │  - CacheManager         │
                          │  ZMQ Push/Pull       │  - TableManager         │
                          │  (DetokenizeMsg)     └────────┬────────────────┘
                          │                               │ ZMQ Pub/Sub (broadcast)
                          │                               ▼
                          │                    ┌─────────────────────┐
                          │                    │  Scheduler (TP Rank 1..N) │
                          │                    │  (同 Rank 0 结构)           │
                          │                    └────────┬────────────────┘
                          │                             │
                          ▼                             ▼
                    ┌──────────────────────────────────────────┐
                    │              Engine (per TP rank)            │
                    │  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
                    │  │ Model    │ │ KV Cache  │ │ GraphRunner │  │
                    │  │ (forward)│ │ (Pool +   │ │ (CUDA Graph)│  │
                    │  │          │ │  RadixTree)│ │             │  │
                    │  └─────────┘ └──────────┘ └─────────────┘  │
                    │  ┌─────────────────┐ ┌────────┐              │
                    │  │ Attention Backend │ │ Sampler │             │
                    │  │ (FlashInfer/FA)   │ │         │             │
                    │  └─────────────────┘ └────────┘              │
                    └──────────────────────────────────────────────┘
```

### 2.2 核心数据结构

以下数据结构是模拟器必须实现的：

#### 2.2.1 SamplingParams

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0       # 0 表示 greedy
    top_k: int = -1                # -1 不限制
    top_p: float = 1.0             # 1.0 不限制
    ignore_eos: bool = False       # 是否忽略 EOS token
    max_tokens: int = 1024         # 最大生成 token 数

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

#### 2.2.2 Req（请求状态）

```python
@dataclass(eq=False)
class Req:
    input_ids: List[int]      # 输入 token 序列（仿真中用 list[int] 替代 tensor）
    table_idx: int            # 在 page_table 中的行索引
    cached_len: int           # 已在 KV cache 中的长度
    output_len: int           # 预期输出长度
    uid: int                  # 全局唯一 ID
    sampling_params: SamplingParams
    cache_handle: 'BaseCacheHandle'  # RadixCache 节点句柄

    def __post_init__(self) -> None:
        # 注意：device_len 和 max_device_len 是普通属性（非 property）
        # complete_one() 需要直接修改 device_len
        self.device_len = len(self.input_ids)       # 当前在设备上的总长度
        self.max_device_len = len(self.input_ids) + self.output_len  # 最大设备长度（初始化时固定，不随 append_host 增长）
        # DP rank 分配（§10.3.3，默认 0 = 单实例无 DP）
        self.dp_rank: int = 0
        # 注意：使用 <= 而非 <，允许全缓存命中（cached_len == device_len，extend_len=0）的情况
        assert 0 <= self.cached_len <= self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:       # 剩余可解码长度
        return self.max_device_len - self.device_len
    @property
    def extend_len(self) -> int:       # 需要扩展的长度
        return self.device_len - self.cached_len
    @property
    def can_decode(self) -> bool:      # 是否可以继续 decode
        return self.remain_len > 0

    def complete_one(self) -> None:    # 完成一个 token 的 decode（在采样前调用）
        self.cached_len = self.device_len
        self.device_len += 1
    def append_host(self, next_token: int) -> None:  # 追加新 token
        self.input_ids = self.input_ids + [next_token]
```

> **关键区别**：`device_len` 和 `max_device_len` 是 `__post_init__` 设置的**普通可变属性**，不是 `@property`。`complete_one()` 直接执行 `self.device_len += 1`，而 `append_host()` 只追加 `input_ids`，不修改 `max_device_len`（它在初始化时固定为 `input_len + output_len`）。

#### 2.2.3 Batch（批处理）

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: List[int] = field(default_factory=list)     # 由 scheduler 填充（仿真中用 list 替代 tensor）
    positions: List[int] = field(default_factory=list)     # 由 scheduler 填充
    out_loc: List[int] = field(default_factory=list)        # KV cache 写入位置，由 scheduler 填充
    padded_reqs: List[Req] = field(default_factory=list)   # CUDA Graph padding 后的请求列表
    attn_metadata: Any = None       # 由 attention backend 填充

    @property
    def is_prefill(self) -> bool: return self.phase == "prefill"
    @property
    def is_decode(self) -> bool: return self.phase == "decode"
    @property
    def size(self) -> int: return len(self.reqs)
    @property
    def padded_size(self) -> int: return len(self.padded_reqs)
```

#### 2.2.4 Context（全局上下文）

```python
@dataclass
class Context:
    page_size: int
    page_table: List[List[int]] | None = field(default=None, init=False)     # 后续设置
    attn_backend: 'BaseAttnBackend' = field(default=None, init=False) # 后续设置
    moe_backend: 'BaseMoeBackend' = field(default=None, init=False)   # 后续设置
    kv_cache: 'BaseKVCachePool' = field(default=None, init=False)     # 后续设置
    _batch: Batch | None = field(default=None, init=False)            # 当前活跃的 batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch"
        return self._batch
```

> 注意：`page_table`、`attn_backend`、`moe_backend`、`kv_cache` 使用 `field(init=False)` 模式，构造时只传 `page_size`，后续由 MockEngine 赋值。

### 2.3 请求生命周期

一个请求从进入到完成的完整状态机：

```
用户请求到达 API Server
    │
    ▼ (HTTP POST /v1/chat/completions)
TokenizeMsg → TokenizerManager 将文本转为 input_ids
    │
    ▼ (ZMQ Push → Scheduler backend addr)
Scheduler.recv_requests() — Rank 0 接收，广播到其他 Rank
    │
    ▼
Scheduler._process_one_msg(UserMsg)
    │ 检查 input_len < max_seq_len
    │ 调整 max_tokens = min(max_tokens, max_seq_len - input_len)
    ▼
PrefillManager.add_one_req(msg) — 加入 pending_list
    │
    ▼ (等待下一次调度)
PrefillManager.schedule_next_batch(prefill_budget)
    │ 1. 创建 PrefillAdder(token_budget=prefill_budget, reserved_size=inflight_tokens)
    │ 2. 遍历 pending_list，对每个请求：
    │    a. CacheManager.match_req(req) → 匹配 RadixCache 前缀
    │    b. 估算 needed = extend_len + output_len
    │    c. 检查 available_size >= needed + reserved_size
    │    d. CacheManager.lock(handle) — 锁定前缀缓存
    │    e. TableManager.allocate() — 分配 table slot
    │    f. 复制 cached tokens 到 token_pool
    │    g. 决定是否 chunked (chunk_size < remain_len)
    │ 3. 返回 Batch(reqs, phase="prefill")
    ▼
Scheduler._prepare_batch(batch)
    │ 1. GraphRunner.pad_batch(batch) — padding 到 CUDA graph 尺寸
    │ 2. CacheManager.allocate_paged(req) — 逐 req 分配 KV cache 页
    │ 3. 计算 positions: [cached_len, device_len) 的连续序列
    │ 4. 计算 input_mapping: (table_idx, positions) → token_pool 索引
    │ 5. 计算 write_mapping: (table_idx, device_len or -1)
    │ 6. batch.out_loc = page_table[input_mapping]
    │ 7. AttnBackend.prepare_metadata(batch)
    ▼
Scheduler._forward(forward_input)
    │ 1. batch.input_ids = token_pool[input_mapping]
    │ 2. Engine.forward_batch(batch, sample_args)
    │    a. if GraphRunner.can_use_cuda_graph(batch): graph_runner.replay(batch)
    │    else: model.forward()
    │    b. 对每个 req: req.complete_one()
    │    c. Sampler.sample(logits, args) → next_tokens
    │    d. next_tokens_gpu → next_tokens_cpu (async copy + event)
    │ 3. token_pool[write_mapping] = next_tokens_gpu
    │ 4. DecodeManager.filter_reqs(batch.reqs) — 更新可 decode 集合
    ▼
Scheduler._process_last_data(last_data)
    │ 1. 等待 copy_done_event.synchronize()
    │ 2. 对每个 req:
    │    a. 如果是 ChunkedReq: 跳过（继续 prefill）
    │    b. req.append_host(next_token)
    │    c. finished = not req.can_decode or (next_token == eos and not ignore_eos)
    │    d. 如果 finished:
    │       - DecodeManager.remove_req(req)
    │       - _free_req_resources(req): TableManager.free + CacheManager.cache_req(finished=True)
    │    e. 如果 prefill 且未完成: CacheManager.cache_req(req, finished=False)
    │ 3. 发送 DetokenizeMsg 列表回 TokenizerManager
    ▼
TokenizerManager → detokenize → 返回给用户
```

---

## 3. 仿真 vs 模拟分类

### 3.1 分类原则

- **仿真（Simulate）**：忠实地重新实现组件的逻辑和算法，但不依赖 GPU。这些组件的行为对系统层性能有直接影响，必须精确模拟。
- **模拟（Emulate/Mock）**：用轻量级 stub 替代，只保留接口契约和行为签名，不实现内部计算。这些组件的计算行为不属于系统层研究范围。
- **排除（Exclude）**：模型本身的 Transformer 层计算不在模拟器范围内。

### 3.2 分类总表

| 组件 | 分类 | 理由 |
|------|------|------|
| API Server (FastAPI) | 模拟 | HTTP 路由不影响调度逻辑，用简单 TCP/socket stub |
| TokenizerManager | 模拟 | tokenize/detokenize 是确定性映射，用字典查表 mock |
| Scheduler 主循环 | **仿真** | 调度逻辑是核心研究对象 |
| PrefillManager | **仿真** | prefill 调度策略、chunked prefill、token budget 直接影响性能 |
| DecodeManager | **仿真** | decode 集合管理、inflight token 计算影响调度决策 |
| CacheManager | **仿真** | KV cache 页分配、前缀匹配、eviction 是核心系统行为 |
| TableManager | **仿真** | page_table 和 token_pool 的分配/释放是基础设施 |
| RadixCache (RadixPrefixCache) | **仿真** | RadixTree 前缀匹配、插入、分裂、eviction 是核心算法 |
| KV Cache Pool (MHAKVCachePool) | 模拟 | 实际 K/V tensor 存储不需要真实数据，只需内存记账 |
| Engine | 模拟 | 模型 forward 的协调逻辑保留，实际计算用 mock |
| Model.forward() | 排除 | 模型计算不在范围内 |
| GraphRunner (CUDA Graph) | **仿真** | batch size 分桶、replay 决策、padding 逻辑需仿真 |
| Sampler | 模拟 | 采样逻辑用 simple random/greedy mock |
| Attention Backend | 模拟 | FlashInfer/FA 的 kernel 调用用 stub |
| MoE Backend | 模拟 | MoE routing 用 mock |
| ZMQ 通信 | 模拟 | 用 in-process 队列替代 ZMQ |
| NCCL/TP 通信 | 模拟 | 单进程内不需要真实 TP 通信 |
| 内存管理（显存预算） | **仿真** | mem_fraction_static、num_pages 计算、OOM 预测需仿真 |
| Overlap Scheduling | **仿真** | 双 stream 重叠是关键调度优化，需仿真时序 |
| 模型权重加载 | 排除 | 不需要真实权重 |
| 模型配置（ModelConfig） | 模拟 | 用参数描述模型特征（num_layers, hidden_size, num_kv_heads 等） |

### 3.3 仿真组件详细规格

以下组件需要完整仿真，即重新实现其全部逻辑：

#### 3.3.1 Scheduler 主循环

**职责**：从消息队列接收请求，调度下一批 batch，执行 forward，处理上一批结果。

**仿真要点**：

1. **overlap_loop 时序仿真**：原始实现使用双 CUDA stream（scheduler stream + engine stream）实现重叠。仿真中需要用时间步模型：
   - 每个 tick 分为 CPU phase（调度+消息处理）和 GPU phase（forward+采样）
   - overlap 模式下，上一批的 post-processing 与当前批的 pre-scheduling 重叠
   - 非 overlap 模式下，串行执行

2. **调度优先级**：prefill 优先于 decode
   ```python
   def _schedule_next_batch(self) -> ForwardInput | None:
       batch = (
           self.prefill_manager.schedule_next_batch(self.prefill_budget)
           or self.decode_manager.schedule_next_batch()
       )
       return self._prepare_batch(batch) if batch else None
   ```

3. **消息处理**：
   - `UserMsg`: 创建 PendingReq，加入 prefill_manager
   - `AbortBackendMsg`: 从 prefill_manager 或 decode_manager 中移除
   - `ExitMsg`: 退出循环
   - `BatchBackendMsg`: 递归处理

4. **结果处理 (_process_last_data)**：
   - 区分 ChunkedReq（跳过，继续 prefill）和普通 Req
   - 判断 finished：`not can_decode` 或 `next_token == eos`
   - finished 时释放资源：TableManager.free + CacheManager.cache_req(finished=True)
   - prefill 非完成时：CacheManager.cache_req(finished=False)

**仿真接口**（完整实现见 9.4 和 9.11 节）：
```python
class SimScheduler:
    def __init__(self, config: 'SimulatorConfig'):
        self.engine = MockEngine(config)
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.finished_reqs: Set[Req] = set()
        self.eos_token_id = config.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.overlap_enabled = config.enable_overlap
        self.offline_mode = config.offline_mode
        self.last_data: ForwardData | None = None  # overlap 模式专用
        self.last_batch: Batch | None = None  # 供外部指标收集

    def run_tick(self, incoming_msgs: List[BaseBackendMsg]) -> List[DetokenizeMsg]:
        """执行一个调度 tick，返回需要回传的结果消息"""
        if self.overlap_enabled:
            return self._overlap_tick(incoming_msgs)
        else:
            return self._normal_tick(incoming_msgs)

    # _normal_tick / _overlap_tick / _process_last_data / _forward / _prepare_batch
    # / _schedule_next_batch / _free_req_resources 的完整实现见 9.11 节
```

#### 3.3.2 PrefillManager + PrefillAdder

**职责**：管理待 prefill 的请求队列，根据 token budget 和 cache 可用性决定哪些请求可以进入当前 batch。

**仿真要点**：

1. **PendingReq 管理**：维护一个 FIFO 队列
2. **Chunked Prefill**：当一个请求的 extend_len > token_budget 时，只处理一部分（chunk_size = min(token_budget, remain_len)），创建 ChunkedReq
3. **资源估算**：
   - `estimated_len = extend_len + output_len`（请求需要的总 token 数）
   - 检查 `estimated_len + reserved_size <= cache_manager.available_size`
   - `reserved_size` 初始为 `decode_manager.inflight_tokens`（当前 decode 在途 token 数 + 每 req 预留 1 page）
4. **前缀匹配**：`CacheManager.match_req(req)` 返回 `MatchResult(cuda_handle)`，其中 `cached_len` 表示已缓存长度
5. **ChunkedReq 特殊行为**：
   - `can_decode` 永远返回 False
   - `append_host` 抛出 NotImplementedError（不应被采样）
   - 在 `_process_last_data` 中被跳过

**仿真接口**：
```python
@dataclass
class PrefillAdder:
    token_budget: int          # 当前 tick 的 prefill token 预算
    reserved_size: int         # 已预留的 token 数（inflight decode）
    cache_manager: CacheManager
    table_manager: TableManager

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # 1. 如果是已 chunked 的请求，继续处理剩余部分
        # 2. 对新请求：
        #    a. match_req → 获取 cached_len
        #    b. 估算 needed = extend_len + output_len
        #    c. 检查 available_size >= needed + reserved_size
        #    d. lock handle
        #    e. allocate table_idx
        #    f. 复制 cached tokens + page entries 到 token_pool/page_table
        #    g. 复制 extend tokens 到 token_pool（供 _forward 读取 batch.input_ids）
        #    h. 决定 chunk_size, 创建 Req 或 ChunkedReq
```

#### 3.3.3 DecodeManager

**职责**：管理可 decode 的请求集合，生成 decode batch。

**仿真要点**：

1. **running_reqs 集合**：`filter_reqs` 在每次 forward 后更新：`{req for req in running_reqs ∪ new_reqs if req.can_decode}`
2. **inflight_tokens 计算**：
   ```python
   @property
   def inflight_tokens(self) -> int:
       tokens_reserved = (page_size - 1) * len(running_reqs)  # 每 req 预留 1 page
       return sum(req.remain_len for req in running_reqs) + tokens_reserved
   ```
3. **调度策略**：按 uid 排序，生成 `Batch(reqs=sorted_list, phase="decode")`
4. **abort**：通过 uid 查找并移除

#### 3.3.4 CacheManager

**职责**：封装 KV cache 页分配、前缀缓存管理、eviction。

**仿真要点**：

1. **free_slots 管理**：维护可用页的物理位置列表。页对齐：`free_slots = [0, page_size, 2*page_size, ...]`
2. **available_size 计算**：
   ```python
   @property
   def available_size(self) -> int:
       return prefix_cache.size_info.evictable_size + len(free_slots) * page_size
   ```
3. **页分配 (allocate_paged)**：
   - 对每个 req，计算需要的页数：`last_page = ceil(device_len / page_size) - ceil(cached_len / page_size)`
   - 如果 free_slots 不足，触发 eviction
   - 将分配的物理位置写入 page_table
4. **cache_req 逻辑**（最复杂）：
   ```python
   def cache_req(self, req: Req, *, finished: bool):
       insert_ids = req.input_ids[:req.cached_len]
       page_indices = page_table[req.table_idx, :req.cached_len]
       old_handle = req.cache_handle
       # 插入前缀到 RadixCache
       cached_len, new_handle = prefix_cache.insert_prefix(insert_ids, page_indices)
       # 解锁旧 handle
       self.unlock(old_handle)
       # 释放已存在于缓存中的重复部分
       self._free(page_indices[old_handle.cached_len:cached_len])
       if finished:
           # 释放尾部
           self._free(page_indices[new_handle.cached_len:])
       else:
           # 保留尾部，更新 handle
           req.cache_handle = new_handle
           self.lock(new_handle)
   ```
5. **lazy_free_region**：在一个上下文范围内收集所有 free 操作，退出时批量合并 free_slots
6. **eviction 触发**：`_allocate` 中，如果 `needed_pages > len(free_slots)`，调用 `prefix_cache.evict((needed_pages - free_pages) * page_size)`

#### 3.3.5 RadixPrefixCache (RadixTree)

**职责**：基于 RadixTree 的前缀缓存，支持前缀匹配、插入、分裂、eviction。

**仿真要点**：

1. **RadixTreeNode 结构**：
   - `children: Dict[key, RadixTreeNode]` — key 由 `key_fn(tokens)` 生成（page_size=1 时为 token 值，page_size>1 时为 tuple）
   - `ref_count: int` — 引用计数，>0 时不可 evict
   - `timestamp: int` — LRU 时间戳
   - `_key: List[int]` — 该节点存储的 token 序列（仿真中用 list[int] 替代 tensor）
   - `_value: List[int]` — 对应的 KV cache 物理位置（页索引列表）

2. **match_prefix（前缀匹配）**：
   - 从 root 开始，逐层匹配子节点的 key
   - 如果部分匹配（`match_len < node.length`），执行 `split_at(match_len)`
   - 返回 `RadixCacheHandle(cached_len, node)`

3. **insert_prefix（插入）**：
   - 先 walk tree 找到最长匹配
   - 如果 `prefix_len < insert_len`，创建新节点
   - `insert_len = align_down(len(input_ids), page_size)` — 只插入页对齐的部分
   - 返回 `InsertResult(cached_len, handle)`

4. **split_at（节点分裂）**：
   - 创建新节点，继承前 `pos` 个 token
   - 原节点缩进到 `[pos:]`
   - 新节点成为原节点父节点的子节点，原节点成为新节点的子节点

5. **evict（LRU eviction）**：
   - 收集所有 `ref_count == 0` 的叶子节点
   - 用最小堆按 timestamp 排序
   - 逐个弹出并释放，直到满足 size 需求
   - 弹出后如果父节点变为叶子且 `ref_count == 0`，加入堆

6. **lock/unlock**：
   - lock: 从 handle.node 向上遍历到 root，`ref_count += 1`，从 evictable 移入 protected
   - unlock: 反向，`ref_count -= 1`，ref_count 归零时从 protected 移入 evictable

#### 3.3.6 TableManager

**职责**：管理 page_table 和 token_pool 的 slot 分配。

**仿真要点**：

1. **page_table 结构**：`[max_running_req + 1, max_seq_len]` 的 int32 矩阵（+1 for dummy req）
2. **token_pool**：`[max_running_req + 1, max_seq_len]` 的 int32 矩阵，存储每个 table_idx 下的 token IDs（注意：源码中 token_pool 在 TableManager 内部创建，不是构造参数）
3. **allocate/free**：维护可用 table_idx 的栈/集合
   ```python
   class TableManager:
       def __init__(self, max_running_req: int, page_table):
           self.page_table = page_table
           # token_pool 在内部创建，与 page_table 相同 shape
           self.token_pool = [[0] * page_table.shape[1]
                              for _ in range(max_running_req + 1)]
           self.free_table_indices = list(range(max_running_req))  # 可用 table_idx
       def allocate(self) -> int:
           return self.free_table_indices.pop()
       def free(self, table_idx: int):
           self.free_table_indices.append(table_idx)
       @property
       def available_size(self) -> int:
           return len(self.free_table_indices)
   ```

#### 3.3.7 GraphRunner (CUDA Graph)

**职责**：管理 CUDA Graph 的捕获与 replay，决定何时使用 graph。

**仿真要点**：

1. **batch size 分桶**：
   ```python
   def _determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory):
       if cuda_graph_bs is not None:
           return cuda_graph_bs
       # 默认: [1, 2, 4] + range(8, cuda_graph_max_bs+1, 8)
       return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
   ```

2. **can_use_cuda_graph**：
   ```python
   def can_use_cuda_graph(self, batch: Batch) -> bool:
       return batch.is_decode and batch.size <= self.max_graph_bs
   ```

3. **pad_batch**：找到最小的 `bs >= batch.size` 的 graph 尺寸，用 dummy_req 填充
   ```python
   def pad_batch(self, batch: Batch):
       padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size) \
           if self.can_use_cuda_graph(batch) else batch.size
       batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
   ```

4. **replay**：仿真中只需记录 replay 调用，不需要真实 graph 执行
   - 模拟成本：replay 的 GPU 时间开销远小于 eager forward
   - 可用配置参数控制：`graph_replay_cost_ticks` vs `eager_forward_cost_ticks`

5. **capture 仿真**：
   - 不需要真实捕获
   - 只需记录哪些 batch size 被捕获
   - 可选：模拟 capture 的内存开销

#### 3.3.8 Overlap Scheduling 时序（DEPRECATED）

**职责**：仿真 CPU 调度与 GPU 计算的重叠。

> **[DEPRECATED]** 本节为初步设计，已被第 9.4 节的细化模型取代。9.4 提供了更忠实的 `last_data` 延迟处理机制和正确的执行顺序。本节保留作为概念性参考，**实现时请以 9.4 节为准**。

**仿真要点**：

原始实现使用两个 CUDA stream：
- `self.stream`（scheduler stream）：消息处理、batch 准备、结果处理
- `self.engine.stream`（engine stream）：model forward、采样

overlap_loop 的执行模式：
```
Tick N:
  1. [scheduler stream] 接收消息 + 调度下一批 (CPU phase)
  2. [engine stream] forward 当前批 (GPU phase) ← 与 step 1 的前半部分重叠
  3. [scheduler stream] 处理上一批结果 (CPU phase) ← 与 step 2 的后半部分重叠
```

仿真模型（DEPRECATED — 勿实现，仅作概念参考）：

```python
# ⚠️ 以下代码使用已废弃的 last_gpu_finish_tick 概念，请勿实现。
# 正确实现见 §9.4 的 _overlap_tick，使用 last_data 延迟处理机制。
def run_tick_overlap(self, incoming_msgs):
    # Phase 1: CPU - 接收消息 + 调度 (与上一批 GPU 计算重叠)
    for msg in incoming_msgs:
        self._process_one_msg(msg)
    forward_input = self._schedule_next_batch()

    # Phase 2: GPU - forward (如果上一批还在 GPU 上，需等待)
    if self.last_gpu_finish_tick > self.current_tick:
        self.current_tick = self.last_gpu_finish_tick  # 等待 GPU

    forward_output = None
    if forward_input:
        forward_output = self._forward(forward_input)
        self.last_gpu_finish_tick = self.current_tick + forward_output.gpu_duration

    # Phase 3: CPU - 处理结果 (可以与下一批 GPU 重叠)
    reply = self._process_last_data(forward_input, forward_output)
    self.current_tick += 1  # CPU tick
    return reply
```

#### 3.3.9 内存预算仿真

**职责**：模拟 SGLang 的显存分配策略。

**仿真要点**：

1. **辅助函数定义**：
   ```python
   def estimate_model_memory(model_config, dtype_size: int) -> int:
       """估算模型权重占用的显存（bytes）"""
       # 粗略估算：每层参数量 ≈ hidden² × 12（QKV + FFN + embed）
       params = model_config.num_layers * model_config.hidden_size * model_config.hidden_size * 12
       return params * dtype_size

   def estimate_graph_buffer(cuda_graph_bs: list, model_config) -> int:
       """估算 CUDA Graph buffer 占用的显存（bytes）"""
       if not cuda_graph_bs:
           return 0
       max_bs = max(cuda_graph_bs)
       # 每层 buffer ≈ max_bs × hidden × 4（中间激活、logits 等）
       return max_bs * model_config.hidden_size * model_config.num_layers * 4
   ```

2. **mem_fraction_static 计算**：
   ```python
   def calculate_memory_budget(config, model_config, total_memory):
       """计算可分配的页数。total_memory 为 GPU 总显存（字节）。"""
       # 模型权重占用
       model_memory = estimate_model_memory(model_config, config.dtype_size)
       # CUDA Graph buffer 占用（需先计算，从可用内存中扣除）
       graph_buffer = estimate_graph_buffer(config.cuda_graph_bs, model_config)
       # 剩余可用 = 比例预算 - 模型权重 - graph buffer
       available = int(config.memory_ratio * total_memory) - model_memory - graph_buffer
       # KV cache 每页大小
       # div_even 返回每 GPU 的 KV head 分布列表，sum 后得到总 KV head 数
       kv_heads_per_gpu = sum(div_even(model_config.num_kv_heads, config.tp_size, allow_replicate=True))
       cache_per_page = (
           2  # key + value
           * model_config.head_dim
           * kv_heads_per_gpu
           * config.page_size
           * config.dtype_size  # float16=2, bfloat16=2, float8=1
           * model_config.num_layers
       )
       # 可分配的页数（OOM 保护：负数时返回 0，由调用方触发 OOM 处理）
       num_pages = max(0, available // cache_per_page)
       return num_pages, model_memory, graph_buffer
   ```

2. **OOM 预测**：当 `num_pages < 1` 时触发 OOM

> 注意：显存参数（`total_gpu_memory`、`memory_ratio`、`dtype_size`）已统一在 `SimulatorConfig` 中定义（见 §4.2），无需单独的 `SimMemoryConfig`。`calculate_memory_budget` 直接从 `SimulatorConfig` 读取这些字段。

### 3.4 模拟组件详细规格

以下组件用 mock/stub 实现，只保留接口契约：

#### 3.4.0 全局上下文工具函数

```python
_global_ctx = None

def set_global_ctx(ctx):
    """设置全局 Context，供模型层通过 get_global_ctx() 访问"""
    global _global_ctx
    _global_ctx = ctx

def get_global_ctx():
    """获取全局 Context（模型层通过此函数访问当前 batch 和 KV cache）"""
    return _global_ctx
```

#### 3.4.1 MockEngine

```python
class MockEngine:
    """模拟 Engine，不执行真实模型计算"""
    # 注意：本节为接口规格，完整实现见 §9.11

    def __init__(self, config: SimulatorConfig):
        self.model_config = config.model_config  # 模型配置参数
        self.device = "cpu"  # 不需要真实 GPU
        self.stream = MockStream()
        self.dtype = "float16"  # 仿真中固定为 float16
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 内存预算计算
        self.num_pages = self._calculate_num_pages(config)
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)

        # 模拟 KV cache pool（只记账，不存真实数据）
        self.ctx.kv_cache = MockKVCachePool(config.model_config, self.num_pages, config.page_size)

        # 模拟 page_table
        self.page_table = [[0] * self.max_seq_len
                           for _ in range(config.max_running_req + 1)]

        # Mock backends
        self.attn_backend = MockAttnBackend()
        self.ctx.attn_backend = self.attn_backend
        if config.model_config.is_moe:
            self.moe_backend = MockMoeBackend()
            self.ctx.moe_backend = self.moe_backend

        # Mock sampler
        self.sampler = MockSampler(config.model_config.vocab_size, config.mock_sample_mode)

        # dummy_req 用于 CUDA Graph padding
        self.dummy_req = Req(
            input_ids=[0], table_idx=config.max_running_req,
            cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,
        )
        self.dummy_req.device_len = 1
        self.dummy_req.max_device_len = 1
        self.page_table[self.dummy_req.table_idx] = [num_tokens] * self.max_seq_len

        # 仿真 GraphRunner（接收 dummy_req）
        self.graph_runner = SimGraphRunner(config, self.model_config, self.dummy_req)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)  # mock logits
            else:
                logits = self._mock_model_forward(batch)   # mock logits

        for req in batch.reqs:
            if not isinstance(req, ChunkedReq):
                req.complete_one()  # ChunkedReq 不生成 token，跳过

        # logits 行数 = len(batch.reqs)（不含 padded_reqs），切片是防御性操作
        next_tokens = self.sampler.sample(logits[:batch.size], args)
        return ForwardOutput(
            next_tokens_gpu=next_tokens,
            next_tokens_cpu=list(next_tokens),
            copy_done_event=MockEvent(),
        )

    def _mock_model_forward(self, batch: Batch) -> List[List[float]]:
        """生成 mock logits，shape = [batch_size, vocab_size]"""
        import random
        return [[random.random() for _ in range(self.model_config.vocab_size)]
                for _ in range(batch.size)]

    def _calculate_num_pages(self, config: 'SimulatorConfig') -> int:
        """计算可用页数（见 3.3.9 内存预算仿真）"""
        if config.num_pages is not None:
            return config.num_pages
        # 自动计算：基于内存预算
        num_pages, _, _ = calculate_memory_budget(
            config, config.model_config, config.total_gpu_memory
        )
        return num_pages
```

#### 3.4.2 MockKVCachePool

```python
class MockKVCachePool(BaseKVCachePool):
    """模拟 KV cache 存储，只做内存记账"""

    def __init__(self, model_config, num_pages, page_size):
        self._num_pages = num_pages
        self._page_size = page_size
        self._num_layers = model_config.num_layers
        self._head_dim = model_config.head_dim
        self._num_kv_heads = model_config.num_kv_heads
        # 不分配真实 tensor，只记录已用页数
        self.used_pages = 0

    def store_kv(self, k, v, out_loc, layer_id):
        pass  # 不做任何事

    @property
    def num_pages(self): return self._num_pages
    @property
    def page_size(self): return self._page_size
    @property
    def total_capacity(self): return self._num_pages * self._page_size
    @property
    def used_capacity(self): return self.used_pages * self._page_size
```

#### 3.4.3 MockSampler

```python
class MockSampler:
    """模拟采样，不依赖 flashinfer"""
    # 注意：本节为接口规格，完整实现见 §9.11

    def __init__(self, vocab_size: int, mode: str = "random"):
        self.vocab_size = vocab_size
        self.mode = mode  # "greedy" 或 "random"

    def sample(self, logits, args: BatchSamplingArgs) -> List[int]:
        # greedy: argmax; 否则: random
        # args.is_greedy: 当 BatchSamplingArgs.temperatures is None 时为 True
        #   （由 prepare 中 all(p.is_greedy for p in params) 设置）
        # self.mode == "greedy": 全局强制 greedy 模式
        if args.is_greedy or self.mode == "greedy":
            return [max(range(len(row)), key=lambda j: row[j]) for row in logits]
        else:
            import random
            return [random.randint(0, self.vocab_size - 1) for _ in logits]
```

#### 3.4.4 Mock Communication

```python
class MockZmqQueue:
    """用 in-process queue 替代 ZMQ"""
    def __init__(self):
        self._queue = queue.Queue()
    def put(self, msg): self._queue.put(msg)
    def get(self, blocking=True):
        return self._queue.get(block=blocking)
    def empty(self): return self._queue.empty()


class SimCommGroup:
    """通用并行通信组仿真，支持 TP all-reduce / EP all-to-all / PP p2p。
    替代原 MockTPGroup（noop），引入通信成本模型。

    成本模型：cost = latency + data_bytes / bandwidth
    data_bytes 由调用方传入（tensor 大小估算），config 提供带宽参数。
    仿真中不实际传输数据，只记录成本 ticks。
    """
    def __init__(self, group_size: int, config: 'SimulatorConfig',
                 group_type: str = "tp"):
        """group_type: "tp" | "ep" | "pp" — 决定使用哪组成本参数"""
        self.group_size = group_size
        self.config = config
        self.group_type = group_type
        self.comm_ticks_log: List[int] = []  # 通信成本记录（供指标收集）

    def _compute_cost(self, data_bytes: int) -> int:
        """计算通信成本 ticks = latency + data_bytes / bandwidth"""
        if self.group_type == "tp":
            latency = self.config.all_reduce_latency_ticks
            per_byte = self.config.all_reduce_cost_per_byte_ticks
        elif self.group_type == "ep":
            latency = self.config.all_to_all_latency_ticks
            per_byte = self.config.all_to_all_cost_per_byte_ticks
        else:  # pp
            latency = 0
            per_byte = self.config.pp_send_recv_cost_per_byte_ticks
        bw = self.config.comm_bandwidth_bytes_per_tick
        transfer_ticks = int(data_bytes / max(1, bw)) if bw > 0 else 0
        cost = latency + int(data_bytes * per_byte) + transfer_ticks
        self.comm_ticks_log.append(cost)
        return cost

    def all_reduce(self, data_bytes: int) -> int:
        """TP all-reduce：返回通信成本 ticks（仿真中不传输数据）"""
        assert self.group_type == "tp"
        return self._compute_cost(data_bytes)

    def all_to_all(self, data_bytes: int) -> int:
        """EP all-to-all：返回通信成本 ticks"""
        assert self.group_type == "ep"
        return self._compute_cost(data_bytes)

    def send_recv(self, data_bytes: int) -> int:
        """PP stage 间 p2p：返回通信成本 ticks"""
        assert self.group_type == "pp"
        return self._compute_cost(data_bytes)

    def barrier(self) -> None:
        """同步屏障（仿真中为 noop，不产生成本）"""
        pass


# 兼容旧代码：MockTPGroup 保留为 SimCommGroup 的薄包装
class MockTPGroup(SimCommGroup):
    """向后兼容：tp_size=1 时等价于原 noop 行为"""
    def __init__(self):
        super().__init__(group_size=1, config=None, group_type="tp")  # type: ignore
    def all_reduce(self, data_bytes: int = 0) -> int:  # type: ignore
        return 0  # tp_size=1 时无通信成本
    def broadcast(self, tensor, root): return tensor  # 保留旧接口
```

#### 3.4.5 MockAttnBackend

```python
class MockAttnBackend(BaseAttnBackend):
    """模拟 attention backend"""
    def prepare_metadata(self, batch: Batch): pass
    def prepare_for_capture(self, batch: Batch): pass
    def prepare_for_replay(self, batch: Batch): pass
    def init_capture_graph(self, max_seq_len, bs_list): pass
    # forward_extend / forward_decode 不需要实现，因为 model.forward 是 mock
```

#### 3.4.5b SimMoeBackend (EP-aware)

```python
class SimMoeBackend(BaseMoeBackend):
    """模拟 MoE backend，支持 EP 专家分布仿真。
    ep_size=1 时退化为 MockMoeBackend（无通信）。

    EP 仿真要点：
    1. 专家按 ep_size 分片，每个 EP rank 持有 num_experts/ep_size 个专家
    2. forward 时执行 all-to-all 通信将 token 发送到目标专家所在 rank
    3. 计算后执行 all-to-all 反向通信收集结果
    4. 仿真中不实际传输数据，只计算通信成本 ticks
    """
    def __init__(self, model_config: 'ModelConfig', ep_size: int = 1,
                 comm_group: 'SimCommGroup | None' = None,
                 dtype_size: int = 2):
        self.model_config = model_config
        self.ep_size = ep_size
        self.comm_group = comm_group  # group_type="ep" 的 SimCommGroup
        self.dtype_size = dtype_size
        self.call_count = 0
        self.total_tokens = 0
        self.comm_ticks_total = 0

        # 专家分布：每个 EP rank 持有的专家索引范围
        if ep_size > 1:
            self.experts_per_rank = div_even(
                model_config.num_experts, ep_size, allow_replicate=False
            )
            self.local_expert_start = sum(self.experts_per_rank[:0])  # rank 0
            self.local_expert_count = self.experts_per_rank[0]
        else:
            self.experts_per_rank = [model_config.num_experts]
            self.local_expert_start = 0
            self.local_expert_count = model_config.num_experts

    def prepare_metadata(self, batch: 'Batch'):
        """记录 token 数量用于指标收集"""
        self.call_count += 1
        self.total_tokens += batch.size

    def forward(self, hidden_states, batch_size: int) -> Tuple[Any, int]:
        """Mock forward + EP 通信成本计算。
        返回 (hidden_states, comm_cost_ticks)。
        """
        self.prepare_metadata_type(batch_size)  # 记录指标

        comm_cost = 0
        if self.ep_size > 1 and self.comm_group is not None:
            # all-to-all：发送 hidden_states 到目标 EP rank
            # 数据量 = batch_size × hidden_size × dtype_size（每个 token 的 hidden）
            data_bytes = (batch_size * self.model_config.hidden_size * self.dtype_size)
            comm_cost += self.comm_group.all_to_all(data_bytes)
            # 反向 all-to-all：收集结果
            comm_cost += self.comm_group.all_to_all(data_bytes)

        return hidden_states, comm_cost

    def prepare_metadata_type(self, batch_size: int):
        """内部：记录调用"""
        self.call_count += 1
        self.total_tokens += batch_size


# 向后兼容
class MockMoeBackend(SimMoeBackend):
    """旧接口兼容：ep_size=1，无通信"""
    def __init__(self):
        super().__init__(
            model_config=None,  # type: ignore
            ep_size=1,
            comm_group=None,
        )
    def forward(self, hidden_states, batch_size: int = 0) -> Tuple[Any, int]:  # type: ignore
        return hidden_states, 0
```

---

## 4. 模拟器架构设计

### 4.1 模块划分

```
sglang_simulator/
├── __init__.py
├── core/                      # 核心数据结构（仿真）
│   ├── __init__.py
│   ├── data_types.py          # SamplingParams, Req, Batch, Context
│   └── model_config.py        # ModelConfig（模型参数描述）
├── scheduler/                 # 调度器（仿真）
│   ├── __init__.py
│   ├── scheduler.py           # SimScheduler 主循环
│   ├── prefill.py             # PrefillManager + PrefillAdder
│   ├── decode.py              # DecodeManager
│   ├── cache.py               # CacheManager
│   ├── table.py               # TableManager
│   ├── config.py              # SimulatorConfig（统一配置）
│   └── io.py                  # MockZmqQueue 通信
├── kvcache/                   # KV Cache（仿真）
│   ├── __init__.py
│   ├── base.py                # 抽象接口
│   ├── radix_cache.py         # RadixPrefixCache + RadixTreeNode
│   ├── naive_cache.py         # NaiveCacheManager（无前缀缓存）
│   └── mha_pool.py            # MockKVCachePool
├── engine/                    # Engine（模拟）
│   ├── __init__.py
│   ├── engine.py              # MockEngine
│   ├── sample.py              # MockSampler
│   ├── graph.py               # SimGraphRunner
│   └── config.py              # EngineConfig
├── memory/                    # 内存预算（仿真）
│   ├── __init__.py
│   └── budget.py              # 显存预算计算（含 TP/DP 并行修正）
├── parallel/                  # 并行仿真（§10）
│   ├── __init__.py
│   ├── topology.py            # ParallelTopology（rank 映射、层分割）
│   ├── comm_group.py          # SimCommGroup（TP/EP/PP 通信成本）
│   ├── tp.py                  # TPSimulator（all-reduce 注入、内存修正）
│   ├── dp.py                 # DPLoadBalancer + DPRankState（请求分发）
│   ├── pp.py                 # PPPipelineSimulator（1F1B/GPipe/interleaved）
│   └── validate.ts           # validate_parallel_config（启动时验证）
├── mock/                      # 模拟组件
│   ├── __init__.py
│   ├── attention.py           # MockAttnBackend
│   ├── moe.py                 # SimMoeBackend（EP-aware，§3.4.5b）
│   ├── communication.py      # MockZmqQueue, SimCommGroup（§3.4.4）
│   └── model.py               # MockModel.forward
├── server/                    # API Server（模拟）
│   ├── __init__.py
│   ├── api_server.py          # 简化版 HTTP server
│   ├── args.py                # 参数解析
│   └── launch.py              # 进程启动（单进程模式）
├── message/                   # 消息类型（仿真）
│   ├── __init__.py
│   └── types.py               # BaseBackendMsg, UserMsg, DetokenizeMsg, etc.
├── tokenizer/                 # Tokenizer（模拟）
│   ├── __init__.py
│   └── mock_tokenizer.py      # 简单字典查表 tokenizer
├── benchmark/                 # 基准测试
│   ├── __init__.py
│   ├── workload.py            # 工作负载生成器
│   └── metrics.py             # 性能指标收集
└── config.py                  # 全局配置
```

### 4.2 配置系统

```python
@dataclass
class SimulatorConfig:
    # ===== 模型配置 =====
    model_config: ModelConfig  # num_layers, hidden_size, num_kv_heads, head_dim, vocab_size, is_moe

    # ===== 调度配置 =====
    max_running_req: int = 128
    max_seq_len: int = 8192
    max_extend_tokens: int = 8192   # prefill budget
    cache_type: str = "radix"       # "radix" or "naive"

    # ===== KV Cache 配置 =====
    page_size: int = 1
    num_pages: int | None = None    # None = 自动计算

    # ===== 内存配置 =====
    total_gpu_memory: int = 80 * 1024**3  # 80 GiB default
    memory_ratio: float = 0.88
    dtype_size: int = 2  # bytes per element

    # ===== CUDA Graph 配置 =====
    enable_cuda_graph: bool = True
    cuda_graph_bs: list[int] | None = None
    cuda_graph_max_bs: int | None = None
    graph_replay_cost_ticks: int = 1    # GPU ticks per replay
    eager_forward_cost_ticks: int = 10  # GPU ticks per eager forward

    # ===== Overlap Scheduling 配置 =====
    enable_overlap: bool = True
    cpu_schedule_cost_ticks: int = 1    # CPU ticks per scheduling
    cpu_process_result_cost_ticks: int = 1  # CPU ticks per result processing

    # ===== TP 张量并行配置 =====
    tp_size: int = 1
    all_reduce_cost_per_byte_ticks: float = 0.001  # 每 byte all-reduce 耗费 ticks
    all_reduce_latency_ticks: int = 2              # all-reduce 固定延迟
    # TP 通信基础设施（仿真为成本参数，实际为 gloo/nccl/pynccl）
    tp_cpu_group_type: str = "gloo"   # CPU 侧：barrier/broadcast
    tp_gpu_group_type: str = "nccl"    # GPU 侧：all-reduce（pynccl 为异步版）

    # ===== DP 数据并行配置 =====
    dp_size: int = 1
    # 标准 DP 请求分发策略（DataParallelController）
    dp_load_balance_strategy: str = "round_robin"  # "round_robin" | "shortest_queue"
    # DP Attention 模式（MLA 模型专用，与标准 DP 不同）
    enable_dp_attention: bool = False  # 开启后 attention 层 DP，MLP 层大 TP 组
    dp_attention_all_gather_cost_per_byte_ticks: float = 0.0015

    # ===== EP 专家并行配置 =====
    ep_size: int = 1                # EP size（MoE 部分 TP rank 重编号，不增加进程数）
    all_to_all_cost_per_byte_ticks: float = 0.002   # 每 byte all-to-all 耗费 ticks
    all_to_all_latency_ticks: int = 3               # all-to-all 固定延迟
    moe_routing_mode: str = "mock"                  # "mock" | "hash" | "simulated"
    enable_eplb: bool = False        # 是否启用动态专家负载均衡（eplb）

    # ===== CP Context Parallel 配置 =====
    cp_size: int = 1                 # CP size（attention 序列切分，TP rank 重编号）
    cp_all_gather_cost_per_byte_ticks: float = 0.001  # 每 byte all-gather 成本

    # ===== PP 流水并行配置 =====
    pp_size: int = 1
    pp_num_micro_batches: int = 1                   # 每个 pipeline batch 的 micro-batch 数
    pp_send_recv_cost_per_byte_ticks: float = 0.0005  # stage 间 p2p 通信成本
    pp_pipeline_schedule: str = "1f1b"              # "1f1b" | "gpipe" | "interleaved"

    # ===== 通信成本通用配置 =====
    comm_bandwidth_bytes_per_tick: int = 1_000_000  # 模拟互联带宽（bytes/tick）
    comm_overlap_with_compute: bool = True           # 通信是否可与计算重叠

    # ===== 离线模式（不启动 server，直接调用 scheduler）=====
    offline_mode: bool = False

    # ===== Tokenizer =====
    eos_token_id: int = 0           # EOS token ID（默认 0，可按模型配置）

    # ===== 采样配置 =====
    mock_sample_mode: str = "random"   # "random" or "greedy" or "fixed"
    fixed_output_token: int = 0        # for "fixed" mode

    # ===== 仿真控制 =====
    max_ticks: int | None = None       # None = 无限运行
    log_level: str = "INFO"
    enable_metrics: bool = True


@dataclass
class ModelConfig:
    """描述模型特征，不需要真实模型"""
    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    is_moe: bool = False
    num_experts: int = 0              # MoE only
    moe_intermediate_size: int = 0    # MoE only
    moe_top_k: int = 1                # MoE 每个 token 选择的专家数
    intermediate_size: int = 0       # dense MLP
    num_attention_heads: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192


@dataclass
class ParallelTopology:
    """并行拓扑配置，描述 TP×DP×PP 进程网格。

    重要：SGLang 的实际进程数 = tp_size × dp_size × pp_size。
    EP 和 CP 不增加进程数，而是在已有 TP rank 上重新划分 rank：
    - CP：在 TP 组内再切 attention 序列（attn_cp_size 是 tp_size 的子划分）
    - EP：在 MoE 部分重排 TP rank 为 moe_dp/moe_ep/moe_tp 三层
    参见 sglang-note《scheduler-rank-and-process-topology》§三。

    仿真中为简化，将 EP/CP 视为通信成本参数（不影响进程数）。
    """
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1       # MoE EP（TP rank 重编号，不增加进程数）
    pp_size: int = 1
    cp_size: int = 1       # Context Parallel（TP rank 重编号，不增加进程数）
    enable_dp_attention: bool = False  # DP Attention 模式（与标准 DP 不同）

    @property
    def world_size(self) -> int:
        """总进程数 = tp × dp × pp（EP/CP 不增加进程数）"""
        return self.tp_size * self.dp_size * self.pp_size

    @property
    def num_dp_groups(self) -> int:
        """DP 组数 = tp × pp（每个 DP 副本内的进程数）"""
        return self.tp_size * self.pp_size

    @property
    def num_pp_stages(self) -> int:
        """PP stage 数 = pp_size"""
        return self.pp_size

    def rank_to_coord(self, rank: int) -> Tuple[int, int, int]:
        """将线性 rank 映射到 (tp_idx, dp_idx, pp_idx) 三元组。
        排序：tp 在最内层，dp 居中，pp 在最外层。
        gpu_id 公式（单机）：dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
        """
        assert 0 <= rank < self.world_size
        tp_idx = rank % self.tp_size
        remain = rank // self.tp_size
        dp_idx = remain % self.dp_size
        pp_idx = remain // self.dp_size
        return (tp_idx, dp_idx, pp_idx)

    def coord_to_rank(self, tp_idx: int, dp_idx: int, pp_idx: int) -> int:
        """将 (tp_idx, dp_idx, pp_idx) 映射回线性 rank"""
        return (
            pp_idx * (self.dp_size * self.tp_size)
            + dp_idx * self.tp_size
            + tp_idx
        )

    def compute_moe_ranks(self, tp_rank: int) -> Tuple[int, int, int]:
        """从 tp_rank 推导 MoE 层级的 (moe_dp_rank, moe_ep_rank, moe_tp_rank)。
        层级：Global(TP) → MOE_DP → EP → MOE_TP
        对应 SGLang 源码 _compute_parallelism_ranks。
        """
        moe_dp_size = self.dp_size  # 简化：moe_dp = dp
        moe_tp_size = max(1, self.tp_size // moe_dp_size // self.ep_size)
        moe_dp_rank = tp_rank // (self.tp_size // moe_dp_size) if moe_dp_size > 0 else 0
        moe_ep_rank = (
            (tp_rank % (self.tp_size // moe_dp_size))
            // moe_tp_size
        ) if self.ep_size > 0 and moe_tp_size > 0 else 0
        moe_tp_rank = tp_rank % moe_tp_size if moe_tp_size > 0 else 0
        return (moe_dp_rank, moe_ep_rank, moe_tp_rank)

    def compute_attn_ranks(self, tp_rank: int) -> Tuple[int, int]:
        """从 tp_rank 推导 Attention 层级的 (attn_cp_rank, attn_tp_rank)。
        层级：Global(TP) → ATTN_DP → ATTN_CP → ATTN_TP
        attn_tp_size = tp_size / attn_dp_size / attn_cp_size
        """
        attn_dp_size = self.dp_size if self.enable_dp_attention else 1
        attn_tp_size = max(1, self.tp_size // attn_dp_size // self.cp_size)
        attn_cp_rank = (tp_rank // attn_tp_size) % self.cp_size if self.cp_size > 0 else 0
        attn_tp_rank = tp_rank % attn_tp_size
        return (attn_cp_rank, attn_tp_rank)

    def pp_stage_layers(self, num_layers: int) -> List[range]:
        """将 num_layers 按 pp_size 均分到各 stage，返回每个 stage 的层范围列表"""
        base = num_layers // self.pp_size
        remainder = num_layers % self.pp_size
        result: List[range] = []
        offset = 0
        for i in range(self.pp_size):
            count = base + (1 if i < remainder else 0)
            result.append(range(offset, offset + count))
            offset += count
        return result
```

### 4.3 时序仿真模型

模拟器使用 **tick-based** 离散事件仿真：

```python
@dataclass
class SimEvent:
    """仿真事件记录（用于时序分析）"""
    tick: int
    event_type: str
    duration: int = 0


class SimulationClock:
    """全局仿真时钟（可选工具，不参与核心 tick 循环）

    仿真器的核心时序由 run_tick 的调用方控制（每个外部调用 = 1 tick）。
    SimulationClock 仅用于高级分析场景：如需精确追踪 GPU 占用时间线，
    可在 SimScheduler.__init__ 中实例化并在 _forward 后调用 schedule_gpu。
    默认不实例化，不影响仿真正确性。
    """
    def __init__(self):
        self.current_tick = 0
        self.gpu_busy_until = 0  # GPU 何时空闲
        self.events: List[SimEvent] = []

    def advance(self, ticks: int = 1):
        self.current_tick += ticks

    def schedule_gpu(self, duration_ticks: int):
        """安排 GPU 任务，返回完成时间"""
        start = max(self.current_tick, self.gpu_busy_until)
        finish = start + duration_ticks
        self.gpu_busy_until = finish
        return finish

    def can_overlap(self) -> bool:
        """检查当前是否可以重叠（GPU 空闲或 CPU 在等 GPU）"""
        return self.current_tick < self.gpu_busy_until
```

**时序仿真规则**：

| 操作 | 耗费 ticks | 执行者 |
|------|-----------|--------|
| 消息接收 + 处理 | 1 | CPU |
| batch 调度 + 准备 | 1 | CPU |
| CUDA Graph replay | `graph_replay_cost_ticks` | GPU |
| Eager forward | `eager_forward_cost_ticks` | GPU |
| 采样 | 0（与 forward 重叠） | GPU |
| 结果处理 + 回传 | 1 | CPU |

**Overlap 模式时序**：
```
Tick 0: [CPU: recv + schedule] → [GPU: forward(batch_0)]
Tick 1: [CPU: process(batch_0) + recv + schedule] → [GPU: forward(batch_1)]
        ↑ process(batch_0) 与 forward(batch_0) 的尾部重叠
        ↑ recv + schedule 与 forward(batch_0) 的尾部重叠
```

**非 Overlap 模式时序**：
```
Tick 0: [CPU: recv + schedule] → [GPU: forward(batch_0)]
Tick 1: [CPU: process(batch_0)]
Tick 2: [CPU: recv + schedule] → [GPU: forward(batch_1)]
Tick 3: [CPU: process(batch_1)]
```

### 4.4 工作负载生成器

```python
@dataclass
class WorkloadConfig:
    num_requests: int = 1000
    input_len_distribution: str = "uniform"  # "uniform", "normal", "trace"
    input_len_min: int = 100
    input_len_max: int = 1024
    input_len_mean: int = 500
    input_len_std: int = 200
    output_len_distribution: str = "uniform"
    output_len_min: int = 100
    output_len_max: int = 1024
    arrival_rate: float = 10.0  # requests per tick
    arrival_distribution: str = "poisson"  # "poisson", "uniform", "trace"
    shared_prefix_ratio: float = 0.3  # 30% 请求共享前缀
    shared_prefix_len: int = 100


class WorkloadGenerator:
    def generate(self, config: WorkloadConfig) -> List[SimRequest]:
        """生成模拟请求序列"""
        requests = []
        for i in range(config.num_requests):
            input_len = self._sample_len(config.input_len_distribution, ...)
            output_len = self._sample_len(config.output_len_distribution, ...)
            arrival_tick = self._sample_arrival(config)
            input_ids = self._generate_tokens(input_len, config.shared_prefix_ratio, ...)
            requests.append(SimRequest(
                uid=i,
                arrival_tick=arrival_tick,
                input_ids=input_ids,
                output_len=output_len,
                sampling_params=SamplingParams(max_tokens=output_len),
            ))
        return requests
```

### 4.5 性能指标收集

```python
@dataclass
class SimulationMetrics:
    # 吞吐量指标
    total_requests: int = 0
    completed_requests: int = 0
    total_tokens_generated: int = 0
    total_ticks: int = 0

    # 延迟指标
    request_latencies: List[int] = field(default_factory=list)  # per-request ticks
    prefill_latencies: List[int] = field(default_factory=list)
    decode_latencies: List[int] = field(default_factory=list)

    # 调度指标
    prefill_batches: int = 0
    decode_batches: int = 0
    avg_prefill_batch_size: float = 0.0
    avg_decode_batch_size: float = 0.0
    chunked_prefill_count: int = 0

    # Cache 指标
    cache_hit_rate: float = 0.0
    cache_eviction_count: int = 0
    avg_cache_utilization: float = 0.0

    # 内存指标
    peak_memory_usage: int = 0
    oom_count: int = 0

    # GPU 利用率
    gpu_busy_ticks: int = 0
    gpu_idle_ticks: int = 0
    gpu_utilization: float = 0.0  # gpu_busy / total

    # CUDA Graph 指标
    cuda_graph_replay_count: int = 0
    eager_forward_count: int = 0
```

---

## 5. 实现指南

### 5.1 实现优先级

按以下顺序实现，每个阶段可独立测试：

**Phase 1: 核心数据结构 + 基础调度**
1. `core/data_types.py` — Req, Batch, SamplingParams, Context
2. `core/model_config.py` — ModelConfig
3. `message/types.py` — UserMsg, DetokenizeMsg
4. `scheduler/table.py` — TableManager
5. `kvcache/base.py` — 抽象接口
6. `kvcache/naive_cache.py` — 简单 cache（无 RadixTree）
7. `scheduler/cache.py` — CacheManager（先用 naive cache）
8. `scheduler/prefill.py` — PrefillManager + PrefillAdder
9. `scheduler/decode.py` — DecodeManager
10. `scheduler/scheduler.py` — SimScheduler（无 overlap，无 CUDA Graph）
11. `mock/` — 所有 mock 组件

**Phase 2: RadixCache + 前缀缓存**
1. `kvcache/radix_cache.py` — RadixTreeNode, RadixPrefixCache, RadixCacheHandle
2. 更新 `scheduler/cache.py` 使用 radix cache
3. 测试前缀匹配、插入、分裂、eviction

**Phase 3: CUDA Graph + Overlap Scheduling**
1. `engine/graph.py` — SimGraphRunner
2. 更新 `scheduler/scheduler.py` 加入 overlap_loop 逻辑
3. `SimulationClock` — tick-based 时序
4. 测试 overlap vs non-overlap 性能差异

**Phase 4: 内存预算 + 工作负载 + 指标**
1. `memory/budget.py` — 显存预算计算
2. `benchmark/workload.py` — 工作负载生成器
3. `benchmark/metrics.py` — 性能指标收集
4. `server/` — 简化版 API server

**Phase 5: 并行仿真（TP/DP/EP/PP）** — 见 §10 完整规格
1. `parallel/topology.py` — ParallelTopology（rank 映射、层分割）
2. `parallel/comm_group.py` — SimCommGroup（替换 MockTPGroup，§3.4.4）
3. `parallel/tp.py` — TPSimulator（all-reduce 注入、内存修正，§10.2）
4. `parallel/dp.py` — DPLoadBalancer + DPRankState（§10.3）
5. `mock/moe.py` — SimMoeBackend（升级 MockMoeBackend，§3.4.5b + §10.4）
6. `parallel/pp.py` — PPPipelineSimulator（§10.5）
7. `parallel/validate.ts` — validate_parallel_config + calculate_memory_budget_parallel（§10.6）
8. 更新 `MockEngine.forward_batch` 注入 TP/EP/PP 通信成本（§10.2.4, §10.4, §10.5.3）
9. 更新 `SimScheduler.add_request` 注入 DP 请求分发（§10.3.3）
10. 更新 `SimulationMetrics` 新增 ParallelMetrics（§10.7）

> **Phase 5 前置依赖**：Phase 1-4 必须完成。Phase 5 的所有并行模块在 `tp_size=dp_size=ep_size=pp_size=1` 时 noop，不破坏 Phase 1-4 的单实例行为。

### 5.2 测试策略

每个 Phase 完成后的验证点：

| Phase | 验证场景 | 预期结果 |
|-------|---------|---------|
| 1 | 单请求 prefill→decode→finish | 请求正确完成，资源正确释放 |
| 1 | 多请求并发 prefill | 按 FIFO 调度，token budget 正确限制 |
| 1 | 请求 abort | 资源正确释放，不影响其他请求 |
| 2 | 共享前缀的多个请求 | 第二个请求 cache hit，减少 prefill 计算量 |
| 2 | Cache eviction | LRU 正确驱逐，free_slots 正确回收 |
| 2 | 部分前缀匹配 + 节点分裂 | split_at 正确分裂，后续请求正确匹配 |
| 3 | Decode batch 使用 CUDA Graph | can_use_cuda_graph 正确判断，pad_batch 正确填充 |
| 3 | Overlap vs non-overlap | overlap 模式 GPU 利用率更高 |
| 4 | 不同工作负载下的吞吐量 | 高 arrival_rate 下 throughput 饱和 |
| 4 | 内存预算限制 | num_pages 不足时正确触发 eviction 或拒绝新请求 |
| 5 | TP all-reduce 通信成本 | TP>1 时 `tp_comm_ticks > 0`，TP=1 时为 0（§10.8.1） |
| 5 | DP round_robin/least_load 分发 | 请求按策略分配到 DP rank（§10.8.2） |
| 5 | EP 专家分片 + all-to-all | `experts_per_rank` 均匀，`ep_comm_ticks > 0`（§10.8.3） |
| 5 | PP 流水线气泡 + send/recv | `pp_bubble_ticks > 0`，层按 pp_size 均分（§10.8.4） |
| 5 | TP×DP×EP×PP 组合 | `comm_ticks_total = tp+ep+pp` 之和（§10.8.5） |
| 5 | 退化为单实例 | 所有并行 size=1 时零通信成本（§10.9） |
| 5 | 并行配置验证 | 非法配置（如 EP>num_experts）抛 ValueError（§10.6.3） |

### 5.3 关键实现细节

#### 5.3.1 Chunked Prefill 仿真

Chunked prefill 是 SGLang 的核心特性，仿真中必须正确实现：

```python
# PrefillAdder._add_one_req 中的关键逻辑
remain_len = pending_req.input_len - cached_len
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len  # 是否需要分块

if is_chunked:
    # 创建 ChunkedReq
    # 传入完整 input_ids，由 positions 控制实际计算范围 [cached_len, cached_len + chunk_size)
    # 下次 tick 续接时复用同一 input_ids，推进 device_len 即可
    req = ChunkedReq(
        input_ids=pending_req.input_ids,  # 完整传入，不截断
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=0,  # ChunkedReq 不生成 token
        uid=pending_req.uid,
        cache_handle=cache_handle,
        sampling_params=pending_req.sampling_params,
    )
    req.device_len = cached_len + chunk_size
    req.max_device_len = cached_len + chunk_size
    pending_req.chunked_req = req  # 保存以便下次继续
```

**关键约束**：
- `token_budget` 在一个 tick 内递减，不可超过 `prefill_budget`
- `reserved_size` 在一个 tick 内递增，用于防止 prefill 占用过多资源导致 decode OOM
- ChunkedReq 的 `can_decode` 返回 False，不会被加入 DecodeManager
- ChunkedReq 在 `_process_last_data` 中被跳过（不采样、不回传）

#### 5.3.2 Page Table 写入仿真

```python
def _write_page_table(page_table, allocated, allocation_info, page_size):
    """
    将分配的物理位置写入 page_table
    allocation_info: List[(table_idx, first_page, last_page)]
    allocated: 物理位置列表
    """
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos = first_page * page_size
        last_pos = last_page * page_size
        length = last_pos - first_pos
        page_table[table_idx, first_pos:last_pos] = allocated[offset:offset+length]
        offset += length
```

#### 5.3.3 cache_req 的 5 个缓存区域

cache_req 是最复杂的操作，需要正确处理 5 个区域：

```
[0, old_handle.cached_len)               — ① 前部保留：已在 prefix cache 中，无需操作
[old_handle.cached_len, cached_len)       — ② 前部已释放：被其他请求抢先缓存 → 需释放重复页
[cached_len, new_handle.cached_len)       — ③ 新写入：本次新插入 prefix cache
[new_handle.cached_len, req.cached_len)   — ④ 尾部保留：未页对齐，无法插入
                                           finished=True → 释放
                                           finished=False → 保留并更新 handle
[req.cached_len, req.device_len)          — ⑤ 尾部已释放：超出 cached_len 的 forward 部分
                                           finished=True → 随 ④ 一起释放
```

#### 5.3.4 RadixTree key_fn

```python
def _get_key_fn(page_size: int):
    """生成 key_fn：page_size=1 时用单个 token 值，page_size>1 时用前 page_size 个 token 的 tuple"""
    if page_size == 1:
        return lambda tokens: tokens[0]  # 单 token 作为 key
    return lambda tokens: tuple(tokens[:page_size])  # page 作为 key
```

这个设计确保了：
- page_size=1 时，每个 token 是一个 key，树粒度最细
- page_size>1 时，每页是一个 key，减少树深度但增加粒度

#### 5.3.5 Eviction 算法

```python
def evict(self, size: int) -> List[int]:
    """LRU 驱逐，返回释放的页索引列表"""
    if size == 0:
        return []
    # 1. 收集所有 ref_count==0 的叶子节点
    leave_nodes = self._collect_leave_nodes_for_evict()
    # 2. 按 timestamp 最小堆排序（LRU）
    heapq.heapify(leave_nodes)
    evicted_indices: List[int] = []
    evicted_size = 0
    while leave_nodes and evicted_size < size:
        node = heapq.heappop(leave_nodes)
        if node.ref_count > 0:
            continue
        evicted_size += node.length
        evicted_indices.extend(node.value)
        # 从父节点删除
        parent = node.parent
        if parent is not None:
            del parent.children[self.key_fn(node._key)]
        # 如果父节点变为叶子且 ref_count==0，加入堆
        if (parent is not None and not parent.is_root()
                and parent.is_leaf() and parent.ref_count == 0):
            heapq.heappush(leave_nodes, parent)
    # 注意：evict 只返回被驱逐的页索引列表，由调用方（CacheManager.allocate_paged）负责加入 free_slots
    return evicted_indices
```

---

## 6. 扩展能力

### 6.1 可选仿真模块

以下模块可在核心模拟器完成后按需添加：

| 模块 | 仿真难度 | 价值 | 状态 |
|------|---------|------|------|
| **TP 张量并行** | 中 | all-reduce 通信成本、权重/KV 分割、ZMQ 广播 + gloo/nccl 三层 | ✅ 已规格（§10.2, §10.6） |
| **DP 数据并行** | 中 | 标准 DP（请求分发）+ DP Attention（MLA all-gather/slice） | ✅ 已规格（§10.3） |
| **EP 专家并行** | 中 | MoE 专家分片、TP rank 重编号、all-to-all、EPLB | ✅ 已规格（§10.4） |
| **PP 流水并行** | 高 | micro-batch 调度、最后 stage 采样、stage 间通信 | ✅ 已规格（§10.5） |
| **CP Context Parallel** | 中 | 长序列切分、TP rank 重编号、KV all-gather | ✅ 已规格（§10.8） |
| Speculative Decoding | 中 | draft model 预测 N token，target model 验证，KV cache 处理 | 待实现 |
| Constraint Decoding | 中 | X-Grammar PDA，与 GPU 计算重叠 | 待实现 |
| Weight Update (RLHF) | 高 | online_update_weights，NCCL 广播，latency 优化 | 待实现 |
| Multi-turn / Tool Calling | 高 | AgentLoop，请求状态机扩展 | 待实现 |
| HiCache (SSD→DRAM→HBM) | 高 | 分层 KV cache，prefetch 机制 | 待实现 |
| Chunked Prefill + Overlap | 中 | prefill 和 decode 混合 batch | ✅ 已实现（§3.3.8, §9.4） |

### 6.2 可配置策略点

模拟器应支持在以下决策点切换策略：

| 决策点 | 默认策略 | 可选策略 |
|--------|---------|---------|
| prefill vs decode 优先级 | prefill 优先 | decode 优先 / 混合 |
| prefill token budget | 固定值 | 自适应（基于 GPU 利用率） |
| cache eviction | LRU | FIFO / LFU |
| cache 类型 | radix | naive / none |
| CUDA Graph | 启用 | 禁用 / 自定义 bs 列表 |
| overlap scheduling | 启用 | 禁用 |
| batch 调度顺序 | FIFO | SJF (最短作业优先) / 优先级 |
| chunked prefill | 启用 | 禁用 / 自定义 chunk size |
| DP 负载均衡 | round_robin | shortest_queue（§10.3） |
| DP Attention | 禁用 | 启用（MLA 模型专用，§10.3.4） |
| PP 流水线调度 | 1f1b | gpipe / interleaved（§10.5） |
| MoE 路由模式 | mock | hash / simulated（§10.4） |
| EPLB 动态负载均衡 | 禁用 | 启用（§10.4.4） |
| CP Context Parallel | 禁用 | cp_size > 1（§10.8） |

---

## 7. 接口契约总表

### 7.1 仿真组件接口

| 组件 | 核心方法 | 输入 | 输出 |
|------|---------|------|------|
| SimScheduler | `run_tick(msgs)` | `List[BaseBackendMsg]` | `List[DetokenizeMsg]` |
| PrefillManager | `schedule_next_batch(budget)` | `int` | `Batch \| None` |
| PrefillManager | `add_one_req(msg)` | `UserMsg` | `void` |
| PrefillManager | `abort_req(uid)` | `int` | `Req \| None` |
| DecodeManager | `schedule_next_batch()` | — | `Batch \| None` |
| DecodeManager | `filter_reqs(reqs)` | `Iterable[Req]` | `void` |
| DecodeManager | `abort_req(uid)` | `int` | `Req \| None` |
| CacheManager | `match_req(req)` | `PendingReq` | `MatchResult` |
| CacheManager | `allocate_paged(req)` | `Req` | `void` |
| CacheManager | `cache_req(req, finished)` | `Req, bool` | `void` |
| CacheManager | `lock(handle)` / `unlock(handle)` | `BaseCacheHandle` | `void` |
| TableManager | `allocate()` | — | `int` |
| TableManager | `free(table_idx)` | `int` | `void` |
| RadixPrefixCache | `match_prefix(ids)` | `List[int]` | `MatchResult` |
| RadixPrefixCache | `insert_prefix(ids, indices)` | `List[int], List[int]` | `InsertResult` |
| RadixPrefixCache | `evict(size)` | `int` | `List[int]` |
| SimGraphRunner | `can_use_cuda_graph(batch)` | `Batch` | `bool` |
| SimGraphRunner | `pad_batch(batch)` | `Batch` | `void` |
| SimGraphRunner | `replay(batch)` | `Batch` | `List[List[float]] (logits)` |

### 7.2 模拟组件接口

| 组件 | 核心方法 | 输入 | 输出 |
|------|---------|------|------|
| MockEngine | `forward_batch(batch, args)` | `Batch, BatchSamplingArgs` | `ForwardOutput` |
| MockSampler | `prepare(batch)` | `Batch` | `BatchSamplingArgs` |
| MockSampler | `sample(logits, args)` | `List[List[float]], BatchSamplingArgs` | `List[int]` |
| MockKVCachePool | `store_kv(k, v, out_loc, layer_id)` | `List, List, List[int], int` | `void` |
| MockAttnBackend | `prepare_metadata(batch)` | `Batch` | `void` |
| MockZmqQueue | `put(msg)` / `get(blocking)` | `BaseBackendMsg` / `bool` | `void` / `BaseBackendMsg` |

### 7.2b 并行仿真组件接口（§10）

| 组件 | 核心方法 | 输入 | 输出 |
|------|---------|------|------|
| SimCommGroup | `all_reduce(data_bytes)` | `int` | `int` (ticks) |
| SimCommGroup | `all_to_all(data_bytes)` | `int` | `int` (ticks) |
| SimCommGroup | `send_recv(data_bytes)` | `int` | `int` (ticks) |
| TPCommInfraSimulator | `zmq_broadcast(msg_size)` | `int` | `int` (ticks) |
| TPCommInfraSimulator | `cpu_barrier()` | — | `int` (ticks) |
| TPCommInfraSimulator | `gpu_all_reduce(data_bytes)` | `int` | `int` (ticks) |
| TPSimulator | `all_reduce_after_attn(batch_size)` | `int` | `int` (ticks) |
| TPSimulator | `all_reduce_after_mlp(batch_size)` | `int` | `int` (ticks) |
| DataParallelController | `assign_request(req)` | `Req` | `int` (dp_rank) |
| DataParallelController | `allocate_pages(dp_rank, n)` | `int, int` | `bool` |
| DPAttentionSimulator | `simulate_mlp_forward(local_batch_sizes)` | `List[int]` | `int` (ticks) |
| SimMoeBackend | `forward(hidden, batch_size)` | `Any, int` | `Tuple[Any, int]` |
| SimMoeBackend | `_route_tokens(batch_size)` | `int` | `Dict[int, int]` |
| EPLBSimulator | `step(rank_loads)` | `List[int]` | `int` (ticks) |
| PPPipelineSimulator | `simulate_pipeline_forward(batch)` | `Batch` | `int` (ticks) |
| CPSimulator | `simulate_attn_forward(seq_len)` | `int` | `int` (ticks) |
| ParallelTopology | `rank_to_coord(rank)` | `int` | `Tuple[int,int,int]` (tp,dp,pp) |
| ParallelTopology | `compute_moe_ranks(tp_rank)` | `int` | `Tuple[int,int,int]` (moe_dp,ep,moe_tp) |
| ParallelTopology | `compute_attn_ranks(tp_rank)` | `int` | `Tuple[int,int]` (attn_cp,attn_tp) |
| ParallelTopology | `pp_stage_layers(num_layers)` | `int` | `List[range]` |
| validate_parallel_config | `(config, model_config)` | `SimulatorConfig, ModelConfig` | `void` (asserts) |
| calculate_memory_budget_parallel | `(config, model_config, mem)` | `SimulatorConfig, ModelConfig, int` | `Tuple[int,int,int]` |

### 7.3 消息类型

```python
@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict: ...
    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg: ...

@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: List[int]  # 仿真中用 list 替代 tensor
    sampling_params: SamplingParams

@dataclass
class DetokenizeMsg(BaseBackendMsg):
    uid: int
    next_token: int
    finished: bool

@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int

@dataclass
class ExitMsg(BaseBackendMsg):
    pass

@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]
```

---

## 8. 参考文件索引

| 组件 | 参考源码 | 参考文章 |
|------|---------|---------|
| Scheduler 主循环 | `mini-sglang/python/minisgl/scheduler/scheduler.py` | `sglang-note/articles/scheduler-call-graph.zh.md` |
| PrefillManager | `mini-sglang/python/minisgl/scheduler/prefill.py` | `sglang-note/articles/scheduler-get-new-batch-prefill.zh.md` |
| DecodeManager | `mini-sglang/python/minisgl/scheduler/decode.py` | `sglang-note/articles/scheduler-update-running-batch.zh.md` |
| CacheManager | `mini-sglang/python/minisgl/scheduler/cache.py` | `Awesome-ML-SYS-Tutorial/sglang/kvcache-code-walk-through/readme-CN.md` |
| RadixCache | `mini-sglang/python/minisgl/kvcache/radix_cache.py` | `sglang-note/articles/radix-cache-structure.zh.md`, `sglang-note/articles/request-lifecycle-radix-cache.zh.md` |
| TableManager | `mini-sglang/python/minisgl/scheduler/table.py` | — |
| Engine | `mini-sglang/python/minisgl/engine/engine.py` | `sglang-note/articles/model-runner-overview.zh.md` |
| GraphRunner | `mini-sglang/python/minisgl/engine/graph.py` | `sglang-note/articles/basics-cuda-graph.zh.md`, `Awesome-ML-SYS-Tutorial/torch/cuda-graph/readme-2.md` |
| Sampler | `mini-sglang/python/minisgl/engine/sample.py` | — |
| API Server | `mini-sglang/python/minisgl/server/api_server.py` | `Awesome-ML-SYS-Tutorial/sglang/code-walk-through/readme-CN.md` |
| 进程启动 | `mini-sglang/python/minisgl/server/launch.py` | `Awesome-ML-SYS-Tutorial/sglang/code-walk-through/readme-CN.md` |
| ZMQ 通信 | `mini-sglang/python/minisgl/scheduler/io.py` | `sglang-note/articles/scheduler-recv-requests.zh.md` |
| 核心数据结构 | `mini-sglang/python/minisgl/core.py` | — |
| KV Cache Base | `mini-sglang/python/minisgl/kvcache/base.py` | `sglang-note/articles/kvcache-prefetch-and-storage.zh.md` |
| Overlap Scheduling | `mini-sglang/python/minisgl/scheduler/scheduler.py#overlap_loop` | `Awesome-ML-SYS-Tutorial/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md` |
| Scheduler Evolution | — | `Awesome-ML-SYS-Tutorial/sglang/scheduler-evolution/` |
| 内存预算 | — | `Awesome-ML-SYS-Tutorial/sglang/kvcache-code-walk-through/mem-fraction-static.md` |
| 权重更新 | — | `Awesome-ML-SYS-Tutorial/sglang/online-update-weights/readme.md`, `Awesome-ML-SYS-Tutorial/sglang/latency-accelerate-for-weight-updates/readme.md` |
| 投机解码 | — | `Awesome-ML-SYS-Tutorial/sglang/speculative-decoding/speculative-decoding.md` |
| 约束解码 | — | `Awesome-ML-SYS-Tutorial/sglang/constraint-decoding/readme.md` |
| DP Attention | — | `Awesome-ML-SYS-Tutorial/sglang/dp-attention/readme.md` |
| 量化架构 | — | `Awesome-ML-SYS-Tutorial/sglang/quantization/quantization_architecture.md` |
| SGLang Omni | — | `Awesome-ML-SYS-Tutorial/sglang/sglang-omni/` |
| 调度策略 | — | `Awesome-ML-SYS-Tutorial/sglang/scheduler/readme.md`, `Awesome-ML-SYS-Tutorial/sglang/sglang-scheduler/readme.md` |

---

## 9. 附录：辅助数据结构与函数规格

### 9.0 集中 Import 列表

以下是实现仿真器所需的全部标准库导入。各代码段不再重复声明：

```python
from __future__ import annotations
import heapq
import time
import itertools
import math
import queue
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    List, Dict, Tuple, Optional, NamedTuple, Union,
    Set, Any, Literal,
)
from collections import deque
```

### 9.1 PendingReq 数据结构

`PendingReq` 是 PrefillManager 内部使用的待处理请求封装，包含 chunked prefill 的续接状态：

```python
@dataclass
class PendingReq:
    """PrefillManager 内部的待处理请求"""
    uid: int
    input_ids: List[int]           # CPU 端 token 序列
    sampling_params: SamplingParams
    chunked_req: 'ChunkedReq | None' = None # 非 None 表示上一 tick chunked 的请求，需续接

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
```

### 9.1b BatchSamplingArgs / ForwardOutput / ForwardInput 定义

以下辅助类型贯穿 Scheduler 和 Engine：

```python
@dataclass
class BatchSamplingArgs:
    """采样参数的 batch 级聚合，由 Sampler.prepare(batch) 生成"""
    temperatures: List[float] | None = None  # None 表示全部 greedy
    top_k: List[int] | None = None           # None 表示不限制
    top_p: List[float] | None = None         # None 表示不限制

    @property
    def is_greedy(self) -> bool:
        return self.temperatures is None

class ForwardOutput(NamedTuple):
    """Engine.forward_batch 的返回值"""
    next_tokens_gpu: List[int]       # GPU 端 token（仿真中用 list 替代 tensor）
    next_tokens_cpu: List[int]       # CPU 端 token（async copy 的结果）
    copy_done_event: 'MockEvent'     # async copy 完成事件（仿真中为 noop）

class ForwardInput(NamedTuple):
    """Scheduler._prepare_batch 的返回值，传递给 _forward"""
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Tuple[List[int], List[int]]   # (table_idx_list, positions)
    write_tuple: Tuple[List[int], List[int]]   # (table_idx_list, write_pos)

ForwardData = Tuple[ForwardInput, ForwardOutput]
```

**消息类型补充**：

```python
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """批量 DetokenizeMsg 包装，用于多结果回传。
    命名沿用源码 BatchTokenizerMsg（结果通过 tokenizer 队列回传给前端）。"""
    data: List[DetokenizeMsg]
```

**chunked 续接机制**：
- 首次调度时 `chunked_req = None`，走完整分配流程（match_req → allocate → add_one_req）
- 如果 `chunk_size < remain_len`，创建 `ChunkedReq`，设置 `pending_req.chunked_req = req`
- 下次 tick 调度时 `PrefillAdder.try_add_one` 检查 `pending_req.chunked_req is not None`，直接走 `_add_one_req` 续接路径，复用已有的 `cache_handle` 和 `table_idx`
- `schedule_next_batch` 中 chunked 请求被放回 `pending_list` 头部：`self.pending_list = chunked_list + self.pending_list[len(reqs):]`，确保下一 tick 优先续接

### 9.2 _prepare_batch 辅助函数规格

#### _make_positions

```python
def _make_positions(batch: Batch) -> List[int]:
    """
    为 batch 中每个 padded_req 生成位置序列。
    每个 req 的位置 = [cached_len, device_len) 的连续整数。
    所有 req 的位置拼接为一个 flat list。

    示例：
      req0: cached_len=100, device_len=108 → positions [100,101,...,107]
      req1: cached_len=50,  device_len=55  → positions [50,51,52,53,54]
      结果: [100,101,...,107, 50,51,52,53,54]

    注意：源码中接收 device 参数用于将 tensor 传输到 GPU，
    仿真中不需要 device 参数，直接返回 list。
    """
    positions = []
    for req in batch.padded_reqs:
        positions.extend(range(req.cached_len, req.device_len))
    return positions
```

#### _make_input_tuple

```python
def _make_input_tuple(batch: Batch, positions: List[int]) -> Tuple[List[int], List[int]]:
    """
    生成 fancy indexing 二元组 (table_idx_list, positions_list)。
    用于从 token_pool 读取：token_pool[table_idx_list[i], positions_list[i]]

    对每个 padded_req，用其 table_idx 填充其 extend_len 个位置。

    注意：源码中第二个参数是 device（用于 GPU 传输），不是 positions。
    仿真中 positions 已经计算好，直接传入使用。

    示例：
      req0: table_idx=3, extend_len=8 → table_idx_list 填充 8 个 3
      req1: table_idx=5, extend_len=5 → table_idx_list 填充 5 个 5
      结果: ([3,3,...,3, 5,5,5,5,5], [100,...,107, 50,...,54])
    """
    table_idx_list = []
    for req in batch.padded_reqs:
        length = req.extend_len
        table_idx_list.extend([req.table_idx] * length)
    return table_idx_list, positions
```

#### _make_write_tuple

```python
def _make_write_tuple(batch: Batch) -> Tuple[List[int], List[int]]:
    """
    生成写入映射二元组 (table_idx_list, write_pos_list)。
    用于写入 next_tokens：token_pool[table_idx_list[i], write_pos_list[i]] = next_tokens[i]

    只对 batch.reqs（非 padded），每 req 一个写入位置。
    写入位置 = device_len（即新 token 的位置）。
    对 ChunkedReq 使用 -1，表示不可写入。

    示例：
      req0: table_idx=3, can_decode=True,  device_len=108 → write_pos=108
      req1: table_idx=5, can_decode=False (ChunkedReq), device_len=55 → write_pos=-1
      结果: ([3, 5], [108, -1])

    注意：write_pos=-1 的条目在 _forward 中通过 if p >= 0 跳过，不会实际写入。
    """
    table_idx_list = [req.table_idx for req in batch.reqs]
    write_pos_list = [req.device_len if req.can_decode else -1 for req in batch.reqs]
    return table_idx_list, write_pos_list
```

### 9.3 KV Cache 抽象类型完整定义

```python
class BaseKVCachePool(ABC):
    """KV cache 存储池抽象（仿真中为 mock，不做真实存储）"""
    @abstractmethod
    def store_kv(self, k, v, out_loc, layer_id: int) -> None: ...
    @property
    @abstractmethod
    def num_pages(self) -> int: ...
    @property
    @abstractmethod
    def page_size(self) -> int: ...
    @property
    @abstractmethod
    def total_capacity(self) -> int: ...
    @property
    @abstractmethod
    def used_capacity(self) -> int: ...

class BasePrefixCache(ABC):
    """前缀缓存抽象接口"""
    @abstractmethod
    def lock_handle(self, handle: 'BaseCacheHandle', unlock: bool = False) -> None: ...
    @abstractmethod
    def match_prefix(self, input_ids: List[int]) -> 'MatchResult': ...
    @abstractmethod
    def insert_prefix(self, input_ids: List[int], indices: List[int]) -> 'InsertResult': ...
    @abstractmethod
    def evict(self, size: int) -> List[int]: ...
    @abstractmethod
    def reset(self) -> None: ...
    @property
    @abstractmethod
    def size_info(self) -> 'CacheSizeInfo': ...
    @abstractmethod
    def check_integrity(self) -> None: ...

class BaseCacheHandle(ABC):
    """缓存句柄，指向 RadixTree 中的节点"""
    cached_len: int  # 该 handle 对应的已缓存长度
    @abstractmethod
    def get_matched_indices(self) -> List[int]: ...

class MatchResult(NamedTuple):
    """match_prefix 的返回值"""
    cuda_handle: BaseCacheHandle

class InsertResult(NamedTuple):
    """insert_prefix 的返回值"""
    cached_len: int   # 插入前已在缓存中的长度
    cuda_handle: BaseCacheHandle  # 与 MatchResult.cuda_handle 统一命名

# CacheSizeInfo 定义见 §9.8（统一使用 CacheSizeInfo，不单独定义 SizeInfo）

class BaseAttnBackend(ABC):
    """Attention backend 抽象（仿真中为 mock）"""
    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...
    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...
    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...
    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

class BaseAttnMetadata(ABC):
    """Attention metadata 抽象（仿真中为空对象）"""
    pass

class BaseMoeBackend(ABC):
    """MoE backend 抽象（仿真中为 mock）"""
    pass

# Mock 辅助类型
class MockStream:
    """模拟 CUDA stream，仿真中为 noop"""
    def wait_stream(self, other): pass

class MockEvent:
    """模拟 CUDA event，仿真中为 noop"""
    def record(self, stream=None): pass
    def synchronize(self): pass

class BaseTokenizerMsg:
    """Tokenizer 消息基类"""
    def encoder(self) -> Dict: ...
    @staticmethod
    def decoder(json: Dict) -> 'BaseTokenizerMsg': ...
```

#### 工具函数

```python
def align_down(n: int, alignment: int) -> int:
    """将 n 向下对齐到 alignment 的倍数"""
    return n - (n % alignment)

def div_ceil(a: int, b: int) -> int:
    """向上取整除法"""
    return (a + b - 1) // b

def div_even(a: int, b: int, allow_replicate: bool = False) -> List[int]:
    """
    将 a 均分到 b 份，返回每份大小列表。
    allow_replicate=True 时允许 a < b（部分份为 0，用于 TP 下 head 复制）。
    allow_replicate=False 时要求 a >= b，否则抛出 ValueError。
    例: div_even(8, 3) → [3, 3, 2]
    """
    if not allow_replicate and a < b:
        raise ValueError(f"div_even({a}, {b}) with allow_replicate=False requires a >= b")
    if a == 0:
        return [0] * b
    base = a // b
    remainder = a % b
    result = [base + 1] * remainder + [base] * (b - remainder)
    return result
```

### 9.3b NaiveCache 规格

Phase 1 中的 `NaiveCache` 是 `BasePrefixCache` 的简化实现，不使用 RadixTree：

```python
class NaivePrefixCache(BasePrefixCache):
    """无前缀匹配的简单 cache，用于 Phase 1 测试"""
    def __init__(self, num_pages: int, page_size: int):
        self.free_slots = list(range(0, num_pages * page_size, page_size))
        self.evictable_size = 0
        self.protected_size = 0
        # match_prefix 总是返回空 handle（cached_len=0）
        # insert_prefix 不做树操作，只分配页
        # evict 从 free_slots 中返回（但 NaiveCache 无 evictable 内容）
    def match_prefix(self, input_ids) -> MatchResult:
        return MatchResult(NaiveCacheHandle(0))  # cached_len=0
    def insert_prefix(self, input_ids, indices) -> InsertResult:
        # cached_len=0：NaiveCache 不存储前缀，所有页在 finished=True 时通过
        # cache_req 的 _free(page_indices[0:]) 全部回收，避免内存泄漏
        return InsertResult(0, NaiveCacheHandle(0))
    def lock_handle(self, handle, unlock=False): pass  # noop
    def evict(self, size): return []
    def reset(self): pass
    @property
    def size_info(self): return CacheSizeInfo(0, 0)
    def check_integrity(self): pass

@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    cached_len: int = 0  # 必须声明为 dataclass 字段（与 §9.11 一致）

    def get_matched_indices(self) -> List[int]:
        return []  # 无前缀匹配
```

### 9.4 Overlap Scheduling 仿真模型细化

#### 问题：tick 模型与 stream 模型的映射

实际 SGLang 使用两个 CUDA stream：
- **scheduler stream** (`self.stream`)：消息处理、batch 准备、结果处理
- **engine stream** (`self.engine.stream`)：model forward + 采样

overlap_loop 通过 `engine_stream_ctx` 将 forward 切换到 engine stream，与 scheduler stream 的后续操作并行。

**仿真决策**：仿真中**不保留 stream 同步语义**，而是用 `last_data` 参数模拟重叠。

> **注意**：本节取代 3.3.8 节的初步 tick 模型。3.3.8 中的 `last_gpu_finish_tick` 概念已被 `last_data` 延迟处理机制取代，后者更忠实地反映了源码 `overlap_loop` 的语义。4.3 节的 `SimulationClock` 作为可选的 GPU 时序追踪工具，可集成到 `run_tick` 中但不是必须的。

**源码 overlap_loop 执行顺序**：
1. 接收消息 + 调度下一批（scheduler stream）
2. forward 当前批（engine stream，启动后立即返回）
3. 处理上一批结果（scheduler stream，与当前 GPU forward 并行）

```python
class SimScheduler:
    def __init__(self, config):
        # ... 初始化 ...
        self.last_data: ForwardData | None = None  # 上一批的 forward 数据
        self.clock: SimulationClock | None = None  # 可选：GPU 时序追踪，默认不实例化

    def run_tick(self, incoming_msgs: List[BaseBackendMsg]) -> List[DetokenizeMsg]:
        """
        overlap 模式 tick 逻辑（忠实于源码 overlap_loop 的执行顺序）：
        1. 处理消息 + 调度下一批（CPU phase，可与上一批 GPU 重叠）
        2. forward 当前批（GPU phase，启动后立即返回 last_data 保存给下一 tick）
        3. 处理上一批结果（CPU phase，仿真中与 phase 2 串行但逻辑上并行）
        """
        # Phase 1: 消息处理 + 调度
        for msg in incoming_msgs:
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()

        # Phase 2: forward 当前批（启动 GPU 计算）
        forward_output = None
        if forward_input is not None:
            forward_output = self._forward(forward_input)
            # 可选：记录 GPU 时序
            # self.clock.schedule_gpu(self._get_forward_cost(forward_input))

        # Phase 3: 处理上一批结果（与当前 GPU forward 逻辑上并行，仿真中串行执行）
        reply = []
        if self.last_data is not None:
            reply = self._process_last_data(self.last_data)
            self.last_data = None  # 清除，防止重复释放

        # 保存当前批数据给下一 tick
        if forward_output is not None:
            self.last_data = (forward_input, forward_output)

        return reply
```

#### finished_reqs 防止重复释放

```python
def _process_last_data(self, last_data: ForwardData | None) -> List[DetokenizeMsg]:
    if last_data is None:
        return []
    batch = last_data[0].batch
    _, next_tokens_cpu, copy_done = last_data[1]
    copy_done.synchronize()  # 仿真中为 noop，但保留语义

    reply = []
    new_finished_reqs: Set[Req] = set()
    with self.cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue  # chunked 请求的采样结果被丢弃
            next_token = next_tokens_cpu[i]
            req.append_host(next_token)
            finished = not req.can_decode
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # overlap 下可能跨 tick 重复释放，用 finished_reqs 集合去重
            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                new_finished_reqs.add(req)
            elif batch.is_prefill:
                self.cache_manager.cache_req(req, finished=False)

    self.finished_reqs = new_finished_reqs  # 只保留本 tick 的 finished
    return reply
```

**关键时序说明**：
- 在 non-overlap 模式中，`_process_last_data` 在同 tick 内调用，`finished_reqs` 始终为空
- 在 overlap 模式中，`_process_last_data` 在下一 tick 调用，如果上一批的 finished 请求在本批又被处理（理论上不会，但防御性编程），`finished_reqs` 防止重复释放

### 9.5 lazy_free_region 概念说明

> **注意**：本节为概念性说明，完整实现见 §9.11 的 `CacheManager.lazy_free_region`。以下伪代码展示了设计原理：

```python
# 概念伪代码（实际实现见 §9.11，使用 _in_lazy_free 标志而非猴子补丁）
@contextmanager
def lazy_free_region(self):
    """
    在上下文内，所有 _free 调用被收集到 lazy_free_list，
    退出上下文时一次性合并到 free_slots。

    设计原因：cache_req 中多次 _free 调用会导致 free_slots
    反复 extend（O(n) 操作），lazy 模式减少合并次数。

    page-aligned 切片：_free 方法中对 indices 做 [::page_size] 切片，
    确保只取每页的起始位置（page_size>1 时去重）。
    """
    # §9.11 实际实现：设置 _in_lazy_free = True，
    # _free 方法检查该标志，将 indices 收集到 lazy_free_list，
    # 退出时 self.free_slots.extend(self.lazy_free_list)
    ...
```

### 9.6 SchedulerIOMixin 接口

```python
class SchedulerIOMixin:
    """Scheduler 的 IO 通信层，仿真中用 MockZmqQueue 替代 ZMQ"""

    def __init__(self, config: 'SimulatorConfig', tp_cpu_group: MockTPGroup):
        self.tp_cpu_group = tp_cpu_group

        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return

        # 单 rank 模式（tp_size=1）
        self._recv_from_tokenizer = MockZmqQueue()
        self._send_into_tokenizer = MockZmqQueue()

        self.receive_msg = self._recv_msg_single_rank
        self.send_result = self._reply_tokenizer_rank0

    def receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]: ...
    def send_result(self, reply: List[DetokenizeMsg]) -> None: ...
    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get(blocking=True))
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get(blocking=False))
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        if len(reply) == 1:
            self._send_into_tokenizer.put(reply[0])
        elif len(reply) > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式：不接收消息，消息由 run_tick 的参数传入"""
        return []

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式：不发送消息，结果由 run_tick 的返回值传递"""
        pass

    def run_when_idle(self) -> None:
        """scheduler 空闲时的回调（仿真中为 noop，源码中用于处理 pending 消息）"""
        pass
```

### 9.7 available_size 的两次检查（概念说明）

> **注意**：本节说明 `try_add_one` 中两次 `available_size` 检查的设计原理。完整实现见 §9.11 的 `PrefillAdder.try_add_one`，以下为概念性伪代码：

```python
# 概念性伪代码，完整实现见 §9.11 PrefillAdder.try_add_one
def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
    if self.table_manager.available_size == 0:
        return None

    handle = self.cache_manager.match_req(req).cuda_handle
    cached_len = handle.cached_len
    extend_len = req.input_len - cached_len
    estimated_len = extend_len + req.output_len

    # 第一次检查：lock 前
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return None

    # lock 会改变 evictable_size（ref_count 0→1 的节点从 evictable 移入 protected）
    self.cache_manager.lock(handle)

    # 第二次检查：lock 后，available_size 可能减小
    if estimated_len + self.reserved_size > self.cache_manager.available_size:
        return self.cache_manager.unlock(handle)  # 解锁并返回 None

    table_idx = self.table_manager.allocate()
    # 复制 cached 部分的 token 和 page entry
    if cached_len > 0:
        self.table_manager.token_pool[table_idx][:cached_len] = req.input_ids[:cached_len]
        self.table_manager.page_table[table_idx][:cached_len] = handle.get_matched_indices()

    return handle, table_idx
```

### 9.8 RadixTree 关键细节补充

#### RadixTreeNode 类定义

```python
class RadixTreeNode:
    """RadixTree 节点，存储 key（token 序列）和 value（page 索引序列）"""
    def __init__(self, key_fn, timestamp: int = 0):
        self.key_fn = key_fn               # 由 token 序列生成 dict key 的函数
        self.timestamp = timestamp         # LRU 用的时间戳
        self._key: List[int] = []          # 该节点对应的 token 序列
        self._value: List[int] = []        # 该节点对应的 page 索引序列
        self.children: Dict[Any, 'RadixTreeNode'] = {}
        self.parent: 'RadixTreeNode' | None = None
        self.ref_count: int = 0            # 引用计数，>0 表示被锁定（不可驱逐）

    @property
    def length(self) -> int:
        return len(self._key)

    @property
    def value(self) -> List[int]:
        """节点的 page 索引序列（供 get_matched_indices 和 evict 使用）"""
        return self._value

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        """是否为叶子节点（无子节点），供 evict 算法使用"""
        return len(self.children) == 0

    def __lt__(self, other: 'RadixTreeNode') -> bool:
        """供 heapq 按 timestamp 排序（LRU 驱逐）"""
        return self.timestamp < other.timestamp

    def set_key_value(self, key: List[int], value: List[int]) -> None:
        self._key = list(key)
        self._value = list(value)

    def set_parent(self, parent: 'RadixTreeNode') -> None:
        self.parent = parent
        if parent is not None:
            parent.children[self.key_fn(self._key)] = self

    def get_match_len(self, input_ids: List[int]) -> int:
        """比较节点 key 和 input_ids，返回匹配长度"""
        min_len = min(len(self._key), len(input_ids))
        for i in range(min_len):
            if self._key[i] != input_ids[i]:
                return i
        return min_len

    def split_at(self, pos: int) -> 'RadixTreeNode':
        """在位置 pos 分裂节点（实现见下方 split_at 细节）"""
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count  # 继承引用计数
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)  # 原节点成为新节点的子节点
        return new_node
```

#### RadixPrefixCache 类定义

```python
@dataclass
class CacheSizeInfo:
    """缓存大小统计"""
    evictable_size: int = 0   # ref_count=0 的节点 token 数（可被 LRU 驱逐）
    protected_size: int = 0   # ref_count>0 的节点 token 数（被锁定）

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

class RadixPrefixCache(BasePrefixCache):
    """基于 RadixTree 的前缀缓存，仿真核心 KV cache 管理器"""
    def __init__(self, num_pages: int, page_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        # key_fn: page_size=1 时用单个 token 值，page_size>1 时用前 page_size 个 token 的 tuple
        self.key_fn = lambda tokens: tokens[0] if page_size == 1 else tuple(tokens[:page_size])
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root 永远不可驱逐
        self._size_info = CacheSizeInfo(evictable_size=0, protected_size=0)
        # 注意：页分配器（free_slots）由 CacheManager 统一管理，不在 RadixPrefixCache 中维护

    @property
    def size_info(self) -> 'CacheSizeInfo':
        return self._size_info

    def _tree_walk(self, input_ids: List[int]) -> Tuple[RadixTreeNode, int]:
        """遍历树，返回最长前缀匹配的节点和匹配长度"""
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len

    def match_prefix(self, input_ids) -> MatchResult:
        """前缀匹配，返回 MatchResult（包含 RadixCacheHandle）"""
        node, match_len = self._tree_walk(list(input_ids))
        return MatchResult(RadixCacheHandle(cached_len=match_len, node=node))

    def insert_prefix(self, input_ids, indices) -> InsertResult:
        """插入前缀到树中，分配页"""
        # 插入页对齐部分
        insert_len = align_down(len(input_ids), self.page_size)
        if insert_len == 0:
            return InsertResult(0, RadixCacheHandle(0, self.root_node))
        node, match_len = self._tree_walk(input_ids[:insert_len])
        if match_len < insert_len:
            # 创建子节点存储未匹配部分
            child = RadixTreeNode(self.key_fn, time.monotonic_ns())
            child.set_key_value(
                input_ids[match_len:insert_len],
                indices[match_len:insert_len],
            )
            child.set_parent(node)
            self._size_info.evictable_size += child.length  # 更新可驱逐大小
            node = child
        return InsertResult(insert_len, RadixCacheHandle(insert_len, node))

    def lock_handle(self, handle: 'RadixCacheHandle', unlock: bool = False) -> None:
        """锁定/解锁节点，调整 ref_count 和 size_info"""
        node = handle.node
        if unlock:
            node.ref_count -= 1
            if node.ref_count == 0:
                self._size_info.protected_size -= node.length
                self._size_info.evictable_size += node.length
        else:
            if node.ref_count == 0:
                self._size_info.evictable_size -= node.length
                self._size_info.protected_size += node.length
            node.ref_count += 1

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """收集所有 ref_count==0 的叶子节点（可驱逐候选）"""
        result = []
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if not node.is_root() and node.is_leaf() and node.ref_count == 0:
                result.append(node)
            stack.extend(node.children.values())
        return result

    def evict(self, size: int) -> List[int]:
        """LRU 驱逐，返回释放的页索引列表（调用方负责管理 free_slots）"""
        evicted: List[int] = []
        evicted_size = 0
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)  # 按 timestamp 排序（最早的最先驱逐）
        while leave_nodes and evicted_size < size:
            node = heapq.heappop(leave_nodes)
            if node.ref_count > 0:
                continue  # 被锁定，跳过
            # 从树中移除节点
            parent = node.parent
            if parent is not None:
                del parent.children[self.key_fn(node._key)]
            # 回收页索引
            evicted.extend(node.value)
            evicted_size += node.length
            self._size_info.evictable_size -= node.length
            # 合并：如果父节点变成叶子且 ref_count==0，加入候选
            if (parent is not None and not parent.is_root()
                    and parent.is_leaf() and parent.ref_count == 0):
                heapq.heappush(leave_nodes, parent)
        return evicted

    def reset(self) -> None:
        """重置缓存"""
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1
        self._size_info = CacheSizeInfo(0, 0)

    def check_integrity(self) -> None:
        """RadixTree 内部完整性校验（仿真中简化为空检查）"""
        pass
```

#### split_at 的 ref_count 继承

```python
def split_at(self, pos: int) -> 'RadixTreeNode':
    """
    在位置 pos 分裂节点：
    - 新节点继承 [0, pos) 的 key/value 和原 ref_count
    - 原节点缩进到 [pos:]
    - 新节点替代原节点在父节点中的位置
    - 原节点成为新节点的子节点

    关键：new_node.ref_count = self.ref_count
    这保证分裂后引用计数正确传播，locked 的节点不会因为分裂而 unlocked
    """
    parent = self.parent
    new_node = RadixTreeNode(self.key_fn, self.timestamp)
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)
    new_node.ref_count = self.ref_count  # 继承引用计数

    self.set_key_value(self._key[pos:], self._value[pos:])
    self.set_parent(new_node)  # 原节点成为新节点的子节点
    return new_node
```

#### _tree_walk 中的 align_down

```python
def _tree_walk(self, input_ids: List[int]) -> Tuple[RadixTreeNode, int]:
    prefix_len = 0
    indice_len = len(input_ids)
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < indice_len:
        child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
        if child_node is None:
            return node, prefix_len

        node = child_node
        # 关键：align_down 确保只匹配页对齐部分
        match_len = node.get_match_len(input_ids[prefix_len:])
        match_len = align_down(match_len, self.page_size)  # 向下对齐到页边界
        prefix_len += match_len

        if match_len != node.length:
            # 部分匹配，需要分裂
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len

        # 完全匹配，更新 timestamp（LRU）
        node.timestamp = tic

    return node, prefix_len
```

#### root 节点保护

```python
# 在 __init__ 中：
self.root_node = RadixTreeNode(self.key_fn)
self.root_node.ref_count = 1  # root 永远不可驱逐
```

#### get_match_len / fast_compare_key

```python
def get_match_len(self, input_ids: List[int]) -> int:
    """
    比较节点 key 和 input_ids，返回匹配长度。
    仿真中用简单逐元素比较（实际代码用 CUDA kernel fast_compare_key）
    """
    min_len = min(len(self._key), len(input_ids))
    for i in range(min_len):
        if self._key[i] != input_ids[i]:
            return i
    return min_len
```

### 9.9 其他补充

#### dummy_req 与 dummy page

```python
# Engine 初始化时创建 dummy req 用于 CUDA Graph padding
self.dummy_req = Req(
    input_ids=[0],           # dummy token
    table_idx=max_running_req,  # 最后一行，预留给 dummy
    cached_len=0,
    output_len=1,
    uid=-1,                  # 无效 uid
    sampling_params=None,
    cache_handle=None,
)
# dummy req 的 page_table 行指向 dummy page（最后一页）
self.page_table[self.dummy_req.table_idx] = [num_tokens] * self.max_seq_len  # num_tokens = num_pages * page_size
```

#### forward_batch contextmanager

```python
# Context 的 forward_batch 是全局 batch 的设置器
# 模型层通过 get_global_ctx().batch 访问当前 batch
# 这使得 model.forward() 不需要显式接收 batch 参数
@contextmanager
def forward_batch(self, batch: Batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

#### page_table 32 字节对齐

```python
def _align_up_32(num: int) -> int:
    """将 max_seq_len 向上对齐到 32 的倍数，用于 page_table 第二维大小"""
    return (num + 31) // 32 * 32
# page_table shape: [max_running_req + 1, _align_up_32(max_seq_len)]
# 32 字节对齐是为了 GPU memory access efficiency
# 仿真中可忽略此对齐，但建议保留以保持一致性
```

#### check_integrity 接口

```python
def check_integrity(self) -> None:
    """
    CacheManager 完整性校验：
    1. prefix_cache.check_integrity() — RadixTree 内部校验
    2. free_pages + cache_pages == num_pages — 页数守恒
    3. free_slots 全部 page-aligned
    """
    self.prefix_cache.check_integrity()
    cache_pages = self.prefix_cache.size_info.total_size // self.page_size
    if len(self.free_slots) + cache_pages != self.num_pages:
        raise RuntimeError(
            f"CacheManager integrity check failed: "
            f"free_pages({len(self.free_slots)}) + "
            f"cache_pages({cache_pages}) != num_pages({self.num_pages})"
        )
    if self.page_size > 1:
        assert all(s % self.page_size == 0 for s in self.free_slots)
```

#### cuda_graph_max_bs 自动计算

```python
def _determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, total_memory):
    if cuda_graph_bs is not None:
        return cuda_graph_bs  # 用户指定

    total_memory_gb = total_memory / (1024**3)
    if cuda_graph_max_bs is None:
        # 自动选择：H200 (80+ GiB) → 256, 其他 → 160
        cuda_graph_max_bs = 256 if total_memory_gb > 80 else 160

    if cuda_graph_max_bs < 1:
        return []  # 禁用 CUDA Graph

    # 默认分桶: [1, 2, 4, 8, 16, 24, 32, ..., cuda_graph_max_bs]
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```

#### TableManager.free_table_indices vs CacheManager.free_slots 区分

模拟器中有两个不同的空闲资源池，语义不同：
- **TableManager.free_table_indices**：可用 `table_idx` 的集合（请求表行号），`allocate()` 返回一个 int
- **CacheManager.free_slots**：可用页的物理位置列表（page-aligned），用于 KV cache 存储分配

二者索引空间完全不同，不应混淆。本报告统一使用 `free_table_indices`（TableManager）和 `free_slots`（CacheManager）命名。

#### 多模态请求的系统层影响

多模态请求（如 Qwen2.5-VL）在系统层有特殊处理：
1. **图片 token 插入**：多模态请求的 input_ids 包含图片 placeholder token，RadixCache 对图片 token 做哈希优化以匹配共享图片
2. **不同 prefill 模式**：多模态可能需要单独的 image prefill 阶段
3. **M-RoPE**：多模态模型使用 3D RoPE（height/width/time），位置编码不同于纯文本

在模拟器中，多模态可作为**可选扩展**，核心仿真层只需将图片 token 视为普通 token 即可。如需精确模拟 cache hit 率，需为图片 token 使用特殊哈希。

#### MoE mock 的行为契约

MoE Backend 归为模拟组件，但其行为契约需要明确：
1. **路由不影响调度**：MoE 路由是模型内部行为，不改变 Scheduler 的 batch 调度决策
2. **EP（Expert Parallel）影响通信**：在 TP+EP 场景下，MoE 层需要 all-to-all 通信，这会影响 overlap timing。仿真中可添加 `moe_comm_cost_ticks` 参数
3. **Mock 行为**：`MockMoeBackend` 不做任何路由计算，只记录调用次数和 token 数量用于指标收集

#### 权重更新 stub 规格（可选）

即使列为可选模块，权重更新（RLHF 场景）的 stub 应提供以下接口：

```python
class MockWeightUpdater:
    """模拟在线权重更新"""
    def update_weights(self, new_weights_ref: str) -> None:
        """
        模拟权重更新流程：
        1. NCCL 广播新权重到所有 TP rank（仿真中为 noop）
        2. 重新加载 state_dict（仿真中为 noop）
        3. 重新捕获 CUDA Graph（仿真中标记 graph 失效）
        """
        # 标记 CUDA Graph 需要重新捕获
        self.engine.graph_runner.invalidate()
```

#### 仿真保真度已知偏差

模拟器的简化导致的已知偏差（使用时需注意）：

| 简化 | 偏差影响 | 缓解建议 |
|------|---------|---------|
| Mock Sampler 用随机 logits | 共享前缀工作负载的 cache hit 率偏低 | 可配置 token 分布模型 |
| CPU-GPU 传输延迟 = 0 | overlap timing 偏乐观 | 可添加 `transfer_cost_ticks` 参数 |
| CUDA Graph capture 成本 = 0 | 启动阶段行为偏乐观 | 可添加 `capture_cost_ticks` |
| ZMQ 通信延迟 = 0 | 高并发调度偏乐观 | 可添加 `comm_latency_ticks` |
| 无内存碎片 | 内存利用率偏乐观 | 可添加碎片化模型 |
| Tensor 操作成本 = 0 | CPU 调度时间偏乐观 | 可在 tick 模型中加权 |

### 9.10 测试策略补充

#### 工作负载辅助类型

```python
@dataclass
class SimRequest:
    """工作负载生成器产生的模拟请求"""
    uid: int
    arrival_tick: int          # 请求到达的 tick
    input_ids: List[int]
    output_len: int           # 预期输出长度
    sampling_params: SamplingParams

class WorkloadGenerator:
    def _sample_len(self, distribution: str, min_val: int, max_val: int,
                    mean: int = 0, std: int = 0) -> int:
        """从指定分布采样长度"""
        if distribution == "uniform":
            return random.randint(min_val, max_val)
        elif distribution == "normal":
            return max(min_val, min(max_val, int(random.gauss(mean, std))))
        else:
            return mean

    def _sample_arrival(self, config: WorkloadConfig, index: int) -> int:
        """采样请求到达 tick"""
        if config.arrival_distribution == "poisson":
            return int(index / config.arrival_rate)
        else:
            return index

    def _generate_tokens(self, length: int, shared_prefix_ratio: float,
                         shared_prefix_len: int, uid: int) -> List[int]:
        """生成 token 序列，部分请求共享前缀"""
        if shared_prefix_ratio > 0 and uid % 3 == 0:  # 1/3 请求共享前缀
            prefix = list(range(shared_prefix_len))
            suffix = list(range(shared_prefix_len, shared_prefix_len + length - shared_prefix_len))
            return prefix + suffix
        return list(range(length))
```

#### SimulationMetrics 方法补充

```python
class SimulationMetrics:
    # ... 数据字段见 4.5 节 ...

    def record_reply(self, reply: List['DetokenizeMsg'], tick: int) -> None:
        """记录一个 tick 的回复消息"""
        for msg in reply:
            self.total_tokens_generated += 1
            if msg.finished:
                self.completed_requests += 1

    def record_batch(self, batch: 'Batch', gpu_ticks: int) -> None:
        """记录一次 batch forward。GPU busy ticks 由 record_tick 统一管理，
        此处不重复累加，避免双重计数。"""
        if batch.is_prefill:
            self.prefill_batches += 1
            self.avg_prefill_batch_size = (
                (self.avg_prefill_batch_size * (self.prefill_batches - 1) + batch.size)
                / self.prefill_batches
            )
        else:
            self.decode_batches += 1
            self.avg_decode_batch_size = (
                (self.avg_decode_batch_size * (self.decode_batches - 1) + batch.size)
                / self.decode_batches
            )
        # gpu_ticks 参数保留供调用方通过 record_tick(tick, gpu_busy=gpu_ticks) 记录
```

#### 完整测试示例：单请求 prefill → decode → finish

```python
def test_single_request_lifecycle():
    """验证单请求的完整生命周期"""
    import random
    random.seed(42)  # 固定随机种子，确保 mock logits 确定性
    config = SimulatorConfig(
        model_config=ModelConfig(num_layers=32, hidden_size=4096, num_kv_heads=8,
                                  head_dim=128, vocab_size=128256, is_moe=False),
        max_running_req=4, max_seq_len=8192, max_extend_tokens=8192,
        page_size=1, total_gpu_memory=80*1024**3,
        enable_cuda_graph=False, enable_overlap=False,
        mock_sample_mode="greedy",  # 确定性输出
    )
    scheduler = SimScheduler(config)

    # 构造请求（ignore_eos=True 防止随机 logits 的 argmax 命中 EOS 导致提前结束）
    input_ids = list(range(100))  # 100 个 token
    msgs = [UserMsg(uid=0, input_ids=input_ids,
                     sampling_params=SamplingParams(max_tokens=5, temperature=0.0,
                                                    ignore_eos=True))]

    # Tick 1: prefill（生成 1 个 token，device_len: 100→101）
    reply = scheduler.run_tick(msgs)
    assert len(reply) == 1
    assert reply[0].uid == 0
    assert not reply[0].finished  # remain_len = 105-101 = 4 > 0

    # Tick 2-5: decode 4 个 token（共 5 个 = max_tokens）
    # Tick 2: device_len 101→102, remain_len=3, not finished
    # Tick 3: device_len 102→103, remain_len=2, not finished
    # Tick 4: device_len 103→104, remain_len=1, not finished
    # Tick 5: device_len 104→105, remain_len=0, FINISHED
    for i in range(4):
        reply = scheduler.run_tick([])  # 无新请求
        assert len(reply) == 1
        if i < 3:
            assert not reply[0].finished
        else:
            assert reply[0].finished  # 第 4 次 decode 后 remain_len=0，完成

    # 验证资源释放
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == config.max_running_req
    # 前缀仍留在 cache 中（evictable），total_size > 0
    assert scheduler.cache_manager.prefix_cache.size_info.evictable_size > 0
```

#### Overlap vs Non-Overlap 性能对比测试

```python
def test_overlap_vs_non_overlap():
    """对比 overlap 和 non-overlap 模式的 GPU 利用率"""
    import random
    workload = WorkloadGenerator().generate(WorkloadConfig(
        num_requests=100,
        input_len_min=200, input_len_max=500,
        output_len_min=100, output_len_max=200,
        arrival_rate=5.0,  # 每 tick 5 个请求
    ))

    for overlap in [True, False]:
        random.seed(42)  # 每次迭代重置随机种子，确保两种模式使用相同的随机序列
        config = SimulatorConfig(
            model_config=ModelConfig(num_layers=32, hidden_size=4096, num_kv_heads=8,
                                      head_dim=128, vocab_size=128256, is_moe=False),
            enable_overlap=overlap,
            enable_cuda_graph=True,
            graph_replay_cost_ticks=1,
            eager_forward_cost_ticks=10,
        )
        scheduler = SimScheduler(config)
        metrics = SimulationMetrics()

        # 运行工作负载
        for tick in range(1000):
            incoming = [r for r in workload if r.arrival_tick == tick]
            msgs = [UserMsg(r.uid, r.input_ids, r.sampling_params) for r in incoming]
            reply = scheduler.run_tick(msgs)
            metrics.record_reply(reply, tick)
            # 记录本 tick 实际调度的 batch 信息
            # 注意：GPU busy 只在有 batch 时记录，避免与 record_batch 重复计数
            if scheduler.last_batch is not None:
                gpu_ticks = (config.graph_replay_cost_ticks
                             if config.enable_cuda_graph
                             else config.eager_forward_cost_ticks)
                metrics.record_batch(scheduler.last_batch, gpu_ticks)
                metrics.record_tick(tick, gpu_busy=gpu_ticks)
            else:
                metrics.record_tick(tick, gpu_busy=0)  # 无 batch，GPU 空闲

        # Overlap 模式需要额外一个空 tick 刷新 last_data
        if overlap:
            reply = scheduler.run_tick([])
            metrics.record_reply(reply, 1000)

        # 验证
        print(f"Overlap={overlap}: throughput={metrics.completed_requests}/{metrics.total_ticks}")
        print(f"  GPU utilization: {metrics.gpu_utilization:.2%}")
        print(f"  Avg batch size: prefill={metrics.avg_prefill_batch_size}, decode={metrics.avg_decode_batch_size}")

    # 预期：overlap 模式因延迟处理 last_data，相同 tick 内可处理更多 batch，
    # GPU 利用率应高于 non-overlap。注意：mock 不模拟真实 GPU 时序，
    # 实际差异取决于 graph_replay_cost_ticks vs eager_forward_cost_ticks 的配置
```

### 9.11 完整实现代码集

本节汇总所有在前面章节引用但未给出完整代码的函数和类实现。

#### SimScheduler 完整方法集

```python
class SimScheduler(SchedulerIOMixin):
    def __init__(self, config: 'SimulatorConfig'):
        self.engine = MockEngine(config)
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.finished_reqs: Set[Req] = set()
        self.eos_token_id = config.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.overlap_enabled = config.enable_overlap
        self.offline_mode = config.offline_mode
        self.last_data: ForwardData | None = None
        self.last_batch: Batch | None = None  # 供外部指标收集使用
        self.clock: 'SimulationClock | None' = None  # 可选：GPU 时序追踪，默认不实例化
        # 初始化 IO 层
        super().__init__(config, MockTPGroup())

    def run_tick(self, incoming_msgs: List[BaseBackendMsg]) -> List[DetokenizeMsg]:
        if self.overlap_enabled:
            return self._overlap_tick(incoming_msgs)
        else:
            return self._normal_tick(incoming_msgs)

    def _normal_tick(self, incoming_msgs: List[BaseBackendMsg]) -> List[DetokenizeMsg]:
        """非 overlap 模式：串行执行（对应源码 normal_loop）"""
        for msg in incoming_msgs:
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))
        return self._process_last_data(ongoing_data)

    def _overlap_tick(self, incoming_msgs: List[BaseBackendMsg]) -> List[DetokenizeMsg]:
        """overlap 模式：延迟处理上一批结果（对应源码 overlap_loop）

        注意：overlap 模式下，最后一批的 forward 结果保存在 last_data 中，
        需要调用方发送一个额外的空 tick（run_tick([])）来刷新。
        """
        for msg in incoming_msgs:
            self._process_one_msg(msg)
        forward_input = self._schedule_next_batch()

        forward_output = None
        if forward_input is not None:
            forward_output = self._forward(forward_input)

        reply = []
        if self.last_data is not None:
            reply = self._process_last_data(self.last_data)
            self.last_data = None

        if forward_output is not None:
            self.last_data = (forward_input, forward_output)

        return reply

    def _process_last_data(self, last_data: ForwardData | None) -> List[DetokenizeMsg]:
        """处理上一批结果（见 9.4 节完整代码）"""
        if last_data is None:
            return []
        batch = last_data[0].batch
        _, next_tokens_cpu, copy_done = last_data[1]
        copy_done.synchronize()

        reply = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token)
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # 仿真中通过 return reply 将结果交给调用方（run_tick 的返回值）
        # 真实 SGLang 中通过 ZMQ send_result 推入 Detokenizer 队列，
        # 但仿真器不需要 ZMQ 通信，reply 由 run_tick 返回值直接传递给调用方
        return reply

    def _schedule_next_batch(self) -> ForwardInput | None:
        """调度下一批：prefill 优先，否则 decode"""
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is not None:
            self.last_batch = batch  # 保存供外部指标收集
        return self._prepare_batch(batch) if batch else None

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """准备 batch 的所有元数据"""
        self.engine.graph_runner.pad_batch(batch)
        for req in batch.reqs:
            self.cache_manager.allocate_paged(req)
        batch.positions = _make_positions(batch)
        input_mapping = _make_input_tuple(batch, batch.positions)
        write_mapping = _make_write_tuple(batch)
        batch.out_loc = [self.engine.page_table[t][p]
                         for t, p in zip(*input_mapping)]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """执行 forward 并更新 token_pool"""
        batch, sample_args, input_tuple, write_tuple = forward_input
        # 从 token_pool 读取 input_ids
        batch.input_ids = [self.token_pool[t][p]
                           for t, p in zip(*input_tuple)]
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 将 next_tokens 写入 token_pool（跳过 -1 的 chunked 位置）
        write_t, write_p = write_tuple
        for i, (t, p) in enumerate(zip(write_t, write_p)):
            if p >= 0:
                self.token_pool[t][p] = forward_output.next_tokens_gpu[i]
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理消息"""
        if isinstance(msg, BatchBackendMsg):
            for m in msg.data:
                self._process_one_msg(m)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            input_len = len(msg.input_ids)
            max_output_len = self.engine.max_seq_len - input_len
            if max_output_len <= 0:
                return
            # 创建 SamplingParams 副本，避免修改传入对象
            # replace 是 dataclasses.replace，需要 from dataclasses import replace
            sp = msg.sampling_params
            if sp.max_tokens > max_output_len:
                sp = replace(sp, max_tokens=max_output_len)
            if sp.max_tokens <= 0:
                return  # 无需生成 token，跳过
            msg.sampling_params = sp
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)

    def _free_req_resources(self, req: Req) -> None:
        """释放请求的所有资源"""
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)
```

#### ChunkedReq 类定义

```python
class ChunkedReq(Req):
    """Chunked prefill 请求，不应被采样"""
    def append_host(self, next_token: int) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # 避免 ChunkedReq 被加入 DecodeManager
```

#### RadixCacheHandle 类定义

```python
@dataclass
class RadixCacheHandle(BaseCacheHandle):
    """RadixTree 节点缓存句柄"""
    cached_len: int
    node: 'RadixTreeNode'

    def get_matched_indices(self) -> List[int]:
        """从 root 到当前节点路径上所有节点的 value 拼接"""
        node = self.node
        value_list: List[List[int]] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return list(itertools.chain.from_iterable(value_list))
```

#### NaiveCacheHandle 修正

```python
# 与 §9.3b 定义相同，此处为 §9.11 完整实现的一部分
@dataclass(frozen=True)
class NaiveCacheHandle(BaseCacheHandle):
    cached_len: int = 0  # 必须声明为 dataclass 字段

    def get_matched_indices(self) -> List[int]:
        return []
```

#### PrefillManager 完整实现

```python
class PrefillManager:
    """管理待 prefill 的请求队列，根据 token budget 和 cache 可用性调度"""

    def __init__(self, cache_manager: 'CacheManager',
                 table_manager: 'TableManager',
                 decode_manager: 'DecodeManager'):
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.decode_manager = decode_manager
        self.pending_list: List[PendingReq] = []

    def add_batch(self, reqs: List[PendingReq]) -> None:
        """将新请求加入待 prefill 队列"""
        self.pending_list.extend(reqs)

    def add_one_req(self, msg: 'UserMsg') -> None:
        """将 UserMsg 转换为 PendingReq 并加入队列"""
        pending = PendingReq(
            uid=msg.uid,
            input_ids=list(msg.input_ids),
            sampling_params=msg.sampling_params,
        )
        self.pending_list.append(pending)

    def abort_req(self, uid: int) -> 'Req | None':
        """通过 uid 中止待 prefill 的请求，返回已分配的 req（如有）"""
        for i, pending in enumerate(self.pending_list):
            if pending.uid == uid:
                self.pending_list.pop(i)
                return pending.chunked_req  # 返回已分配的 req（如有）
        return None

    def schedule_next_batch(self, token_budget: int) -> Batch | None:
        """调度一个 prefill batch"""
        if not self.pending_list:
            return None

        adder = PrefillAdder(
            token_budget=token_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            decode_manager=self.decode_manager,
        )

        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        i = 0
        while i < len(self.pending_list):
            pending_req = self.pending_list[i]
            result = adder.try_add_one(pending_req)
            if result is None:
                break  # 资源/budget 不足，停止调度，保留未处理请求
            reqs.append(result)
            if isinstance(result, ChunkedReq):
                # chunked 请求剩余部分放回队列头部，携带 chunked_req 以便续接
                remaining = PendingReq(
                    uid=pending_req.uid,
                    input_ids=pending_req.input_ids,
                    sampling_params=pending_req.sampling_params,
                    chunked_req=result,  # 保存 chunked 状态，下次 tick 走续接路径
                )
                chunked_list.append(remaining)
            i += 1

        # chunked_list 优先放回队列头部，确保下一 tick 优先续接
        # self.pending_list[i:] 保留未被调度的请求（因 break 跳出）
        self.pending_list = chunked_list + self.pending_list[i:]

        if not reqs:
            return None
        return Batch(reqs=reqs, phase="prefill")


class PrefillAdder:
    """逐个尝试将请求加入 prefill batch"""

    def __init__(self, token_budget: int, reserved_size: int,
                 cache_manager: 'CacheManager', table_manager: 'TableManager',
                 decode_manager: 'DecodeManager'):
        self.token_budget = token_budget
        self.reserved_size = reserved_size
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.decode_manager = decode_manager
        self.consumed_tokens = 0

    def try_add_one(self, pending_req: PendingReq) -> Req | ChunkedReq | None:
        """尝试将一个 PendingReq 转换为可执行的 Req/ChunkedReq"""
        # 0. 检查是否为 chunked 续接（上一 tick 未完成的请求）
        if pending_req.chunked_req is not None:
            return self._try_add_one_chunked(pending_req)

        # 1. 前缀匹配
        match_result = self.cache_manager.match_req(pending_req)
        handle = match_result.cuda_handle
        cached_len = handle.cached_len
        extend_len = pending_req.input_len - cached_len

        # 2. 资源检查
        estimated_len = extend_len + pending_req.output_len
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # 资源不足

        # 3. token budget 检查
        remaining_budget = self.token_budget - self.consumed_tokens
        if remaining_budget <= 0:
            return None

        # 4. lock handle（改变 evictable_size）
        self.cache_manager.lock(handle)

        # 5. 再次检查（lock 后 available_size 可能减小）
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        # 6. 分配 table_idx
        table_idx = self.table_manager.allocate()

        # 7. 复制 cached 部分的 token 和 page entry
        if cached_len > 0:
            self.table_manager.token_pool[table_idx][:cached_len] = \
                pending_req.input_ids[:cached_len]
            self.table_manager.page_table[table_idx][:cached_len] = \
                handle.get_matched_indices()

        # 7b. 复制 extend 部分的 token 到 token_pool（供 _forward 读取 batch.input_ids）
        # 注意：page_table 的 extend 部分由 _prepare_batch 中的 allocate_paged 填充
        chunk_size = min(extend_len, remaining_budget)
        if chunk_size > 0:
            self.table_manager.token_pool[table_idx][cached_len:cached_len + chunk_size] = \
                pending_req.input_ids[cached_len:cached_len + chunk_size]

        # 8. 决定 chunk_size（复用 7b 已计算的值）
        is_chunked = chunk_size < extend_len

        if is_chunked:
            # 创建 ChunkedReq：只处理 chunk_size 个 token
            req = ChunkedReq(
                input_ids=pending_req.input_ids,
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=0,  # ChunkedReq 不生成 token
                uid=pending_req.uid,
                sampling_params=pending_req.sampling_params,
                cache_handle=handle,
            )
            req.device_len = cached_len + chunk_size
            req.max_device_len = cached_len + chunk_size
        else:
            # 创建完整 Req
            req = Req(
                input_ids=pending_req.input_ids,
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=pending_req.output_len,
                uid=pending_req.uid,
                sampling_params=pending_req.sampling_params,
                cache_handle=handle,
            )
            req.device_len = cached_len + chunk_size
            req.max_device_len = cached_len + extend_len + pending_req.output_len

        # 9. 更新已消耗的 token 预算
        # 注意：页分配在 _prepare_batch 中统一执行（避免重复分配）
        self.consumed_tokens += chunk_size

        # 10. 加入 decode_manager（非 chunked 的请求）
        if not is_chunked:
            self.decode_manager.add_req(req)

        return req

    def _try_add_one_chunked(self, pending_req: PendingReq) -> Req | ChunkedReq | None:
        """续接上一 tick 的 chunked prefill 请求，复用已有的 cache_handle 和 table_idx"""
        prev_req = pending_req.chunked_req
        handle = prev_req.cache_handle
        cached_len = prev_req.device_len  # 上次已处理到的位置
        extend_len = pending_req.input_len - cached_len  # 剩余需处理的 token 数

        # 资源检查
        estimated_len = extend_len + pending_req.output_len
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None

        # token budget 检查
        remaining_budget = self.token_budget - self.consumed_tokens
        if remaining_budget <= 0:
            return None

        # 复用已有的 table_idx（不重新分配）
        table_idx = prev_req.table_idx

        # 决定 chunk_size
        chunk_size = min(extend_len, remaining_budget)
        is_chunked = chunk_size < extend_len

        # 复制本 chunk 的 token 到 token_pool（供 _forward 读取 batch.input_ids）
        if chunk_size > 0:
            self.table_manager.token_pool[table_idx][cached_len:cached_len + chunk_size] = \
                pending_req.input_ids[cached_len:cached_len + chunk_size]

        if is_chunked:
            req = ChunkedReq(
                input_ids=pending_req.input_ids,
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=0,  # ChunkedReq 不生成 token
                uid=pending_req.uid,
                sampling_params=pending_req.sampling_params,
                cache_handle=handle,
            )
            req.device_len = cached_len + chunk_size
            req.max_device_len = cached_len + chunk_size
        else:
            # 最后一个 chunk：转换为完整 Req，可参与 decode
            req = Req(
                input_ids=pending_req.input_ids,
                table_idx=table_idx,
                cached_len=cached_len,
                output_len=pending_req.output_len,
                uid=pending_req.uid,
                sampling_params=pending_req.sampling_params,
                cache_handle=handle,
            )
            req.device_len = cached_len + chunk_size
            req.max_device_len = pending_req.input_len + pending_req.output_len

        # 更新已消耗的 token 预算
        # 注意：页分配在 _prepare_batch 中统一执行（避免重复分配）
        self.consumed_tokens += chunk_size

        if not is_chunked:
            self.decode_manager.add_req(req)

        return req
```

#### DecodeManager 完整实现

```python
class DecodeManager:
    """管理可 decode 的请求集合，生成 decode batch"""

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.running_reqs: Set[Req] = set()

    def add_req(self, req: Req) -> None:
        """将 prefill 完成的请求加入 decode 队列"""
        self.running_reqs.add(req)

    def remove_req(self, req: Req) -> None:
        """从 decode 队列移除已完成的请求"""
        self.running_reqs.discard(req)

    def filter_reqs(self, new_reqs: List[Req]) -> None:
        """每次 forward 后更新 running_reqs：移除已完成的，加入新的"""
        all_reqs = self.running_reqs | set(new_reqs)
        self.running_reqs = {req for req in all_reqs if req.can_decode}

    @property
    def inflight_tokens(self) -> int:
        """当前 decode 在途 token 数（用于 prefill 的 reserved_size 计算）。
        tokens_reserved = (page_size - 1) * len(running_reqs)：
        每个 running req 因页对齐最多浪费 page_size-1 个 token 位置，
        预留此量防止 prefill 分配导致 decode OOM。"""
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """调度一个 decode batch"""
        if not self.running_reqs:
            return None
        sorted_reqs = sorted(self.running_reqs, key=lambda r: r.uid)
        return Batch(reqs=sorted_reqs, phase="decode")

    def abort_req(self, uid: int) -> 'Req | None':
        """通过 uid 中止请求，返回被移除的 Req（与 PrefillManager.abort_req 统一命名）"""
        for req in list(self.running_reqs):
            if req.uid == uid:
                self.running_reqs.discard(req)
                return req
        return None
```

#### CacheManager 完整实现

```python
class CacheManager:
    """封装 KV cache 页分配、前缀缓存管理、eviction"""

    def __init__(self, num_pages: int, page_size: int,
                 page_table: List[List[int]], cache_type: str = "radix"):
        self.num_pages = num_pages
        self.page_size = page_size
        self.page_table = page_table
        self.free_slots: List[int] = list(range(0, num_pages * page_size, page_size))
        self.lazy_free_list: List[int] = []
        self._in_lazy_free = False

        if cache_type == "radix":
            self.prefix_cache: BasePrefixCache = RadixPrefixCache(num_pages, page_size)
        else:
            self.prefix_cache = NaivePrefixCache(num_pages, page_size)

    @property
    def available_size(self) -> int:
        """可用 token 数 = evictable cache + free pages"""
        return (self.prefix_cache.size_info.evictable_size
                + len(self.free_slots) * self.page_size)

    def match_req(self, req: 'PendingReq') -> 'MatchResult':
        """匹配前缀缓存，排除最后一个 token"""
        input_len = req.input_len
        assert input_len > 0
        return self.prefix_cache.match_prefix(req.input_ids[:input_len - 1])

    def lock(self, handle: 'BaseCacheHandle') -> None:
        self.prefix_cache.lock_handle(handle)

    def unlock(self, handle: 'BaseCacheHandle') -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, req: 'Req') -> None:
        """为请求分配页，将物理位置写入 page_table。
        页数由 req.device_len 和 req.cached_len 自动计算，无需传入 extend_len。"""
        device_len = req.device_len
        cached_len = req.cached_len
        last_page = div_ceil(device_len, self.page_size) - div_ceil(cached_len, self.page_size)
        needed_pages = max(0, last_page)

        if needed_pages > len(self.free_slots):
            # 触发 eviction：从 RadixCache 的 evictable 节点中回收页
            evict_size = (needed_pages - len(self.free_slots)) * self.page_size
            evicted = self.prefix_cache.evict(evict_size)
            self.free_slots.extend(evicted)

        # 分配页并写入 page_table
        # 边界情况：如果 eviction 后仍不足（所有页被 locked/protected），
        # 循环会在 free_slots 耗尽时 break，未分配的页位置保持为 0。
        # 此情况在正常负载下不会发生（PrefillAdder 的 available_size 检查会拦截），
        # 但作为防御性编程保留 break。
        for i in range(needed_pages):
            if not self.free_slots:
                break
            page_idx = self.free_slots.pop()  # O(1)，从末尾取页
            # page_table 第二维是 token 位置（非页号），需乘 page_size 转为位置索引
            start_pos = (div_ceil(cached_len, self.page_size) + i) * self.page_size
            for j in range(self.page_size):
                pos = start_pos + j
                if pos < len(self.page_table[req.table_idx]):
                    self.page_table[req.table_idx][pos] = page_idx

    def cache_req(self, req: 'Req', *, finished: bool) -> None:
        """将请求的已计算 token 插入前缀缓存"""
        # page-aligned 切片：只处理完整页
        aligned_len = (req.cached_len // self.page_size) * self.page_size
        insert_ids = req.input_ids[:aligned_len]
        page_indices = self.page_table[req.table_idx][:aligned_len]
        old_handle = req.cache_handle

        # 插入前缀到 RadixCache
        insert_result = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        cached_len = insert_result.cached_len
        new_handle = insert_result.cuda_handle

        # 解锁旧 handle
        if old_handle is not None:
            self.unlock(old_handle)

        # 释放已存在于缓存中的重复部分
        if old_handle is not None:
            self._free(page_indices[old_handle.cached_len:cached_len])

        if finished:
            # 释放尾部
            self._free(page_indices[new_handle.cached_len:])
        else:
            # 保留尾部，更新 handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def _free(self, indices: List[int]) -> None:
        """释放页索引（page-aligned：page_size>1 时只取每页起始位置去重）"""
        if self.page_size > 1:
            indices = indices[::self.page_size]  # 只取每页起始位置
        if self._in_lazy_free:
            self.lazy_free_list.extend(indices)
        else:
            self.free_slots.extend(indices)

    @contextmanager
    def lazy_free_region(self):
        """在上下文内收集 free 操作，退出时批量合并"""
        self._in_lazy_free = True
        self.lazy_free_list = []
        try:
            yield
        finally:
            self._in_lazy_free = False
            self.free_slots.extend(self.lazy_free_list)
            self.lazy_free_list = []

    def check_integrity(self) -> None:
        """完整性校验：页数守恒 + RadixTree 内部校验"""
        # 1. RadixTree 内部完整性校验
        self.prefix_cache.check_integrity()
        # 2. 页数守恒
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                f"CacheManager integrity check failed: "
                f"free_pages({len(self.free_slots)}) + "
                f"cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
```

#### TableManager 完整实现

```python
class TableManager:
    """管理请求表（page_table 的行分配）"""

    def __init__(self, max_running_req: int, page_table: List[List[int]]):
        self.max_running_req = max_running_req
        self.page_table = page_table
        # free_table_indices: 0 ~ max_running_req-1（最后一行 max_running_req 预留给 dummy）
        self.free_table_indices = list(range(max_running_req))
        # token_pool: 每行存储 token 值（仿真中用 list）
        self.token_pool: List[List[int]] = [
            [0] * len(page_table[0]) for _ in range(max_running_req + 1)
        ]

    @property
    def available_size(self) -> int:
        return len(self.free_table_indices)

    def allocate(self) -> int:
        """分配一个 table_idx"""
        if not self.free_table_indices:
            raise RuntimeError("No available table indices")
        return self.free_table_indices.pop()  # O(1)，从末尾取

    def free(self, table_idx: int) -> None:
        """释放一个 table_idx"""
        self.free_table_indices.append(table_idx)
```

#### MockSampler 完整实现

```python
class MockSampler:
    """模拟采样器"""

    def __init__(self, vocab_size: int, mode: str = "random"):
        self.vocab_size = vocab_size
        self.mode = mode

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """从 batch 的 sampling_params 生成 BatchSamplingArgs"""
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params) or self.mode == "greedy":
            return BatchSamplingArgs(temperatures=None)
        return BatchSamplingArgs(
            temperatures=[p.temperature for p in params],
            top_k=[p.top_k for p in params],
            top_p=[p.top_p for p in params],
        )

    def sample(self, logits: List[List[float]], args: BatchSamplingArgs) -> List[int]:
        """生成 token 列表"""
        if args.is_greedy or self.mode == "greedy":
            return [max(range(len(row)), key=lambda j: row[j]) for row in logits]
        else:
            return [random.randint(0, self.vocab_size - 1) for _ in logits]
```

#### SimGraphRunner 完整实现

```python
class SimGraphRunner:
    """仿真版 CUDA Graph Runner"""

    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig',
                 dummy_req: 'Req'):
        self.enable_cuda_graph = config.enable_cuda_graph
        self.graph_bs_list = _determine_cuda_graph_bs(
            config.cuda_graph_bs, config.cuda_graph_max_bs, config.total_gpu_memory
        )
        self.max_graph_bs = max(self.graph_bs_list) if self.graph_bs_list else 0
        self.vocab_size = model_config.vocab_size
        # dummy_req 由 MockEngine 创建并传入（table_idx = max_running_req）
        self.dummy_req = dummy_req

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if not self.enable_cuda_graph:
            return False
        return batch.is_decode and batch.size <= self.max_graph_bs

    def pad_batch(self, batch: Batch) -> None:
        if self.can_use_cuda_graph(batch):
            padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        else:
            padded_size = batch.size
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    def replay(self, batch: Batch) -> List[List[float]]:
        """仿真版 replay：返回随机 logits"""
        import random
        return [[random.random() for _ in range(self.vocab_size)]
                for _ in range(batch.size)]

    def destroy_cuda_graphs(self) -> None:
        pass  # 仿真中无需清理
```

#### MockEngine 完整实现

```python
class MockEngine:
    """模拟 Engine，不执行真实模型计算"""

    def __init__(self, config: 'SimulatorConfig'):
        self.model_config = config.model_config
        self.device = "cpu"
        self.stream = MockStream()
        self.dtype = "float16"
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # 内存预算计算
        self.num_pages = self._calculate_num_pages(config)
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)

        # page_table（仿真中用 list of lists）
        self.page_table = [[0] * self.max_seq_len
                           for _ in range(config.max_running_req + 1)]

        # Mock KV cache pool
        self.ctx.kv_cache = MockKVCachePool(config.model_config, self.num_pages, config.page_size)

        # Mock backends
        self.attn_backend = MockAttnBackend()
        self.ctx.attn_backend = self.attn_backend
        if config.model_config.is_moe:
            self.moe_backend = MockMoeBackend()
            self.ctx.moe_backend = self.moe_backend

        # Mock sampler
        self.sampler = MockSampler(config.model_config.vocab_size, config.mock_sample_mode)

        # dummy_req 用于 CUDA Graph padding（table_idx = max_running_req，即最后一行）
        self.dummy_req = Req(
            input_ids=[0], table_idx=config.max_running_req,
            cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,
        )
        self.dummy_req.device_len = 1
        self.dummy_req.max_device_len = 1
        # dummy page_table 行填充为 num_tokens（标记所有页已使用）
        self.page_table[self.dummy_req.table_idx] = [num_tokens] * self.max_seq_len

        # SimGraphRunner（接收 dummy_req）
        self.graph_runner = SimGraphRunner(config, config.model_config, self.dummy_req)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """模拟 forward_batch"""
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self._mock_model_forward(batch)

        for req in batch.reqs:
            if not isinstance(req, ChunkedReq):
                req.complete_one()  # ChunkedReq 不生成 token，跳过

        # logits 行数 = len(batch.reqs)（不含 padded_reqs），切片是防御性操作
        next_tokens = self.sampler.sample(logits[:batch.size], args)
        return ForwardOutput(
            next_tokens_gpu=next_tokens,
            next_tokens_cpu=list(next_tokens),
            copy_done_event=MockEvent(),
        )

    def _mock_model_forward(self, batch: Batch) -> List[List[float]]:
        """生成 mock logits"""
        import random
        return [[random.random() for _ in range(self.model_config.vocab_size)]
                for _ in range(batch.size)]

    def _calculate_num_pages(self, config: 'SimulatorConfig') -> int:
        """计算可用页数（见 3.3.9 内存预算仿真）"""
        if config.num_pages is not None:
            return config.num_pages
        # 自动计算：基于内存预算（与 §3.4.1 一致）
        num_pages, _, _ = calculate_memory_budget(
            config, config.model_config, config.total_gpu_memory
        )
        return num_pages

    def shutdown(self) -> None:
        pass
```

#### 配置类关系说明

模拟器使用统一的 `SimulatorConfig`，不需要继承层级：

```python
# SimulatorConfig 包含所有配置（4.2 节已定义）
# 不需要单独的 SimEngineConfig 或 SchedulerConfig
# MockEngine 和 SimScheduler 直接接收 SimulatorConfig
# SimulatorConfig 中：
#   - model_config, page_size, num_pages, max_seq_len, max_running_req 等 → Engine 使用
#   - max_extend_tokens, cache_type, enable_overlap 等 → Scheduler 使用
#   - enable_cuda_graph, cuda_graph_bs 等 → GraphRunner 使用
#   - offline_mode → SchedulerIOMixin 使用
#   - total_gpu_memory, memory_ratio, dtype_size → 内存预算使用
```

#### CacheManager.match_req 的 off-by-one 细节

```python
def match_req(self, req: 'PendingReq') -> MatchResult:
    """匹配前缀缓存"""
    input_len = req.input_len
    assert input_len > 0
    # 关键：排除最后一个 token！
    # 因为最后一个 token 可能还没有对应的 KV cache
    # 只匹配 [0, input_len-1) 的前缀
    return self.prefix_cache.match_prefix(req.input_ids[:input_len - 1])
```

#### SimulationMetrics 完整方法集

```python
class SimulationMetrics:
    # ... 数据字段见 4.5 节 ...

    def record_reply(self, reply: List['DetokenizeMsg'], tick: int) -> None:
        for msg in reply:
            self.total_tokens_generated += 1
            if msg.finished:
                self.completed_requests += 1

    def record_batch(self, batch: 'Batch', gpu_ticks: int) -> None:
        """记录 batch 统计信息。GPU busy ticks 由 record_tick 统一管理，
        此处不重复累加 gpu_busy_ticks，避免与 record_tick 双重计数。"""
        if batch.is_prefill:
            self.prefill_batches += 1
            n = self.prefill_batches
            old = self.avg_prefill_batch_size
            self.avg_prefill_batch_size = (old * (n - 1) + batch.size) / n
        else:
            self.decode_batches += 1
            n = self.decode_batches
            old = self.avg_decode_batch_size
            self.avg_decode_batch_size = (old * (n - 1) + batch.size) / n
        # gpu_ticks 参数保留用于调用方通过 record_tick(tick, gpu_busy=gpu_ticks) 记录
        # 此处不直接累加，防止与 record_tick 的 gpu_busy 参数重复计数

    def record_tick(self, tick: int, gpu_busy: int = 0) -> None:
        self.total_ticks = max(self.total_ticks, tick + 1)
        self.gpu_busy_ticks += gpu_busy
        self.gpu_idle_ticks = max(0, self.total_ticks - self.gpu_busy_ticks)
        self.gpu_utilization = self.gpu_busy_ticks / max(1, self.total_ticks)
```

### 9.12 边界情况处理说明

以下是实现时需注意的边界情况及其处理方式：

| 边界情况 | 处理方式 | 位置 |
|----------|----------|------|
| **空 batch 进入 forward** | `_schedule_next_batch` 返回 `None`，`_normal_tick`/`_overlap_tick` 跳过 forward | §9.4 |
| **页耗尽（eviction 后仍不足）** | `allocate_paged` 循环 break，未分配位置保持为 0；正常负载下 `PrefillAdder.available_size` 检查会拦截 | §9.11 CacheManager |
| **RadixCache 空树** | `root_node.ref_count=1` 永不驱逐；`_tree_walk` 返回 `(root, 0)` | §9.11 RadixPrefixCache |
| **ChunkedReq 写入位置 -1** | `_make_write_tuple` 生成 -1，`_forward` 中 `if p >= 0` 跳过 | §9.2, §9.11 |
| **Overlap 模式 last_data 残留** | 调用方需发送空 tick `run_tick([])` 刷新 | §9.4 `_overlap_tick` |
| **dummy_req 进入采样** | `pad_batch` 只影响 `padded_reqs`，不影响 `batch.reqs`，采样仅处理 `batch.reqs` | §9.11 SimGraphRunner |
| **所有 running_reqs 完成** | `decode_manager.schedule_next_batch` 返回 `None`，转 prefill 调度 | §9.11 DecodeManager |
| **全缓存命中（cached_len == input_len）** | `Req.__post_init__` 使用 `<=` 断言允许 `cached_len == device_len`；`chunk_size=0`，`complete_one` 仍推进 `device_len` 生成首 token | §2.2.2, §9.11 PrefillAdder |
| **NaiveCache 页回收** | `insert_prefix` 返回 `NaiveCacheHandle(0)`，确保 `finished=True` 时 `_free(page_indices[0:])` 回收全部页 | §9.3b |
| **内存预算为负（模型过大）** | `calculate_memory_budget` 使用 `max(0, ...)` 返回 0 页，`graph_buffer` 从 available 中扣除 | §3.3.9 |
| **token_pool extend 部分未初始化** | `try_add_one` 和 `_try_add_one_chunked` 均复制 extend tokens 到 `token_pool`，确保 `batch.input_ids` 正确 | §9.11 PrefillAdder |
| **并行 size 全为 1** | 所有并行模块 noop，`comm_ticks_total=0`，等价于纯单实例 | §10.11 |
| **EP>1 但非 MoE 模型** | `validate_parallel_config` 抛 `ValueError`，启动即失败 | §10.7.3 |
| **EP>tp_size** | `validate_parallel_config` 抛 `ValueError`（EP 是 TP rank 重编号，不能超过 tp_size） | §10.4.1, §10.7.3 |
| **CP>tp_size** | `validate_parallel_config` 抛 `ValueError`（CP 是 TP rank 重编号） | §10.8.1, §10.7.3 |
| **DP 副本页耗尽** | `allocate_pages` 返回 `False`，请求等待或被拒绝 | §10.3.2 |
| **DP Attention + 非 MLA 模型** | `enable_dp_attention=True` 但 `num_kv_heads>1` 时发出 warning（DP Attention 专为 MLA 设计） | §10.3.4 |
| **PP micro-batch size=0** | `_split_micro_batches` 跳过空 micro-batch，不产生通信成本 | §10.5.2 |
| **PP 中间 stage 采样** | `is_pp_last=False` 时返回 `is_intermediate=True`，不调用 sampler | §10.5.3 |
| **TP KV heads 不整除** | `div_even(allow_replicate=True)` 复制余数，发出 warning | §10.7.3 |
| **CUDA Graph + 并行** | CUDA Graph replay 路径跳过 TP/PP 通信仿真（graph 内已固化） | §10.2.4, §10.5.3 |
| **EPLB 负载均衡触发** | `enable_eplb=True` 且负载方差 > 均值 10% 时，`step()` 返回重平衡成本 | §10.4.4 |
| **ZMQ 广播成本** | TP>1 时 primary rank 的 ZMQ 广播成本计入 `zmq_broadcast_ticks` | §10.6.1 |

---

## 10. 并行仿真扩展（TP/DP/EP/PP 完整规格）

本章描述 TP/DP/EP/PP 四种并行策略在仿真器中的完整规格。前序章节已定义基础设施（§4.2 `SimulatorConfig`/`ParallelTopology`、§3.4.4 `SimCommGroup`、§3.4.5b `SimMoeBackend`），本章定义**仿真流程集成**——即各并行策略如何与单实例 `SimScheduler`/`MockEngine` 的 forward 路径交互、通信成本注入点、多 rank 协调逻辑、以及指标收集。

### 10.1 设计原则与架构映射

| 原则 | 说明 |
|------|------|
| **单实例调度器不变** | `SimScheduler` 仍为单实例，所有并行逻辑在 `MockEngine.forward_batch` 和 `SimModelForward` 中注入 |
| **通信成本仿真** | 不实际传输数据，只通过 `SimCommGroup` 计算通信 ticks 并累加到 GPU busy ticks |
| **内存分割** | TP/EP/PP 各自影响显存预算：TP 分割权重+KV、EP 分割专家、PP 分割层 |
| **多 rank 逻辑折叠** | 仿真器不模拟多个进程，而是将多 rank 行为折叠为单进程内的通信成本 + 状态记录 |
| **零运行时依赖** | 不依赖 NCCL/MPI，通信成本纯计算 |
| **退化为单实例** | `tp_size=dp_size=ep_size=pp_size=1` 时，所有并行模块 noop，与 §3 核心仿真完全一致 |

**与 SGLang 真实架构对应**：

```
SGLang 真实多 rank                    仿真器单进程
┌─────────────────────────┐          ┌────────────────────────────┐
│ Scheduler rank 0         │          │ SimScheduler (单实例)        │
│  ├─ TP group [0..tp-1]   │  ──→     │  ├─ SimCommGroup("tp")     │
│  ├─ DP group [0..dp-1]   │  ──→     │  ├─ DPLoadBalancer           │
│  ├─ EP group [0..ep-1]   │  ──→     │  ├─ SimMoeBackend           │
│  └─ PP stage [0..pp-1]   │  ──→     │  └─ PPPipelineSimulator     │
└─────────────────────────┘          └────────────────────────────┘
```

### 10.2 TP 张量并行仿真

#### 10.2.1 仿真目标

从 `MockTPGroup`（noop）升级为真实通信成本仿真，覆盖：
1. **权重分割**：`num_attention_heads` / `tp_size`、`intermediate_size` / `tp_size`、KV heads `div_even(num_kv_heads, tp_size)`
2. **all-reduce 注入**：每个 TP-grouped 算子后注入 all-reduce 成本
3. **内存预算修正**：权重内存 ÷ `tp_size`，KV cache 头数 ÷ `tp_size`

#### 10.2.2 TP 通信组与内存修正

```python
class TPSimulator:
    """TP 张量并行仿真器，封装通信组与内存修正逻辑。
    由 MockEngine 持有，在 forward 路径中调用。
    tp_size=1 时所有方法 noop（成本为 0）。
    """
    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig'):
        self.config = config
        self.model_config = model_config
        self.tp_size = config.tp_size
        self.comm_group = SimCommGroup(
            group_size=config.tp_size,
            config=config,
            group_type="tp",
        )
        # 内存修正：本地（单 rank）可见的参数量
        if self.tp_size > 1:
            self.local_num_heads = self._div_heads(model_config.num_attention_heads)
            self.local_num_kv_heads = self._div_heads(model_config.num_kv_heads)
            self.local_intermediate = model_config.intermediate_size // self.tp_size
        else:
            self.local_num_heads = model_config.num_attention_heads
            self.local_num_kv_heads = model_config.num_kv_heads
            self.local_intermediate = model_config.intermediate_size

    def _div_heads(self, total_heads: int) -> int:
        """TP 下 attention heads 分割，使用 div_even 保证均衡"""
        result = div_even(total_heads, self.tp_size, allow_replicate=False)
        return result[0]  # rank 0 视角

    def all_reduce_after_attn(self, batch_size: int) -> int:
        """attention 后 all-reduce。数据量 = batch × hidden × dtype。"""
        if self.tp_size <= 1:
            return 0
        data_bytes = batch_size * self.model_config.hidden_size * self.config.dtype_size
        return self.comm_group.all_reduce(data_bytes)

    def all_reduce_after_mlp(self, batch_size: int) -> int:
        """MLP 后 all-reduce。数据量 = batch × hidden × dtype。"""
        if self.tp_size <= 1:
            return 0
        data_bytes = batch_size * self.model_config.hidden_size * self.config.dtype_size
        return self.comm_group.all_reduce(data_bytes)

    def total_comm_ticks(self) -> int:
        return sum(self.comm_group.comm_ticks_log)
```

#### 10.2.3 TP 内存预算修正

`calculate_memory_budget`（§3.3.9）需在 TP>1 时修正权重内存：

```python
def calculate_memory_budget_with_tp(
    config: 'SimulatorConfig',
    model_config: 'ModelConfig',
    total_gpu_memory: int,
) -> Tuple[int, int, int]:
    """TP-aware 内存预算。
    权重内存 ÷ tp_size（每个 rank 只需持有 1/tp 的权重）。
    KV cache 头数 ÷ tp_size（每个 rank 只需持有 1/tp 的 KV heads）。
    """
    tp = config.tp_size
    # 权重内存（修正：每个 TP rank 只需 1/tp 的权重）
    weight_bytes = _estimate_weight_bytes(model_config) // tp
    # 非权重视为常量
    non_weight_bytes = _estimate_non_weight_bytes(model_config)
    # 可用于 KV cache 的内存
    available = int(total_gpu_memory * config.memory_ratio)
    kv_budget = max(0, available - weight_bytes - non_weight_bytes)
    # 每 token 每 layer 的 KV bytes（修正：头数 ÷ tp）
    local_kv_heads = div_even(model_config.num_kv_heads, tp, allow_replicate=False)[0]
    bytes_per_token_per_layer = (
        2 * local_kv_heads * model_config.head_dim * config.dtype_size
    )
    bytes_per_token = bytes_per_token_per_layer * model_config.num_layers
    num_tokens = kv_budget // max(1, bytes_per_token)
    num_pages = num_tokens // config.page_size
    graph_buffer = _calc_graph_buffer(config)
    num_pages = max(0, num_pages - graph_buffer)
    return num_pages, num_tokens, kv_budget
```

#### 10.2.4 TP 仿真流程（forward 路径集成）

```python
class MockEngine:
    # ... 已有初始化 ...
    def __init__(self, config, model_config):
        # ... 原有初始化 ...
        self.tp_sim = TPSimulator(config, model_config)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """TP 通信成本注入到 forward 路径"""
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
                tp_comm = self.tp_sim.total_comm_ticks()  # graph 内已记录
            else:
                logits, tp_comm = self._mock_model_forward_with_tp(batch)

        # 通信成本累加到 GPU busy ticks（通过 metrics）
        if tp_comm > 0:
            self.metrics.comm_ticks_total += tp_comm
            self.metrics.tp_comm_ticks += tp_comm
        # ... 采样逻辑不变 ...

    def _mock_model_forward_with_tp(self, batch: Batch) -> Tuple[List[List[float]], int]:
        """逐层 forward，每层注入 TP 通信成本"""
        total_comm = 0
        for _ in range(self.model_config.num_layers):
            total_comm += self.tp_sim.all_reduce_after_attn(batch.size)
            if self.model_config.is_moe and self.moe_backend is not None:
                hs, moe_comm = self.moe_backend.forward(None, batch.size)
                total_comm += moe_comm
            else:
                total_comm += self.tp_sim.all_reduce_after_mlp(batch.size)
        logits = self._mock_model_forward(batch)
        return logits, total_comm
```

#### 10.2.5 TP 指标

`SimulationMetrics` 新增字段：

```python
class SimulationMetrics:
    # ... 已有字段 ...
    tp_comm_ticks: int = 0           # TP all-reduce 总成本
    tp_all_reduce_count: int = 0      # all-reduce 调用次数
```

### 10.3 DP 数据并行仿真

#### 10.3.1 两种 DP 模式

SGLang 有**两种完全不同的 DP 模式**，仿真器必须分别覆盖：

| 模式 | 触发条件 | 核心机制 | 通信 |
|------|----------|----------|------|
| **标准 DP** (`DataParallelController`) | `dp_size > 1` | 请求分发到各 DP 副本，各自独立调度 | 无 NCCL（各副本独立） |
| **DP Attention** (`enable_dp_attention=True`) | MLA 模型 + `dp_size > 1` | Attention 层 DP（每 rank 独立 KV），MLP/MoE 层大 TP 组 | all-gather/slice 每 layer |

**关键区别**（参见 Awesome-ML-SYS-Tutorial `dp-attention/readme.md`）：
- 标准 DP：各副本完全独立，通过 `DataParallelController` 用 ZMQ 分发请求
- DP Attention：同一进程组内，Attention 层用 `ReplicatedLinear`（每 DP rank 持有全量 attention 权重），MLP 层用 `all_gather → forward → slice` 模式

#### 10.3.2 标准 DP — 请求分发器

```python
class DPRankState:
    """单个 DP 副本的状态"""
    def __init__(self, rank: int, num_pages: int, page_size: int):
        self.rank = rank
        self.num_pages = num_pages
        self.page_size = page_size
        self.used_pages = 0
        self.running_reqs: List[int] = []
        self.queue_len: int = 0  # waiting_queue 长度（shortest_queue 策略用）

    @property
    def load(self) -> int:
        """当前负载 = 已用页数 + 队列长度"""
        return self.used_pages + self.queue_len


class DataParallelController:
    """标准 DP 请求分发器（对应 SGLang DataParallelController）。
    通过 ZMQ 与各 DP 副本的 scheduler 通信。
    策略：round_robin / shortest_queue（对应 SGLang 源码）。
    dp_size=1 时所有请求分配到 rank 0（等价于无 DP）。
    """
    def __init__(self, config: 'SimulatorConfig', total_num_pages: int):
        self.config = config
        self.dp_size = config.dp_size
        self.strategy = config.dp_load_balance_strategy  # "round_robin" | "shortest_queue"
        self.round_robin_idx = 0
        pages_per_rank = div_even(total_num_pages, self.dp_size, allow_replicate=False)
        self.ranks: List[DPRankState] = [
            DPRankState(i, pages_per_rank[i], config.page_size)
            for i in range(self.dp_size)
        ]

    def assign_request(self, req: 'Req') -> int:
        """为新请求分配 DP 副本，返回 rank 索引"""
        if self.dp_size <= 1:
            return 0
        if self.strategy == "round_robin":
            rank = self.round_robin_idx % self.dp_size
            self.round_robin_idx += 1
        elif self.strategy == "shortest_queue":
            # 选择队列最短的副本（对应 SGLang shortest_queue 策略）
            rank = min(range(self.dp_size), key=lambda i: self.ranks[i].load)
        else:
            raise ValueError(f"Unknown DP strategy: {self.strategy}")
        self.ranks[rank].running_reqs.append(req.uid)
        req.dp_rank = rank
        return rank

    def release_request(self, req: 'Req') -> None:
        """请求完成时从 DP 副本移除"""
        if self.dp_size <= 1:
            return
        rank = getattr(req, 'dp_rank', 0)
        if rank < len(self.ranks):
            self.ranks[rank].running_reqs = [
                uid for uid in self.ranks[rank].running_reqs if uid != req.uid
            ]

    def allocate_pages(self, dp_rank: int, num_pages: int) -> bool:
        rank_state = self.ranks[dp_rank]
        if rank_state.used_pages + num_pages > rank_state.num_pages:
            return False
        rank_state.used_pages += num_pages
        return True

    def free_pages(self, dp_rank: int, num_pages: int) -> None:
        rank_state = self.ranks[dp_rank]
        rank_state.used_pages = max(0, rank_state.used_pages - num_pages)
```

#### 10.3.3 标准 DP 与调度器集成

```python
class SimScheduler:
    def __init__(self, config, model_config):
        # ... 原有初始化 ...
        self.dp_controller = DataParallelController(config, self.num_pages)

    def add_request(self, req: Req) -> None:
        """添加请求时分配 DP 副本"""
        dp_rank = self.dp_controller.assign_request(req)
        req.dp_rank = dp_rank
        self.waiting_queue.append(req)

    def _free_request_resources(self, req: Req) -> None:
        """释放请求资源时同时释放 DP 副本资源"""
        self.dp_controller.release_request(req)
        self.cache_manager.free_req(req)
```

#### 10.3.4 DP Attention 仿真（MLA 模型专用）

当 `enable_dp_attention=True` 时，仿真 MLA 模型的 DP Attention 机制。

**核心机制**（参见 `dp-attention/readme.md`）：
1. Attention 层：每个 DP rank 独立处理自己的请求子集，KV cache 不复制（各 rank 独立）
2. MLP/MoE 层：所有 DP rank 的 hidden_states **all-gather** 成一个完整 batch → forward → **slice** 回各 rank
3. 非 MLP 层的 TP size = 1（每个 DP rank 内），MLP 层的 TP size = dp_size（大 TP 组）

```python
class DPAttentionSimulator:
    """DP Attention 仿真器（MLA 模型专用）。
    对应 SGLang 的 enable_dp_attention 机制。

    仿真要点：
    1. Attention 层不通信（每 rank 独立 KV cache）
    2. MLP/MoE 层前后插入 all-gather / slice 通信成本
    3. global_num_tokens = sum(各 DP rank 的 batch_size)
    4. 数据流：local_hidden → all_gather → global_hidden → MLP → slice → local_hidden
    """
    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig'):
        self.config = config
        self.model_config = model_config
        self.dp_size = config.dp_size if config.enable_dp_attention else 1
        # all-gather 通信组（group_type="dp_attn"）
        self.comm_group = SimCommGroup(
            group_size=self.dp_size,
            config=config,
            group_type="dp_attn",
        ) if self.dp_size > 1 else None
        self.total_comm_ticks = 0

    def simulate_mlp_forward(self, local_batch_sizes: List[int]) -> int:
        """仿真 MLP 层的 all-gather → forward → slice 通信成本。
        local_batch_sizes: 各 DP rank 的本地 batch_size 列表。
        返回通信成本 ticks。
        """
        if self.dp_size <= 1 or self.comm_group is None:
            return 0
        # all-gather: 各 rank 的 hidden_states 汇聚
        # 数据量 = sum(local_batch_sizes) × hidden_size × dtype_size
        total_tokens = sum(local_batch_sizes)
        gather_bytes = total_tokens * self.model_config.hidden_size * self.config.dtype_size
        # all-gather 成本（用 _compute_cost 中的 dp_attn 参数）
        gather_cost = self.comm_group._compute_cost(gather_bytes)
        # slice 不产生通信成本（本地切片）
        self.total_comm_ticks += gather_cost
        return gather_cost


# SimCommGroup 扩展：支持 dp_attn group_type
# 在 §3.4.4 SimCommGroup._compute_cost 中追加：
#   elif self.group_type == "dp_attn":
#       latency = self.config.all_reduce_latency_ticks
#       per_byte = self.config.dp_attention_all_gather_cost_per_byte_ticks
```

#### 10.3.5 DP 内存预算修正

```python
def calculate_memory_budget_with_dp(
    config: 'SimulatorConfig', model_config: 'ModelConfig', total_gpu_memory: int
) -> Tuple[int, int, int]:
    """DP-aware 内存预算。
    标准 DP：权重在 DP rank 间复制（不分割），KV cache 按 dp_size 分区。
    DP Attention：attention 权重复制（每 DP rank 全量），MLP 权重 ÷ tp_size（大 TP 组）。
    """
    dp = config.dp_size
    if config.enable_dp_attention:
        # DP Attention: attention 权重复制，MLP 权重 ÷ tp_size
        attn_weight = _estimate_attn_weight_bytes(model_config)
        mlp_weight = _estimate_mlp_weight_bytes(model_config) // config.tp_size
        weight_bytes = attn_weight + mlp_weight
    else:
        # 标准 DP: 全量权重复制
        weight_bytes = _estimate_weight_bytes(model_config)
    non_weight_bytes = _estimate_non_weight_bytes(model_config)
    available = int(total_gpu_memory * config.memory_ratio)
    kv_budget = max(0, available - weight_bytes - non_weight_bytes)
    kv_budget_per_rank = kv_budget // dp
    # 标准 DP: KV heads 不除（每 rank 独立完整 KV cache）
    # DP Attention: KV heads 不除（MLA 本身 num_kv_heads=1，不需要再分）
    bytes_per_token_per_layer = (
        2 * model_config.num_kv_heads * model_config.head_dim * config.dtype_size
    )
    bytes_per_token = bytes_per_token_per_layer * model_config.num_layers
    num_tokens = kv_budget_per_rank // max(1, bytes_per_token)
    num_pages = num_tokens // config.page_size
    graph_buffer = _calc_graph_buffer(config)
    num_pages = max(0, num_pages - graph_buffer)
    return num_pages, num_tokens, kv_budget_per_rank
```

#### 10.3.6 DP 指标

```python
class ParallelMetrics:
    # ... 其他字段 ...
    # 标准 DP
    dp_rank_load: List[int] = field(default_factory=list)  # 各 DP 副本最终负载
    # DP Attention
    dp_attn_comm_ticks: int = 0          # DP Attention all-gather 总成本
    dp_attn_all_gather_count: int = 0     # all-gather 调用次数
```

### 10.4 EP 专家并行仿真

#### 10.4.1 EP rank 结构（TP rank 重编号）

**关键概念**（参见 sglang-note《scheduler-rank-and-process-topology》§2.5）：
EP **不增加进程数**，而是在 MoE 层将 TP rank 重新划分为三层：
```
层级：Global(TP) → MOE_DP → EP → MOE_TP
```
即：`tp_rank` → `(moe_dp_rank, moe_ep_rank, moe_tp_rank)`

各层含义：
- `moe_dp_size`：MoE 数据并行副本数（复用 DP 副本）
- `ep_size`：专家并行数（每个 EP rank 持有 `num_experts/ep_size` 个专家）
- `moe_tp_size`：MoE 内部张量并行数（`tp_size / moe_dp_size / ep_size`）

`ParallelTopology.compute_moe_ranks(tp_rank)` 封装了此映射。

#### 10.4.2 仿真目标

`SimMoeBackend`（§3.4.5b）已实现专家分片与 all-to-all 成本计算。本节补充**路由仿真**、**EPLB 动态负载均衡**和**与 forward 路径的集成**。

#### 10.4.3 MoE 路由仿真模式

`config.moe_routing_mode` 决定路由行为：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `"mock"` | 不路由，仅记录 token 数和通信成本 | 基准仿真（默认） |
| `"hash"` | `expert_idx = hash(token_id) % num_experts`，统计跨 rank 比例 | 负载分布仿真 |
| `"simulated"` | 随机选择 `moe_top_k` 个专家，计算跨 EP rank 的 token 比例 | 真实路由仿真 |

```python
class SimMoeBackend(BaseMoeBackend):
    # ... §3.4.5b 已有的 __init__ 和 forward ...

    def _route_tokens(self, batch_size: int) -> Dict[int, int]:
        """路由仿真：返回 {target_ep_rank: token_count} 映射。
        只有 moe_routing_mode != "mock" 时执行真实路由。
        """
        if self.model_config is None:
            return {0: batch_size}
        routing_mode = getattr(self.config, 'moe_routing_mode', 'mock') \
            if self.config else 'mock'
        if routing_mode == "mock":
            # mock 模式：所有 token "留在本地"，但仍计算 all-to-all 成本
            return {0: batch_size}
        top_k = self.model_config.moe_top_k
        rank_distribution: Dict[int, int] = {}
        for token_idx in range(batch_size):
            if routing_mode == "hash":
                expert = hash(token_idx) % self.model_config.num_experts
            else:  # "simulated"
                import random
                expert = random.randint(0, self.model_config.num_experts - 1)
            # 找到该专家所在的 EP rank
            ep_rank = self._expert_to_rank(expert)
            rank_distribution[ep_rank] = rank_distribution.get(ep_rank, 0) + top_k
        return rank_distribution

    def _expert_to_rank(self, expert_idx: int) -> int:
        """将专家索引映射到 EP rank"""
        cumulative = 0
        for rank, count in enumerate(self.experts_per_rank):
            cumulative += count
            if expert_idx < cumulative:
                return rank
        return 0

    def forward(self, hidden_states, batch_size: int) -> Tuple[Any, int]:
        """完整 EP forward：路由 + all-to-all 成本"""
        self.call_count += 1
        self.total_tokens += batch_size
        comm_cost = 0
        if self.ep_size > 1 and self.comm_group is not None:
            # 路由分布
            distribution = self._route_tokens(batch_size)
            # 计算需要发送到其他 rank 的 token 数据量
            remote_tokens = sum(c for r, c in distribution.items() if r != 0)
            # all-to-all：发送 remote_tokens 的 hidden
            data_bytes = remote_tokens * self.model_config.hidden_size * self.dtype_size
            comm_cost += self.comm_group.all_to_all(data_bytes)
            # 反向 all-to-all
            comm_cost += self.comm_group.all_to_all(data_bytes)
            self.comm_ticks_total += comm_cost
        return hidden_states, comm_cost
```

#### 10.4.4 EPLB 动态专家负载均衡

当 `enable_eplb=True` 时，仿真 SGLang 的 `eplb` 模块——按运行时统计动态调整专家分布：

```python
class EPLBSimulator:
    """动态专家负载均衡仿真器。
    对应 SGLang 的 eplb 模块。

    机制：
    1. 收集每个 EP rank 的 token 负载统计
    2. 按 step 间隔重新分配专家到各 EP rank
    3. 仿真中不实际迁移专家权重，只记录重平衡成本
    """
    def __init__(self, config: 'SimulatorConfig', num_experts: int, ep_size: int):
        self.enabled = config.enable_eplb
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.rebalance_interval = 100  # 每 100 步重平衡一次
        self.step_count = 0
        self.rebalance_count = 0
        self.rebalance_cost_ticks = 10  # 每次重平衡的固定成本

    def step(self, rank_loads: List[int]) -> int:
        """每步调用，返回重平衡成本（0 表示不重平衡）"""
        if not self.enabled or self.ep_size <= 1:
            return 0
        self.step_count += 1
        if self.step_count % self.rebalance_interval != 0:
            return 0
        # 检查是否需要重平衡（负载方差超过阈值）
        avg = sum(rank_loads) / len(rank_loads)
        variance = sum((l - avg) ** 2 for l in rank_loads) / len(rank_loads)
        if variance < avg * 0.1:  # 方差小于均值的 10%，不重平衡
            return 0
        self.rebalance_count += 1
        return self.rebalance_cost_ticks
```

#### 10.4.5 EP 指标

```python
class ParallelMetrics:
    # ... 其他字段 ...
    ep_comm_ticks: int = 0           # EP all-to-all 总成本
    ep_all_to_all_count: int = 0    # all-to-all 调用次数
    ep_cross_rank_tokens: int = 0   # 跨 rank 路由的 token 总数
    eplb_rebalance_count: int = 0   # EPLB 重平衡次数
    eplb_rebalance_ticks: int = 0   # EPLB 重平衡总成本
```

### 10.5 PP 流水并行仿真

#### 10.5.1 仿真目标与关键约束

PP 将模型层按 `pp_size` 分割到多个 stage，采用 micro-batch 流水线调度。仿真目标：
1. **层分割**：`num_layers` 按 `pp_size` 分配到各 stage（`ParallelTopology.pp_stage_layers`）
2. **micro-batch 调度**：支持 `1f1b` / `gpipe` / `interleaved` 三种流水线调度
3. **stage 间 send/recv**：每个 micro-batch 在 stage 间传递 hidden_states，产生通信成本
4. **pipeline bubble**：仿真流水线气泡对吞吐的影响

**关键约束**（参见 sglang-note《scheduler-rank-and-process-topology》§5.2-5.3）：
- **只有 PP=0, TP=0 rank 从 ZMQ 拉取请求**，通过 `broadcast_pyobj` 广播到同 PP 组的其他 rank，再通过 `point_to_point_pyobj` 传给下一 PP stage
- **只有最后一个 PP stage 做采样**（`is_pp_last = (pp_rank == pp_size - 1)`），中间 stage 只返回 hidden_states
- **PP 通信组是跨 stage 的**：同一 DP rank、同一 TP rank、不同 PP rank 的进程属于同一个 PP group

#### 10.5.2 PP 流水线仿真器

```python
class PPPipelineSimulator:
    """PP 流水并行仿真器。
    pp_size=1 时所有方法 noop（退化为单 stage）。
    仿真中不实际分割模型，只仿真流水线调度时序和通信成本。
    """
    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig'):
        self.config = config
        self.model_config = model_config
        self.pp_size = config.pp_size
        self.schedule = config.pp_pipeline_schedule
        self.num_micro_batches = config.pp_num_micro_batches
        # 通信组
        self.comm_group = SimCommGroup(
            group_size=config.pp_size,
            config=config,
            group_type="pp",
        ) if self.pp_size > 1 else None
        # 每个 stage 的层范围
        self.stage_layers = ParallelTopology(
            pp_size=self.pp_size
        ).pp_stage_layers(model_config.num_layers)
        # 流水线状态
        self.stage_buffers: List[List] = [[] for _ in range(self.pp_size)]
        # 仿真 tick 记录
        self.bubble_ticks = 0
        self.comm_ticks_total = 0

    def simulate_pipeline_forward(self, batch: 'Batch') -> int:
        """仿真整个 pipeline forward，返回总通信成本 ticks。
        不实际执行 forward（MockEngine 已有 mock forward），
        只仿真 stage 间通信和流水线气泡。
        """
        if self.pp_size <= 1:
            return 0
        total_comm = 0
        micro_batches = self._split_micro_batches(batch)
        if self.schedule == "gpipe":
            total_comm = self._simulate_gpipe(micro_batches)
        elif self.schedule == "1f1b":
            total_comm = self._simulate_1f1b(micro_batches)
        elif self.schedule == "interleaved":
            total_comm = self._simulate_interleaved(micro_batches)
        else:
            raise ValueError(f"Unknown PP schedule: {self.schedule}")
        self.comm_ticks_total += total_comm
        return total_comm

    def _split_micro_batches(self, batch: 'Batch') -> List['Batch']:
        """将 batch 按 num_micro_batches 分割为 micro-batch 列表。
        仿真中不实际分割数据，只记录 micro-batch 的 size。
        """
        if self.num_micro_batches <= 1:
            return [batch]
        micro_size = batch.size // self.num_micro_batches
        remainder = batch.size % self.num_micro_batches
        micro_batches = []
        for i in range(self.num_micro_batches):
            size = micro_size + (1 if i < remainder else 0)
            # 仿真 micro-batch：只记录 size，不复制数据
            micro_batch = type(batch)(size=size)  # type: ignore
            micro_batches.append(micro_batch)
        return micro_batches

    def _stage_send_recv_cost(self, micro_batch_size: int) -> int:
        """计算单个 stage 间 send/recv 的通信成本。
        数据量 = micro_batch_size × hidden_size × dtype_size。
        """
        if self.comm_group is None:
            return 0
        data_bytes = micro_batch_size * self.model_config.hidden_size * self.config.dtype_size
        return self.comm_group.send_recv(data_bytes)

    def _simulate_gpipe(self, micro_batches: List) -> int:
        """GPipe 调度：所有 micro-batch 依次走完所有 stage。
        气泡 = (pp_size - 1) × 单 stage 计算时间。
        """
        total_comm = 0
        for mb in micro_batches:
            for stage in range(self.pp_size - 1):
                total_comm += self._stage_send_recv_cost(mb.size)
        # GPipe 气泡
        self.bubble_ticks += (self.pp_size - 1) * self.config.eager_forward_cost_ticks
        return total_comm

    def _simulate_1f1b(self, micro_batches: List) -> int:
        """1F1B 调度：每个 stage 交替执行 forward 和 backward。
        气泡 = (pp_size - 1) × 单 stage 计算时间（比 GPipe 少）。
        仿真中只仿真 forward 通信（不仿真 backward）。
        """
        total_comm = 0
        n = len(micro_batches)
        for stage in range(self.pp_size - 1):
            for mb_idx in range(n):
                total_comm += self._stage_send_recv_cost(micro_batches[mb_idx].size)
        # 1F1B 气泡（比 GPipe 略少）
        self.bubble_ticks += (self.pp_size - 1) * self.config.eager_forward_cost_ticks
        return total_comm

    def _simulate_interleaved(self, micro_batches: List) -> int:
        """Interleaved 调度：将 micro-batch 分成多组，在 stage 间交错执行。
        气泡最少，但通信次数更多。
        """
        total_comm = 0
        n = len(micro_batches)
        # interleaved 的通信次数 ≈ GPipe × interleaved_factor
        for stage in range(self.pp_size - 1):
            for mb_idx in range(n):
                total_comm += self._stage_send_recv_cost(micro_batches[mb_idx].size)
        # interleaved 气泡（比 1F1B 更少）
        self.bubble_ticks += max(0, (self.pp_size - 1) // 2) * self.config.eager_forward_cost_ticks
        return total_comm
```

#### 10.5.3 PP 与 forward 路径集成

```python
class MockEngine:
    def __init__(self, config, model_config, pp_rank: int = 0):
        # ... 原有初始化 ...
        self.pp_rank = pp_rank
        self.is_pp_last = (pp_rank == config.pp_size - 1)
        self.pp_sim = PPPipelineSimulator(config, model_config)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """PP 通信成本注入 + 最后 stage 采样"""
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
                pp_comm = 0  # CUDA Graph 内不仿真 PP
            else:
                logits = self._mock_model_forward(batch)
                pp_comm = self.pp_sim.simulate_pipeline_forward(batch)

        if pp_comm > 0:
            self.metrics.parallel.pp_comm_ticks += pp_comm
            self.metrics.parallel.pp_bubble_ticks += self.pp_sim.bubble_ticks
            self.metrics.parallel.comm_ticks_total += pp_comm

        # 关键：只有最后一个 PP stage 做采样
        if not self.is_pp_last:
            # 中间 stage 返回 hidden_states（不采样）
            return ForwardOutput(logits=logits, sampled_ids=None, is_intermediate=True)
        # 最后 stage 正常采样
        next_token_ids = self.sampler.sample(logits, args)
        # ... 返回最终输出 ...
```

#### 10.5.4 PP 指标

```python
class SimulationMetrics:
    # ... 已有字段 ...
    pp_comm_ticks: int = 0        # PP send/recv 总成本
    pp_bubble_ticks: int = 0      # 流水线气泡 ticks
    pp_micro_batches: int = 0     # 执行的 micro-batch 总数
```

### 10.6 TP 通信基础设施仿真

#### 10.6.1 SGLang 实际通信架构

SGLang 的 TP 通信分为三层（参见 mini-sglang `scheduler/io.py` 和 `engine/engine.py`）：

| 层 | 实现 | 用途 |
|----|------|------|
| **ZMQ 多 rank** | `ZmqSubQueue` / `ZmqPubQueue` | primary rank 从 tokenizer 接收消息后广播给其他 TP rank |
| **CPU 侧 group** | `gloo` backend (`tp_cpu_group`) | barrier、broadcast（元数据同步） |
| **GPU 侧 group** | `nccl` 或 `pynccl` (`tp_device_group`) | all-reduce（梯度/激活同步） |

```python
class TPCommInfraSimulator:
    """TP 通信基础设施仿真器。
    仿真 ZMQ 广播 + gloo barrier + nccl all-reduce 的三层成本。
    tp_size=1 时所有方法 noop。
    """
    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig'):
        self.config = config
        self.model_config = model_config
        self.tp_size = config.tp_size
        # CPU 侧 gloo group 成本
        self.cpu_group_type = config.tp_cpu_group_type  # "gloo"
        # GPU 侧 nccl group 成本
        self.gpu_group_type = config.tp_gpu_group_type  # "nccl" or "pynccl"
        # SimCommGroup 用于 all-reduce 成本计算
        self.comm_group = SimCommGroup(
            group_size=self.tp_size,
            config=config,
            group_type="tp",
        )
        self.zmq_broadcast_ticks = 0
        self.barrier_ticks = 0

    def zmq_broadcast(self, msg_size: int) -> int:
        """仿真 ZMQ 广播成本（primary rank → 其他 TP rank）。
        成本 = msg_size / bandwidth（无 NCCL 延迟，纯 TCP 传输）。
        """
        if self.tp_size <= 1:
            return 0
        # ZMQ 传输成本：简化为 msg_size / bandwidth
        cost = msg_size // max(1, self.config.comm_bandwidth_bytes_per_tick)
        self.zmq_broadcast_ticks += cost
        return cost

    def cpu_barrier(self) -> int:
        """仿真 gloo barrier 成本（固定延迟）"""
        if self.tp_size <= 1:
            return 0
        cost = 1  # barrier 固定 1 tick
        self.barrier_ticks += cost
        return cost

    def gpu_all_reduce(self, data_bytes: int) -> int:
        """仿真 nccl all-reduce 成本（调用 SimCommGroup）"""
        return self.comm_group.all_reduce(data_bytes)
```

### 10.7 并行组合与初始化

#### 10.7.1 通信组初始化

```python
def init_parallel_groups(config: 'SimulatorConfig',
                          model_config: 'ModelConfig'
                          ) -> Dict[str, Any]:
    """根据 SimulatorConfig 初始化所有并行仿真组件。
    返回组件字典，由 MockEngine 和 SimScheduler 使用。
    """
    topology = ParallelTopology(
        tp_size=config.tp_size,
        dp_size=config.dp_size,
        ep_size=config.ep_size,
        pp_size=config.pp_size,
        cp_size=config.cp_size,
        enable_dp_attention=config.enable_dp_attention,
    )
    return {
        "topology": topology,
        "tp_comm": TPCommInfraSimulator(config, model_config),
        "tp_sim": TPSimulator(config, model_config),
        "dp_controller": DataParallelController(config, 0),  # num_pages 由 scheduler 填入
        "dp_attn_sim": DPAttentionSimulator(config, model_config)
            if config.enable_dp_attention else None,
        "pp_sim": PPPipelineSimulator(config, model_config),
        "cp_sim": CPSimulator(config, model_config)
            if config.cp_size > 1 else None,
        "eplb_sim": EPLBSimulator(config, model_config.num_experts, config.ep_size)
            if config.enable_eplb else None,
        "moe_backend": (
            SimMoeBackend(
                model_config, config.ep_size,
                SimCommGroup(config.ep_size, config, "ep"),
            )
            if model_config.is_moe else None
        ),
    }
```

#### 10.7.2 内存预算组合修正

当多种并行同时启用时，内存预算需组合修正：

```python
def calculate_memory_budget_parallel(
    config: 'SimulatorConfig',
    model_config: 'ModelConfig',
    total_gpu_memory: int,
) -> Tuple[int, int, int]:
    """组合并行内存预算。
    修正规则：
    - 标准 TP：权重 ÷ tp_size，KV heads ÷ tp_size
    - DP Attention：attention 权重复制，MLP 权重 ÷ tp_size
    - 标准 DP：KV cache 总量 ÷ dp_size（各副本独立）
    - EP：不影响 KV cache（MoE 权重在权重预算中）
    - CP：不影响 KV cache（序列切分不改变每层 KV 大小）
    - PP：不影响 KV cache（层分割不影响每层 KV）
    """
    tp = config.tp_size
    dp = config.dp_size
    if config.enable_dp_attention:
        # DP Attention: attention 权重复制，MLP 权重 ÷ tp_size
        attn_weight = _estimate_attn_weight_bytes(model_config)
        mlp_weight = _estimate_mlp_weight_bytes(model_config) // tp
        weight_bytes = attn_weight + mlp_weight
    else:
        # 标准 TP: 全量权重 ÷ tp_size
        weight_bytes = _estimate_weight_bytes(model_config) // tp
    non_weight_bytes = _estimate_non_weight_bytes(model_config)
    available = int(total_gpu_memory * config.memory_ratio)
    kv_budget = max(0, available - weight_bytes - non_weight_bytes)
    # DP 分区
    kv_budget_per_rank = kv_budget // dp
    # TP 分割 KV heads
    local_kv_heads = div_even(model_config.num_kv_heads, tp, allow_replicate=True)[0]
    bytes_per_token_per_layer = (
        2 * local_kv_heads * model_config.head_dim * config.dtype_size
    )
    bytes_per_token = bytes_per_token_per_layer * model_config.num_layers
    num_tokens = kv_budget_per_rank // max(1, bytes_per_token)
    num_pages = num_tokens // config.page_size
    graph_buffer = _calc_graph_buffer(config)
    num_pages = max(0, num_pages - graph_buffer)
    return num_pages, num_tokens, kv_budget_per_rank
```

#### 10.7.3 并行约束验证

```python
def validate_parallel_config(config: 'SimulatorConfig',
                              model_config: 'ModelConfig') -> None:
    """验证并行配置合法性，启动时调用。"""
    assert config.tp_size >= 1, "tp_size must be >= 1"
    assert config.dp_size >= 1, "dp_size must be >= 1"
    assert config.ep_size >= 1, "ep_size must be >= 1"
    assert config.pp_size >= 1, "pp_size must be >= 1"
    assert config.cp_size >= 1, "cp_size must be >= 1"
    # EP 仅对 MoE 模型有效
    if config.ep_size > 1 and not model_config.is_moe:
        raise ValueError("EP size > 1 requires is_moe=True in ModelConfig")
    # EP 是 TP rank 重编号：ep_size 不能超过 tp_size
    if config.ep_size > config.tp_size:
        raise ValueError(
            f"ep_size ({config.ep_size}) > tp_size ({config.tp_size}). "
            f"EP is a re-ranking of TP ranks, cannot exceed tp_size."
        )
    # CP 是 TP rank 重编号：cp_size 不能超过 tp_size
    if config.cp_size > config.tp_size:
        raise ValueError(
            f"cp_size ({config.cp_size}) > tp_size ({config.tp_size}). "
            f"CP is a re-ranking of TP ranks, cannot exceed tp_size."
        )
    # EP size 不能超过专家数
    if config.ep_size > model_config.num_experts:
        raise ValueError(
            f"ep_size ({config.ep_size}) > num_experts ({model_config.num_experts})"
        )
    # PP size 不能超过层数
    if config.pp_size > model_config.num_layers:
        raise ValueError(
            f"pp_size ({config.pp_size}) > num_layers ({model_config.num_layers})"
        )
    # TP size 不能超过 attention heads
    if config.tp_size > model_config.num_attention_heads:
        raise ValueError(
            f"tp_size ({config.tp_size}) > num_attention_heads "
            f"({model_config.num_attention_heads})"
        )
    # KV heads 必须能被 TP 整除（或使用复制）
    kv_per_rank = div_even(model_config.num_kv_heads, config.tp_size, allow_replicate=True)
    if len(set(kv_per_rank)) > 1:
        import warnings
        warnings.warn("KV heads not evenly divisible by tp_size, using replication")
```

### 10.8 CP Context Parallel 仿真

#### 10.8.1 CP rank 结构（TP rank 重编号）

**关键概念**（参见 sglang-note《scheduler-rank-and-process-topology》§2.4）：
CP **不增加进程数**，而是在 Attention 层将 TP rank 重新划分为两层：
```
层级：Global(TP) → ATTN_DP → ATTN_CP → ATTN_TP
```
即：`tp_rank` → `(attn_cp_rank, attn_tp_rank)`

各层含义：
- `attn_dp_size`：Attention 数据并行（启用 DP Attention 时 = dp_size，否则 = 1）
- `cp_size`：Context Parallel 数（将长序列按 cp_size 切分）
- `attn_tp_size`：Attention 内部张量并行数（`tp_size / attn_dp_size / cp_size`）

`ParallelTopology.compute_attn_ranks(tp_rank)` 封装了此映射。

#### 10.8.2 CPSimulator

```python
class CPSimulator:
    """Context Parallel 仿真器。
    对应 SGLang 的 attn_cp_size 机制。

    机制：
    1. 长序列按 cp_size 切分为多个 chunk
    2. 每个 CP rank 处理一个 chunk 的 attention 计算
    3. attention 前后通过 all-gather 同步完整序列的 KV
    4. 仿真中不实际切分序列，只计算通信成本
    """
    def __init__(self, config: 'SimulatorConfig', model_config: 'ModelConfig'):
        self.config = config
        self.model_config = model_config
        self.cp_size = config.cp_size
        self.comm_group = SimCommGroup(
            group_size=self.cp_size,
            config=config,
            group_type="cp",
        ) if self.cp_size > 1 else None
        self.total_comm_ticks = 0

    def simulate_attn_forward(self, seq_len: int) -> int:
        """仿真 CP attention 的 all-gather 通信成本。
        seq_len: 完整序列长度。
        CP 通信：all-gather KV cache（每 rank 只有 1/cp_size 的 KV）。
        """
        if self.cp_size <= 1 or self.comm_group is None:
            return 0
        # KV all-gather 数据量 = seq_len × num_kv_heads × head_dim × dtype × num_layers
        kv_bytes = (
            seq_len * self.model_config.num_kv_heads
            * self.model_config.head_dim * self.config.dtype_size
            * self.model_config.num_layers
        )
        cost = self.comm_group._compute_cost(kv_bytes)
        self.total_comm_ticks += cost
        return cost
```

> SimCommGroup._compute_cost 需追加 `cp` group_type：
> ```python
> elif self.group_type == "cp":
>     latency = self.config.all_reduce_latency_ticks
>     per_byte = self.config.cp_all_gather_cost_per_byte_ticks
> ```

### 10.9 并行仿真完整指标汇总

`SimulationMetrics` 新增的并行相关字段汇总：

```python
@dataclass
class ParallelMetrics:
    """并行仿真指标子结构，嵌入 SimulationMetrics"""
    # TP
    tp_comm_ticks: int = 0
    tp_all_reduce_count: int = 0
    # DP
    dp_rank_load: List[int] = field(default_factory=list)
    dp_rebalance_count: int = 0
    # EP
    ep_comm_ticks: int = 0
    ep_all_to_all_count: int = 0
    ep_cross_rank_tokens: int = 0
    # PP
    pp_comm_ticks: int = 0
    pp_bubble_ticks: int = 0
    pp_micro_batches: int = 0
    # 通用
    comm_ticks_total: int = 0  # 所有并行通信成本总和

    def summary(self) -> Dict[str, Any]:
        return {
            "tp_comm_ticks": self.tp_comm_ticks,
            "tp_all_reduce_count": self.tp_all_reduce_count,
            "dp_rank_load": self.dp_rank_load,
            "dp_rebalance_count": self.dp_rebalance_count,
            "ep_comm_ticks": self.ep_comm_ticks,
            "ep_all_to_all_count": self.ep_all_to_all_count,
            "ep_cross_rank_tokens": self.ep_cross_rank_tokens,
            "pp_comm_ticks": self.pp_comm_ticks,
            "pp_bubble_ticks": self.pp_bubble_ticks,
            "pp_micro_batches": self.pp_micro_batches,
            "comm_ticks_total": self.comm_ticks_total,
        }
```

### 10.10 并行仿真测试策略

#### 10.10.1 TP 测试用例

```python
def test_tp_comm_cost():
    """TP>1 时应产生 all-reduce 通信成本"""
    config = SimulatorConfig(
        model_config=ModelConfig(num_layers=2, hidden_size=128, ...),
        tp_size=2, all_reduce_latency_ticks=2,
    )
    engine = MockEngine(config, config.model_config)
    batch = make_test_batch(size=4)
    engine.forward_batch(batch, make_sampling_args())
    assert engine.metrics.parallel.tp_comm_ticks > 0
    assert engine.metrics.parallel.tp_all_reduce_count > 0

def test_tp_memory_budget():
    """TP>1 时页数应约为 TP=1 的 tp_size 倍"""
    config_tp1 = SimulatorConfig(model_config=mc, tp_size=1)
    config_tp2 = SimulatorConfig(model_config=mc, tp_size=2)
    pages1, _, _ = calculate_memory_budget_parallel(config_tp1, mc, mem)
    pages2, _, _ = calculate_memory_budget_parallel(config_tp2, mc, mem)
    assert pages2 >= pages1 * 1.8  # 容许 10% overhead
```

#### 10.10.2 DP 测试用例

```python
def test_dp_round_robin():
    """round_robin 策略应均匀分配"""
    config = SimulatorConfig(model_config=mc, dp_size=4,
                            dp_load_balance_strategy="round_robin")
    controller = DataParallelController(config, total_pages=1000)
    for i in range(8):
        req = make_req(uid=i)
        rank = controller.assign_request(req)
        assert rank == i % 4

def test_dp_shortest_queue():
    """shortest_queue 应分配到最空闲的副本"""
    config = SimulatorConfig(model_config=mc, dp_size=2,
                            dp_load_balance_strategy="shortest_queue")
    controller = DataParallelController(config, total_pages=1000)
    # 先填充 rank 0
    controller.ranks[0].used_pages = 500
    req = make_req(uid=0)
    rank = controller.assign_request(req)
    assert rank == 1  # 应分配到空闲的 rank 1

def test_dp_attention_all_gather():
    """DP Attention 模式应产生 all-gather 通信成本"""
    config = SimulatorConfig(model_config=mc, dp_size=2,
                            enable_dp_attention=True)
    dp_attn_sim = DPAttentionSimulator(config, mc)
    comm = dp_attn_sim.simulate_mlp_forward([4, 4])  # 两个 DP rank 各 4 token
    assert comm > 0  # 有 all-gather 成本
```

#### 10.10.3 EP 测试用例

```python
def test_ep_expert_distribution():
    """EP>1 时专家应均匀分片"""
    mc = ModelConfig(num_experts=8, is_moe=True, moe_top_k=2, hidden_size=64, ...)
    config = SimulatorConfig(model_config=mc, ep_size=2)
    moe = SimMoeBackend(mc, ep_size=2,
        comm_group=SimCommGroup(2, config, "ep"))
    assert moe.experts_per_rank == [4, 4]
    assert moe._expert_to_rank(3) == 0
    assert moe._expert_to_rank(5) == 1

def test_ep_comm_cost():
    """EP>1 时 forward 应产生 all-to-all 通信成本"""
    mb = make_test_batch(size=4)
    _, comm = moe.forward(None, 4)
    assert comm > 0
```

#### 10.10.4 PP 测试用例

```python
def test_pp_bubble():
    """PP>1 时应产生流水线气泡"""
    config = SimulatorConfig(model_config=mc, pp_size=2,
                            pp_num_micro_batches=4,
                            pp_pipeline_schedule="1f1b")
    pp_sim = PPPipelineSimulator(config, mc)
    batch = make_test_batch(size=8)
    comm = pp_sim.simulate_pipeline_forward(batch)
    assert comm > 0  # 有 send/recv 成本
    assert pp_sim.bubble_ticks > 0  # 有气泡

def test_pp_stage_layers():
    """层应按 pp_size 均分"""
    topo = ParallelTopology(pp_size=4)
    layers = topo.pp_stage_layers(8)
    assert [len(r) for r in layers] == [2, 2, 2, 2]
```

#### 10.10.5 组合并行测试

```python
def test_tp_dp_ep_pp_cp_combined():
    """TP=4, DP=2, EP=2, PP=2, CP=2 组合仿真。
    注意：ep_size=2 和 cp_size=2 都是 tp_size=4 的重编号，不增加进程数。
    world_size = tp × dp × pp = 4 × 2 × 2 = 16（不是 4×2×2×2×2=32）
    """
    mc = ModelConfig(num_layers=8, num_attention_heads=8,
                     num_kv_heads=8, hidden_size=64,
                     is_moe=True, num_experts=4, moe_top_k=1, ...)
    config = SimulatorConfig(
        model_config=mc, tp_size=4, dp_size=2, ep_size=2, pp_size=2,
        cp_size=2, pp_num_micro_batches=2,
    )
    validate_parallel_config(config, mc)  # 不抛异常
    assert ParallelTopology(
        tp_size=4, dp_size=2, ep_size=2, pp_size=2, cp_size=2
    ).world_size == 16  # tp×dp×pp，不是 32
    engine = MockEngine(config, mc)
    batch = make_test_batch(size=4)
    engine.forward_batch(batch, make_sampling_args())
    m = engine.metrics.parallel
    assert m.tp_comm_ticks > 0
    assert m.ep_comm_ticks > 0
    assert m.pp_comm_ticks > 0
    assert m.comm_ticks_total == (
        m.tp_comm_ticks + m.ep_comm_ticks + m.pp_comm_ticks
    )
```

### 10.11 退化为单实例的验证

```python
def test_parallel_degradation():
    """所有并行 size=1 时应退化为纯单实例，零通信成本"""
    mc = ModelConfig(num_layers=2, hidden_size=64, ...)
    config = SimulatorConfig(model_config=mc, tp_size=1, dp_size=1,
                            ep_size=1, pp_size=1, cp_size=1)
    engine = MockEngine(config, mc)
    batch = make_test_batch(size=4)
    engine.forward_batch(batch, make_sampling_args())
    m = engine.metrics.parallel
    assert m.tp_comm_ticks == 0
    assert m.ep_comm_ticks == 0
    assert m.pp_comm_ticks == 0
    assert m.comm_ticks_total == 0
```

### 10.12 TS strict 实现要点

1. **类型定义**：所有 Python dataclass → TS interface/class，`List` → `T[]`，`Dict` → `Map`/`Record`
2. **Optional 字段**：`num_pages: int | None` → `num_pages: number | null`
3. **Tuple 返回**：`Tuple[int, int, int]` → `[number, number, number]` 或具名 interface
4. **零运行时依赖**：不引入 NCCL/MPI/任何通信库，通信成本纯算术
5. **枚举**：`group_type: str` → `type GroupType = "tp" | "ep" | "pp" | "dp_attn" | "cp"`
6. **strict 模式**：所有字段显式初始化，禁用 `any`（`// @ts-ignore` 仅用于兼容层）
7. **EP/CP 重编号**：TS 实现中 `compute_moe_ranks` / `compute_attn_ranks` 是纯计算函数，不涉及进程间通信

---

## 11. 总结

本报告基于对 `sglang-note`（源码阅读笔记，40+ 篇文章）、`mini-sglang`（约 5000 行的 SGLang 紧凑实现）和 `Awesome-ML-SYS-Tutorial/sglang`（架构博客，20+ 篇文章）的全面研究，提取了 SGLang 推理框架的完整系统层行为规格。

**核心发现**：
1. **仿真范围**：Scheduler 全链路（prefill/decode 调度、cache 管理、table 管理）、RadixCache 前缀缓存算法、CUDA Graph 分桶与 replay 决策、Overlap Scheduling 时序、内存预算计算 — 这些是系统层研究的核心
2. **模拟范围**：模型 forward、attention kernel、sampler、ZMQ/NCCL 通信 — 这些用 mock 替代，不影响系统层行为
3. **关键算法**：Chunked Prefill（token budget 控制）、RadixTree（前缀匹配+分裂+LRU eviction）、cache_req 的 5 区域管理、Overlap Scheduling 的双 stream 重叠
4. **并行仿真**（§10）：
   - **TP**（§10.2, §10.6）：all-reduce 通信成本 + 权重/KV 分割 + ZMQ 广播 + gloo/nccl 三层通信基础设施
   - **DP**（§10.3）：标准 DP（`DataParallelController` 请求分发 round_robin/shortest_queue）+ DP Attention（MLA 模型专用 all-gather/slice 机制）
   - **EP**（§10.4）：MoE 专家分片 + TP rank 重编号（moe_dp/moe_ep/moe_tp 三层）+ all-to-all + EPLB 动态负载均衡
   - **PP**（§10.5）：micro-batch 调度 + 最后 stage 采样 + stage 间通信 + 气泡仿真
   - **CP**（§10.8）：Context Parallel 长序列切分 + TP rank 重编号 + KV all-gather
   - **进程数 = tp × dp × pp**（EP/CP 是 TP rank 重编号，不增加进程数）
   - 所有并行通过 `SimCommGroup` 统一通信成本模型，`size=1` 时退化为纯单实例

**实现路径**（§5.1）：Phase 1-4 覆盖单实例核心（数据结构→RadixCache→CUDA Graph/Overlap→内存预算/指标），Phase 5 覆盖并行扩展（TP/DP/EP/PP/CP），每个 Phase 可独立测试。

**TS strict 实现要点**（§10.12）：所有 Python dataclass → TS interface，零运行时依赖（通信成本纯算术），strict 模式禁用 `any`，`compute_moe_ranks`/`compute_attn_ranks` 为纯计算函数。

模拟器采用 tick-based 离散事件仿真，支持可配置的调度策略、并行拓扑（TP×DP×PP 进程网格 + EP/CP rank 重编号 + DP Attention）、工作负载和性能指标收集，可在无 GPU 环境下研究 SGLang 的系统层优化策略和并行通信开销。
