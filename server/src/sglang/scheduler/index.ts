// scheduler — K1: TableManager + S2: PrefillAdder/PrefillManager/DecodeManager
// P6+S3: SimScheduler（ParallelGroups 接入 + EPLB 集成 + globalStep + §9.11 完整调度循环）
// S5: SimulationClock + Overlap Scheduling（last_data 延迟 + 空 tick 刷新 + 高水位背压）

import { Req, Batch, SamplingParams, BatchSamplingArgs } from "../core";
import type { ForwardOutput } from "../core";
import { ChunkedReq, PendingReq } from "../entities";
import { CacheManager } from "../cache";
import type { BaseCacheHandle } from "../cache";
import type { ParallelGroups } from "../parallel/groups";
import type { SimulationMetrics } from "../metrics";
import { MockEngine } from "../engine";
import type {
  SimulatorConfig,
  SimRequestMsg,
  SimRespMsg,
  SchedulerMsg,
  UserMsg,
  AbortMsg,
  BatchSchedulerMsg,
  ForwardInput,
} from "../types";

/** 管理 page_table 行分配（§3.3.6 / §9.11 完整实现） */
export class TableManager {
  readonly maxRunningReq: number;
  pageTable: number[][];
  tokenPool: number[][];
  freeTableIndices: number[];

  constructor(maxRunningReq: number, pageTable: number[][]) {
    this.maxRunningReq = maxRunningReq;
    this.pageTable = pageTable;
    // 最后一行预留给 dummy req
    this.freeTableIndices = Array.from(
      { length: maxRunningReq }, (_, i) => i
    );
    this.tokenPool = Array.from(
      { length: maxRunningReq + 1 },
      () => new Array(pageTable[0].length).fill(0)
    );
  }

  get availableSize(): number {
    return this.freeTableIndices.length;
  }

  allocate(): number {
    if (this.freeTableIndices.length === 0) {
      throw new Error("No available table indices");
    }
    return this.freeTableIndices.pop()!;
  }

  free(tableIdx: number): void {
    this.freeTableIndices.push(tableIdx);
  }
}

// ===== S5: SimulationClock（§4.1 / §4.3） =====

/** 仿真时钟事件记录 */
export interface SimEvent {
  tick: number;
  eventType: "gpu_start" | "gpu_end" | "cpu_schedule" | "cpu_process";
  duration: number;
}

/**
 * SimulationClock — GPU 时序追踪时钟（§4.1 / §4.3）
 *
 * 提供 tick 计数、advance、GPU 任务调度、重叠检测及 tick 回调队列功能。
 * 默认不在 SimScheduler 中实例化；需要时通过配置或构造器选项启用。
 */
export class SimulationClock {
  currentTick: number = 0;
  gpuBusyUntil: number = 0;
  events: SimEvent[] = [];
  private _tickCallbacks: Array<(tick: number) => void> = [];

  /** 推进时钟 deltaTicks 个 tick；单调不回退 */
  advance(deltaTicks: number = 1): void {
    if (deltaTicks <= 0) {
      throw new Error(`advance(deltaTicks) requires deltaTicks > 0, got ${deltaTicks}`);
    }
    this.currentTick += deltaTicks;
    for (const cb of this._tickCallbacks) {
      cb(this.currentTick);
    }
  }

  /** 安排 GPU 任务，返回完成时间（tick 号） */
  scheduleGpu(durationTicks: number): number {
    const start = Math.max(this.currentTick, this.gpuBusyUntil);
    this.gpuBusyUntil = start + durationTicks;
    this.events.push({ tick: start, eventType: "gpu_start", duration: durationTicks });
    this.events.push({ tick: this.gpuBusyUntil, eventType: "gpu_end", duration: 0 });
    return this.gpuBusyUntil;
  }

  /** GPU 当前是否繁忙（可重叠检测） */
  canOverlap(): boolean {
    return this.currentTick < this.gpuBusyUntil;
  }

  /** 注册 tick 回调，返回取消注册函数 */
  onTick(callback: (tick: number) => void): () => void {
    this._tickCallbacks.push(callback);
    return () => {
      const idx = this._tickCallbacks.indexOf(callback);
      if (idx !== -1) {
        this._tickCallbacks.splice(idx, 1);
      }
    };
  }
}

// ===== S2: PrefillAdder（§9.5 / §9.7 / §9.11） =====

/**
 * PrefillAdder — 逐个尝试将请求加入 prefill batch（§9.5 / §9.11）
 *
 * 实现两次 available_size 检查（lock 前宽松、lock 后严格）、
 * token budget 管理、chunked prefill 分块与续接。
 */
export class PrefillAdder {
  private readonly tokenBudget: number;
  private readonly cacheManager: CacheManager;
  private readonly tableManager: TableManager;
  private readonly decodeManager: DecodeManager;
  consumedTokens: number;

  constructor(
    tokenBudget: number,
    cacheManager: CacheManager,
    tableManager: TableManager,
    decodeManager: DecodeManager,
  ) {
    this.tokenBudget = tokenBudget;
    this.cacheManager = cacheManager;
    this.tableManager = tableManager;
    this.decodeManager = decodeManager;
    this.consumedTokens = 0;
  }

  /** 当前 token budget 剩余 */
  get remainingBudget(): number {
    return this.tokenBudget - this.consumedTokens;
  }

  /** decodeManager 的 reservedSize（inflightTokens） */
  get reservedSize(): number {
    return this.decodeManager.inflightTokens;
  }

  /**
   * 尝试将 pendingReq 加入 prefill batch（§9.11 步骤 0-10）
   * @returns Req | ChunkedReq 加入成功；null 资源/budget 不足
   */
  tryAddOne(pendingReq: PendingReq): Req | ChunkedReq | null {
    // 步骤 0: chunked 续接路径
    if (pendingReq.chunkedReq !== null) {
      return this._tryAddOneChunked(pendingReq);
    }

    // 步骤 1: 前缀匹配
    const matchResult = this.cacheManager.matchReq(pendingReq);
    const handle = matchResult.cudaHandle;
    const cachedLen = handle.cachedLen;
    const prefixLen = pendingReq.inputLen;
    const extendLen = prefixLen - cachedLen;

    // 步骤 2: 第一次 available_size 检查（宽松，lock 前）
    // estimatedLen 包含 extend 部分 + decode 输出长度（§9.11 L3208）
    const estimatedLen = extendLen + pendingReq.outputLen;
    if (estimatedLen + this.reservedSize > this.cacheManager.availableSize) {
      return null;
    }

    // 步骤 3: token budget 检查
    if (this.remainingBudget <= 0) {
      return null;
    }

    // 步骤 4: lock
    this.cacheManager.lock(handle);

    try {
      // 步骤 5: 第二次 available_size 检查（严格，lock 后）
      if (estimatedLen + this.reservedSize > this.cacheManager.availableSize) {
        this.cacheManager.unlock(handle);
        return null;
      }

      // 步骤 6: 分配 table_idx
      let tableIdx: number;
      try {
        tableIdx = this.tableManager.allocate();
      } catch (_err) {
        this.cacheManager.unlock(handle);
        return null;
      }

      try {
        // 步骤 7: 复制 cached 部分的 token 和 page entry
        const matchedIndices = handle.getMatchedIndices();
        for (let i = 0; i < matchedIndices.length && i < this.tableManager.pageTable[tableIdx].length; i++) {
          this.tableManager.pageTable[tableIdx][i] = matchedIndices[i];
        }

        // 步骤 7b: 复制 extend 部分的 token 到 token_pool
        for (let i = 0; i < extendLen; i++) {
          this.tableManager.tokenPool[tableIdx][cachedLen + i] = pendingReq.inputIds[cachedLen + i];
        }

        // 步骤 8: 决定 chunk_size
        const chunkSize = Math.min(extendLen, this.remainingBudget);
        const isChunked = extendLen > chunkSize;

        // 步骤 9: 更新 consumedTokens
        this.consumedTokens += chunkSize;

        if (isChunked) {
          // 返回 ChunkedReq
          const creq = new ChunkedReq({
            rid: pendingReq.rid,
            inputIds: pendingReq.inputIds,
            samplingParams: pendingReq.samplingParams,
          });
          creq.deviceLen = cachedLen + chunkSize;
          creq.maxDeviceLen = cachedLen + chunkSize;
          (creq as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle = handle;
          (creq as unknown as { tableIdx: number }).tableIdx = tableIdx;
          return creq;
        } else {
          // 步骤 10: 非 chunked 的请求加入 decodeManager
          const req = new Req({
            rid: pendingReq.rid,
            inputIds: pendingReq.inputIds,
            samplingParams: pendingReq.samplingParams,
          });
          req.deviceLen = prefixLen;
          req.maxDeviceLen = cachedLen + extendLen + pendingReq.outputLen;
          (req as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle = handle;
          (req as unknown as { tableIdx: number }).tableIdx = tableIdx;
          this.decodeManager.addReq(req);
          return req;
        }
      } catch (err) {
        this.tableManager.free(tableIdx);
        this.cacheManager.unlock(handle);
        throw err;
      }
    } catch (err) {
      // allocate 的 unlock 已在内部处理；此 catch 仅防御性
      throw err;
    }
  }

  /**
   * Chunked prefill 续接路径（§9.11）
   * 续接中不调用 lock（handle 已锁定），不重新分配 tableIdx
   */
  private _tryAddOneChunked(pendingReq: PendingReq): Req | ChunkedReq | null {
    const prevReq = pendingReq.chunkedReq!;
    const cachedLen = prevReq.deviceLen;
    const extendLen = pendingReq.inputLen - cachedLen;
    const tableIdx = (prevReq as unknown as { tableIdx: number }).tableIdx;
    const handle = (prevReq as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle!;

    // 资源检查（续接路径同样包含 outputLen，§9.11 L3290）
    if (extendLen + pendingReq.outputLen + this.reservedSize > this.cacheManager.availableSize) {
      return null;
    }

    // token budget 检查
    if (this.remainingBudget <= 0) {
      return null;
    }

    // 复制本 chunk 的 extend token 到 token_pool
    const chunkSize = Math.min(extendLen, this.remainingBudget);
    const isChunked = extendLen > chunkSize;

    for (let i = 0; i < chunkSize; i++) {
      this.tableManager.tokenPool[tableIdx][cachedLen + i] = pendingReq.inputIds[cachedLen + i];
    }

    // 更新 consumedTokens
    this.consumedTokens += chunkSize;

    if (isChunked) {
      const creq = new ChunkedReq({
        rid: pendingReq.rid,
        inputIds: pendingReq.inputIds,
        samplingParams: pendingReq.samplingParams,
      });
      creq.deviceLen = cachedLen + chunkSize;
      creq.maxDeviceLen = cachedLen + chunkSize;
      (creq as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle = handle;
      (creq as unknown as { tableIdx: number }).tableIdx = tableIdx;
      return creq;
    } else {
      // 最后一个 chunk 转为完整 Req
      const req = new Req({
        rid: pendingReq.rid,
        inputIds: pendingReq.inputIds,
        samplingParams: pendingReq.samplingParams,
      });
      req.deviceLen = pendingReq.inputLen;
      req.maxDeviceLen = pendingReq.inputLen + pendingReq.outputLen;
      (req as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle = handle;
      (req as unknown as { tableIdx: number }).tableIdx = tableIdx;
      this.decodeManager.addReq(req);
      return req;
    }
  }
}

// ===== S2: DecodeManager（§9.2 / §9.11） =====

/**
 * DecodeManager — 管理可 decode 的请求集合（§9.2 / §9.11）
 *
 * 负责 runningReqs 管理、inflightTokens 计算、decode batch 生成。
 */
export class DecodeManager {
  private readonly _pageSize: number;
  runningReqs: Set<Req>;

  constructor(pageSize: number) {
    this._pageSize = pageSize;
    this.runningReqs = new Set();
  }

  /** 添加请求到 runningReqs */
  addReq(req: Req): void {
    this.runningReqs.add(req);
  }

  /** 从 runningReqs 移除请求 */
  removeReq(req: Req): void {
    this.runningReqs.delete(req);
  }

  /** 按 uid 查找并移除请求，返回被移除的请求或 null */
  abortReq(uid: number): Req | null {
    for (const req of this.runningReqs) {
      if (req.rid === uid) {
        this.runningReqs.delete(req);
        return req;
      }
    }
    return null;
  }

  /**
   * forward 后过滤请求（§9.11）
   * runningReqs = (runningReqs ∪ newReqs) 中 canDecode 为 true 的子集
   */
  filterReqs(newReqs: Req[]): void {
    const merged = new Set([...this.runningReqs, ...newReqs]);
    this.runningReqs = new Set();
    for (const req of merged) {
      if (req.canDecode) {
        this.runningReqs.add(req);
      }
    }
  }

  /**
   * inflightTokens 计算（§9.11）
   * tokens_reserved = (pageSize - 1) * len(runningReqs)
   * inflightTokens = sum(remainLen) + tokens_reserved
   */
  get inflightTokens(): number {
    let remainSum = 0;
    for (const req of this.runningReqs) {
      remainSum += req.remainLen;
    }
    const tokensReserved = (this._pageSize - 1) * this.runningReqs.size;
    return remainSum + tokensReserved;
  }

  /**
   * 生成 decode batch（§9.11）
   * 按 rid 排序 runningReqs，构造 Batch
   */
  scheduleNextBatch(): Batch | null {
    if (this.runningReqs.size === 0) {
      return null;
    }

    const sorted = [...this.runningReqs].sort((a, b) => a.rid - b.rid);
    const batch = new Batch();
    for (const req of sorted) {
      batch.reqs.set(req.rid, req);
      batch.readyIds.push(req.rid);
    }
    batch.numDecodeTokens = sorted.length;
    return batch;
  }
}

// ===== S2: PrefillManager（§9.2 / §9.5 / §9.11） =====

/**
 * PrefillManager — 管理待 prefill 的请求队列（§9.2 / §9.5 / §9.11）
 *
 * 通过 PrefillAdder 逐个调度，chunked 请求剩余部分放回队列头部优先续接。
 */
export class PrefillManager {
  pendingList: PendingReq[];
  private readonly cacheManager: CacheManager;
  private readonly tableManager: TableManager;
  private readonly decodeManager: DecodeManager;

  constructor(
    cacheManager: CacheManager,
    tableManager: TableManager,
    decodeManager: DecodeManager,
  ) {
    this.pendingList = [];
    this.cacheManager = cacheManager;
    this.tableManager = tableManager;
    this.decodeManager = decodeManager;
  }

  /** 添加单个请求到 pendingList 尾部 */
  addOneReq(msg: SimRequestMsg): void {
    const sp = msg.samplingParams ?? new SamplingParams({ maxNewTokens: msg.outputLen });
    const pr = new PendingReq({
      rid: msg.uid,
      inputIds: msg.inputIds,
      samplingParams: sp,
    });
    this.pendingList.push(pr);
  }

  /** 批量添加请求到 pendingList 尾部 */
  addBatch(msgs: SimRequestMsg[]): void {
    for (const msg of msgs) {
      this.addOneReq(msg);
    }
  }

  /**
   * 按 uid 移除 pending 请求
   * @returns 被移除请求的 chunkedReq（如有），不存在则返回 null
   */
  abortReq(uid: number): ChunkedReq | null {
    for (let i = 0; i < this.pendingList.length; i++) {
      if (this.pendingList[i].rid === uid) {
        const removed = this.pendingList.splice(i, 1)[0];
        return removed.chunkedReq;
      }
    }
    return null;
  }

  /**
   * 调度下一个 prefill batch（§9.5 / §9.11）
   * @param tokenBudget 本 tick 可用的 prefill token budget
   * @returns Batch 或 null（无请求或资源不足）
   */
  scheduleNextBatch(tokenBudget: number): Batch | null {
    if (this.pendingList.length === 0) {
      return null;
    }

    const adder = new PrefillAdder(
      tokenBudget,
      this.cacheManager,
      this.tableManager,
      this.decodeManager,
    );

    const reqs: Req[] = [];
    const chunkedList: PendingReq[] = [];
    let i = 0;

    for (; i < this.pendingList.length; i++) {
      const pendingReq = this.pendingList[i];
      const result = adder.tryAddOne(pendingReq);

      if (result === null) {
        break;
      }

      if (result instanceof ChunkedReq) {
        // 构造新的 PendingReq 携带 chunkedReq，放回队列头部续接
        const continueReq = new PendingReq({
          rid: pendingReq.rid,
          inputIds: pendingReq.inputIds,
          samplingParams: pendingReq.samplingParams,
          chunkedReq: result,
        });
        chunkedList.push(continueReq);
        reqs.push(result);
      } else {
        reqs.push(result);
      }
    }

    // 更新 pendingList：chunked 续接优先 + 未调度的请求
    this.pendingList = [...chunkedList, ...this.pendingList.slice(i)];

    if (reqs.length === 0) {
      return null;
    }

    const batch = new Batch();
    for (const req of reqs) {
      batch.reqs.set(req.rid, req);
      batch.readyIds.push(req.rid);
    }
    batch.extendInputTokens = adder.consumedTokens;
    return batch;
  }
}

// ===== S3: SchedulerIOMixin（§9.6 L2193-2241） =====

/**
 * SchedulerIOMixin — 调度器消息 I/O 管理层（§9.6）
 *
 * 负责：
 * - receiveMsg(): 接收消息；offline 模式下返回空数组
 * - sendResult(): 发送结果；offline 模式下为 noop
 * - syncAllRanks(): 同步所有 rank；tp_size=1 时为 noop
 * - step(): 推进一步消息处理
 */
export class SchedulerIOMixin {
  protected readonly config: SimulatorConfig;
  protected readonly _incomingQueue: SchedulerMsg[];
  protected readonly _outgoingQueue: SimRespMsg[];

  constructor(config: SimulatorConfig) {
    this.config = config;
    this._incomingQueue = [];
    this._outgoingQueue = [];
  }

  /** 接收消息；offline 模式下返回空数组 */
  receiveMsg(): SchedulerMsg[] {
    if (this.config.offlineMode) return [];
    const msgs = [...this._incomingQueue];
    this._incomingQueue.length = 0;
    return msgs;
  }

  /** 发送结果；offline 模式下为 noop */
  sendResult(reply: SimRespMsg[]): void {
    if (this.config.offlineMode) return;
    this._outgoingQueue.push(...reply);
  }

  /** 同步所有 rank；tp_size=1 时为 noop */
  syncAllRanks(): void {
    // 仿真中单 rank 为 noop
  }

  /** 推进一步消息处理（非 offline 模式下由外部驱动） */
  step(): void {
    // offline 模式下消息通过 runTick 参数传入，不需要主动拉取
  }
}

// ===== P6+S3: SimScheduler（§9.11 完整调度循环 + P6 ParallelGroups 集成） =====

/**
 * Forward 返回数据，用于 overlap tick 中延迟处理
 */
interface _OverlapData {
  forwardInput: ForwardInput;
  forwardOutput: ForwardOutput;
}

/**
 * SimScheduler — 仿真调度器（§9.11 L2888-3058 + P6 ParallelGroups 集成）
 *
 * 继承 SchedulerIOMixin，实现完整的调度循环：
 * _processOneMsg → _scheduleNextBatch → _forward → _processLastData
 *
 * 合并后的统一调度器同时具备：
 * - §9.11 完整调度循环（4种消息类型、_prepareBatch 6步、tokenPool读写、copyDoneEvent.synchronize()）
 * - P6 并行特性（ParallelGroups 接入、EPLB tick末尾、globalStep、overlap tick、DP 分发）
 *
 * 构造器两种模式：
 * - 简单模式：new SimScheduler(config) — 内部创建 MockEngine/TableManager/CacheManager
 * - 完整模式：new SimScheduler(config, { prefillManager, decodeManager, ... }) — 注入依赖
 */
export class SimScheduler extends SchedulerIOMixin {
  readonly engine: MockEngine;
  readonly tableManager: TableManager;
  readonly cacheManager: CacheManager;
  readonly decodeManager: DecodeManager;
  readonly prefillManager: PrefillManager;
  finishedReqs: Set<Req>;
  readonly eosTokenId: number;
  readonly tokenPool: number[][];
  readonly prefillBudget: number;
  readonly overlapEnabled: boolean;
  lastBatch: Batch | null;
  lastForwardOutput: ForwardOutput | null;

  // P6: 并行扩展字段
  private readonly _groups: ParallelGroups | null;
  private readonly _simMetrics: SimulationMetrics | null;
  private _globalStep: number = 0;
  private _lastOverlapData: _OverlapData | null = null;

  // S5: Overlap Scheduling 扩展字段
  private _tickCounter: number = 0;
  private _overlapWaitTicks: number;
  private _eagerExtraDelayTicks: number;
  private _idleCounter: number = 0;
  private _lastDataPending: boolean = false;
  private _lastDataAckTick: number = 0;
  private _clock: SimulationClock | null;
  private _highWatermark: number;

  constructor(
    config: SimulatorConfig,
    opts?: {
      prefillManager?: PrefillManager;
      decodeManager?: DecodeManager;
      parallelGroups?: ParallelGroups;
      simMetrics?: SimulationMetrics;
      engine?: MockEngine;
      cacheManager?: CacheManager;
      tableManager?: TableManager;
      clock?: SimulationClock;
    },
  ) {
    super(config);

    // P6: 并行扩展
    const groups = opts?.parallelGroups ?? null;
    this._groups = groups;
    this._simMetrics = opts?.simMetrics ?? null;

    // S3: 核心组件 — 支持外部注入或内部创建
    this.engine = opts?.engine ?? new MockEngine(config);
    this.tableManager = opts?.tableManager ?? new TableManager(config.maxRunningReq, this.engine.pageTable);
    this.cacheManager = opts?.cacheManager ?? new CacheManager(
      this.engine.numPages, config.pageSize, this.engine.pageTable, config.cacheType
    );
    this.decodeManager = opts?.decodeManager ?? new DecodeManager(config.pageSize);
    this.prefillManager = opts?.prefillManager ?? new PrefillManager(
      this.cacheManager, this.tableManager, this.decodeManager
    );

    this.finishedReqs = new Set();
    this.eosTokenId = config.eosTokenId;
    this.tokenPool = this.tableManager.tokenPool;
    this.prefillBudget = config.maxExtendTokens;
    this.overlapEnabled = config.enableOverlap;
    this.lastBatch = null;
    this.lastForwardOutput = null;

    // S5: Overlap Scheduling 配置
    this._overlapWaitTicks = config.tokenRecvDelayTicks;
    this._eagerExtraDelayTicks = config.eagerForwardExtraDelayTicks;
    this._highWatermark = config.messagesHighWatermark;

    // S5: SimulationClock 初始化
    if (opts?.clock !== undefined) {
      this._clock = opts.clock;
    } else if (config.enableOverlap && config.enableMetrics) {
      this._clock = new SimulationClock();
    } else {
      this._clock = null;
    }
  }

  // ===== P6: 并行扩展属性 =====

  /** 当前全局步数 */
  get globalStep(): number {
    return this._globalStep;
  }

  /** 获取 ParallelGroups（只读） */
  get groups(): ParallelGroups | null {
    return this._groups;
  }

  /** 获取 SimulationClock（只读） */
  get clock(): SimulationClock | null {
    return this._clock;
  }

  /** 获取当前 tick 计数器 */
  get tickCounter(): number {
    return this._tickCounter;
  }

  /**
   * 添加请求到调度器
   *
   * P6 扩展：若 dpSize > 1，先通过 dpController.select_rank_for_request 分配 DP rank
   */
  addRequest(msg: SimRequestMsg): void {
    // DP 请求分发
    if (this._groups && this.config.dpSize > 1) {
      const neededPages = 1; // 简化：每个请求至少需要 1 页
      const rank = this._groups.dpController.select_rank_for_request(neededPages);
      if (rank === null) {
        // OOM：拒绝请求
        return;
      }
      const dpRank = rank.rank;
      const sp = msg.samplingParams ?? new SamplingParams({ maxNewTokens: msg.outputLen });
      const pr = new PendingReq({
        rid: msg.uid,
        inputIds: msg.inputIds,
        samplingParams: sp,
        dpRank,
      });
      this.prefillManager.pendingList.push(pr);
      return;
    }
    this.prefillManager.addOneReq(msg);
  }

  // ===== §9.11 调度循环入口 =====

  /**
   * 执行一个调度 tick（§9.11）
   * @param incoming 进入的请求消息列表
   * @returns 响应消息列表
   */
  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    if (this.overlapEnabled) {
      return this._overlapTick(incoming);
    }
    return this._normalTick(incoming);
  }

  /**
   * _normalTick — 串行调度循环（对齐 §9.11 L2916-2924）
   * 1. for msg in incoming: _processOneMsg(msg)
   * 2. forwardInput = _scheduleNextBatch()
   * 3. ongoingData = null
   * 4. if forwardInput !== null: ongoingData = [forwardInput, _forward(forwardInput)]
   * 5. return _processLastData(ongoingData)
   * 6. EPLB maybe_rebalance（tick 末尾，§10.4.4）
   * 7. 递增 globalStep
   */
  private _normalTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // 1. 消息处理：将 SimRequestMsg 转换为 UserMsg 后处理
    for (const msg of incoming) {
      const userMsg: UserMsg = {
        tag: "req_in",
        uid: msg.uid,
        inputIds: msg.inputIds,
        samplingParams: msg.samplingParams,
        outputLen: msg.outputLen,
      };
      this._processOneMsg(userMsg);
    }

    // 2. 调度下一个 batch
    const forwardInput = this._scheduleNextBatch();

    // 3-4. forward
    let ongoingData: [ForwardInput, ForwardOutput] | null = null;
    if (forwardInput !== null) {
      const forwardOutput = this._forward(forwardInput);
      ongoingData = [forwardInput, forwardOutput];
    }

    // 5. 结果处理
    const replies = this._processLastData(ongoingData);

    // 6. EPLB maybe_rebalance（§10.4.4）
    this._maybeEplbRebalance();

    // 7. 递增 globalStep
    this._globalStep += 1;

    // S5: 推进 SimulationClock
    this._clock?.advance(1);

    return replies;
  }

  /**
   * _overlapTick — 重叠调度循环（§9.4 / §9.11 + S5 完整 Overlap Scheduling）
   *
   * Phase 1: 处理消息
   * Phase 2: 处理上一批结果（带 last_data 延迟检查）— 先处理释放 GPU
   * Phase 3: 调度下一批（仅当 GPU 空闲时，含高水位背压检查）
   * Phase 4: forward 当前批（GPU phase）
   * Phase 5: 保存当前批数据（带延迟计算）
   * Phase 6: 空闲 tick 刷新检查
   * Phase 7: 递增 tickCounter + EPLB + globalStep + advance clock
   */
  private _overlapTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // Phase 1: 消息处理
    for (const msg of incoming) {
      const userMsg: UserMsg = {
        tag: "req_in",
        uid: msg.uid,
        inputIds: msg.inputIds,
        samplingParams: msg.samplingParams,
        outputLen: msg.outputLen,
      };
      this._processOneMsg(userMsg);
    }

    // Phase 2: 处理上一批结果（带 last_data 延迟检查）
    // 移到调度之前：先释放 GPU，再调度新批
    const replies: SimRespMsg[] = [];
    if (this._lastOverlapData !== null && this._tickCounter >= this._lastDataAckTick) {
      replies.push(...this._processLastData([
        this._lastOverlapData.forwardInput,
        this._lastOverlapData.forwardOutput,
      ]));
      this._lastOverlapData = null;
      this._lastDataPending = false;
    }

    // Phase 3: 调度下一批（仅当 GPU 空闲时）
    // GPU 忙 = 上一批结果尚未处理 → 不调度新 forward，防止 _lastOverlapData 被覆盖
    const gpuBusy = this._lastOverlapData !== null;
    let forwardInput: ForwardInput | null;
    if (gpuBusy || (this._highWatermark > 0 && this._incomingQueue.length > this._highWatermark)) {
      forwardInput = null;
    } else {
      forwardInput = this._scheduleNextBatch();
    }

    // Phase 4: forward 当前批
    const forwardOutput = forwardInput
      ? this._forward(forwardInput)
      : null;

    // Phase 5: 保存当前批数据（带延迟计算）
    if (forwardOutput !== null && forwardInput !== null) {
      this._lastOverlapData = { forwardInput, forwardOutput };
      const isGraphCapture = forwardOutput.isGraphCapture ?? false;
      if (isGraphCapture) {
        this._lastDataAckTick = this._tickCounter + this._overlapWaitTicks;
      } else {
        this._lastDataAckTick = this._tickCounter + this._overlapWaitTicks + this._eagerExtraDelayTicks;
      }
      this._lastDataPending = true;

      // S5: 记录 GPU 占用到 clock
      const gpuDuration = (forwardOutput.prefillBatchTime ?? 0) + (forwardOutput.decodeBatchTime ?? 0);
      if (gpuDuration > 0) {
        this._clock?.scheduleGpu(gpuDuration);
      }
    }

    // Phase 6: 空闲 tick 刷新检查（仅当 GPU 空闲时才算空闲）
    if (incoming.length === 0 && forwardInput === null && !gpuBusy) {
      this._idleCounter += 1;
      if (this.config.idleCountForFlush > 0 && this._idleCounter >= this.config.idleCountForFlush) {
        this._forcePrefillSchedule();
        this._idleCounter = 0;
      }
    } else {
      this._idleCounter = 0;
    }

    // Phase 7: 递增 tickCounter + EPLB + globalStep + advance clock
    this._tickCounter += 1;
    this._maybeEplbRebalance();
    this._globalStep += 1;
    this._clock?.advance(1);

    return replies;
  }

  /**
   * S5: _forcePrefillSchedule — 空闲 tick 时强制调度 prefill
   * 仅触发 prefill 调度路径，不单独调度 decode batch。
   * 续接完成的请求通过 PrefillAdder 自动加入 decodeManager，
   * 下一 tick 的 decode 阶段自然调度之。
   */
  private _forcePrefillSchedule(): void {
    const batch = this.prefillManager.scheduleNextBatch(this.prefillBudget);
    if (batch === null) return;

    const forwardInput = this._prepareBatch(batch);
    const forwardOutput = this._forward(forwardInput);

    // 保存结果到 _lastOverlapData，由下一 tick 的 Phase 3 处理
    this._lastOverlapData = { forwardInput, forwardOutput };
    const isGraphCapture = forwardOutput.isGraphCapture ?? false;
    if (isGraphCapture) {
      this._lastDataAckTick = this._tickCounter + this._overlapWaitTicks;
    } else {
      this._lastDataAckTick = this._tickCounter + this._overlapWaitTicks + this._eagerExtraDelayTicks;
    }
    this._lastDataPending = true;

    // 记录 GPU 占用
    const gpuDuration = (forwardOutput.prefillBatchTime ?? 0) + (forwardOutput.decodeBatchTime ?? 0);
    if (gpuDuration > 0) {
      this._clock?.scheduleGpu(gpuDuration);
    }
  }

  // ===== §9.11 调度循环核心方法 =====

  /**
   * _processOneMsg — 处理单条消息（对齐 §9.11 L3027-3052）
   * 四种消息类型：BatchSchedulerMsg / ExitMsg / UserMsg / AbortMsg
   */
  private _processOneMsg(msg: SchedulerMsg): void {
    switch (msg.tag) {
      case "batch": {
        const batchMsg = msg as BatchSchedulerMsg;
        for (const sub of batchMsg.data) {
          this._processOneMsg(sub);
        }
        break;
      }
      case "exit":
        throw new Error("ExitSignal");
      case "req_in": {
        const userMsg = msg as UserMsg;
        this._handleUserMsg(userMsg);
        break;
      }
      case "abort": {
        const abortMsg = msg as AbortMsg;
        this._handleAbortMsg(abortMsg);
        break;
      }
    }
  }

  /** 处理用户请求消息（§9.11 L3037-3046） */
  private _handleUserMsg(msg: UserMsg): void {
    const inputLen = msg.inputIds.length;
    const maxOutputLen = this.engine.maxSeqLen - inputLen;
    if (maxOutputLen <= 0) return;

    // 构造 SimRequestMsg 并加入 prefillManager
    const sp = msg.samplingParams ?? new SamplingParams({ maxNewTokens: msg.outputLen });
    if (sp.maxNewTokens > maxOutputLen) {
      (sp as { maxNewTokens: number }).maxNewTokens = maxOutputLen;
    }
    if (sp.maxNewTokens <= 0) return;

    // P6: 通过 addRequest 以支持 DP 分发
    this.addRequest({
      tag: "req_in",
      uid: msg.uid,
      inputIds: msg.inputIds,
      samplingParams: sp,
      outputLen: sp.maxNewTokens,
    });
  }

  /** 处理中止请求消息（§9.11 L3047-3052） */
  private _handleAbortMsg(msg: AbortMsg): void {
    const chunked = this.prefillManager.abortReq(msg.uid);
    if (chunked !== null) {
      this._freeReqResources(chunked);
      return;
    }
    const req = this.decodeManager.abortReq(msg.uid);
    if (req !== null) {
      this._freeReqResources(req);
    }
  }

  /**
   * _scheduleNextBatch — 调度下一个 batch（对齐 §9.11 L2984-2992）
   * prefill 优先，否则 decode
   */
  private _scheduleNextBatch(): ForwardInput | null {
    const batch = this.prefillManager.scheduleNextBatch(this.prefillBudget)
      || this.decodeManager.scheduleNextBatch();

    if (batch === null) return null;

    this.lastBatch = batch;
    return this._prepareBatch(batch);
  }

  /**
   * _prepareBatch — 准备 batch 元数据（对齐 §9.11 L2994-3010）
   * 1. padBatch
   * 2. allocatePaged
   * 3. 计算 positions
   * 4. 计算 input_tuple / write_tuple
   * 5. 计算 outLoc
   * 6. prepareMetadata
   * 7. 构造 ForwardInput
   */
  private _prepareBatch(batch: Batch): ForwardInput {
    // 1. padBatch（S4: 使用 simGraphRunner）
    this.engine.simGraphRunner.padBatch(batch);

    // 2. allocatePaged + 3-6 计算
    const inputTableIdx: number[] = [];
    const inputPositions: number[] = [];
    const writeTableIdx: number[] = [];
    const writePositions: number[] = [];
    const outLocList: number[] = [];

    for (const req of batch.reqs.values()) {
      const tableIdx = (req as unknown as { tableIdx: number }).tableIdx;
      const cachedLen = req.cachedLen;

      // allocatePaged
      this.cacheManager.allocatePaged({
        deviceLen: req.deviceLen,
        cachedLen: cachedLen,
        tableIdx: tableIdx,
      });

      // input positions: [cachedLen, deviceLen) for prefill; [deviceLen-1, deviceLen) for decode
      const isPrefill = req.extendLen > 0;
      if (isPrefill) {
        for (let pos = cachedLen; pos < req.deviceLen; pos++) {
          inputTableIdx.push(tableIdx);
          inputPositions.push(pos);
        }
      } else {
        inputTableIdx.push(tableIdx);
        inputPositions.push(req.deviceLen - 1);
      }

      // write mapping: for non-ChunkedReq, write at deviceLen position
      if (!(req instanceof ChunkedReq)) {
        writeTableIdx.push(tableIdx);
        writePositions.push(req.deviceLen);
        outLocList.push(req.deviceLen);
      } else {
        // ChunkedReq: no write position (skip in forward)
        writeTableIdx.push(-1);
        writePositions.push(-1);
        outLocList.push(-1);
      }
    }

    // Fill positions array on batch
    batch.positions = inputPositions;
    batch.outLoc = outLocList;

    // 6. prepareMetadata
    this.engine.mockAttnBackend.prepareMetadata(batch);

    // 7. Prepare sampleArgs
    const sampleArgs = this.engine.mockSampler.prepare(batch);

    return {
      batch,
      sampleArgs,
      inputTuple: [inputTableIdx, inputPositions],
      writeTuple: [writeTableIdx, writePositions],
    };
  }

  /**
   * _forward — 执行 forward（对齐 §9.11 L3012-3025）
   * 1. 从 tokenPool 读取 input_ids 填入 batch.inputIds
   * 2. 调用 engine.forward_batch（内部调用 forwardBatch 含完整并行层循环）
   * 3. 将 next_tokens 写入 tokenPool（跳过 ChunkedReq 位置：write_p < 0 时跳过）
   * 4. 调用 decodeManager.filterReqs
   */
  private _forward(forwardInput: ForwardInput): ForwardOutput {
    const { batch, sampleArgs } = forwardInput;
    const [inputTableIdx, inputPositions] = forwardInput.inputTuple;
    const [writeTableIdx, writePositions] = forwardInput.writeTuple;

    // 1. 从 tokenPool 读取 input_ids
    const inputIds: number[] = [];
    for (let i = 0; i < inputTableIdx.length; i++) {
      const tidx = inputTableIdx[i];
      const pos = inputPositions[i];
      if (tidx >= 0 && tidx < this.tokenPool.length && pos >= 0 && pos < this.tokenPool[tidx].length) {
        inputIds.push(this.tokenPool[tidx][pos]);
      } else {
        inputIds.push(0);
      }
    }
    batch.inputIds = inputIds;

    // 2. 调用 forward_batch（内部调用 forwardBatch 含完整并行层循环）
    const forwardOutput = this.engine.forward_batch(batch, sampleArgs);
    this.lastForwardOutput = forwardOutput;

    // 2b. 同步 CUDA Graph / Eager 计数器到 _simMetrics
    // （engine 内部的 simMetrics 是独立实例，此处同步到 Simulator 级别）
    if (this._simMetrics) {
      if (batch.extendInputTokens > 0) {
        this._simMetrics.recordEagerForward();
      }
      if (batch.numDecodeTokens > 0) {
        if (forwardOutput.isGraphCapture) {
          this._simMetrics.recordCudaGraphReplay();
        } else {
          this._simMetrics.recordEagerForward();
        }
      }
    }

    // 3. 将 next_tokens 写入 tokenPool
    const nextTokens = forwardOutput.nextTokensCpu;
    if (nextTokens !== null) {
      let tokenIdx = 0;
      const reqsArray = [...batch.reqs.values()];
      for (let i = 0; i < writeTableIdx.length; i++) {
        const wTidx = writeTableIdx[i];
        const wPos = writePositions[i];
        if (wTidx < 0) continue;  // ChunkedReq: skip
        if (tokenIdx < nextTokens.length) {
          const req = reqsArray[i];
          const writePos = wPos;
          if (wTidx < this.tokenPool.length && writePos >= 0 && writePos < this.tokenPool[wTidx].length) {
            this.tokenPool[wTidx][writePos] = nextTokens[tokenIdx];
          }
          tokenIdx++;
        }
      }
    }

    // 4. filterReqs
    this.decodeManager.filterReqs([...batch.reqs.values()]);

    return forwardOutput;
  }

  /**
   * _processLastData — 处理上一轮 forward 结果（对齐 §9.11 L2950-2982）
   * 1. ongoingData === null → return []
   * 2. copyDoneEvent.synchronize()
   * 3. lazy_free_region 上下文内遍历 batch.reqs
   * 4. 更新 finishedReqs
   */
  private _processLastData(
    ongoingData: [ForwardInput, ForwardOutput] | null,
  ): SimRespMsg[] {
    if (ongoingData === null) return [];

    const forwardInput = ongoingData[0];
    const forwardOutput = ongoingData[1];
    const batch = forwardInput.batch;

    // 2. copyDoneEvent.synchronize()（对齐 §9.11 L2956）
    forwardOutput.copyDoneEvent.synchronize();

    const reply: SimRespMsg[] = [];
    const newFinishedReqs: Set<Req> = new Set();

    // 3. lazy_free_region 上下文
    this.cacheManager.beginLazyFree();
    try {
      const nextTokens = forwardOutput.nextTokensCpu;
      let tokenIdx = 0;

      for (const req of batch.reqs.values()) {
        // ChunkedReq 跳过（对齐 §9.11 L2962-2963）
        if (req instanceof ChunkedReq) continue;
        if (req.finished) continue;

        const nextToken = nextTokens !== null && tokenIdx < nextTokens.length
          ? nextTokens[tokenIdx]
          : 0;
        tokenIdx++;

        // 追加 token
        req.appendHost(nextToken);

        // 判断 finished
        const isEos = !req.samplingParams.ignoreEos && nextToken === this.eosTokenId;
        const isFinished = !req.canDecode || isEos;

        // 构造响应
        reply.push({
          tag: isFinished ? "resp_done" : "resp_token",
          uid: req.rid,
          nextToken,
          finished: isFinished,
        });

        if (isFinished && !this.finishedReqs.has(req)) {
          req.finished = true;
          req.finishReason = isEos ? "eos" : "length";
          this.decodeManager.removeReq(req);
          this._freeReqResources(req);
          newFinishedReqs.add(req);
        } else if (!isFinished && batch.extendInputTokens > 0) {
          // 非 finished 的 prefill 请求调用 cacheReq(finished=false)
          this.cacheManager.cacheReq({
            inputIds: req.inputIds,
            cachedLen: req.cachedLen,
            tableIdx: (req as unknown as { tableIdx: number }).tableIdx,
            cacheHandle: (req as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle,
          }, false);
        }
      }
    } finally {
      this.cacheManager.endLazyFree();
    }

    // 4. 更新 finishedReqs（对齐 §9.11 L2978）
    this.finishedReqs = newFinishedReqs;

    return reply;
  }

  /**
   * _freeReqResources — 释放请求资源（对齐 §9.11 L3054-3057）
   */
  private _freeReqResources(req: Req): void {
    const tableIdx = (req as unknown as { tableIdx: number }).tableIdx;
    this.tableManager.free(tableIdx);
    this.cacheManager.freeCache({
      inputIds: req.inputIds,
      cachedLen: req.cachedLen,
      tableIdx: tableIdx,
      cacheHandle: (req as unknown as { cacheHandle: BaseCacheHandle | null }).cacheHandle,
    });
  }

  // ===== P6: EPLB 集成 =====

  /**
   * EPLB maybe_rebalance 调用
   *
   * 在 tick 末尾调用（§10.4.4），条件：
   * - groups.eplbSim 非 null
   * - groups.moeBackend 非 null
   */
  private _maybeEplbRebalance(): void {
    if (this._groups?.eplbSim && this._groups.moeBackend) {
      const result = this._groups.eplbSim.maybe_rebalance(
        this._globalStep,
        this._groups.moeBackend.expertLoadCounts,
        this._groups.moeBackend,
      );
      if (result.shouldRebalance && this._simMetrics) {
        this._simMetrics.parallel.epRebalanceCostTicks += result.rebalanceTicks;
      }
    }
  }
}

// ===== P6 向后兼容：SimSchedulerImpl 作为 SimScheduler 的类型别名 =====
/** @deprecated 使用 SimScheduler 代替 */
export const SimSchedulerImpl = SimScheduler;
