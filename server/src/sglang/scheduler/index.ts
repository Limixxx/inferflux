// scheduler — K1: TableManager + S2: PrefillAdder/PrefillManager/DecodeManager
// P6: SimScheduler 实现（ParallelGroups 接入 + EPLB 集成 + globalStep）

import { Req, Batch, SamplingParams } from "../core";
import { ChunkedReq, PendingReq } from "../entities";
import type { CacheManager } from "../cache";
import type { BaseCacheHandle } from "../cache";
import type { SimRequestMsg, SimRespMsg } from "../types";
import type { ParallelGroups } from "../parallel/groups";
import type { SimulationMetrics } from "../metrics";

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

// ===== P6: SimScheduler 实现 =====

/**
 * SimScheduler — 仿真调度器（P6 ParallelGroups 集成）
 *
 * 在 S1 的调度逻辑基础上，接入 ParallelGroups：
 * - 构造器接收 optional ParallelGroups
 * - add_request 路径中集成 dpController.select_rank_for_request
 * - tick 末尾调用 EPLB maybe_rebalance（§10.4.4）
 * - 维护 _globalStep 供 EPLB 使用
 */
export class SimSchedulerImpl {
  private readonly _config: import("../types").SimulatorConfig;
  private readonly _prefillManager: PrefillManager;
  private readonly _decodeManager: DecodeManager;
  private readonly _groups: ParallelGroups | null;
  private readonly _simMetrics: SimulationMetrics | null;
  private _globalStep: number = 0;

  constructor(
    config: import("../types").SimulatorConfig,
    prefillManager: PrefillManager,
    decodeManager: DecodeManager,
    parallelGroups?: ParallelGroups,
    simMetrics?: SimulationMetrics,
  ) {
    this._config = config;
    this._prefillManager = prefillManager;
    this._decodeManager = decodeManager;
    this._groups = parallelGroups ?? null;
    this._simMetrics = simMetrics ?? null;
  }

  /** 当前全局步数 */
  get globalStep(): number {
    return this._globalStep;
  }

  /** 获取 ParallelGroups（只读） */
  get groups(): ParallelGroups | null {
    return this._groups;
  }

  /**
   * 添加请求到调度器
   *
   * P6 扩展：若 dpSize > 1，先通过 dpController.select_rank_for_request 分配 DP rank
   */
  addRequest(msg: SimRequestMsg): void {
    // DP 请求分发
    if (this._groups && this._config.dpSize > 1) {
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
      this._prefillManager.pendingList.push(pr);
      return;
    }
    this._prefillManager.addOneReq(msg);
  }

  /**
   * 执行一个调度 tick
   * @param incoming 进入的请求消息列表
   * @returns 响应消息列表
   */
  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    if (this._config.enableOverlap) {
      return this._overlap_tick(incoming);
    }
    return this._normal_tick(incoming);
  }

  /**
   * Normal（非 overlap）tick
   *
   * 步骤：
   * 1. 处理 incoming 消息 → add_request
   * 2. 调度 prefill/decode batch
   * 3. forward + 结果处理
   * 4. EPLB maybe_rebalance（tick 末尾）
   * 5. 递增 globalStep
   */
  private _normal_tick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // 步骤 1：处理入队消息
    for (const msg of incoming) {
      this.addRequest(msg);
    }

    // 步骤 2 & 3：调度 + forward + 结果处理（桩实现，后续 S1 Issue 完善）
    const replies: SimRespMsg[] = [];

    // 步骤 4：EPLB maybe_rebalance（§10.4.4）
    this._maybeEplbRebalance();

    // 步骤 5：递增 globalStep
    this._globalStep += 1;

    return replies;
  }

  /**
   * Overlap tick
   *
   * 与 _normal_tick 类似，但支持 overlap scheduling。
   * 步骤末尾同样调用 EPLB + 递增 globalStep。
   */
  private _overlap_tick(incoming: SimRequestMsg[]): SimRespMsg[] {
    // 步骤 1：处理入队消息
    for (const msg of incoming) {
      this.addRequest(msg);
    }

    // 步骤 2 & 3：overlap 调度 + forward + 结果处理（桩实现，后续 S1 Issue 完善）
    const replies: SimRespMsg[] = [];

    // 步骤 4：EPLB maybe_rebalance（§10.4.4）
    this._maybeEplbRebalance();

    // 步骤 5：递增 globalStep
    this._globalStep += 1;

    return replies;
  }

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
