// scheduler — K1: TableManager + S2: PrefillAdder/PrefillManager/DecodeManager + S3: SchedulerIOMixin/SimScheduler

import { Req, Batch, SamplingParams, BatchSamplingArgs } from "../core";
import { ChunkedReq, PendingReq } from "../entities";
import { CacheManager } from "../cache";
import type { BaseCacheHandle } from "../cache";
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

// ===== S3: SimScheduler（§9.11 L2888-3058） =====

/**
 * SimScheduler — 仿真调度器（§9.11 L2888-3058）
 *
 * 继承 SchedulerIOMixin，实现完整的 normal_tick 调度循环：
 * _processOneMsg → _scheduleNextBatch → _forward → _processLastData
 *
 * S3 仅实现 normal_tick；overlap 为 S5 范围
 * overlapEnabled=true 时降级为 normal_tick，确保 end-to-end 可跑通
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

  constructor(config: SimulatorConfig) {
    super(config);
    this.engine = new MockEngine(config);
    this.tableManager = new TableManager(config.maxRunningReq, this.engine.pageTable);
    this.cacheManager = new CacheManager(
      this.engine.numPages, config.pageSize, this.engine.pageTable, config.cacheType
    );
    this.decodeManager = new DecodeManager(config.pageSize);
    this.prefillManager = new PrefillManager(
      this.cacheManager, this.tableManager, this.decodeManager
    );
    this.finishedReqs = new Set();
    this.eosTokenId = config.eosTokenId;
    this.tokenPool = this.tableManager.tokenPool;
    this.prefillBudget = config.maxExtendTokens;
    this.overlapEnabled = config.enableOverlap;
    this.lastBatch = null;
  }

  /**
   * 执行一个调度 tick
   * S3: 仅实现 normal_tick；overlap 为 S5 范围
   * 当 overlapEnabled=true 时降级为 normal_tick，确保 end-to-end 可跑通
   */
  runTick(incoming: SimRequestMsg[]): SimRespMsg[] {
    return this._normalTick(incoming);
  }

  /**
   * _normalTick — 串行调度循环（对齐 §9.11 L2916-2924）
   * 1. for msg in incoming: _processOneMsg(msg)
   * 2. forwardInput = _scheduleNextBatch()
   * 3. ongoingData = null
   * 4. if forwardInput !== null: ongoingData = [forwardInput, _forward(forwardInput)]
   * 5. return _processLastData(ongoingData)
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
    let ongoingData: [ForwardInput, import("../core").ForwardOutput] | null = null;
    if (forwardInput !== null) {
      const forwardOutput = this._forward(forwardInput);
      ongoingData = [forwardInput, forwardOutput];
    }

    // 5. 结果处理
    return this._processLastData(ongoingData);
  }

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

    this.prefillManager.addOneReq({
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
   * 2. 调用 engine.forward_batch
   * 3. 将 next_tokens 写入 tokenPool（跳过 ChunkedReq 位置：write_p < 0 时跳过）
   * 4. 调用 decodeManager.filterReqs
   */
  private _forward(forwardInput: ForwardInput): import("../core").ForwardOutput {
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

    // 2. 调用 forward_batch
    const forwardOutput = this.engine.forward_batch(batch, sampleArgs);

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
    ongoingData: [ForwardInput, import("../core").ForwardOutput] | null,
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
}
