// core — S1: SamplingParams/Req/Batch + 工具函数

// ===== 工具函数（§3.4.1 / §9.2） =====

/** 将 n 向下对齐到 alignment 的倍数 */
export function alignDown(n: number, alignment: number): number {
  return n - (n % alignment);
}

/** 向上取整除法 */
export function divCeil(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

/**
 * 将 a 均分到 b 份，返回每份大小列表。
 * allowReplicate=true 时允许 a < b（部分份为 0，用于 TP 下 head 复制）。
 * allowReplicate=false 时要求 a >= b，否则抛出 Error。
 * 例: divEven(8, 3) → [3, 3, 2]
 */
export function divEven(a: number, b: number, allowReplicate: boolean = false): number[] {
  if (a === 0) return new Array(b).fill(0);
  if (!allowReplicate && a < b) {
    throw new Error(`divEven(${a}, ${b}) with allowReplicate=false requires a >= b`);
  }
  const base = Math.floor(a / b);
  const remainder = a % b;
  return [...Array(remainder).fill(base + 1), ...Array(b - remainder).fill(base)];
}

/** 返回每种 dtype 每个元素占用的字节数 */
export function bytesPerElement(dtype: "float32" | "float16" | "bfloat16"): number {
  switch (dtype) {
    case "float32": return 4;
    case "float16": return 2;
    case "bfloat16": return 2;
  }
}

// ===== SamplingParams 类（§9.2） =====

/** 采样参数的数据类型 */
export type SamplingDtype = "float32" | "float16" | "bfloat16";

/** 采样参数构造选项 */
export interface SamplingParamsOpts {
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  frequencyPenalty?: number;
  repetitionPenalty?: number;
  minP?: number;
  stopTokenIds?: number[];
  skipSpecialTokens?: boolean;
  dtype?: SamplingDtype;
}

/**
 * 采样参数（对应 §2.2.1 / §9.2）
 * S1 将 S0 的 interface 升级为完整 class 实现
 */
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
  readonly dtype: SamplingDtype;

  constructor(opts?: SamplingParamsOpts) {
    this.maxNewTokens = opts?.maxNewTokens ?? 1024;
    this.temperature = opts?.temperature ?? 0.0;
    this.topP = opts?.topP ?? 1.0;
    this.topK = opts?.topK ?? -1;
    this.frequencyPenalty = opts?.frequencyPenalty ?? 0.0;
    this.repetitionPenalty = opts?.repetitionPenalty ?? 1.0;
    this.minP = opts?.minP ?? 0.0;
    this.stopTokenIds = opts?.stopTokenIds ?? [];
    this.skipSpecialTokens = opts?.skipSpecialTokens ?? true;
    this.dtype = opts?.dtype ?? "float16";
  }

  /** 是否为贪婪采样：temperature≤0 或 topK=1 且 topP=1.0 */
  get isGreedy(): boolean {
    return (this.temperature <= 0 || this.topK === 1) && this.topP === 1.0;
  }
}

// ===== Req 类（§2.2.2 / §9.2） =====

/** Req 构造选项 */
export interface ReqOpts {
  rid: number;
  inputIds: number[];
  samplingParams: SamplingParams;
  originInputLen?: number;
  promptLogprobStartPos?: number;
}

/**
 * 请求状态（对应 §2.2.2 / §9.2）
 * deviceLen 和 maxDeviceLen 是普通可变属性，非 getter
 */
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

  // 可变属性（非 getter），与 §2.2.2 一致
  deviceLen: number;
  maxDeviceLen: number;

  constructor(opts: ReqOpts) {
    this.rid = opts.rid;
    this.originInputLen = opts.originInputLen ?? opts.inputIds.length;
    this.inputIds = [...opts.inputIds];
    this.outputIds = [];
    this.promptLogprobStartPos = opts.promptLogprobStartPos ?? -1;
    this.samplingParams = opts.samplingParams;
    this.finished = false;
    this.finishReason = null;
    this.samplingCounter = 0;
    this.maxNewTokens = opts.samplingParams.maxNewTokens;

    this.deviceLen = opts.inputIds.length;
    this.maxDeviceLen = opts.inputIds.length + opts.samplingParams.maxNewTokens;

    this.dpRank = 0;
  }

  /** 剩余可解码长度 */
  get remainLen(): number {
    return this.maxDeviceLen - this.deviceLen;
  }

  /** 需要扩展的长度（外部 cachedLen 管理） */
  get extendLen(): number {
    return this.deviceLen - this.cachedLen;
  }

  /** 是否可以继续 decode */
  get canDecode(): boolean {
    return this.remainLen > 0;
  }

  /**
   * cachedLen — 由外部 CacheManager/TableManager 管理
   * Req 本身不持有此状态，默认返回 deviceLen（无缓存场景）
   * 子类 ChunkedReq 可能覆写
   */
  get cachedLen(): number {
    return this.deviceLen;
  }

  /** 完成一个 token 的 decode（在采样前调用） */
  completeOne(): void {
    this.deviceLen += 1;
    this.samplingCounter += 1;
  }

  /** 追加新 token（采样后调用） */
  appendHost(nextToken: number): void {
    this.inputIds = [...this.inputIds, nextToken];
    this.outputIds.push(nextToken);
  }
}

// ===== Batch 类（§2.2.3 / §9.2） =====

/**
 * 批处理（对应 §2.2.3 / §9.2）
 * reqs 使用 Map<number, Req> 按 rid 索引
 */
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

  constructor() {
    this.reqs = new Map();
    this.initLen = 0;
    this.promptTokens = 0;
    this.extendInputTokens = 0;
    this.extendOutputTokens = 0;
    this.numDecodeTokens = 0;
    this.hasIdleReqs = false;
    this.readyIds = [];
    this.nextId = 0;
    this.schedulerThinkingBatch = false;
  }

  /** 从 readyIds 取下一个就绪请求 */
  nextReadyReq(): Req | undefined {
    if (this.readyIds.length === 0) return undefined;
    const id = this.readyIds.shift()!;
    return this.reqs.get(id);
  }

  /** 从 readyIds 取下一个请求但不移除 */
  nextBatchReq(): Req | undefined {
    if (this.readyIds.length === 0) return undefined;
    const id = this.readyIds[0];
    return this.reqs.get(id);
  }
}
