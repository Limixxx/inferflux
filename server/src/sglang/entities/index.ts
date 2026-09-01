// entities — S1: ChunkedReq/PendingReq

import { Req, SamplingParams } from "../core";

// ===== ChunkedReq 类（§9.11） =====

/**
 * Chunked prefill 请求（对应 §9.11）
 * 继承 Req，覆盖 canDecode 返回 false，appendHost 抛出错误
 */
export class ChunkedReq extends Req {
  get canDecode(): boolean {
    return false;
  }

  appendHost(_nextToken: number): never {
    throw new Error("ChunkedReq should not be sampled");
  }
}

// ===== PendingReq 数据结构（§9.1） =====

/** PendingReq 构造选项 */
export interface PendingReqOpts {
  rid: number;
  inputIds: number[];
  samplingParams: SamplingParams;
  priority?: number;
  nextScheduledTime?: number;
  chunkedReq?: ChunkedReq | null;
  dpRank?: number;
}

/**
 * PrefillManager 内部的待处理请求（对应 §9.1）
 * chunkedReq 非 null 时表示上一 tick chunked 的请求，需续接
 */
export class PendingReq {
  readonly rid: number;
  readonly priority: number;
  nextScheduledTime: number;
  inputIds: number[];
  samplingParams: SamplingParams;
  chunkedReq: ChunkedReq | null;
  dpRank: number;

  constructor(opts: PendingReqOpts) {
    this.rid = opts.rid;
    this.priority = opts.priority ?? 0;
    this.nextScheduledTime = opts.nextScheduledTime ?? 0;
    this.inputIds = [...opts.inputIds];
    this.samplingParams = opts.samplingParams;
    this.chunkedReq = opts.chunkedReq ?? null;
    this.dpRank = opts.dpRank ?? 0;
  }

  /** 输入 token 长度 */
  get inputLen(): number {
    return this.inputIds.length;
  }

  /** 预期输出长度（来自 samplingParams.maxNewTokens） */
  get outputLen(): number {
    return this.samplingParams.maxNewTokens;
  }
}
