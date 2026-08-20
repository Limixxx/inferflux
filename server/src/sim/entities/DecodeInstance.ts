import { SimParams, SimRequest, ISimEngine } from "../../shared/types";
import { cellSizeOf, clamp } from "../../shared/utils";
import { NTR_MAX, NTR_CLIP, NTR_DECAY_STEPS, PERREQ_REF_CTX } from "../../shared/constants";

/** A single Decode server instance with its own queues, KV pool, and batch. */
export class DecodeInstance {
  id: number;
  preallocQ: SimRequest[] = [];
  transferQ: SimRequest[] = [];
  waitingQ: SimRequest[] = [];
  running: SimRequest[] = [];
  kvUsed = 0;
  nextStepAt: number | null = null;
  draining = false;
  ntr = 0;
  retractGlow = -1e9;

  constructor(id: number) {
    this.id = id;
  }

  maxTokens(P: SimParams): number {
    return Math.max(1, Math.floor(P.kvGbD * 2**30 / cellSizeOf(P)));
  }

  /** Decode step latency: base + per-req KV read (scales with context length). */
  decodeStepMs(P: SimParams): number {
    const perTokMs = P.decodeMsPerReq / PERREQ_REF_CTX;
    let kvMs = 0;
    for (const r of this.running) kvMs += perTokMs * (r.inputLen + r.tokensOut);
    return P.decodeMsBase + kvMs;
  }

  pendingLoad(): number {
    return this.preallocQ.length + this.transferQ.length +
           this.waitingQ.length + this.running.length;
  }

  isEmpty(): boolean {
    return this.preallocQ.length === 0 && this.transferQ.length === 0 &&
           this.waitingQ.length === 0 && this.running.length === 0;
  }

  /** KV headroom reserved for in-flight decodes (SGLang rem_total_token_offset). */
  reservedOffset(): number {
    let off = 0;
    for (const r of this.running)
      off += Math.min(r.outputLen - r.tokensOut, NTR_CLIP) * this.ntr;
    return off;
  }

  /** SGLang estimate_new_token_ratio_after_retract. */
  raiseNtrAfterRetract(): void {
    let decoded = 0, projected = 0;
    for (const r of this.running) {
      decoded += r.tokensOut;
      projected += Math.min(r.outputLen, NTR_CLIP);
    }
    const frac = projected > 0 ? decoded / projected : 0;
    this.ntr = clamp(Math.max(this.ntr, 1 - frac), this.ntr, NTR_MAX);
  }

  /** SGLang retract_decode: evict the request with fewest decoded tokens. */
  retractDecode(P: SimParams, engine: ISimEngine, now: number): number {
    const cap = this.maxTokens(P);
    let count = 0;
    while (this.running.length > 1 && this.kvUsed + this.running.length > cap) {
      let vi = 0;
      for (let i = 1; i < this.running.length; i++) {
        const a = this.running[i], b = this.running[vi];
        if (a.tokensOut < b.tokensOut ||
            (a.tokensOut === b.tokensOut && a.inputLen > b.inputLen)) vi = i;
      }
      const victim = this.running[vi];
      this.kvUsed -= (victim.inputLen + victim.tokensOut);
      this.running.splice(vi, 1);
      victim.retracted = true;
      victim.stage = "d_waiting";
      this.waitingQ.unshift(victim);
      count++;
    }
    if (count) {
      this.raiseNtrAfterRetract();
      this.retractGlow = now;
      engine.retractTotal += count;
    }
    return count;
  }
}
