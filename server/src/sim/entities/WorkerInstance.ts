import { SimParams, SimRequest, ISimEngine } from "../../shared/types";
import { cellSizeOf, clamp, fullPrefillMs, chunkPrefillMs } from "../../shared/utils";
import { NTR_MAX, NTR_CLIP, NTR_DECAY_STEPS, PERREQ_REF_CTX } from "../../shared/constants";

/**
 * WorkerInstance — agg-mode unified worker.
 *
 * Models SGLang's make_batch mixed-batch scheduling: a single `running` array
 * holds both prefill (w_prefill / w_chunked_prefill) and decode (w_decode)
 * requests that share one GPU iteration. Step latency is
 *   step = max(prefill_compute, decode_step)
 * mirroring the overlap of compute-bound prefill and memory-bound decode on
 * the same GPU. There is no TransferLink and no P/D handshake; prefill
 * completes in-place and the request immediately transitions to decode,
 * reusing the same KV pool.
 */
export class WorkerInstance {
  id: number;
  waitingQ: SimRequest[] = [];      // queued for make_batch admission
  running: SimRequest[] = [];       // unified batch: w_prefill / w_chunked_prefill / w_decode
  kvUsed = 0;
  draining = false;
  ntr = 0;
  retractGlow = -1e9;
  nextStepAt: number | null = null;

  constructor(id: number) {
    this.id = id;
  }

  maxTokens(P: SimParams): number {
    return Math.max(1, Math.floor(P.kvGb * 2**30 / cellSizeOf(P)));
  }

  /** Decode step latency — counts only w_decode requests' KV read cost. */
  decodeStepMs(P: SimParams): number {
    const perTokMs = P.decodeMsPerReq / PERREQ_REF_CTX;
    let kvMs = 0;
    for (const r of this.running)
      if (r.stage === "w_decode") kvMs += perTokMs * (r.inputLen + r.tokensOut);
    return P.decodeMsBase + kvMs;
  }

  /**
   * Unified step latency (one GPU iteration of make_batch):
   *  - If batch contains prefill requests: step = max(prefill_compute, decode_step)
   *    (prefill is compute-bound, decode is memory-bound → they overlap on GPU)
   *  - If batch is decode-only: step = decode_step
   */
  stepLatencyMs(P: SimParams): number {
    let prefillMs = 0;
    for (const r of this.running) {
      if (r.stage === "w_prefill") {
        prefillMs = Math.max(prefillMs, fullPrefillMs(P, r));
      } else if (r.stage === "w_chunked_prefill") {
        prefillMs = Math.max(prefillMs, chunkPrefillMs(P, r, r.chunkOffset));
      }
    }
    const decMs = this.decodeStepMs(P);
    return prefillMs > 0 ? Math.max(prefillMs, decMs) : decMs;
  }

  pendingLoad(): number {
    return this.waitingQ.length + this.running.length;
  }

  isEmpty(): boolean {
    return this.waitingQ.length === 0 && this.running.length === 0;
  }

  /** KV headroom reserved for in-flight decodes (mirrors DecodeInstance). */
  reservedOffset(): number {
    let off = 0;
    for (const r of this.running)
      if (r.stage === "w_decode")
        off += Math.min(r.outputLen - r.tokensOut, NTR_CLIP) * this.ntr;
    return off;
  }

  /** SGLang estimate_new_token_ratio_after_retract. */
  raiseNtrAfterRetract(): void {
    let decoded = 0, projected = 0;
    for (const r of this.running) {
      if (r.stage !== "w_decode") continue;
      decoded += r.tokensOut;
      projected += Math.min(r.outputLen, NTR_CLIP);
    }
    const frac = projected > 0 ? decoded / projected : 0;
    this.ntr = clamp(Math.max(this.ntr, 1 - frac), this.ntr, NTR_MAX);
  }

  /**
   * SGLang retract_decode: evict the w_decode request with fewest decoded tokens.
   * Prefill requests (w_prefill / w_chunked_prefill) are NOT evictable — they
   * are mid-computation on the GPU.
   */
  retractDecode(P: SimParams, engine: ISimEngine, now: number): number {
    const cap = this.maxTokens(P);
    let count = 0;
    while (this.running.length > 1 && this.kvUsed + this.running.length > cap) {
      let vi = -1;
      for (let i = 0; i < this.running.length; i++) {
        const a = this.running[i];
        if (a.stage !== "w_decode") continue;
        if (vi === -1) { vi = i; continue; }
        const b = this.running[vi];
        if (a.tokensOut < b.tokensOut ||
            (a.tokensOut === b.tokensOut && a.inputLen > b.inputLen)) vi = i;
      }
      if (vi === -1) break;  // no evictable decode request
      const victim = this.running[vi];
      this.kvUsed -= (victim.inputLen + victim.tokensOut);
      this.running.splice(vi, 1);
      victim.retracted = true;
      victim.stage = "w_waiting";
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
