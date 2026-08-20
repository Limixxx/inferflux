import { SimParams, SimRequest, KVPOLL, ISimEngine, Gauges } from "../shared/types";
import { TICK, HANDSHAKE_RTT_MS, NTR_MAX, NTR_CLIP, NTR_DECAY_STEPS } from "../shared/constants";
import { mulberry32, expSample, RNG } from "../shared/rng";
import { clamp, cellSizeOf, chunkPrefillMs, chunkTokens } from "../shared/utils";
import { PrefillInstance } from "./entities/PrefillInstance";
import { DecodeInstance } from "./entities/DecodeInstance";
import { makeRequest, resetReqSeq } from "./entities/Request";
import { selectByPolicy, RRCounter } from "./LoadBalancer";
import { MetricsCollector } from "./MetricsCollector";
import { DEFAULTS } from "../shared/presets";

/**
 * SimEngine — the core PD-disaggregation simulation engine.
 * Mirrors SGLang's real PD-disaggregation request lifecycle:
 *   tokenizer → router(bootstrap_room) →
 *   P: PrefillBootstrapQueue → waiting_queue → chunked prefill → inflight/KV-transfer
 *   D: DecodePreallocQueue → DecodeTransferQueue → waiting_queue → prebuilt(first
 *      token) → decode_loop (continuous batching) → detokenizer
 */
export class SimEngine implements ISimEngine {
  P: SimParams;
  rng: RNG;
  metrics: MetricsCollector;
  now = 0;
  pList: PrefillInstance[] = [];
  dList: DecodeInstance[] = [];
  pSeq = 0;
  dSeq = 0;
  responding: SimRequest[] = [];
  allActive: Set<SimRequest> = new Set();
  rrP: RRCounter = { i: 0 };
  rrD: RRCounter = { i: 0 };
  nextArrival = 0;
  linkBusyMs = 0;
  lastGaugeAt = 0;
  inflightIntegral = 0;
  debugMaxKvRatio = 0;
  tokenConservationOk = true;
  retractTotal = 0;

  constructor(P: SimParams, seed = 12345) {
    this.P = P;
    this.rng = mulberry32(seed);
    this.metrics = new MetricsCollector();
    this.reset();
  }

  reset(): void {
    this.now = 0;
    this.pList = [];
    this.dList = [];
    this.pSeq = 0;
    this.dSeq = 0;
    this.responding = [];
    this.allActive = new Set();
    this.rrP = { i: 0 };
    this.rrD = { i: 0 };
    this.nextArrival = this.P.arrivalDist === "uniform"
      ? 1000 / Math.max(0.01, this.P.qps)
      : expSample(this.rng, 1000 / Math.max(0.01, this.P.qps));
    this.linkBusyMs = 0;
    this.lastGaugeAt = 0;
    this.inflightIntegral = 0;
    this.debugMaxKvRatio = 0;
    this.tokenConservationOk = true;
    this.retractTotal = 0;
    resetReqSeq();
    this.metrics.reset(0);
    this.syncTopology();
  }

  syncTopology(): void {
    const P = this.P;
    while (this.pList.length < P.numP) this.pList.push(new PrefillInstance(this.pSeq++));
    while (this.dList.length < P.numD) this.dList.push(new DecodeInstance(this.dSeq++));
    for (let i = 0; i < this.pList.length; i++) this.pList[i].draining = i >= P.numP;
    for (let i = 0; i < this.dList.length; i++) this.dList[i].draining = i >= P.numD;
    while (this.pList.length > P.numP && this.pList[this.pList.length - 1].isEmpty())
      this.pList.pop();
    while (this.dList.length > P.numD && this.dList[this.dList.length - 1].isEmpty())
      this.dList.pop();
  }

  advance(dtSim: number): void {
    const target = this.now + dtSim;
    while (this.now < target) {
      this.now = Math.min(target, this.now + TICK);
      this.step();
    }
  }

  step(): void {
    const P = this.P, now = this.now;
    this.syncTopology();

    /* -- arrivals → router fans out to P AND D concurrently -- */
    while (this.nextArrival <= now) {
      const r = makeRequest(this, this.nextArrival);
      this.allActive.add(r);
      const activeP = this.pList.filter(p => !p.draining);
      const activeD = this.dList.filter(d => !d.draining);
      const best = selectByPolicy(P.lbPolicyP, activeP, p => p.pendingLoad(), this.rng, this.rrP);
      const d = selectByPolicy(P.lbPolicyD, activeD, x => x.pendingLoad(), this.rng, this.rrD);
      r.p = best; r.d = d;
      const tokMs = r.inputLen * P.tokenizeUsPerTok / 1000;
      r.stamps.tokenized = this.nextArrival + tokMs;
      r.stage = "p_bootstrap"; r.kvPoll = KVPOLL.Bootstrapping;
      r.readyAt = r.stamps.tokenized + HANDSHAKE_RTT_MS;
      r.dReadyAt = r.stamps.tokenized + HANDSHAKE_RTT_MS;
      best.bootstrapQ.push(r);
      d.preallocQ.push(r);
      this.nextArrival += P.arrivalDist === "uniform"
        ? 1000 / Math.max(0.01, P.qps)
        : expSample(this.rng, 1000 / Math.max(0.01, P.qps));
    }

    /* -- Prefill instances -- */
    for (const p of this.pList) {
      const ready = p.bootstrapQ.filter(r => r.readyAt <= now && r.dPrealloc);
      if (ready.length) {
        const rdy = new Set(ready);
        for (const r of ready) {
          r.stamps.bootstrapDone = now;
          r.stage = "p_waiting"; r.kvPoll = KVPOLL.WaitingForInput;
          p.waitingQ.push(r);
        }
        p.bootstrapQ = p.bootstrapQ.filter(r => !rdy.has(r));
      }
      while (p.slots.length < 1 && p.waitingQ.length) {
        const r = p.waitingQ[0];
        const cap = p.maxTokens(P);
        if (p.kvUsed + r.inputLen > cap) break;
        p.waitingQ.shift();
        p.kvUsed += r.inputLen;
        this.debugMaxKvRatio = Math.max(this.debugMaxKvRatio, p.kvUsed / cap);
        r.stamps.pQueueExit = now;
        r.stage = "p_prefill";
        r.chunksTotal = Math.max(1, Math.ceil(r.uncachedLen / P.chunkSize));
        r.bytesTotal = r.uncachedLen * cellSizeOf(P);
        p.slots.push({ req: r, busyUntil: now + chunkPrefillMs(P, r, 0) });
      }
      for (let si = p.slots.length - 1; si >= 0; si--) {
        const slot = p.slots[si];
        while (slot.busyUntil <= now) {
          const r = slot.req;
          r.chunksComputed++;
          this.enqueueReadyChunks(p, r);
          if (r.chunksComputed >= r.chunksTotal) {
            r.stamps.prefillDone = slot.busyUntil;
            r.stage = "p_transfer"; r.kvPoll = KVPOLL.Transferring;
            p.inflight.push(r);
            p.slots.splice(si, 1);
            break;
          } else {
            slot.busyUntil += chunkPrefillMs(P, r, r.chunksComputed);
          }
        }
      }
      const link = p.link;
      for (;;) {
        if (link.current && link.doneAt <= now) {
          const it = link.current;
          it.req.chunksTransferred++;
          it.req.bytesDone += it.bytes;
          link.current = null;
          if (it.req.chunksTransferred >= it.req.chunksTotal && it.req.stage === "p_transfer")
            this.completeTransfer(p, it.req, it.doneAtWas!);
        }
        if (!link.current && link.queue.length) {
          const it = link.queue.shift()!;
          const dur = P.transferOverheadMs + it.bytes / (P.bandwidthGBs * 2**30) * 1000;
          link.current = it;
          link.startAt = Math.max(now - TICK, link.doneAt);
          link.doneAt = link.startAt + dur;
          it.doneAtWas = link.doneAt;
          continue;
        }
        break;
      }
      if (link.current) this.linkBusyMs += TICK;
    }

    /* -- Decode instances -- */
    for (const d of this.dList) {
      if (d.preallocQ.length && d.preallocQ.some(r => r.dReadyAt <= now)) {
        const still: SimRequest[] = [];
        let blocked = false;
        const cap = d.maxTokens(P);
        for (const r of d.preallocQ) {
          if (blocked || r.dReadyAt > now) { still.push(r); continue; }
          const projNew = Math.min(r.outputLen, NTR_CLIP);
          const need = r.inputLen + projNew;
          const rem = cap - d.kvUsed - d.reservedOffset();
          if (need > rem) { still.push(r); blocked = true; continue; }
          d.kvUsed += r.inputLen;
          this.debugMaxKvRatio = Math.max(this.debugMaxKvRatio, d.kvUsed / cap);
          r.dPrealloc = true;
          r.stamps.preallocDone = now;
          d.transferQ.push(r);
          this.enqueueReadyChunks(r.p, r);
        }
        d.preallocQ.length = 0;
        for (const r of still) d.preallocQ.push(r);
      }

      if (d.ntr < P.newTokenRatio) d.ntr = P.newTokenRatio;
      const admit = (at: number) => {
        const cap = d.maxTokens(P);
        while (d.waitingQ.length && d.running.length < P.maxRunning) {
          const r = d.waitingQ[0];
          const joinKv = r.retracted ? (r.inputLen + r.tokensOut) : 1;
          if (d.kvUsed + joinKv + (d.running.length + 1) > cap) break;
          d.waitingQ.shift();
          d.kvUsed += joinKv;
          this.debugMaxKvRatio = Math.max(this.debugMaxKvRatio, d.kvUsed / cap);
          r.stamps.dQueueExit = at;
          r.stage = "d_running";
          if (!r.retracted) {
            r.tokensOut = 1;
            r.stamps.firstToken = at + P.detokenizeMs;
            this.metrics.bkTokens++; this.metrics.totalTokens++;
          } else {
            r.retracted = false;
          }
          r.lastTokenT = at;
          d.running.push(r);
        }
      };
      if (d.running.length === 0 && d.waitingQ.length) {
        admit(now);
        d.nextStepAt = now + d.decodeStepMs(P);
      }
      if (d.running.length === 0) {
        d.nextStepAt = null;
        if (d.isEmpty()) d.ntr = P.newTokenRatio;
        continue;
      }
      while (d.nextStepAt !== null && d.nextStepAt <= now) {
        const stepAt = d.nextStepAt;
        const cap = d.maxTokens(P);
        const retracted = d.retractDecode(P, this, stepAt);
        if (d.running.length === 0) { d.nextStepAt = null; break; }
        for (let i = d.running.length - 1; i >= 0; i--) {
          const r = d.running[i];
          r.tokensOut++;
          d.kvUsed += 1;
          r.lastTokenT = stepAt;
          this.metrics.bkTokens++; this.metrics.totalTokens++;
          if (r.tokensOut >= r.outputLen) {
            r.stamps.lastToken = stepAt + P.detokenizeMs;
            d.running.splice(i, 1);
            d.kvUsed -= (r.inputLen + r.tokensOut);
            if (r.tokensOut !== r.outputLen) this.tokenConservationOk = false;
            r.stage = "response"; r.readyAt = stepAt + P.detokenizeMs;
            this.responding.push(r);
          }
        }
        this.debugMaxKvRatio = Math.max(this.debugMaxKvRatio, d.kvUsed / cap);
        if (!retracted)
          d.ntr = Math.max(P.newTokenRatio, d.ntr - (NTR_MAX - P.newTokenRatio) / NTR_DECAY_STEPS);
        admit(stepAt);
        if (d.running.length === 0) { d.nextStepAt = null; break; }
        d.nextStepAt = stepAt + d.decodeStepMs(P);
      }
    }

    /* -- D detok → tokenizer manager → router → client: complete -- */
    for (let i = this.responding.length - 1; i >= 0; i--) {
      const r = this.responding[i];
      if (r.readyAt > now) continue;
      this.responding.splice(i, 1);
      r.stamps.detokDone = r.readyAt;
      r.stage = "done";
      this.allActive.delete(r);
      this.metrics.record(r, now);
    }

    this.inflightIntegral += this.allActive.size * TICK;
    this.metrics.tick(now, this);
  }

  enqueueReadyChunks(p: PrefillInstance | null, r: SimRequest): void {
    if (!r.dPrealloc || !p) return;
    while (r.chunksQueued < r.chunksComputed) {
      const toks = chunkTokens(this.P, r.uncachedLen, r.chunksQueued);
      p.link.queue.push({ req: r, bytes: toks * cellSizeOf(this.P), tokens: toks });
      r.chunksQueued++;
    }
  }

  completeTransfer(p: PrefillInstance, r: SimRequest, doneAt: number): void {
    p.kvUsed -= r.inputLen;
    const ii = p.inflight.indexOf(r);
    if (ii >= 0) p.inflight.splice(ii, 1);
    r.stamps.transferDone = doneAt;
    r.kvPoll = KVPOLL.Success;
    r.stage = "d_waiting";
    const d = r.d;
    const ti = d.transferQ.indexOf(r);
    if (ti >= 0) d.transferQ.splice(ti, 1);
    d.waitingQ.push(r);
  }

  sampleGauges(): Gauges {
    let pQueue = 0, dQueue = 0, running = 0, kvPu = 0, kvPc = 0, kvDu = 0, kvDc = 0;
    let kvDpre = 0;
    for (const p of this.pList) {
      pQueue += p.bootstrapQ.length + p.waitingQ.length;
      kvPu += p.kvUsed; kvPc += p.maxTokens(this.P);
    }
    for (const d of this.dList) {
      dQueue += d.preallocQ.length + d.transferQ.length + d.waitingQ.length;
      running += d.running.length;
      kvDu += d.kvUsed; kvDc += d.maxTokens(this.P);
      for (const r of d.transferQ) kvDpre += r.inputLen;
      for (const r of d.waitingQ)  kvDpre += r.retracted ? 0 : r.inputLen;
    }
    const dtMs = Math.max(1, this.now - this.lastGaugeAt);
    const link = this.linkBusyMs / (dtMs * Math.max(1, this.pList.length));
    this.linkBusyMs = 0; this.lastGaugeAt = this.now;
    return {
      pQueue, dQueue, running,
      kvP: kvPc ? kvPu / kvPc : 0, kvD: kvDc ? kvDu / kvDc : 0,
      kvDpre: kvDc ? kvDpre / kvDc : 0,
      link: clamp(link, 0, 1),
      inflight: this.allActive.size,
    };
  }
}

/* ====================== Headless runner =========================== */

/** Run the engine headlessly for `seconds` of simulated time. */
export function runHeadless(overrides: Partial<SimParams>, seconds: number, seed = 42): SimEngine {
  const P = Object.assign({}, DEFAULTS, overrides) as SimParams;
  const eng = new SimEngine(P, seed);
  for (let ts = 0; ts < seconds * 1000; ts += 1000) eng.advance(1000);
  return eng;
}
