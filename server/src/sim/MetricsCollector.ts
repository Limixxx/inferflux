import { SimRequest, MetricSnapshot, Gauges, ISimEngine, SimMode } from "../shared/types";
import { RING_MAX, SERIES_LEN, BUCKET_MS, SERIES_KEYS, BD_KEYS_DISAGG, BD_KEYS_AGG } from "../shared/constants";

/** Ring entry for completed requests (percentile tracking). */
interface RingEntry {
  ttft: number;
  tpot: number;
  e2e: number;
  bd: number[];
  dHandshake: number;
  t: number;
}

/** Return the BD_KEYS variant for a given deployment mode. */
function bdKeysFor(mode: SimMode): readonly string[] {
  return mode === "agg" ? BD_KEYS_AGG : BD_KEYS_DISAGG;
}

/**
 * MetricsCollector — accumulates per-request latency breakdowns into a ring
 * buffer for percentile calculation, and rolls 1-second gauge buckets into
 * time-series arrays for sparkline rendering.
 *
 * Mode-aware: pd-disagg uses 7-column breakdown (tokenize/bootstrap/pQueue/
 * prefill/transfer/dQueue/detok); agg uses 4-column (tokenize/queue/prefill/
 * detok). Mode is set via `setMode()` and the breakdown arrays are
 * re-initialized on mode change.
 */
export class MetricsCollector {
  ring!: (RingEntry | undefined)[];
  ringN = 0;
  ringIdx = 0;
  series: Record<string, Float64Array> = {};
  bdSeries: Float64Array[] = [];
  seriesHead = 0;
  seriesCount = 0;
  bucketStart = 0;
  bkComp = 0;
  bkTokens = 0;
  bkTtftSum = 0;
  bkTpotSum = 0;
  bkE2eSum = 0;
  bkDhsSum = 0;
  bkBd: number[] = [];
  totalCompleted = 0;
  totalTokens = 0;
  mode: SimMode = "pd-disagg";

  constructor() { this.reset(0); }

  /** Set the deployment mode; re-initializes breakdown arrays if mode changes. */
  setMode(mode: SimMode): void {
    if (this.mode === mode) return;
    this.mode = mode;
    const keys = bdKeysFor(mode);
    this.bkBd = new Array(keys.length).fill(0);
    this.bdSeries = [];
    for (let j = 0; j < keys.length; j++)
      this.bdSeries.push(new Float64Array(SERIES_LEN).fill(NaN));
  }

  reset(now: number): void {
    this.ring = new Array(RING_MAX);
    this.ringN = 0;
    this.ringIdx = 0;
    this.series = {};
    for (const k of SERIES_KEYS) {
      this.series[k] = new Float64Array(SERIES_LEN).fill(NaN);
    }
    this.seriesHead = 0;
    this.seriesCount = 0;
    this.bucketStart = now;
    this.bkComp = 0;
    this.bkTokens = 0;
    this.bkTtftSum = 0;
    this.bkTpotSum = 0;
    this.bkE2eSum = 0;
    this.bkDhsSum = 0;
    const keys = bdKeysFor(this.mode);
    this.bkBd = new Array(keys.length).fill(0);
    this.bdSeries = [];
    for (let j = 0; j < keys.length; j++)
      this.bdSeries.push(new Float64Array(SERIES_LEN).fill(NaN));
    this.totalCompleted = 0;
    this.totalTokens = 0;
  }

  record(r: SimRequest, now: number, mode: SimMode = "pd-disagg"): void {
    const s = r.stamps;
    const ttft = s.firstToken - s.recv;
    const tpot = r.outputLen > 1 ? (s.lastToken - s.firstToken) / (r.outputLen - 1) : 0;
    const e2e = s.detokDone - s.recv;
    this.bkE2eSum = (this.bkE2eSum || 0) + e2e;
    let bd: number[];
    let dHandshake: number;
    if (mode === "agg") {
      // 4-column agg breakdown: tokenize / queue / prefill / detok
      bd = [
        s.tokenized - s.recv,
        s.wQueueExit - s.tokenized,
        s.wPrefillDone - s.wQueueExit,
        s.firstToken - s.wPrefillDone,
      ];
      dHandshake = 0;  // no D-side handshake in agg mode
    } else {
      // 7-column pd-disagg breakdown
      bd = [
        s.tokenized - s.recv,
        s.bootstrapDone - s.tokenized,
        s.pQueueExit - s.bootstrapDone,
        s.prefillDone - s.pQueueExit,
        s.transferDone - s.prefillDone,
        s.dQueueExit - s.transferDone,
        s.firstToken - s.dQueueExit,
      ];
      dHandshake = s.preallocDone - s.tokenized;
    }
    this.ring[this.ringIdx] = { ttft, tpot, e2e, bd, dHandshake, t: now };
    this.ringIdx = (this.ringIdx + 1) % RING_MAX;
    if (this.ringN < RING_MAX) this.ringN++;
    this.bkComp++;
    this.bkTokens += r.outputLen;
    this.bkTtftSum += ttft;
    this.bkTpotSum += tpot;
    this.bkDhsSum = (this.bkDhsSum || 0) + dHandshake;
    const keys = bdKeysFor(this.mode);
    for (let j = 0; j < keys.length; j++) this.bkBd[j] += bd[j];
    this.totalCompleted++;
    this.totalTokens += r.outputLen;
  }

  tick(now: number, engine: ISimEngine): void {
    while (now - this.bucketStart >= BUCKET_MS) {
      this.bucketStart += BUCKET_MS;
      const g = engine.sampleGauges();
      const put = (k: string, v: number) => { this.series[k][this.seriesHead] = v; };
      put("ttft", this.bkComp ? this.bkTtftSum / this.bkComp : NaN);
      put("tpot", this.bkComp ? this.bkTpotSum / this.bkComp : NaN);
      put("e2e", this.bkComp ? this.bkE2eSum / this.bkComp : NaN);
      put("rps", this.bkComp * 1000 / BUCKET_MS);
      put("tps", this.bkTokens * 1000 / BUCKET_MS);
      // Mode-aware gauge routing — agg mode zeros out pd-disagg-only gauges and
      // vice versa, so consumers can rely on the same series keys either way.
      if (this.mode === "agg") {
        put("pQueue", 0); put("dQueue", 0); put("running", g.running);
        put("kvP", 0); put("kvD", 0); put("kvDpre", 0); put("link", 0);
        put("dHandshake", 0);
        put("inflight", g.inflight);
        put("wQueue", g.wQueue); put("kvW", g.kvW);
      } else {
        put("pQueue", g.pQueue); put("dQueue", g.dQueue); put("running", g.running);
        put("kvP", g.kvP); put("kvD", g.kvD); put("kvDpre", g.kvDpre); put("link", g.link);
        put("dHandshake", this.bkComp ? this.bkDhsSum / this.bkComp : NaN);
        put("inflight", g.inflight);
        put("wQueue", 0); put("kvW", 0);
      }
      const keys = bdKeysFor(this.mode);
      for (let j = 0; j < keys.length; j++)
        this.bdSeries[j][this.seriesHead] = this.bkComp ? this.bkBd[j] / this.bkComp : NaN;
      this.seriesHead = (this.seriesHead + 1) % SERIES_LEN;
      if (this.seriesCount < SERIES_LEN) this.seriesCount++;
      this.bkComp = 0; this.bkTokens = 0; this.bkTtftSum = 0; this.bkTpotSum = 0;
      this.bkE2eSum = 0; this.bkDhsSum = 0;
      this.bkBd.fill(0);
    }
  }

  snapshot(now?: number, windowMs?: number): MetricSnapshot | null {
    const total = this.ringN;
    if (total === 0) return null;
    const cutoff = (windowMs != null && now != null) ? now - windowMs : -Infinity;
    const ttftA: number[] = [], tpotA: number[] = [], e2eA: number[] = [], dhsA: number[] = [];
    const keys = bdKeysFor(this.mode);
    const bdSum = new Array(keys.length).fill(0);
    for (let i = 0; i < total; i++) {
      const e = this.ring[i]!;
      if (e.t != null && e.t < cutoff) continue;
      ttftA.push(e.ttft); tpotA.push(e.tpot); e2eA.push(e.e2e);
      dhsA.push(e.dHandshake || 0);
      for (let j = 0; j < keys.length; j++) bdSum[j] += e.bd[j];
    }
    const n = ttftA.length;
    if (n === 0) return null;
    const ttfts = Float64Array.from(ttftA), tpots = Float64Array.from(tpotA);
    const e2es = Float64Array.from(e2eA), dhss = Float64Array.from(dhsA);
    ttfts.sort(); tpots.sort(); e2es.sort(); dhss.sort();
    const pct = (arr: Float64Array, p: number) => arr[Math.min(n - 1, Math.floor(p * n))];
    const avg = (arr: Float64Array) => { let s = 0; for (const v of arr) s += v; return s / n; };
    return {
      n,
      ttft: { avg: avg(ttfts), p50: pct(ttfts, .5), p99: pct(ttfts, .99) },
      tpot: { avg: avg(tpots), p50: pct(tpots, .5), p99: pct(tpots, .99) },
      e2e:  { avg: avg(e2es),  p50: pct(e2es, .5),  p99: pct(e2es, .99) },
      dHandshake: { avg: avg(dhss), p99: pct(dhss, .99) },
      bd: bdSum.map(v => v / n),
    };
  }

  latestSeries(k: string): number[] {
    const out: number[] = [];
    const start = (this.seriesHead - this.seriesCount + SERIES_LEN) % SERIES_LEN;
    for (let i = 0; i < this.seriesCount; i++)
      out.push(this.series[k][(start + i) % SERIES_LEN]);
    return out;
  }

  recentBreakdown(lastN: number): number[] {
    const keys = bdKeysFor(this.mode);
    const out = new Array(keys.length).fill(0);
    let any = false;
    for (let j = 0; j < keys.length; j++) {
      let sum = 0, cnt = 0;
      for (let back = 1; back <= Math.min(lastN, this.seriesCount); back++) {
        const v = this.bdSeries[j][(this.seriesHead - back + SERIES_LEN) % SERIES_LEN];
        if (!Number.isNaN(v)) { sum += v; cnt++; }
      }
      if (cnt) { out[j] = sum / cnt; any = true; }
    }
    if (any) return out;
    const s = this.snapshot();
    return s ? s.bd : out;
  }
}
