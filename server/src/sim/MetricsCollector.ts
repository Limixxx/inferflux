import { SimRequest, MetricSnapshot, Gauges, ISimEngine } from "../shared/types";
import { RING_MAX, SERIES_LEN, BUCKET_MS, SERIES_KEYS, BD_KEYS } from "../shared/constants";

/** Ring entry for completed requests (percentile tracking). */
interface RingEntry {
  ttft: number;
  tpot: number;
  e2e: number;
  bd: number[];
  dHandshake: number;
  t: number;
}

/**
 * MetricsCollector — accumulates per-request latency breakdowns into a ring
 * buffer for percentile calculation, and rolls 1-second gauge buckets into
 * time-series arrays for sparkline rendering.
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

  constructor() { this.reset(0); }

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
    this.bkBd = new Array(BD_KEYS.length).fill(0);
    this.bdSeries = [];
    for (let j = 0; j < BD_KEYS.length; j++)
      this.bdSeries.push(new Float64Array(SERIES_LEN).fill(NaN));
    this.totalCompleted = 0;
    this.totalTokens = 0;
  }

  record(r: SimRequest, now: number): void {
    const s = r.stamps;
    const ttft = s.firstToken - s.recv;
    const tpot = r.outputLen > 1 ? (s.lastToken - s.firstToken) / (r.outputLen - 1) : 0;
    const e2e = s.detokDone - s.recv;
    this.bkE2eSum = (this.bkE2eSum || 0) + e2e;
    const bd = [
      s.tokenized - s.recv,
      s.bootstrapDone - s.tokenized,
      s.pQueueExit - s.bootstrapDone,
      s.prefillDone - s.pQueueExit,
      s.transferDone - s.prefillDone,
      s.dQueueExit - s.transferDone,
      s.firstToken - s.dQueueExit,
    ];
    const dHandshake = s.preallocDone - s.tokenized;
    this.ring[this.ringIdx] = { ttft, tpot, e2e, bd, dHandshake, t: now };
    this.ringIdx = (this.ringIdx + 1) % RING_MAX;
    if (this.ringN < RING_MAX) this.ringN++;
    this.bkComp++;
    this.bkTokens += r.outputLen;
    this.bkTtftSum += ttft;
    this.bkTpotSum += tpot;
    this.bkDhsSum = (this.bkDhsSum || 0) + dHandshake;
    for (let j = 0; j < BD_KEYS.length; j++) this.bkBd[j] += bd[j];
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
      put("pQueue", g.pQueue); put("dQueue", g.dQueue); put("running", g.running);
      put("kvP", g.kvP); put("kvD", g.kvD); put("kvDpre", g.kvDpre); put("link", g.link);
      put("dHandshake", this.bkComp ? this.bkDhsSum / this.bkComp : NaN);
      put("inflight", g.inflight);
      for (let j = 0; j < BD_KEYS.length; j++)
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
    const bdSum = new Array(BD_KEYS.length).fill(0);
    for (let i = 0; i < total; i++) {
      const e = this.ring[i]!;
      if (e.t != null && e.t < cutoff) continue;
      ttftA.push(e.ttft); tpotA.push(e.tpot); e2eA.push(e.e2e);
      dhsA.push(e.dHandshake || 0);
      for (let j = 0; j < BD_KEYS.length; j++) bdSum[j] += e.bd[j];
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
    const out = new Array(BD_KEYS.length).fill(0);
    let any = false;
    for (let j = 0; j < BD_KEYS.length; j++) {
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
