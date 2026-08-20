export type RNG = () => number;

export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return function(): number {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function expSample(rng: RNG, mean: number): number {
  return -Math.log(1 - rng()) * mean;
}

export function lognormalSample(rng: RNG, mean: number, sigma: number): number {
  const mu = Math.log(mean) - sigma * sigma / 2;
  const u1 = rng(), u2 = rng();
  const z = Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  return Math.exp(mu + sigma * z);
}

export function sampleLen(rng: RNG, mean: number, dist: "fixed" | "uniform" | "lognormal"): number {
  let v: number;
  if (dist === "fixed") v = mean;
  else if (dist === "uniform") v = mean * (0.5 + rng());
  else v = lognormalSample(rng, mean, 0.5);
  return Math.max(1, Math.round(v));
}
