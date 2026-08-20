import { LBPolicy } from "../shared/types";
import { RNG } from "../shared/rng";

/** Round-robin counter object. */
export interface RRCounter { i: number; }

/**
 * Load-balancing policy — mirrors sgl-model-gateway policies.
 * Returns the chosen instance from `pool` (already filtered to non-draining),
 * using `loadOf(inst)` as the load metric and a per-role round-robin counter.
 */
export function selectByPolicy<T>(
  policy: LBPolicy,
  pool: T[],
  loadOf: (inst: T) => number,
  rng: RNG,
  rr: RRCounter,
): T {
  if (pool.length === 1) return pool[0];
  switch (policy) {
    case "round_robin":
      return pool[rr.i++ % pool.length];
    case "random":
      return pool[Math.floor(rng() * pool.length)];
    case "power_of_two": {
      const a = pool[Math.floor(rng() * pool.length)];
      let b = pool[Math.floor(rng() * pool.length)];
      if (b === a) b = pool[(pool.indexOf(a) + 1) % pool.length];
      return loadOf(a) <= loadOf(b) ? a : b;
    }
    case "least":
    default: {
      let best = pool[0], bl = loadOf(best);
      for (let i = 1; i < pool.length; i++) {
        const l = loadOf(pool[i]);
        if (l < bl) { bl = l; best = pool[i]; }
      }
      return best;
    }
  }
}
