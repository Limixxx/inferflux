// metrics — SimulationMetrics

import { ParallelMetrics } from "../parallel";

/**
 * 仿真指标集合
 *
 * P0: 嵌入 ParallelMetrics 子结构，收集并行通信与负载指标。
 * 后续 Issue 将补充调度、缓存、延迟等其他指标。
 */
export class SimulationMetrics {
  readonly parallel: ParallelMetrics = new ParallelMetrics();

  reset(): void {
    this.parallel.reset();
  }
}
