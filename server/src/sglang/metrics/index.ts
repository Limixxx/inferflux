// metrics — SimulationMetrics

import { ParallelMetrics } from "../parallel";

/**
 * 仿真指标集合
 *
 * P0: 嵌入 ParallelMetrics 子结构，收集并行通信与负载指标。
 * P6: 新增 toJSON() 方法暴露并行指标汇总。
 */
export class SimulationMetrics {
  readonly parallel: ParallelMetrics = new ParallelMetrics();

  reset(): void {
    this.parallel.reset();
  }

  /** 将并行指标汇总为 JSON 可序列化对象 */
  toJSON(): Record<string, unknown> {
    return {
      parallel: this.parallel.summary(),
    };
  }
}
