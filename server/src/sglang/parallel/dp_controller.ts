// parallel/dp_controller.ts — P2a: DataParallelController — 标准 DP（§10.3）

import { divEven } from "../core";

/**
 * 单个 DP 副本状态
 *
 * 维护 rank 索引、负载（等待中请求数）、已分配页数与页容量上限。
 */
export class DPRankState {
  readonly rank: number;
  load: number;                        // 等待中的请求数
  pages_allocated: number;             // 已分配的页数
  readonly pages_capacity: number;     // 该副本的页容量上限

  constructor(rank: number, pages_capacity: number) {
    this.rank = rank;
    this.load = 0;
    this.pages_allocated = 0;
    this.pages_capacity = pages_capacity;
  }

  /** 当前可用页数 */
  get pages_available(): number {
    return this.pages_capacity - this.pages_allocated;
  }
}

/**
 * 标准 DP 请求分发器（§10.3）
 *
 * 支持 round_robin / shortest_queue 两种分发策略，
 * 按 pages_per_rank = total_num_pages // dp_size 为每个副本划分独立 KV 页池。
 */
export class DataParallelController {
  readonly strategy: "round_robin" | "shortest_queue";
  readonly dp_size: number;
  private readonly _ranks: DPRankState[];
  private _round_robin_idx: number = 0;

  constructor(
    dp_size: number,
    total_num_pages: number,
    strategy: "round_robin" | "shortest_queue",
  ) {
    if (dp_size < 1) {
      throw new Error(`dp_size must be >= 1, got ${dp_size}`);
    }
    this.dp_size = dp_size;
    this.strategy = strategy;

    const pagesPerRank = divEven(total_num_pages, dp_size);
    this._ranks = pagesPerRank.map((cap, i) => new DPRankState(i, cap));
  }

  /** 获取所有副本状态（只读） */
  get ranks(): ReadonlyArray<DPRankState> {
    return this._ranks;
  }

  /**
   * 为新请求选择 DP 副本并分配页。
   *
   * 两步操作：先 allocate_pages(needed_pages) 成功 → 设 req.dp_rank = rank；记录 load += 1。
   * 失败返回 null（OOM）。
   */
  select_rank_for_request(needed_pages: number): DPRankState | null {
    // dp_size=1 时退化 noop：始终选择 rank 0
    if (this.dp_size <= 1) {
      const r = this._ranks[0];
      if (r.pages_available >= needed_pages) {
        r.pages_allocated += needed_pages;
        r.load += 1;
        return r;
      }
      return null;
    }

    // 多副本：选出候选 rank
    let candidateIdx: number;
    if (this.strategy === "round_robin") {
      candidateIdx = this._round_robin_idx % this.dp_size;
      this._round_robin_idx += 1;
    } else {
      // shortest_queue：选 load 最小的副本，相同 load 选 index 最小的
      candidateIdx = 0;
      let minLoad = this._ranks[0].load;
      for (let i = 1; i < this.dp_size; i++) {
        if (this._ranks[i].load < minLoad) {
          minLoad = this._ranks[i].load;
          candidateIdx = i;
        }
      }
    }

    // 尝试 allocate_pages
    const candidate = this._ranks[candidateIdx];
    if (candidate.pages_available >= needed_pages) {
      candidate.pages_allocated += needed_pages;
      candidate.load += 1;
      return candidate;
    }
    return null;
  }

  /**
   * 为指定副本分配页。
   * 条件：pages_capacity - pages_allocated >= needed_pages。
   * 成功返回 true，失败返回 false。
   */
  allocate_pages(rank_idx: number, needed_pages: number): boolean {
    const r = this._ranks[rank_idx];
    if (r.pages_available >= needed_pages) {
      r.pages_allocated += needed_pages;
      return true;
    }
    return false;
  }

  /**
   * 释放指定副本的页并减少负载。
   * pages_allocated 回写；load -= 1。
   * 包含越界保护：pages_allocated = max(0, pages_allocated - freed_pages)。
   */
  free_pages(rank_idx: number, freed_pages: number): void {
    const r = this._ranks[rank_idx];
    r.pages_allocated = Math.max(0, r.pages_allocated - freed_pages);
    r.load = Math.max(0, r.load - 1);
  }
}
