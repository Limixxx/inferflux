// scheduler — K1: TableManager

/** 管理 page_table 行分配（§3.3.6 / §9.11 完整实现） */
export class TableManager {
  readonly maxRunningReq: number;
  pageTable: number[][];
  tokenPool: number[][];
  freeTableIndices: number[];

  constructor(maxRunningReq: number, pageTable: number[][]) {
    this.maxRunningReq = maxRunningReq;
    this.pageTable = pageTable;
    // 最后一行预留给 dummy req
    this.freeTableIndices = Array.from(
      { length: maxRunningReq }, (_, i) => i
    );
    this.tokenPool = Array.from(
      { length: maxRunningReq + 1 },
      () => new Array(pageTable[0].length).fill(0)
    );
  }

  get availableSize(): number {
    return this.freeTableIndices.length;
  }

  allocate(): number {
    if (this.freeTableIndices.length === 0) {
      throw new Error("No available table indices");
    }
    return this.freeTableIndices.pop()!;
  }

  free(tableIdx: number): void {
    this.freeTableIndices.push(tableIdx);
  }
}
