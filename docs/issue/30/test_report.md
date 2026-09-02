# Issue #30 — P6 并行组合集成 测试报告

**日期**: 2026-09-02
**测试命令**: `npx ts-node src/test/sglang-p6.test.ts`
**测试结果**: 45/45 通过 ✓

## 测试用例总览

| 组 | 用例数 | 状态 | 说明 |
|----|--------|------|------|
| Case 1: size=1 退化 noop | 3 | ✓ | 全并行维度=1 时组件创建与零通信验证 |
| Case 2: 多并行组合 | 3 | ✓ | tp=4,dp=2,ep=2,pp=2,cp=2+MLA+MoE 全组件验证 |
| Case 3: DP Attention | 4 | ✓ | enableDpAttention && useMla 条件创建验证 |
| Case 4: PP bubble 比例 | 3 | ✓ | 1f1b vs gpipe bubble 比例验证 |
| Case 5: validateParallelConfig | 10 | ✓ | 7 条约束 + 边界条件覆盖 |
| Case 6: 类型检查与导出 | 7 | ✓ | TypeScript strict + SimScheduler 集成验证 |
| **Case 7: MockSampler 集成** | **7** | **✓** | **驳回修复：forwardBatch 使用 MockSampler 而非旧 Sampler** |
| 边界条件 B1-B8 | 8 | ✓ | OOM/大规模/EPLB tick/PP intermediate |

## Case 7 详细 — MockSampler 集成验证

本轮新增 7 个测试，专门验证驳回修复（forwardBatch 使用 MockSampler + 传递 sampleArgs）：

| 用例 | 描述 | 结果 |
|------|------|------|
| C7-T1 | forwardBatch 使用 MockSampler 而非旧 Sampler（samplingCounter=0） | ✓ |
| C7-T2 | mockSampler.prepare 返回有效 BatchSamplingArgs | ✓ |
| C7-T3 | forwardBatchReq 内部使用 MockSampler.prepare | ✓ |
| C7-T4 | forwardBatchSeqLen 内部使用 MockSampler.prepare | ✓ |
| C7-T5 | forwardBatch 显式传入 sampleArgs 覆盖 prepare | ✓ |
| C7-T6 | PP last stage forwardBatch 使用 MockSampler | ✓ |
| C7-T7 | PP intermediate stage 跳过采样，sampledIds=null | ✓ |

## 类型检查 & 构建

- `npx tsc --noEmit` — 无错误 ✓
- `npm run build` — 构建成功 ✓
