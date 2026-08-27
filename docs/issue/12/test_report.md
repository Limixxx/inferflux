---
issue_number: 12
issue_type: Feature
test_date: 2026-08-27
test_result: pass
---

# Issue #12 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | estimateModelMemory 使用默认 ModelConfig + dtypeSize=2 | ✅ pass |
| T2 | estimateGraphBuffer cudaGraphBs=null 返回 0 | ✅ pass |
| T3 | estimateGraphBuffer cudaGraphBs=[1,2,4,8] + 默认 ModelConfig | ✅ pass |
| T4 | estimateGraphBuffer cudaGraphBs=[] 空数组返回 0 | ✅ pass |
| T5 | calculateMemoryBudget 默认配置（80GiB, 0.88, tp=1）numPages > 0 | ✅ pass |
| T6 | calculateMemoryBudget 返回值类型正确 MemoryBudgetResult | ✅ pass |
| T7 | calculateMemoryBudget tp=2 与 tp=1 结果一致性 | ✅ pass |
| T7b | calculateMemoryBudget tp=4 numKvHeads=2 (GQA with TP replication) | ✅ pass |
| T8 | calculateMemoryBudget OOM (totalGpuMemory=1) numPages=0 且警告 | ✅ pass |
| T9 | calculateMemoryBudget 忽略 config.numPages | ✅ pass |
| T10 | calculateMemoryBudget pageSize=16 numPages < pageSize=1 | ✅ pass |
| T11 | calculateMemoryBudget divEven(8, 3, true) 正确使用 | ✅ pass |
| B1 | totalGpuMemory=1 byte → numPages=0, OOM warning | ✅ pass |
| B2 | memoryRatio=0 → numPages=0 | ✅ pass |
| B3 | memoryRatio=1.0 → all memory available minus model+graph | ✅ pass |
| B4 | dtypeSize=1 → more pages than dtypeSize=2 | ✅ pass |
| B5 | cudaGraphBs null vs [] vs [1,2,4] | ✅ pass |
| B6 | tpSize > numKvHeads (tp=4, numKvHeads=2) | ✅ pass |
| B7 | numKvHeads=1 (MLA scenario) | ✅ pass |
| B8 | numLayers=0 → cachePerPage=0, numPages=0 with warning | ✅ pass |

## 类型检查
- 结果: pass (`npx tsc --noEmit` 零错误)

## 失败用例详情（如有）
无

## 边界条件覆盖
- totalGpuMemory 极小（1 byte）→ numPages=0, OOM 警告 ✅
- memoryRatio = 0 → numPages=0 ✅
- memoryRatio = 1.0 → 所有显存可用 ✅
- dtypeSize = 1（float8）→ 更多页数 ✅
- cudaGraphBs = null vs [] vs [1,2,4] ✅
- tpSize > numKvHeads（tp=4, numKvHeads=2）✅
- numKvHeads = 1（MLA 场景）✅
- numLayers = 0 → 除零保护返回 numPages=0 ✅
