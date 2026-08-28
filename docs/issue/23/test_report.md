---
issue_number: 23
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #23 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | 全 size=1 生成有效结果，graphBuffer 与基础版一致 | ✅ pass |
| T2 | TP>1 权重修正 | ✅ pass |
| T3 | TP>1 KV heads 分割 (tp=2, numKvHeads=8) | ✅ pass |
| T4 | DP>1 KV budget 分割 | ✅ pass |
| T5 | DP Attention KV 乘数 | ✅ pass |
| T6 | EP>1 MoE 权重修正 | ✅ pass |
| T7 | EP>1 非 MoE 不修正 | ✅ pass |
| T8 | PP>1 权重修正 | ✅ pass |
| T9 | CP>1 KV 乘数 | ✅ pass |
| T10 | 组合并行 tp=8,dp=2,ep=2,pp=2,cp=2 | ✅ pass |
| T11 | parallelCorrections 字段填充 | ✅ pass |
| T12 | OOM 场景 | ✅ pass |
| T13 | DP Attention attention 权重复制 | ✅ pass |
| T14 | 合法配置通过 | ✅ pass |
| T15 | 约束 1：world_size 内部一致性 | ✅ pass |
| T16 | 约束 2：EP>1 但非 MoE | ✅ pass |
| T17 | 约束 3：cp_size 不整除 tp_size | ✅ pass |
| T18 | 约束 4：(tp/cp) 不整除 ep_size | ✅ pass |
| T19 | 约束 5：pp_size > numLayers | ✅ pass |
| T20 | 约束 6：DP Attention 但非 MLA | ✅ pass |
| T21 | 约束 7：mem_fraction 越界 | ✅ pass |
| T22 | 警告：KV heads 不整除 | ✅ pass |
| T23 | 多错误同时返回 | ✅ pass |
| T24 | 全默认配置通过 | ✅ pass |
| B1 | totalGpuMemory=1 byte → numPages=0, OOM warning | ✅ pass |
| B2 | memoryRatio=0 → numPages=0 | ✅ pass |
| B3 | numKvHeads=1 (MLA scenario) | ✅ pass |
| B4 | numLayers=0 → numPages=0 | ✅ pass |
| B5 | epSize>1 但 not MoE → validate error, budget no EP division | ✅ pass |
| B6 | cpSize = tpSize → full CP split | ✅ pass |
| B7 | ppSize = numLayers → each stage has 1 layer | ✅ pass |
| B8 | ppSize > numLayers → validate error | ✅ pass |

## 类型检查
- 结果: pass（新增代码无编译错误，已有编译错误与本次修改无关）

## 回归测试
- K5 测试: 20/20 pass
- P0 测试: 37/37 pass

## 失败用例详情（如有）
无

## 边界条件覆盖
- totalGpuMemory 极小（1 byte）→ OOM，numPages=0 ✅
- memoryRatio = 0 → available 为负 → numPages = 0 ✅
- numKvHeads = 1（MLA 场景）→ divEven 正确处理 ✅
- numLayers = 0 → bytesPerToken = 0 → 除零保护，numPages = 0 ✅
- epSize > 1 但 isMoe = false → validate 报错，budget 不除 epSize ✅
- cpSize = tpSize → attn_tp_size = 1（全部 rank 用于 CP 切分）✅
- ppSize = numLayers → 每 stage 恰好 1 层 ✅
- ppSize > numLayers → 某 stage 0 层 → validate 报错 ✅
