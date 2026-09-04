---
issue_number: 7
issue_type: Feature
test_date: 2026-09-04
test_result: pass
---

# Issue #7 测试报告

## 验收测试结果

| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T-D1 | GET /health returns 200 | ✅ pass |
| T-D1b | GET /health body.ok === true | ✅ pass |
| T-D2 | GET /state returns 200 | ✅ pass |
| T-D2a | state has scheduler field | ✅ pass |
| T-D2b | state has parallel field | ✅ pass |
| T-D2c | state has metrics field | ✅ pass |
| T-D3 | POST /command step returns 200 | ✅ pass |
| T-D3a | step advances tickCounter by dt | ✅ pass |
| T-D3b | pause sets paused=true | ✅ pass |
| T-D3c | reset sets tickCounter to 0 | ✅ pass |
| T-D4 | POST /params tpSize=4 returns 200 | ✅ pass |
| T-D4a | after /params tpSize, config.tpSize === 4 | ✅ pass |
| T-D5 | POST /preset fullCombo returns 200 | ✅ pass |
| T-D5a | fullCombo sets (4,2,2,2,2) | ✅ pass |
| T-D5b | epSize=2 with isMoe=false returns 400 | ✅ pass |
| B-single | single preset: all parallel dims = 1 | ✅ pass |
| B-step-no-start | step works without prior start | ✅ pass |
| B-step-no-start-a | tickCounter incremented | ✅ pass |

## 类型检查
- 结果: pass (tsc --noEmit, exit code 0, no errors)

## 回归测试结果

| 测试文件 | 通过 | 失败 |
|----------|------|------|
| sglang-s0 | 22 | 0 |
| sglang-s1 | 26 | 0 |
| sglang-s2 | 48 | 0 |
| sglang-s3 | 运行通过 | 0 |
| sglang-s4 | 运行通过 | 0 |
| sglang-s5 | 运行通过 | 0 |
| sglang-s6 | 39 | 0 |
| sglang-k1 | 23 | 0 |
| sglang-k2 | 31 | 0 |
| sglang-k3 | 35 | 0 |
| sglang-k4 | 41 | 0 |
| sglang-k5 | 20 | 0 |
| sglang-p0 | 37 | 0 |
| sglang-p1a | 25 | 0 |
| sglang-p1b | 32 | 0 |
| sglang-p2a | 24 | 0 |
| sglang-p2b | 16 | 0 |
| sglang-p3a | 37 | 0 |
| sglang-p3b | 25 | 0 |
| sglang-pp | 39 | 0 |
| sglang-p5 | 23 | 0 |
| sglang-p6 | 45 | 0 |
| sglang-service | 18 | 0 |
| verify-metrics-http | 运行通过 | 0 |
| **合计** | **549+** | **0** |

## Bug 修复

修复了 `CacheManager` 中 `cacheType === "radix"` 时抛出 "not implemented" 错误的 bug：
- **根因**：`cache_manager.ts` 第 59-60 行，当 `cacheType === "radix"` 时直接 throw，未使用已实现的 `RadixPrefixCache`
- **修复**：将 `throw` 替换为 `this.prefixCache = new RadixPrefixCache(numPages, pageSize)`
- **同步更新**：`sglang-k3.test.ts` 中 `T_extra` 测试从"期望抛异常"改为"验证 radix 构造成功"
