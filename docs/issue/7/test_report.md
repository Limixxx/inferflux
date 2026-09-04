---
issue_number: 7
issue_type: Feature
test_date: 2026-09-04
test_result: pass
---

# Issue #7 测试报告

## 驳回意见修复验证

| # | 驳回问题 | 修复措施 | 结果 |
|---|---------|---------|------|
| 1 | cache 模块重构未在 PR body 中声明（与"零代码改动"矛盾） | 回滚 cache 模块重构，恢复 cache/index.ts 为原始版本（包含类定义），删除 cache/base.ts，恢复 4 个文件的 import 路径 | ✅ pass |

## 验收测试结果

### §3 演示系统验收测试（sglang-service.test.ts）

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

**小计: 18 通过 / 0 失败**

### 一致性回归测试（22 个 sglang-*.test.ts + verify-metrics-http.test.ts）

| 测试文件 | 通过 | 失败 | 测试文件 | 通过 | 失败 |
|---------|------|------|---------|------|------|
| sglang-s0 | 22 | 0 | sglang-p0 | 37 | 0 |
| sglang-s1 | 26 | 0 | sglang-p1a | 25 | 0 |
| sglang-s2 | 48 | 0 | sglang-p1b | 32 | 0 |
| sglang-s3 | 52 | 0 | sglang-p2a | 24 | 0 |
| sglang-s4 | 46 | 0 | sglang-p2b | 16 | 0 |
| sglang-s5 | 23 | 0 | sglang-p3a | 37 | 0 |
| sglang-s6 | 39 | 0 | sglang-p3b | 25 | 0 |
| sglang-k1 | 23 | 0 | sglang-pp | 39 | 0 |
| sglang-k2 | 31 | 0 | sglang-p5 | 23 | 0 |
| sglang-k3 | 35 | 0 | sglang-p6 | 45 | 0 |
| sglang-k4 | 41 | 0 | verify-metrics-http | ALL PASS | 0 |
| sglang-k5 | 20 | 0 | | | |

**回归测试合计: 638+ 通过 / 0 失败**

## 类型检查

- 结果: ✅ pass（`npx tsc --noEmit` 无错误）

## 边界条件覆盖

- 全并行维度 = 1（single 预设）: rank 网格 1×1×1，通信成本全 0 ✅
- 仿真未 start 直接 step: 离线单步正常工作 ✅
- 并行配置非法（epSize=2 + isMoe=false）: 返回 400 + 错误明细 ✅
