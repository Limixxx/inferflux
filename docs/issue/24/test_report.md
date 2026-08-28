---
issue_number: 24
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #24 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | DPRankState 初始化：rank=0, pages_capacity=100, load=0, pages_allocated=0 | ✅ pass |
| T2 | pages_available 计算：pages_capacity=100, pages_allocated=30 → pages_available=70 | ✅ pass |
| T3 | dp_size=1 退化 noop：select_rank_for_request 始终返回 rank 0 | ✅ pass |
| T4 | dp_size=1 页不足返回 null：pages_capacity=5, needed_pages=10 → null | ✅ pass |
| T5 | pages_per_rank 均分：total=100, dp_size=3 → [34, 33, 33]（divEven 语义） | ✅ pass |
| T6 | allocate_pages 成功：pages_available >= needed_pages → true | ✅ pass |
| T7 | allocate_pages 失败：pages_available < needed_pages → false | ✅ pass |
| T8 | free_pages 回写：pages_allocated 减少, load 减少 | ✅ pass |
| T9 | free_pages 越界保护：free_pages(999) → pages_allocated = max(0, ...) 不为负 | ✅ pass |
| T10 | round_robin 轮询均匀：dp_size=4, 连续 8 次请求 → [0,1,2,3,0,1,2,3] | ✅ pass |
| T11 | round_robin 分配失败不影响轮询索引 | ✅ pass |
| T12 | shortest_queue 选最小负载：选最先出现的最小值 | ✅ pass |
| T13 | shortest_queue 负载均衡：连续请求均匀分布到各副本 | ✅ pass |
| T14 | 所有副本页不足返回 null | ✅ pass |
| T15 | 部分副本页不足仍可分配 | ✅ pass |
| T16 | 完整生命周期：select → allocate → free 后状态回到初始 | ✅ pass |
| T17 | 大量请求压力测试：dp_size=8, 1000 次请求 round_robin 均匀分布 | ✅ pass |
| T18 | ParallelMetrics 回填验证：ranks.map 可正确获取 load 和 pages_allocated | ✅ pass |
| B1 | dp_size=0 构造函数抛出 Error | ✅ pass |
| B2 | total_num_pages=0 任何分配均返回 null | ✅ pass |
| B3 | needed_pages=0 select 成功，load+1，pages_allocated 不变 | ✅ pass |
| B4 | freed_pages=0 不改变 pages_allocated，但 load-1 | ✅ pass |
| B5 | round_robin_idx 溢出使用 % dp_size 取模 | ✅ pass |
| B6 | shortest_queue 多副本 load 相同选 index 最小 | ✅ pass |

## 类型检查
- 结果: pass
- 说明: dp_controller.ts 及 parallel 模块无类型错误；存在其他 Issue 遗留的 TableManager 导出缺失（sglang-k1.test.ts / sglang-s0.test.ts），与本 Issue 无关

## 失败用例详情（如有）
无

## 边界条件覆盖
- dp_size=0 抛出 Error ✅
- total_num_pages=0 全部分配失败 ✅
- needed_pages=0 分配成功但 pages 不变 ✅
- freed_pages=0 pages 不变但 load-1 ✅
- round_robin_idx 溢出取模 ✅
- shortest_queue 相同 load 选最小 index ✅
