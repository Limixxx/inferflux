---
issue_number: 15
issue_type: Feature
test_date: 2026-08-28
test_result: pass
---

# Issue #15 测试报告

## 验收测试结果
| 用例编号 | 测试描述 | 结果 |
|----------|---------|------|
| T1 | RadixTreeNode 构造与基本属性 | ✅ pass |
| T2 | RadixTreeNode.setKeyValue | ✅ pass |
| T3 | RadixTreeNode.getMatchLen | ✅ pass |
| T4 | RadixTreeNode.splitAt 基本分裂 | ✅ pass |
| T5 | RadixTreeNode.splitAt refCount 继承 | ✅ pass |
| T6 | RadixTreeNode.setParent 双向引用 | ✅ pass |
| T7 | RadixTreeNode.isLeaf/isRoot | ✅ pass |
| T8 | RadixPrefixCache 构造 | ✅ pass |
| T9 | matchPrefix 完全命中 | ✅ pass |
| T10 | matchPrefix 部分命中 | ✅ pass |
| T11 | matchPrefix 未命中 | ✅ pass |
| T12 | insertPrefix 新序列插入 | ✅ pass |
| T13 | insertPrefix 已存在序列 | ✅ pass |
| T14 | 同一前缀重放命中率 100% | ✅ pass |
| T15 | split_at 正确性 | ✅ pass |
| T16 | lockHandle 锁定 | ✅ pass |
| T17 | lockHandle 解锁 | ✅ pass |
| T18 | lockHandle 多次锁定 | ✅ pass |
| T19 | evict 基本驱逐 | ✅ pass |
| T20 | evict 驱逐不影响命中节点 | ✅ pass |
| T21 | evict 父节点合并 | ✅ pass |
| T22 | handle lock 阻止驱逐 | ✅ pass |
| T23 | sizeInfo 一致性 | ✅ pass |
| T24 | RadixCacheHandle.getMatchedIndices | ✅ pass |
| T25 | reset 重置缓存 | ✅ pass |
| T26 | checkIntegrity 正常 | ✅ pass |
| T27 | checkIntegrity 检测不一致 | ✅ pass |
| T28 | pageSize=1 的 keyFn | ✅ pass |
| T29 | pageSize>1 的 keyFn | ✅ pass |
| T30 | insertPrefix 空序列 | ✅ pass |
| B1 | 空树 matchPrefix | ✅ pass |
| B2 | 单 token 插入与匹配 | ✅ pass |
| B3 | evict 请求量大于可驱逐量 | ✅ pass |
| B4 | evict(0) | ✅ pass |
| B5 | 连续 lock/unlock 配对 | ✅ pass |
| B6 | 未 unlock 就再次 lock | ✅ pass |
| B7 | 插入长度非页对齐 | ✅ pass |
| B8 | 完全相同序列重复插入 | ✅ pass |
| B9 | 多条共享前缀的序列 | ✅ pass |
| T_extra | RadixPrefixCache extends BasePrefixCache | ✅ pass |
| T_extra | RadixCacheHandle extends BaseCacheHandle | ✅ pass |

## 类型检查
- 结果: pass（K4 相关文件无类型错误；已有的 k1/s0 测试文件的 TableManager 错误为预存问题，非 K4 引入）

## 失败用例详情（如有）
无

## 边界条件覆盖
- 空树 matchPrefix 返回 cachedLen=0
- 单 token pageSize=1 正常工作
- evict 超额请求仅驱逐可驱逐量
- evict(0) 返回空数组无副作用
- 连续 lock/unlock 配对 refCount 归零
- 未 unlock 再 lock refCount 累加正确
- 非页对齐插入自动 alignDown
- 重复插入不创建新节点
- 多条共享前缀序列正确共享分支节点
