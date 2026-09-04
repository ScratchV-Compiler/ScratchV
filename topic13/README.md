# 课题 13：窥孔优化器 — 完成目录

> **状态**：✅ 已完成（2026-08-01）  
> **负责人模块**：汇编层窥孔优化（`--peephole-asm`）  
> **主实现**：[`scratchv/backend/asm_peephole.py`](../scratchv/backend/asm_peephole.py)

本目录标明：**本工作对应 ScratchV 课题 13**，并索引全部相关文档与产物路径。

---

## 文档位置（必读）

| 类型 | 读者 | 路径 |
|------|------|------|
| **设计文档** | 人（架构/审查/汇报） | [`docs/topics/13-窥孔优化器-设计文档.md`](../docs/topics/13-窥孔优化器-设计文档.md) |
| **赛道 A 设计文档** | Parser 复用 / review 跟进 | [`docs/topics/13-窥孔优化器-赛道A-设计文档.md`](../docs/topics/13-窥孔优化器-赛道A-设计文档.md) |
| **赛道 A 开发文档** | 实现与验收清单 | [`docs/topics/13-窥孔优化器-赛道A-开发文档.md`](../docs/topics/13-窥孔优化器-赛道A-开发文档.md) |
| **开发文档** | AI / 维护者 | [`docs/topics/archive/topic13_asm_peephole_guide.md`](../docs/topics/archive/topic13_asm_peephole_guide.md) |
| **新手教程** | 入门学习 | [`docs/topics/13-窥孔优化器.md`](../docs/topics/13-窥孔优化器.md) |
| **课题提案（归档）** | 原始任务说明 | [`docs/topics/archive/课题13：窥孔优化器.md`](../docs/topics/archive/课题13：窥孔优化器.md) |
| **课程 HTML** | 浏览器阅读 | [`docs/topics/html/13-窥孔优化器.html`](../docs/topics/html/13-窥孔优化器.html) |
| **前后对比报告** | 效果数据 | [`benchmark_reports/peephole_compare.md`](../benchmark_reports/peephole_compare.md) |
| **课题索引入口** | 全课题地图 | [`docs/topics/INDEX.md`](../docs/topics/INDEX.md)（第 13 项） |

---

## 代码与测试

| 类型 | 路径 |
|------|------|
| 主实现 | `scratchv/backend/asm_peephole.py` |
| 编译器集成 | `scratchv/compiler.py`（`_run_asm_passes`）、`scratchv/main.py`（`--peephole-asm`） |
| 单元测试 | `tests/test_asm_peephole.py` |
| 集成测试 | `tests/test_asm_peephole_integration.py` |
| 压力测试 | `tests/test_asm_peephole_stress.py` |
| 黑盒测试 | `tests/test_asm_peephole_blackbox.py` |
| 黑盒样例 | `tests/fixtures/asm_peephole/` |
| 性能基准 | `benchmarks/bench_asm_peephole.py` |
| 前后对比脚本 | `benchmarks/compare_peephole.py` |

---

## 交付摘要

- 默认规则：**8 条**（已移除不健全的「假交换删除」）
- 测试：**84 / 84 PASSED**
- 效果：DSL 基准约 -0.43%；合成高冗余约 -25%；综合样例约 -46%

### 一键复验

```bash
cd /home/z/ScratchV-main   # 或你的仓库根目录
source .venv/bin/activate
python -m pytest tests/test_asm_peephole*.py -q
python benchmarks/compare_peephole.py --markdown benchmark_reports/peephole_compare.md
```

---

## 阅读顺序建议

1. 本页（定位课题与路径）  
2. [新手教程](../docs/topics/13-窥孔优化器.md)  
3. [设计文档](../docs/topics/13-窥孔优化器-设计文档.md)  
4. [对比报告](../benchmark_reports/peephole_compare.md)  
5. 改代码时再看 [AI 开发文档](../docs/topics/archive/topic13_asm_peephole_guide.md)  
