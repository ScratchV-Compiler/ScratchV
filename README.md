# ⚡ ScratchV

**从 ONNX 到 RISC-V 机器码 — 一个你完全看得懂的 AI 编译器。**

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/ScratchV-Compiler/ScratchV/actions/workflows/ci.yml/badge.svg)](https://github.com/ScratchV-Compiler/ScratchV/actions)
[![Docs](https://img.shields.io/badge/docs-课程站点-blue)](https://scratchv-compiler.github.io/ScratchV/docs/)
[![Poster](https://img.shields.io/badge/📢-项目海报-orange)](https://scratchv-compiler.github.io/ScratchV/ScratchV.html)

---

> 📢 **[查看项目海报](https://scratchv-compiler.github.io/ScratchV/ScratchV.html)** — 了解项目招募信息、3个月学习路线、课题精选

## 🚀 我是新手，从哪开始？

```
1. 克隆代码     git clone https://github.com/ScratchV-Compiler/ScratchV.git
2. 安装依赖     cd ScratchV && pip install -e .
3. 打开课程     xdg-open docs/topics/html/index.html
```

或者一步步来：

| 步骤 | 看什么 | 时间 |
|------|--------|------|
| 🔧 搭环境 | [00-环境搭建指南](docs/00-环境搭建指南.md) | 15 min |
| 💡 理解概念 | [01-编译器概念入门](docs/01-编译器概念入门.md) | 15 min |
| 🏃 动手试试 | [02-快速上手教程](docs/02-快速上手教程.md) | 30 min |
| 📊 看懂数据 | [03-指标解读指南](docs/03-指标解读指南.md) | 15 min |
| 🐛 遇到问题 | [04-故障排除FAQ](docs/04-故障排除FAQ.md) | 随时查 |

> 🌐 **在线课程**: [在浏览器中学习](https://scratchv-compiler.github.io/ScratchV/docs/) — 带进度追踪、搜索、深色主题的交互式课程

---

## ⚡ 快速命令

```bash
make quick-start    # 打印新手引导
make test           # 运行全部测试
make bench-topic06  # 运行课题 06 TinyFive 性能测试套件
make bench-cnn      # 编译 CNN 模型 + 性能估算
make bench-ci       # 完整 CI 对比 (ScratchV vs LLVM)
make bench-reports  # 生成 Dashboard + 优化历史
```

---

## 📂 项目结构（核心目录）

```
ScratchV/
├── scratchv/             ← 编译器源码
│   ├── frontend/         ← ONNX 解析 / DSL 解析
│   ├── ir/               ← 中间表示 (三地址码 SSA)
│   ├── optimizer/        ← 5 个优化 pass
│   ├── backend/          ← RISC-V 代码生成 / LLVM 后端
│   ├── standalone/       ← 零依赖独立编译器 + 仿真工具
│   ├── ci/               ← CI 编排 + 性能仪表盘
│   └── analysis/         ← CFG 构建 / IR 验证
├── docs/                 ← 📚 完整文档体系
│   ├── 00~04-*.md        ← 新手入门 5 篇
│   ├── topics/           ← 30 个模块详解
│   └── topics/html/      ← 🌐 交互式课程站点
├── benchmarks/           ← 通用基准与课题 06 性能基线
├── tests/                ← 单元测试及课题 06 的 23 个 DSL 用例
├── scripts/              ← 工具脚本及课题 06 测试驱动
└── models/               ← 测试用 ONNX 模型
```

---

## 📊 核心数据 (cnn.onnx)

| 指标 | LLVM (float32) | ScratchV (Q16.16) | 追赶进度 |
|------|---------------|-------------------|---------|
| 静态指令 | 1,102 | 749 | ✅ 0.68× |
| 动态指令 | 18.5 亿 | 32.2 亿 | 🟡 1.74× |
| 优化前 | — | 78.0 亿 | 已减 58.7% |

> 📈 实时数据: [性能仪表盘](https://scratchv-compiler.github.io/ScratchV/dashboard.html)

---

## 🗺️ 编译器管线

```
ONNX 模型 (.onnx)
    │
    ├─→ ScratchV 原生路径 (Q16.16 定点 → RV32IM 机器码)
    │   ONNX解析 → MemoryPlan → CNNRISCVGenerator → .bin
    │
    └─→ LLVM 路径 (float32 → LLVM IR → RV64FD 汇编)
        ONNX解析 → LLVMCNNGenerator → .ll → llc → .s
```

---

## 📚 全部文档

| 入口 | 说明 |
|------|------|
| [📘 课程首页](docs/topics/html/index.html) | 交互式课程（推荐新手） |
| [📖 文档导航](docs/INDEX.md) | 全部 Markdown 文档索引 |
| [🏗️ 架构总览](docs/ARCHITECTURE.md) | ONNX→RISC-V 双路径详解 |
| [📊 性能仪表盘](https://scratchv-compiler.github.io/ScratchV/dashboard.html) | LLVM vs ScratchV 对比 |
| [课题 06 测试套件](docs/topics/06-性能测试套件使用说明.md) | TinyFive 正确性验证、Benchmark 与回归报告 |
| [📢 项目海报](https://scratchv-compiler.github.io/ScratchV/ScratchV.html) | 招募信息 + 3个月学习路线 + 课题精选 |

---

## 💬 加入社区

**QQ 群：`1106852304`**

有问题？想交流？欢迎加入 ScratchV 编译器学习群，一起讨论、一起成长。

---

## 🤝 贡献

ScratchV 是零基础友好的教育项目。查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

## 📄 许可

MIT License — 详见 [LICENSE](LICENSE)
