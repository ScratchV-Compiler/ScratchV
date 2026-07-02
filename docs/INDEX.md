# ScratchV 文档导航

> 🚀 **新用户？** 按顺序阅读，零基础友好。

---

## 第一步：入门（30 分钟）

| 序号 | 文档 | 内容 | 时间 |
|------|------|------|------|
| 00 | [环境搭建指南](00-环境搭建指南.md) | 安装 Python、克隆代码、配置环境 | 15 min |
| 01 | [编译器概念入门](01-编译器概念入门.md) | 翻译官/乐高/办公桌类比讲解编译原理 | 15 min |
| 02 | [快速上手教程](02-快速上手教程.md) | DSL→IR→汇编→二进制，完整走一遍 | 30 min |

---

## 第二步：理解指标

| 序号 | 文档 | 内容 |
|------|------|------|
| 03 | [指标解读指南](03-指标解读指南.md) | 静态/动态指令、Cache、Dashboard 解读 |

---

## 第三步：深入模块（30 Topics）

详见 [topics/INDEX.md](topics/INDEX.md) — 按入门→中级→高级分级。
🌐 **在线浏览**: [课题指南网页版](topics/index.html) — 卡片式浏览 30 个课题，按难度过滤。

| 难度 | 数量 | 代表模块 |
|------|------|---------|
| 🔰 入门 | 15 个 | 汇编美化、窥孔优化、指令计数、CFG 构建、Cache 分析 |
| 📗 中级 | 9 个 | DSL 增强器、IR 系统、指令选择、寄存器分配、RISC-V 编译器 |
| 📕 高级 | 3 个 | LLVM 编译器、扩展指令选择、IR 验证 |
| ⬜ 规划中 | 3 个 | 循环展开、函数内联、SIMD 向量化 |

---

## 参考文档

| 文档 | 用途 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 完整架构说明：ONNX→RISC-V 双路径 |
| [04-故障排除FAQ](04-故障排除FAQ.md) | 环境/编译/测试/CI/Git 常见问题速查 |
| [developer_guide.md](developer_guide.md) | 开发者指南：代码规范、测试、CI |
| [optimization_guide.md](optimization_guide.md) | 优化指南：从哪里入手、怎么验证 |
| [verification.md](verification.md) | 验证方法：如何确认编译结果正确 |
| [CODING_STANDARDS.md](CODING_STANDARDS.md) | 代码风格规范 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南 |

---

## 项目主页

- [ScratchV.md](ScratchV.md) — 项目介绍（招募用）
- [ScratchV.html](ScratchV.html) — 项目介绍（网页版）

---

> 💡 **不知道看哪个？** 从 [00-环境搭建指南](00-环境搭建指南.md) 开始，一步步来。
>
> 🐛 **遇到问题？** 先查 [04-故障排除FAQ](04-故障排除FAQ.md)。
