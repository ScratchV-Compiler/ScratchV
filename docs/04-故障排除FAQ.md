# 04 — 故障排除 FAQ

> **目标**: 快速解决 ScratchV 使用过程中的常见问题。
> **使用方式**: Ctrl+F 搜索你的报错关键词。

---

## 目录

1. [环境问题](#1-环境问题)
2. [编译问题](#2-编译问题)
3. [测试问题](#3-测试问题)
4. [CI 问题](#4-ci-问题)
5. [Git 问题](#5-git-问题)
6. [性能问题](#6-性能问题)

---

## 1. 环境问题

### Q: `pip install -e .` 报 `error: externally-managed-environment`

**原因**: 系统的 Python 不允许直接 pip install（常见于 Ubuntu 24.04+、macOS Homebrew）。

**解决**:
```bash
# 方案 A: 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate
pip install -e .

# 方案 B: 加 --break-system-packages（不推荐，可能搞乱系统 Python）
pip install -e . --break-system-packages
```

---

### Q: `python3: command not found`

**原因**: 你的系统上 Python 命令可能是 `python` 而不是 `python3`。

**解决**:
```bash
# 先看看你有哪些 Python
which python python3 python3.12 2>/dev/null

# 如果没有 python3，创建别名
alias python3=python   # 临时
# 或者安装 Python 3
sudo apt install python3   # Ubuntu
brew install python@3.12   # macOS
```

---

### Q: `ModuleNotFoundError: No module named 'scratchv'`

**原因**: 没有安装 ScratchV 包，或虚拟环境没激活。

**解决**:
```bash
# 确认虚拟环境已激活（提示符有 (venv)）
which python3    # 应该指向 .../ScratchV/venv/bin/python3

# 重新安装
pip install -e .
```

---

### Q: `pip install tinyfive` 失败

**原因**: `tinyfive` 可能需要编译 RISC-V 仿真器的依赖。

**解决**:
```bash
# 先安装编译工具
sudo apt install build-essential python3-dev   # Ubuntu
xcode-select --install                          # macOS

# 再安装
pip install tinyfive
```

如果还是失败，TinyFive 是可选的——核心编译器不需要它。

---

### Q: 国内 `pip install` 太慢

**解决**:
```bash
# 使用清华镜像
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者永久设置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. 编译问题

### Q: `FileNotFoundError: models/graph/cnn.onnx`

**原因**: 模型文件不存在。可能需要先下载或生成。

**解决**:
```bash
# 确认文件是否存在
ls -la models/graph/cnn.onnx

# 如果不存在，检查是否在正确的目录
pwd   # 应该在 ScratchV 项目根目录
```

---

### Q: 编译 CNN 模型时 `MemoryError` 或 OOM

**原因**: 某些仿真模式（如 ProfiledMachine）会加载整个 binary 到内存并逐条仿真，可能消耗大量内存。

**解决**:
```bash
# 只用估算模式（不仿真，内存小）
python3 scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx \
    -o /tmp/cnn.bin --estimate --report

# 不用 --tinyfive（TinyFive 仿真也需要内存）
```

---

### Q: 生成的 RISC-V 代码跑不起来

**检查清单**:
1. 确认目标架构是 RV32IM（32 位整数）还是 RV64FD（64 位浮点）
2. 如果是 ScratchV 原生路径，确认 Q16.16 乘法后是否做了 `srai 16` 截断
3. 确认地址空间分配不重叠（MemoryPlan 是否正确）
4. 尝试用 `--asm output.s` 输出汇编，人工检查

---

### Q: `--compare` 报 `llvmlite not found`

**原因**: LLVM 对比功能需要 `llvmlite` 包。

**解决**:
```bash
pip install llvmlite
```

如果 `llvmlite` 安装失败（尤其是在非 x86 架构上），可以跳过对比功能，只用 ScratchV 原生路径。

---

## 3. 测试问题

### Q: `make test` 失败

**第一步**: 看具体哪个测试失败了：

```bash
# 运行测试并看详细信息
python3 -m pytest tests/ -v --tb=long
```

**常见原因**:

| 失败模式 | 可能原因 | 解决 |
|---------|---------|------|
| `ImportError` | 依赖没装 | `pip install -e .` |
| `AssertionError` | 代码改动引入了 bug | 检查你最近的修改 |
| `FileNotFoundError` | 测试需要的文件缺失 | 确认在项目根目录运行 |
| 超时/hang | 测试死循环 | Ctrl+C 中断，检查最近的循环修改 |

---

### Q: 某个特定测试一直过不了

**解决**:
```bash
# 单独跑这个测试，看详细信息
python3 -m pytest tests/test_specific.py::test_name -v --tb=long

# 加 print 调试
python3 -m pytest tests/test_specific.py::test_name -v -s
```

`-s` 参数让 print 输出显示在终端（默认被 pytest 捕获）。

---

### Q: `make bench` 失败

**原因**: benchmark 通常依赖更多文件（模型、测试数据）。

**解决**:
```bash
# 确认模型文件存在
ls models/graph/cnn.onnx models/single_op/

# 如果单算子模型不存在，先生成
make split-models
```

---

## 4. CI 问题

### Q: GitHub Actions CI 失败

**第一步**: 看 CI 日志，找到第一个报错。

**常见 CI 问题**:

| 错误 | 原因 | 解决 |
|------|------|------|
| `python: command not found` | CI 环境 Python 命令不同 | 确认 CI 配置使用 `python3.12` |
| `git clone` 超时 | GitHub 网络不稳定 | 使用本地 mirror 或重试 |
| OOM (Out of Memory) | CI runner 内存不够 | 减少并行度，或 skip 大内存测试 |
| 测试 hang | 某个测试死循环 | 加 timeout 限制 |

---

### Q: 本地测试通过但 CI 不通过

**常见原因**:
- **Python 版本不同**: CI 固定用 Python 3.12，检查你本地版本：`python3 --version`
- **依赖版本不同**: CI 可能装了不同版本的 onnx/numpy
- **文件路径**: CI 的工作目录可能和本地不同

**解决**: 尽量用 `make bench-ci` 模拟 CI 的完整流程。

---

### Q: `make bench-ci` 生成的 dashboard 不显示数据

**原因**: JSON 数据文件可能为空或格式不对。

**解决**:
```bash
# 检查 JSON 输出
cat benchmark_reports/ci_data.json | python3 -m json.tool | head -50

# 确认数据有内容
python3 -c "
import json
with open('benchmark_reports/ci_data.json') as f:
    data = json.load(f)
print(f'Models: {len(data.get(\"models\", []))}')
print(f'Keys: {list(data.keys())}')
"
```

---

## 5. Git 问题

### Q: `git push` 报 `Permission denied`

**原因**: SSH key 没配置，或没权限推送到仓库。

**解决**:
```bash
# 检查 remote URL
git remote -v

# 如果是 HTTPS，可能需要配置 token
# 如果是 SSH，检查 key
ssh -T git@github.com
```

---

### Q: 合并冲突 (merge conflict)

**解决**:
```bash
# 看哪些文件冲突
git status

# 手动编辑冲突文件，删除 <<<<<<<, =======, >>>>>>> 标记
# 然后
git add <冲突文件>
git commit
```

---

### Q: `git clone` 太慢或超时

**解决**:
```bash
# 配置代理
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

# 或者用浅克隆（只取最近一次提交）
git clone --depth 1 https://github.com/ScratchV-Compiler/ScratchV.git
```

---

## 6. 性能问题

### Q: 为什么 ScratchV 编译的代码比 LLVM 慢？

这是预期的。核心原因：

1. **Q16.16 定点** vs **float32**: 每个乘加需要 ~12 条整数指令 vs LLVM 的 ~2 条浮点指令
2. **无循环优化**: 没有 LLVM -O3 的自动循环展开/交换/向量化
3. **地址计算开销**: 没有 GEP 指令，需要手动 MUL+ADD 链

详情见 [03-指标解读指南.md](03-指标解读指南.md) 第 5 节。

---

### Q: 编译太慢怎么办？

ScratchV 的编译速度通常不是瓶颈（CNN 模型几秒就能编译完）。如果觉得慢：

```bash
# 不用仿真，只用估算
--estimate    # 快，瞬间完成
# 不用
--tinyfive    # 慢，需要逐条仿真
```

---

### Q: 我的优化没看到效果

**检查清单**:
1. 优化的是内层循环吗？（外层循环里的改动能被内层频率放大）
2. 跑 benchmark 确认了吗？不要凭感觉
3. 静态指令数减少了吗？（如果静态指令减少但动态指令没减少，说明优化在错误的地方）
4. 有没有引入新的开销？（比如少了一条 `add` 但多了三条 `lw`）

---

## 还没解决？

1. 搜索 [GitHub Issues](https://github.com/ScratchV-Compiler/ScratchV/issues) — 看有没有人遇到过同样的问题
2. 如果没找到，创建新的 Issue，附上：
   - 你的操作系统和 Python 版本
   - 完整的错误信息（复制粘贴，不要截图）
   - 你执行了什么命令
   - 你期望什么结果，实际发生了什么

---

> 🔗 相关文档：[00-环境搭建指南.md](00-环境搭建指南.md) | [03-指标解读指南.md](03-指标解读指南.md) | [developer_guide.md](developer_guide.md)
