# W1 环境搭建文档 — RISC-V + QEMU

## 目录

1. [环境概述](#1-环境概述)
2. [前置条件](#2-前置条件)
3. [安装步骤](#3-安装步骤)
   - 3.1 [安装 RISC-V 交叉编译器](#31-安装-risc-v-交叉编译器)
   - 3.2 [安装并配置 QEMU 用户态模拟器](#32-安装并配置-qemu-用户态模拟器)
   - 3.3 [验证安装](#33-验证安装)
4. [项目结构](#4-项目结构)
5. [编译与运行 RISC-V 程序](#5-编译与运行-risc-v-程序)
   - 5.1 [汇编程序](#51-汇编程序)
   - 5.2 [C 程序](#52-c-程序)
   - 5.3 [使用 Makefile](#53-使用-makefile)
6. [RV32IMF 架构说明](#6-rv32imf-架构说明)
7. [常见问题](#7-常见问题)
8. [参考资源](#8-参考资源)

---

## 1. 环境概述

本环境为 **ScratchV** 项目提供 **W1 阶段**所需的 RISC-V 开发环境，目标是：

- 使用 **RV32IMF** 指令集（32-bit RISC-V + 整数乘法 + 单精度浮点）
- 通过 **QEMU 用户态模拟** 运行 RISC-V 程序（无需硬件开发板）
- 支持手写汇编（`.s`）和 C 语言的交叉编译

### 环境规格

| 项目 | 内容 |
|------|------|
| 宿主机架构 | x86_64 |
| 操作系统 | Ubuntu 24.04 LTS |
| RISC-V 工具链 | `gcc-riscv64-linux-gnu` (Ubuntu 13.3.0) |
| APT 源 | 清华大学镜像源 (推荐) |
| QEMU | qemu-user-static (8.2.2), 包含 qemu-riscv32 用户态模拟器 |
| 目标 ABI | `ilp32f` (32-bit 整数 + 硬浮点调用约定) |
| 目标架构 | `rv32imf` (RV32I + M 扩展 + F 扩展) |

---

## 2. 前置条件

- x86_64 架构的 Linux 系统（推荐 Ubuntu 24.04）
- sudo 权限（用于安装工具链）
- 网络连接（用于下载 QEMU 预编译包）

## 3. 安装步骤

> 如果你是全新环境，按以下步骤操作即可完整复现。

### 3.1 安装 RISC-V 交叉编译器

```bash
sudo apt update
sudo apt install -y gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu
```

验证安装：

```bash
riscv64-linux-gnu-gcc --version
# 输出: riscv64-linux-gnu-gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

riscv64-linux-gnu-as --version | head -1
riscv64-linux-gnu-ld --version | head -1
riscv64-linux-gnu-objdump --version | head -1
```

> **说明**：安装的虽然是 `riscv64-linux-gnu-*` 工具链，但通过 `-march=rv32imf -mabi=ilp32f` 标志可以编译 32-bit RISC-V 程序。

### 3.2 安装 QEMU 用户态模拟器

推荐使用 APT 从清华源安装 `qemu-user-static`：

```bash
sudo apt install -y qemu-user-static
```

> **关于清华源**：本文档假设你已全局配置清华 APT 镜像源。如未配置，可参考[清华大学 Ubuntu 镜像源说明](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)。配置后，更新包列表：`sudo apt update`

验证安装：

```bash
which qemu-riscv32-static
qemu-riscv32-static --version
# 输出: qemu-riscv32 version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.16)
```

**备选方案**（如网络不畅）：从 Debian 仓库直接下载预编译二进制

```bash
# 设置项目目录
PROJECT_DIR=/path/to/your/aic
mkdir -p $PROJECT_DIR/toolchain

# 下载并解压（需要网络访问 Debian 源）
wget -O /tmp/qemu-user-static.deb \
  "http://ftp.debian.org/debian/pool/main/q/qemu/qemu-user-static_8.2+dfsg-4_amd64.deb"

dpkg-deb -x /tmp/qemu-user-static.deb /tmp/qemu-extract
cp /tmp/qemu-extract/usr/bin/qemu-riscv32-static $PROJECT_DIR/toolchain/qemu-riscv32
chmod +x $PROJECT_DIR/toolchain/qemu-riscv32
rm -rf /tmp/qemu-user-static.deb /tmp/qemu-extract
```

### 3.3 验证安装

完整的端到端测试会编译并运行一个 RV32IMF 程序，验证整数运算、乘法、浮点运算的正确性。

手动执行编译和运行：

```bash
# 编译：将汇编源码编译为 RV32IMF 可执行文件
riscv64-linux-gnu-gcc \
  -march=rv32imf \
  -mabi=ilp32f \
  -nostdlib -static \
  -o tests/test_rv32imf \
  tests/test_rv32imf.s

# 运行：使用 QEMU 用户态模拟执行
/path/to/toolchain/qemu-riscv32 tests/test_rv32imf

# 检查退出码：0 表示全部测试通过
echo $?   # 应输出 0
```

---

## 4. 项目结构

```
aic/
├── env_setup.sh            # 环境变量设置脚本 (source 后可用别名)
├── Makefile                # 编译/运行/反汇编 快捷命令
├── docs/
│   └── w1_environment_setup.md   # 本文件
├── toolchain/
│   └── qemu-riscv32        # QEMU RISC-V 32-bit 用户态模拟器
└── tests/
    ├── test_rv32imf.s      # 汇编测试程序 (RV32IMF)
    ├── test_rv32imf.c      # C 语言测试程序 (RV32IMF)
    ├── test_rv32imf        # 汇编测试编译产物
    └── test_rv32imf_c      # C 测试编译产物
```

---

## 5. 编译与运行 RISC-V 程序

### 5.1 汇编程序

**完整编译命令**：

```bash
riscv64-linux-gnu-gcc \
  -march=rv32imf \
  -mabi=ilp32f \
  -nostdlib -static \
  -o output_program \
  your_source.s
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `-march=rv32imf` | 目标架构：RV32I + M(乘除) + F(单精度浮点) |
| `-mabi=ilp32f` | ABI：32-bit 整数指针 + 硬浮点（浮点参数通过 f 寄存器传递） |
| `-nostdlib` | 不使用标准库（因我们使用 `_start` 而非 `main`） |
| `-static` | 静态链接 |

> **关于 `_start` 和 `main`**：QEMU 用户态模式下，程序入口可以是 `_start`（需要 `-nostdlib`）或 `main`。使用 `_start` 更简洁，无需链接 C 运行时库。

**汇编程序模板**：

```asm
.section .text
.globl _start

_start:
    # 你的代码写在这里

    # 退出程序 (Linux syscall: exit)
    li a0, 0          # 退出码 0
    li a7, 93         # SYS_exit
    ecall
```

### 5.2 C 程序

```bash
riscv64-linux-gnu-gcc \
  -march=rv32imf \
  -mabi=ilp32f \
  -nostdlib -ffreestanding -static \
  -o output_program \
  your_source.c
```

**C 程序模板**：

```c
void _start(void) {
    // 你的代码
    int a = 10, b = 20;
    int c = a + b;     // 测试整数加法

    // 浮点运算
    float x = 1.5f, y = 2.5f;
    float z = x + y;   // 测试浮点加法

    // 退出
    __asm__ volatile ("li a0, 0; li a7, 93; ecall");
}
```

### 5.3 使用 Makefile

项目提供了 Makefile 方便操作，需要先安装 `make`：

```bash
sudo apt install -y make
```

常用命令：

```bash
# 编译所有测试
make all

# 运行所有测试
make test

# 仅运行汇编测试 / 仅运行 C 测试
make asm_test
make c_test

# 反汇编查看生成的指令
make disasm

# 查看文件头信息
make elfinfo

# 清理编译产物
make clean
```

---

## 6. RV32IMF 架构说明

### 指令集子集

| 子集 | 全称 | 本项目使用的典型指令 |
|------|------|---------------------|
| **RV32I** | 32-bit 整数基础指令 | `add`, `sub`, `and`, `or`, `li`, `lui`, `bne`, `ecall` |
| **M** | 整数乘除扩展 | `mul` |
| **F** | 单精度浮点扩展 | `fadd.s`, `fmul.s`, `fmv.w.x`, `fmv.x.w` |

### 寄存器约定 (RV32IMF Calling Convention)

| 寄存器 | ABI 名称 | 用途 | Caller/Callee Saved |
|--------|----------|------|---------------------|
| x0 | zero | 硬编码 0 | — |
| x1 | ra | 返回地址 | Caller |
| x2 | sp | 栈指针 | Callee |
| x5-x7 | t0-t2 | 临时寄存器 | Caller |
| x8-x9 | s0-s1 | 保存寄存器 | Callee |
| x10-x17 | a0-a7 | 参数/返回值 | Caller |
| x18-x27 | s2-s11 | 保存寄存器 | Callee |
| x28-x31 | t3-t6 | 临时寄存器 | Caller |
| f0-f7 | ft0-ft7 | 浮点临时 | Caller |
| f8-f9 | fs0-fs1 | 浮点保存 | Callee |
| f10-f17 | fa0-fa7 | 浮点参数/返回值 | Caller |
| f18-f27 | fs2-fs11 | 浮点保存 | Callee |
| f28-f31 | ft8-ft11 | 浮点临时 | Caller |

### Linux 退出 syscall

```asm
li a0, <exit_code>    # 退出码
li a7, 93            # SYS_exit (Linux RISC-V 系统调用号)
ecall                # 触发系统调用
```

---

## 7. 常见问题

### Q1: `multiple definition of _start` 错误

**原因**：链接时默认包含了 C 运行时库的 `crt1.o`，其中也定义了 `_start`。  
**解决**：添加 `-nostdlib` 标志，告诉链接器不使用标准启动文件。

### Q2: `ABI is incompatible with that of the selected emulation` 错误

```
target emulation `elf64-littleriscv' does not match `elf32-littleriscv'
```

**原因**：`riscv64-linux-gnu-gcc` 默认是 64-bit 目标，而 `-march=rv32imf` 生成 32-bit 代码。

**解决**：同时指定 `-march=rv32imf -mabi=ilp32f`，但必须配合 `-nostdlib` 使用，否则链接器会尝试用 64-bit 的 crt 库链接 32-bit 的目标文件。

### Q3: qemu-riscv32 报 `Permission denied`

**解决**：`chmod +x /path/to/qemu-riscv32`

### Q4: 浮点运算结果不对

**可能原因**：

1. 编译时忘记 `-mabi=ilp32f`（使用 `ilp32` 而非 `ilp32f` 会导致浮点参数通过整数寄存器传递，造成误解）
2. 单精度和双精度混淆：RV32F 仅支持 `float`（32-bit），`double`（64-bit）需要 D 扩展

---

## 8. 参考资源

- [RISC-V 指令集手册 (Volume I: Unprivileged)](https://riscv.org/technical/specifications/)
- [RISC-V Assembly Programmer's Manual](https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm-manual.adoc)
- [QEMU RISC-V User Mode Emulation](https://www.qemu.org/docs/master/user/main.html)
- [riscv-collab/riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [RISC-V Linux syscall 列表](https://github.com/torvalds/linux/blob/master/arch/riscv/include/uapi/asm/unistd.h)

---

## 附录：详细安装日志 (2026-05-25)

本环境在以下配置上搭建并验证通过：

- 系统：Ubuntu 24.04 LTS (x86_64)
- 工具链：gcc-riscv64-linux-gnu 13.3.0 (apt 从清华源安装)
- APT 源：清华大学开源软件镜像站
- QEMU：8.2.2 (apt 从清华源安装 qemu-user-static)
- 测试：RV32I 整数运算 + RV32M 乘法 + RV32F 浮点加/乘法 —— 全部通过

### 快速复现（使用清华源）

```bash
# 1. 配置清华 APT 源（如未配置）
# 参考: https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
# 编辑 /etc/apt/sources.list，替换所有源为 https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ 开头的镜像

# 2. 安装工具链和 QEMU (sudo)
sudo apt update
sudo apt install -y gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu qemu-user-static

# 3. 验证安装
riscv64-linux-gnu-gcc --version
qemu-riscv32-static --version

# 4. 编译并测试（可选）
riscv64-linux-gnu-gcc -march=rv32imf -mabi=ilp32f -nostdlib -static -o /tmp/test /tmp/test.s
qemu-riscv32-static /tmp/test
echo "Exit: $?"    # 应输出 0
```

### APT 源配置建议

清华大学镜像源速度快，推荐配置。修改 `/etc/apt/sources.list`：

```bash
sudo sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo apt update
```

或者直接编辑文件，参考清华源[帮助页面](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)的最新配置。
