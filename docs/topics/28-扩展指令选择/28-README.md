# Topic 28 使用指南

本课题完成整数算子扩展及标量 f32/f64 的 RISC-V F/D 指令选择。交付与完成条件见[交付与验收](28-交付与验收.md)，详细合同见[技术设计](28-扩展指令选择-设计文档.md)。

## 支持的操作和类型

所有新增能力通过 `--extended-isel` 启用。

| 操作 | i32 | f32 | f64 |
|---|---|---|---|
| `const/add/sub/mul/div/neg` | RV32IM 整数指令 | F 扩展 `.s` | D 扩展 `.d` |
| `rem/mod` | 有符号 `rem`；`mod` 为别名 | 不支持 | 不支持 |
| `min/max/abs` | 整数指令序列 | `fmin.s/fmax.s/fabs.s` | `fmin.d/fmax.d/fabs.d` |
| `sqrt` | 不支持 | `fsqrt.s` | `fsqrt.d` |
| IR `load/store` 与 spill | `lw/sw`，4 字节 | `flw/fsw`，4 字节 | `fld/fsd`，8 字节 |
| 比较 `== != < > <= >=`、条件分支 | GPR 比较/分支 | FPR 比较，GPR 布尔结果 | FPR 比较，GPR 布尔结果 |
| 单值返回 | `a0` | `fa0` | `fa0` |

DSL 常量写成 `const(1.5, dtype:f64)`；省略 dtype 时为 f32。IR 的类型系统位于 `scratchv/ir/types.py`，以 `DataType.FLOAT32/FLOAT64` 区分两种精度，仓库没有名为 `type_system.py` 的模块。ONNX 标量 `Add/Sub/Mul/Div/Sqrt/Min/Max` 保留 FLOAT/DOUBLE 类型。

| ISA | ABI | 类型 |
|---|---|---|
| `rv32im` | `ilp32` | i32 |
| `rv32imf` | `ilp32f` | i32、f32 |
| `rv32imfd` | `ilp32d` | i32、f32、f64 |

`naive/greedy/linear` 三种分配器均支持 GPR/FPR 分类分配和按类型 spill。不同 dtype 不做隐式转换；不支持的 ISA/ABI/type 组合明确报错。

## 示例与验收

在已安装项目依赖的 Python 环境中，从仓库根目录运行：

```bash
python -m scratchv.main benchmarks/topic28/i32_div_rem.dsl \
  --extended-isel --target-isa rv32im --target-abi ilp32 \
  --reg-alloc greedy -o /tmp/topic28_i32.s
python -m scratchv.main benchmarks/topic28/f64_sum.dsl \
  --extended-isel --target-isa rv32imfd --target-abi ilp32d \
  --reg-alloc linear -o /tmp/topic28_f64.s
clang --target=riscv32 -march=rv32imfd -mabi=ilp32d \
  -c /tmp/topic28_f64.s -o /tmp/topic28_f64.o

SCRATCHV_REQUIRE_RISCV_EXECUTION=1 python -m pytest -q tests/test_topic28_acceptance.py
```

四个交付示例分别是 [i32 除法和余数](../../../benchmarks/topic28/i32_div_rem.dsl)（−5）、[f32 求和](../../../benchmarks/topic28/f32_sum.dsl)（3.5）、[f64 求和](../../../benchmarks/topic28/f64_sum.dsl)（6.75）、[f64 点积](../../../benchmarks/topic28/f64_dot.dsl)（15）。验收测试自动汇编、链接并通过 QEMU 执行，无需 RISC-V C 运行库。工具与完整回归命令见[验收说明](28-交付与验收.md#验收方法)。

## 数值语义与边界

- 整数除法向零截断，余数与被除数同号：`div(-17,5)=-3`，`mod(-17,5)=-2`。除零及 `INT32_MIN/-1` 使用 RV32M 定义，不插入陷阱。
- 浮点算术和平方根固定使用 `rne`（最近值、相同距离取偶数）。DSL 的 `min/max` 使用 RISC-V 的 NaN、符号零及异常标志语义，不能用 Python `min/max` 直接替代。
- ONNX 浮点 Min/Max 保留 NaN：任意输入为 NaN，结果就是 NaN，变长输入逐对保留这条规则；运算产生规范化的 quiet NaN，sNaN 设置 invalid 标志。非 NaN 输入继续使用 RISC-V 的符号零规则（Min 选择 −0，Max 选择 +0）；本次 NaN 修复不改变这项既有目标约定。
- 硬浮点交付产物是合法 F/D 汇编，执行验证使用 QEMU；内置 encoder/emulator 目前仅支持 RV32IM。
- ONNX 硬浮点入口处理标量，支持至多一个标量浮点参数；每个函数的栈帧上限为 2032 字节。
- ONNX initializer 与 graph input 同名时，当前接口冻结 initializer，不生成可覆盖参数；不提供运行时覆盖。这是既有入口政策，ONNX 规范允许实现选择是否提供覆盖。
- `--extended-isel` 与 `--dag-isel`、LLVM 后端互斥。生产 linear 分配器为 `regalloc_linear.py`。
