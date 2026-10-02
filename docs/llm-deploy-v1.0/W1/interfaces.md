# 接口基线 v1.1：待团队冻结确认

> 上游：[开发计划.md](../开发计划.md) §2.2、§4.3 W1。
> 状态（2026-10-02）：**按当前实现核对的候选基线，团队确认未完成**。
> `docs:interfaces` 只检查文档结构；“有版本号”不等于冻结会议已通过。正式决议见 §7。

## 0. 四类接口与路线

| # | 接口 | 当前实现边界 |
|---|---|---|
| 一 | 前端 | ONNX 文件 → shared IR `Program`，权重通过 `parser.initializers` 单独绑定 |
| 二 | IR | 基础张量算子图、形状/dtype/attrs 契约、verifier 和 NumPy 解释器 |
| 三 | 后端 | `CompilerDriver(backend="tensor-c")` → 显式张量循环 C → Zig/LLVM → RV64GC ELF |
| 四 | 运行时 | 指针数组模型 ABI；QEMU virt 裸机原始输入装载与 UART 输出协议 |

上述新路径已经本地数值验证。原 `riscv` 标量选择器及旧 `llvm` 路径不由本探测证明；旧选择器的 MatMul 占位/张量与 FP32 覆盖缺口仍保留。不能将“仓库有 LLVM 后端”或 CNN standalone 可运行等同于 Qwen FP32 已支持。

## 1. 冻结与变更规则

正式冻结前由 E1–E5 核对本文，E2 记录结论、日期和提交。冻结后接口修改需 PR、E2 批准与团队通知；兼容扩展升次版本，破坏性变更升主版本。消费方负责验证：E2 验前端，E3 验 IR，E4 验编译产物，E5 验端到端数值。不得绕过共享 IR 直接从 ONNX 另建隐藏的执行语义。

本次修订替换旧草案的“融合节点强制”“INT32 三指针入口”“partial RoPE”提议。这是纠正草案与实际实现的冲突，不伪称团队已经批准这些变化。

## 2. 接口一：前端（ONNX → IR）

```python
parser = ONNXParser()
program = parser.parse(model_path)
result = IRInterpreter(program).run(inputs, initializers=parser.initializers)
```

- `parse(model_path: str) -> Program` 保持主入口；`initializers` 是名称到 NumPy 数组的映射。仅传 `Program` 而丢失数组绑定不能运行带权重图。
- 外部 ONNX 权重分片相对于模型文件目录解析。ONNX 结构检查与完整权重载入不是一回事；完整图数值执行必须准备真实分片，不能把 LFS pointer 当权重。
- 不支持的 op/domain、属性或形状应显式报错。禁止跳过节点或替换成零张量来通过探测。
- 当前以导出图的基础算子组合表达 RMSNorm、RoPE、SwiGLU 与 GQA。无需为证明正确性强制新增 `RMSNORM/ROPE/SWIGLU/ATTENTION` opcode；后续融合须保留等价语义，并同时更新 verifier、解释器、后端与对照用例。
- 固定 batch=1、L=256、FP32、无 KV Cache。ONNX schema 中的 IDs 为 INT64，不能未经明确转换在前端改成 INT32。

PR #89 合成图和官方 Qwen3 小模型是两种不同输入：前者无 Q/K RMSNorm且为 partial RoPE；后者保留官方实现的 Q/K RMSNorm、全 head_dim RoPE 和独立 head_dim。完整 Qwen 图也应按固定 revision 的官方配置解释。

## 3. 接口二：共享 IR 数据结构与算子契约

权威定义在 `scratchv/ir/types.py`，逐算子检查和 NumPy 语义在 `scratchv/verification/ir_numpy_ops.py`，解释器在 `scratchv/verification/ir_interpreter.py`。文档不再复制易过期的 opcode 数量表。

`Value` 的名称/dtype/shape、`Instruction` 的 opcode/operands/attrs、`Function` 参数与返回值、`Program` globals 是跨阶段共享信息。SSA 名称唯一；输入、初始化数组、返回数组的 dtype/shape 必须与定义一致。新增 opcode 或语义属性必须同步其消费者，不能假定只改 enum 后 verifier 和 pass 就自动兼容。

| 语义 | 当前基线 |
|---|---|
| FP32 舍入 | 常量折叠遵循算子 dtype 的逐步舍入，不用 Python double 折叠后只在末尾截断 |
| 形状 | 本部署路线要求编译期确定，广播、Reshape、Slice、Gather 等按各自契约检查 |
| RMSNorm | 沿对应末轴归一化；保留官方模型 Q/K 独立 RMSNorm，eps=`1e-6` |
| RoPE | Qwen3 为**全 head_dim** 旋转；theta=`1e6`，positions 固定 `0..255` |
| GQA | 按组重复 KV 头；小模型 Q/KV=4/2，完整模型 16/8；不能混淆 head_dim 与 hidden/heads |
| mask | FP32 加性因果 + key-padding，允许位置 0、屏蔽位置 `finfo(float32).min` |
| 返回 | 当前执行接口取单个返回张量；诊断图以 Flatten+Concat 打包 29 个检查点，schema 保存偏移/形状 |

解释器支持循环/分支，不意味着所有后端也支持。本次 `tensor-c` 只接受单函数直线图；控制流、动态形状或不支持的 dtype 应在编译时失败。融合 Attention 是可选优化，不是当前正确性验收的前置；从基础算子模式识别融合仍然可行。

## 4. 接口三：后端（IR → RISC-V）

```python
driver = CompilerDriver(
    CompilerConfig(backend="tensor-c", optimize_level="all", verify_ir=True)
)
result = driver.compile("model.onnx", "model.c")
# result.success 为真后使用 driver.tensor_artifact 和 driver.initializers
```

标准 CLI：

```bash
python -X utf8 -B -m scratchv.main model.onnx --backend tensor-c --optimize all --verify-ir -o model.c
```

`TensorCArtifact` 提供 `source`、有序 `inputs: tuple[TensorSpec, ...]`、`output`、`workspace_bytes`、`constant_bytes`、`function_name` 与 `compile_flags`。`TensorSpec` 包含 name/dtype/shape，并可读取 numpy_dtype/nbytes/size；dtype 是 IR `DataType`，不是任意字符串。

代码生成器将静态权重放只读段，中间张量放有界静态 arena，按 SSA 最后使用复用空间；不在栈上分配大张量。默认工作区上限 256 MiB。编译 flags 包含 `-fno-fast-math -ffp-contract=off -fno-strict-aliasing`；数学函数使用 Zig 随附 musl libm，不用粗略近似。

目标为 RV64GC、LP64D、小端、64 位指针。Zig/LLVM 承担 C 到机器码，ScratchV 负责 ONNX/IR、优化、循环生成与内存规划。Zig 命令采用 `riscv64-linux-musl` 以获得数学库，但 guest 入口、链接布局与执行均为裸机，不运行 Linux，也不依赖 Linux syscall。

编译验证 `none/basic/all` 优化后的 IR，QEMU 验证 `none/all` 的普通图与诊断图。`--verify-ir` 只证明 IR 契约；最终数值必须用探测脚本执行 ELF。旧 `--verify` 不会执行这份 C 产物，不能用它替代 QEMU gate。

## 5. 接口四：运行时 FFI

### 5.1 模型调用 ABI

```c
int scratchv_run(const void *const inputs[], void *output);
```

在 LP64D ABI 下，a0 是输入指针数组的地址，a1 是输出缓冲区地址，返回 a0 为状态码。输入顺序由 artifact.inputs 定义，不能靠字段名字猜测。调用者提供匹配 dtype/shape、连续、适当对齐的数组；输出不得覆盖输入或内部工作区。当前返回码 0 成功，1 指针错误、2 数值错误、3 Gather 越界。Python 运行器会先验证输入形状/dtype，收到非零 guest 状态或损坏帧即报错。

权重和工作区是生成函数的私有静态存储；函数**不可重入**，不支持并发共享 arena。

| 张量 | dtype | 小模型 shape | 完整导出 shape |
|---|---|---|---|
| input_ids | INT64 | `[1,256]` | `[1,256]` |
| attention_mask | FP32 | `[1,1,256,256]` | `[1,1,256,256]` |
| logits | FP32 | `[1,256,128]` | `[1,256,151936]` |

这是模型调用 ABI，不是 HTTP/tokenizer 接口。mask 不是二维 0/1 mask。完整 logits 为 155,582,464 bytes（约 148.4 MiB）；当前小模型内存验证不能覆盖完整模型。

### 5.2 构建和 QEMU 传输

`scratchv/runtime/riscv_tensor.py` 提供 `discover_toolchain`、`build_riscv_tensor`、`run_riscv_tensor`。工具可从环境变量或仓库 `output/tools/` 发现；缺工具显式失败，不隐式下载安装。

当前 QEMU 使用 virt/TCG、512 MiB RAM、无 BIOS/OS，自带启动汇编和链接脚本。输入通过 raw loader 写入保留区域（地址 `0x9c000000`，容量 64 MiB），输出通过 UART 二进制帧返回。帧包含标识、状态、长度、校验和和结束标记；解析拒绝截断/损坏/异常状态，超时失败并清理本次子进程。该传输是现有探测协议，不应误写为通用 Linux FFI 或 mmap。

### 5.3 后续运行时边界

Tokenizer、采样、生成循环和完整模型权重装载仍待实现/验收。当前小模型权重静态嵌入；完整 0.6B 权重不应直接套用 512 MiB guest 配置。W4 需明确大权重装载策略、容量检查、生命周期及分块输出；选择 Linux mmap 时须另建 Linux guest/user-mode 路线，裸机不具备该系统调用。

## 6. 能力与剩余缺口

| 项目 | 当前状态 | 后续责任 |
|---|---|---|
| 前端基础算子与 IR 数值 | 已补齐本次图所需算子与用例，真实两层图通过 | E1/E2 继续完整图对照 |
| 旧选择器张量 MatMul/FP32 | 未修复；新路径不经过它 | E3：保留范围说明，是否另行补齐由团队决定 |
| tensor-c / RV64 运行器 | 本地及545e696的[Linux部署任务](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/36983000988/job/110761571841)通过，含28次两层QEMU执行与artifacts上传 | E3/E4/E5：第二人独立复现、后续提交Checks；不外推通用主CI状态 |
| 完整模型 IR 数值与容量 | 未由小模型验收 | E2/E4/E5：逐步扩大配置及预训练权重验证 |
| 完整 ONNX 门禁 | 历史导出、本地verify，以及545e696的[Linux download/ORT重型任务](https://github.com/yuki-328/ScratchV/actions/runs/36983119833/job/110761955767)均通过 | E1/E5：第二人复现及后续提交Checks；不冒称本轮重新导出或全部CI通过 |
| Tokenizer / 生成 / 大权重加载 | 当前探测不覆盖 | E4：W4 实现及专项验收 |

## 7. 团队待决议与确认

以下均为**候选决定，待团队确认**，不自动勾选：

- [ ] D1：接受 RV64GC/LP64D + QEMU virt 裸机为当前部署探测路线；保留其他后端范围。
- [ ] D2：接受输入指针数组/单输出 ABI、不可重入及 UART 传输边界。
- [ ] D3：接受 ONNX INT64 IDs；如另需 INT32 接口，显式设计转换与范围校验。
- [ ] D4：接受 tensor-c + Zig/LLVM FP32 实现；不宣称原标量选择器已修复。
- [ ] D5：接受基础算子图作为正确性基线，融合节点作为后续优化。
- [ ] D6：确认小模型静态嵌入边界，另行设计完整权重加载和内存计划。
- [ ] D7：确认接口版本、变更审批及消费方测试责任。

| 项 | 记录 |
|---|---|
| 会议日期 / 对应 commit | 待填写 |
| E1 前端 / E2 IR / E3 后端 / E4 运行时 / E5 测试确认 | 待逐人填写 |
| 尚未同意的条款 / 后续责任人与截止时间 | 待填写 |
| 最终冻结版本 | 待 E2 确认后填写 |

## 8. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 原草案 | v0.1 / v1.0 提议 | 四接口勘察与初始融合节点/三指针入口设想；未完成团队冻结 |
| 2026-10-02 | v1.1 候选 | 按实际实现校正基础图、全 RoPE、INT64 与指针数组 ABI、tensor-c/RV64 裸机及验收边界 |
