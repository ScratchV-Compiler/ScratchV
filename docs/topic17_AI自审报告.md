# Topic17 AI 自审报告：伪指令、寄存器泄露与执行正确性

## 1. 审查结论

本轮没有仅依靠“汇编文本看起来合理”作结论，而是按“语义表 → 分配 → spill/reload 重写 → 真实编码 → 执行结果”逐层验证。自审发现并修复了 5 类真实问题，其中最重要的是高压力 CFG 在 join 处读取错误寄存器，以及 TinyFive 验证适配层的 `LW` 只返回低 8 位。最终全量测试为 `555 passed`。

当前可以确认支持的整数伪指令范围是：

| 伪指令 | 分配语义 | 真实指令转换 | 验证 |
|---|---|---|---|
| `mv rd, rs` | `rd=def`，`rs=use` | `addi rd, rs, 0` | 编码等价 + TinyFive 执行 |
| `li rd, imm` | `rd=def`，立即数不占寄存器 | 小立即数 `addi`；大立即数 `lui`/`addi` | RV32 边界执行 |
| `max rd, rs1, rs2` | `rd=def`，两源为 use；右侧仅允许寄存器或立即数 0 | `bge` + 两条 copy + `j` | 双路径、负数、源/目标重叠 |
| `bnez rs, label` | `rs=use`，terminator | `bne rs, x0, label` | 编码等价 + taken/not-taken |
| `j label` | 无寄存器，terminator | `jal x0, label` | 编码等价 + 执行 |
| `call label` | 隐式定义 `ra`，caller-saved clobber | 本地目标 `jal ra, label` | 编码等价 + ABI spill 检查 |
| label | 无 def/use | 真实汇编标签 | CFG/编码 |

浮点伪指令目前只有分配语义元数据，不属于 RV32IM encoder 的可执行支持范围；本报告不把它们描述成“已经完整支持”。

## 2. 支持方式

### 2.1 单一语义来源

`scratchv/backend/machine_semantics.py` 为每个 `MachineOp` 显式记录 operand 的 def/use、立即数位置、控制流属性、隐式寄存器和 ABI clobber。两个 linear-scan 与 greedy 路径都读取同一份语义，避免对 `dst/src1/src2` 字段名进行猜测。模块还会检查是否有新增 opcode 漏填语义。

### 2.2 CFG 与活跃性

`scratchv/backend/regalloc_cfg.py` 按标签、条件分支、直接跳转和 fallthrough 恢复基本块，再迭代计算 `live_in/live_out`。当前仍使用保守的单段 live interval，不利用 lifetime hole，但不会因为值只在后继块使用就过早释放。

### 2.3 可执行 spill/reload

`scratchv/backend/regalloc_rewrite.py` 同时跟踪 vreg 所在寄存器、寄存器 owner 和栈槽是否包含最新值。所有 source 先完成 materialize，再选择 destination。目标可以复用 source 寄存器；若 source 后续仍活跃，则先 `sw` 保存旧值。高压力 CFG 对 edge-live 值使用固定栈槽作为前驱边之间的共同位置，避免路径相关状态泄露到 join。

### 2.4 ABI 与真实编码

call 前仅保存 call 后仍活跃且位于 caller-saved 寄存器的值，call 后使对应 resident 映射失效。`scratchv/backend/riscv_encoder.py` 将整数伪指令展开成 RV32IM 机器指令，严格拒绝未知寄存器、未定义目标、`max` 非零立即数，以及没有空闲临时寄存器时的 branch-immediate 展开，避免静默覆盖活值。

## 3. 自审发现的问题与修复

1. **TinyFive `LW` 验证假失败**：依赖包在当前 NumPy 上对 `uint8` 移位会截断高 24 位。适配层现在绑定兼容 `LW`，直接按 little-endian signed i32 读取四字节，并增加 `0x12345678` store/load 回归。
2. **双源 reload 冲突**：两个 spilled source 可能被装进同一物理寄存器。现在先保护全部 source，物理池不足时明确失败。
3. **destination 复用仍存活 source**：两寄存器机器上，三操作数指令必须允许 rd 与某个 source 重叠。现在在覆盖前保存旧 source，再执行指令，后续按需 reload。
4. **CFG join 错误映射**：某个分支上的定义曾被放到临时空闲寄存器，而 join 按全局映射读取另一个寄存器。现在定义保持全局 assignment，高压力 CFG 的 edge-live 值在每条前驱边规范化到栈槽；栈槽状态在块入口重新建立，不跨源代码顺序继承。
5. **伪指令展开冲突/静默 clobber**：`max` 的内部标签可能和用户标签重名；branch-immediate 在所有 `t0`–`t6` 已使用时曾回退覆盖 `t6`。现在内部标签避让用户标签，无可用 scratch 时明确报错。

同时修复了同一 vreg 的纯重定义被误判为“需要保存旧值”的问题，保证 CNN 在 pressure peak 11、19 个物理寄存器时仍为 0 spill。

## 4. 寄存器泄露验证

这里的“泄露”指分配完成后仍出现 vreg 名称，而不是内存资源泄漏。验证采用三层防线：

1. 对 `input_tensor`、`maximum_value` 等不符合 `v0` 正则的任意名称做 token 级检查，防止只检查 `%` 或 `v\d+` 漏报；
2. 检查 greedy 输出的每个 operand，不允许 `kind == "vreg"`；
3. 所有可执行汇编交给严格 encoder。未知寄存器不会被默认为某个物理寄存器，而是直接报错。

当前专项用例未发现分配后 vreg 泄露。

## 5. 验证矩阵与结果

- 伪指令与手写真实指令的二进制等价：`mv`、小/大 `li`、`bnez`、`j`、本地 `call`；
- TinyFive 执行：`mv`、RV32 全范围边界 `li`、`max` 双路径/负数/源目标别名、分支 taken/not-taken；
- 24 个随机直线程序：12 个固定 seed × 两个 linear-scan 版本，每个程序包含 6 个常量与 24 个随机 `add/sub/xor/and`，结果与 Python 的 RV32 模 2^32 参考语义一致；
- CFG 执行差分：两种分配器 × taken/not-taken，在两寄存器压力下结果均与参考一致；
- benchmark 对齐：

| 场景 | 物理寄存器 | pressure peak | excess | spill slots | spill stores | reloads | 汇编有效 |
|---|---:|---:|---:|---:|---:|---:|---|
| Simple | 8 | 5 | 0 | 0 | 0 | 0 | 未执行编码（合成压力 IR） |
| Dense | 5 | 29 | 24 | 28 | 63 | 75 | 未执行编码（合成压力 IR） |
| CNN | 19 | 11 | 0 | 0 | 0 | 0 | 是 |

全量：`555 passed`，`git diff --check` 无 whitespace error。

## 6. 尚未过度承诺的边界

- TinyFive 执行使用 ScratchV 自己的 encoder，能验证分配与模拟执行，但不是完全独立的工具链 oracle；合入前应再用 GNU RISC-V assembler/objdump 与 Spike 或 QEMU 做交叉验证。
- `call` 仅支持 flat binary 内的本地 JAL 范围目标；外部符号、远调用和 relocation 未实现。
- `max` 的立即数右操作数目前只支持 0，非零立即数会明确报错。
- 浮点伪指令只有 def/use 元数据，RV32IM encoder/TinyFive 路径未证明其真实编码与执行。
- linear-scan 使用保守单段 interval；spill 正确性已验证，但不是最优分配。
- 任意手写 MachineInstr 若固定使用 `t0`/`x5` 等分配池内物理寄存器，仍需要 fixed-register interference 建模；当前正式 selector 只固定使用 `a0`、`zero`、`ra` 等不在 19 个分配寄存器池中的 ABI 寄存器。
- greedy 当前是线性启发式，不是完整的路径敏感 CFG 分配器；P1 已保证 eviction/reload 与 call clobber 的基本正确性，但复杂分支应继续以 linear-scan 路径为主。
