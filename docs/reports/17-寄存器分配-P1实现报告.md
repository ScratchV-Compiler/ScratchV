# Topic17 P1 实现报告：CFG 活跃性与可执行溢出

## 1. 本阶段结论

P1 已完成两项 Wiki 交付目标：

1. 线性分配路径按标签和 terminator 恢复机器基本块，计算 successor、predecessor、live-in 和 live-out，并把数据流结果用于活跃区间修正与块间值携带。
2. greedy 分配器在寄存器被复用后会删除旧映射、插入 spill `sw`，并在旧值再次使用前插入 reload `lw`，不再出现“只有 store、没有 reload”的假溢出。

同时修复了 P0 和 AI 自审中暴露的执行错误：同一条双源指令的两个 spilled vreg 不再被依次 reload 到同一个寄存器；目标寄存器可以在先保存旧源值后安全复用仍存活的 source；高压力 CFG 的各条前驱边通过固定栈槽交接值，join 块不再读取错误的物理寄存器。若物理池本身小于指令要求的不同源寄存器数，分配器会明确失败，而不是生成静默错误结果。

## 2. CFG 与活跃性

`scratchv/backend/regalloc_cfg.py` 从扁平 `LsInstruction` 序列恢复基本块：

- leader：入口、标签、terminator 后一条指令；
- 条件分支：目标边 + fallthrough 边；
- `j`/`jal`：已知直接目标边；
- `jalr`：间接目标，不猜测 successor；
- `call`：不是 terminator，保留 fallthrough。

每个基本块先计算局部 `uses` 和 `defines`，再迭代求解：

```text
live_out[B] = union(live_in[S])，S 属于 successors[B]
live_in[B]  = uses[B] union (live_out[B] - defines[B])
```

当前分配器仍使用保守的单段 live interval。CFG 数据流保证穿过“本块没有局部 use”的值仍覆盖块边界；进一步利用 lifetime holes 复用寄存器属于后续优化，不影响本阶段正确性。

## 3. Spill 重写

`scratchv/backend/regalloc_rewrite.py` 统一服务两个 linear-scan 版本，实际跟踪：

- `vreg -> resident register`；
- `register -> current owner`；
- 栈槽中是否保存了该 vreg 的最新值；
- 当前指令全部 source operand 的保护集合。

重写顺序为：保护所有当前源 → 为缺失源选择不同寄存器并 reload → 选择可与 source 重叠的目标寄存器（旧 source 仍存活时先保存）→ 发射指令 → 必要时写回栈槽。高压力 CFG 会在所有前驱边上把 edge-live 值规范化到固定栈槽，块入口不继承源代码顺序中的临时状态。这样既避免了“先 reload A 到 t0，再 reload B 到 t0，最后执行 `add t0,t0,t0`”，也避免了某个分支把定义写入临时寄存器、join 却按全局映射读取另一个寄存器。

## 4. CALL ABI

`MachineOp.CALL` 使用语义表中的 caller-saved clobber 集：`ra`、`a0`–`a7`、`t0`–`t6`。

- linear：call 前只保存位于 clobbered 寄存器且 call 后仍存活的值；call 后使这些 resident 映射失效，后续按需 reload；
- greedy：采用相同 clobber 信息保存和失效映射；
- `s0`–`s11` 中的 live value 不产生 caller-save；
- 本地 `call label` 在 flat encoder 中展开为真实 `jal ra, label`，并复用严格的未定义标签检查。超过 JAL 范围的外部/远符号仍应交给支持 ELF relocation 的正式汇编链接流程。

## 5. Greedy 修复

修复点包括：

- eviction 后删除旧的 `vreg -> physical register` 映射；
- source 再次出现时，从对应栈槽 reload；
- 当前指令的其他 source register 不参与 victim 选择；
- 最后一次使用后的寄存器及时释放；
- spill/reload 使用标准 `sw rs, offset(sp)` / `lw rd, offset(sp)` 操作数顺序；
- CNN 在 19 寄存器银行下不再因为简单循环复用制造假 spill。

## 6. 验证结果

| 场景 | 寄存器 | pressure peak | excess | slots | spill stores | reloads |
|---|---:|---:|---:|---:|---:|---:|
| Simple | 8 | 5 | 0 | 0 | 0 | 0 |
| Dense | 5 | 29 | 24 | 28 | 63 | 75 |
| CNN | 19 | 11 | 0 | 0 | 0 | 0 |

专项测试覆盖：

- if/else/join CFG 的 successor 与 live-in；
- if/else 两条路径在两寄存器压力下均与参考结果一致；
- 两寄存器高压程序在 TinyFive 上执行结果为 7；
- 24 个随机直线程序（12 seeds × 两种 linear-scan）与 Python RV32 参考语义一致；
- 一寄存器无法表达两个不同 source 时明确报错；
- caller-saved 值跨 call 的 store/reload；
- callee-saved 值跨 call 不产生额外访存；
- greedy 发生 eviction 后存在配对 reload；
- `call` 与 `jal ra,label` 编码一致。
- `mv`、`li`、`max`、`bnez`、`j` 的编码/执行等价性及任意名称 vreg 不泄露；
- `li` 覆盖 RV32 有符号边界，`max` 覆盖目标与源重叠、内部标签冲突；
- TinyFive `LW` 四字节读取经过独立回归，避免验证器只读低 8 位造成假失败。

全量测试结果：`555 passed`。详细审查过程见 `docs/topic17_AI自审报告.md`。

## 7. 后续边界

P1 解决的是静态分配和静态 spill site 的正确性。运行时动态 spill 次数仍取决于循环执行次数，需要 Spike/QEMU/TinyFive 的执行轨迹计数。进一步优化方向包括 lifetime holes/split intervals、栈槽复用、成本感知 victim，以及正式 ELF relocation/链接支持。
