# W4 运行接口

## 权重与地址

`scratchv/runtime/weight_bundle.py` 将生成 artifact 的有序 `external_weights` 与
`external_initializers` 写为 `manifest.json`、`weights.bin`。清单记录版本、小端、
64 字节对齐、总字节数、SHA-256，以及每个张量的名称、dtype、shape、偏移和长度。
模型原始分片另按 `probes/w1_qwen3_export/manifest.json` 校验；bundle 包含 IR literal，
不能把 ONNX initializer 的顺序当作调用顺序。

写入使用新目录和有界缓冲区；拒绝重复或错误绑定、非法数值、路径链接、溢出、截断及
损坏。调用方在读取、生成、打包期间须保持源映射和文件稳定。前后身份检查能发现常见
变化，不是防止恶意并发改写的文件锁。

地址布局单独预留代码、栈、输入、完整输出、workspace、权重和余量，校验 uint64
溢出及区域边界。固定模型原始权重合计 2,384,201,728 bytes，完整输出为 155,582,464
bytes；实际 bundle、workspace、地址和 guest 大小以生成的布局为准。guest 容量不等于
宿主进程 RSS，增加 QEMU `-m` 不能代替完整规划。

`weight_transport.py` 保持标准 bundle 不变，生成至多 512 MiB 的连续 raw-loader 分片，
绕开部分 QEMU/host 的单文件读取上限。启动前和退出后分别核对分片及拼接摘要；成功
回收仅删除本次创建的运输副本，保留标准 bundle 和运输清单。准备或运行失败保留诊断。

`copied_bytes` 与每段 `written_bytes` 统计已完整确认的写入块，不能作为失败后的文件
长度。`retained_bytes` 通过不跟随链接的文件身份/长度观测得到；无法完整观测时为
`null`，已知小计在 `retained_observed_bytes`，原因在 `retained_observation_errors`。
观测失败保持 FAIL 并保留原始操作异常，不能把未知当成 0。清理前校验创建归属与文件
身份，拒绝删除未成功创建、被替换或变为链接的路径。

## 外置 C ABI

```c
int scratchv_run_external(const void *const inputs[],
                         const void *const weights[],
                         void *workspace, size_t workspace_bytes,
                         void *output);
```

小模型的二参数 `scratchv_run(inputs, output)` 继续存在。调用方应根据 artifact 选择
入口，不将两种签名混用。host 通过装载/传输与 guest 交互，不能在宿主直接调用 RV64 函数。

| 参数 | 调用合同 |
|---|---|
| `inputs` | 按 `artifact.inputs` 排列；每个缓冲区满足对应 dtype/shape/nbytes，连续、目标小端、按元素大小对齐 |
| `weights` | 按 `artifact.external_weights` 排列，包含生成的 literal；整个同步调用期间只读且有效 |
| `workspace` | 非零需求时至少分配 `artifact.workspace_bytes`，起点按 8 字节对齐；显式容量参数不得小于需求 |
| `output` | 独立缓冲区，大小和类型取自 `artifact.output`；只有返回值为 0 才能使用结果 |
| 指针表 | 数量足够，按 `_Alignof(void *)` 对齐；表本身和所指数据均保持有效 |

只读输入/权重允许共享有效存储；output 和使用的 workspace 必须彼此分离，并与只读
张量及指针表分离。并发调用需要各自独立的 workspace 和 output。workspace 中间值
按最后使用点复用，不能当作持久结果保存；顺序调用返回后可复用，但旧输出需另行保留。

裸指针不携带实际分配容量、shape/dtype 或表长度，生成期与 Python/manifest 边界校验
这些元数据，调用方保证底层分配。非空检查不能证明指针有效；保留 ndarray 引用也不能
防止调用方手动关闭 mmap。只有空表可以传空表指针，零 workspace 可以传 NULL；当前
零元素张量/output 仍需要非空、正确对齐的占位地址。

## 编译与执行

`TensorCCodegen` 的 external 模式生成不内嵌大权重的循环，可用 `kernel_calls=True`
拆成独立 helper，控制完整图编译规模。`matmul_policy` 显式选择 sequential 或
blocked_fma；后者按 128 项 K 分块、块内显式融合乘加、块间 FP32 相加。保持非
fast-math、禁止隐式 FMA 合并，不能将两种累加顺序声称为逐位等价。

`riscv_external.py` 验证 ELF 架构、ABI、入口、装载段和布局，启动裸机 QEMU。guest
输出完成帧后，host 用 QMP 停止并转储精确大小输出，再退出；输出不是由 host C/NumPy
替算。实际 QEMU 在完整运行前须通过严格 FMA 位模式/FCSR 检查。

编译失败、超时、取消、协议错误、非有限输出、文件校验或清理错误均保持 FAIL；附加
清理错误不能覆盖原始错误。报告文件本身写入失败也不能保留可误读的 PASS。
当前完成通知仅在 QEMU 平台验证；未来真实硬件需单独确认设备内存屏障、MMIO 和缓存一致性。

## 观测字段

- `forward_wall_s`：从调用 QEMU 的 Popen 前到完成帧通过验证，含启动与装载。
- `qemu_wall_s`：同一起点到观测进程退出，含输出转储。
- `qemu.elapsed_s`：从运输准备前到进程/运输清理及运输报告保存后，不含函数前段的
  ELF/bundle 校验、输入打包/写入和最后 `run.json` 发布。
- 主报告 `stages_seconds.<case>_qemu`：完整运行器调用耗时。
- `sampled_peak_qemu_rss_bytes`：仅 QEMU 进程的采样 RSS；不包含父进程，不是硬性内存上限。

这些时间不能互换，也不能用 QEMU TCG 墙钟推算真实芯片周期或吞吐。
