# ScratchV engineering memory

[2026-10-19] 合并上游张量执行管线时，ConstantFolder 默认沿用 checked IR 的溢出错误语义；扩展 RV32 编译路径显式启用 rv32_integer_semantics，保留 i32 DIV/ABS 溢出的硬件结果。浮点目标仍保留舍入与 fflags 保护。ONNX 默认维持张量形状和数据绑定，只有扩展 RV32 路径将单元素初始化器作为标量常量；常量位模式须贯通解释器、优化器及独立审计。未知维度的零哨兵只用于标量后端拒绝，不能成为张量输出的精确形状；真实 dim_value=0 必须保留。形状推断会在代码生成前拒绝声明矛盾，测试应保留这一更早的拒绝边界。适用于 Topic 28 与 W1/W2/W3 管线共存。

[2026-10-20] 循环上界常量跨回边存活会增加局部寄存器分配的栈流量；应在循环头按需生成，小立即数可用有符号 SLTI 比较。仅当紧邻后继只有当前块的唯一 fallthrough 入边时，greedy 才能保留寄存器映射；分支目标、合流与回边仍须使用规范栈副本。性能验收使用 Topic 06 的实际 TinyFive 动态计数及既有成本门槛，同时用独立汇编器/QEMU 核对比较边界与两侧分支结果。

[2026-10-23] 合并上游具名 IR 绑定规则时，参数和指令结果须优先于引用上的常量提示；绑定名应在选择器公共逐函数入口初始化，覆盖重写函数/ABI 降低的扩展选择器。多函数寄存器命名空间会重命名机器操作数，集成测试应核对绑定与立即数的区别及实际执行结果，同时保留原始 IR 名称不变。

[2026-10-14] 项目规范要求 pytest 测试类组织；新增测试使用 Test... 类，辅助函数保持模块级。迁移既有测试时保留参数化装饰器、fixture 参数和多行 DSL/汇编字符串；对比 AST 与去掉类名后的收集用例，防止整理时改变逻辑或漏收集。适用于测试组织调整。

[2026-10-14] LLVM 浮点常量按声明精度编码后用 bitcast 表示，避免十进制格式丢失有效位及加零丢失负零；显式 bit_pattern 必须保留并验证。LLVM 验证通过仍可能算错，回归需实际执行并核对返回位模式。LLVM 优化同样不能用宿主精度绕过 f32 输入舍入，适用于 typed DSL、ONNX 常量及显式 pass。

[2026-10-13] ONNX 浮点 Min/Max 的 NaN 传播不同于 RISC-V fmin/fmax 选择正常数字的规则，IR 必须保留来源要求，不能只按同名算子映射。用 ONNX 官方参考实现核对合法模型，再实际执行目标程序；同时检查 qNaN/sNaN 标志、普通数/无穷大和既有 DSL 行为。NaN 补偿运算只能在 NaN 路径执行，否则额外加法会引入溢出或 invalid；值参照工具也要遵循同一 IR 标记和目标符号零约定。

[2026-10-20] 课题 01 的解析成功和块数检查不能证明控制流执行正确；验收须核对比较契约、分支合流、循环变量更新与零次路径。当前 DSLInterpreter 的分支处理为空操作，不可作为控制流执行 oracle；未更新条件变量的旧 while 示例仅适合解析测试。benchmark 语料须显式初始化累加器，并采用上游与当前解析器均支持的语法；模拟错误即使用例没有 expected 文件也必须判失败，不能依赖未定义变量自动补零。

[2026-09-13] DSL benchmark 的详细日志按用例使用默认关闭的 details/summary 折叠，汇总表、失败检查和性能提示保留在外层；GitHub Markdown 与本地 HTML 均需支持，HTML 中的诊断源码必须转义。

[2026-09-13] DSL 错误标记缩进须按实际行号前缀的可见宽度计算，不能固定为 6；覆盖 9/10、99/100 位数边界及空格/tab、有色和无色输出，避免多位行号导致 caret 左移。ANSI 控制序列不计入宽度。

[2026-09-13] 维护者要求课题测试与 benchmark 接入原有 `.github/workflows/ci.yml` 的 test/benchmark jobs，不新建独立 pipeline；pytest tests/ 已自动发现 DSL 测试，报告复用现有 artifacts。适用于本仓库课题 PR 的 CI 接入。

[2026-09-13] DSL 诊断 benchmark 应用同一脚本和固定正确输入，在独立进程中导入基线与当前 checkout，并检查实际模块路径；同时比较源码与 IR 摘要。适用于课题 9 前端回归，避免 editable install 导致两侧都测到当前代码；错误输入的新增诊断能力单独验收，性能目标达标状态与功能通过状态分开报告。

[2026-09-19] 前端语义增强会有意改变固定 DSL 语料的 IR；基线 benchmark 应按用例显式列出允许变化，继续拒绝未列出的 IR 差异，并保留同语料、诊断正确与性能阈值检查。适用于多个前端课题共用同一 A/B benchmark 的 CI。

[2026-09-19] 课题专项 benchmark 应接入原有 CI job，摘要表常显，逐用例源码、IR 与汇编放入默认关闭的 details/summary，并同时产出 JSON/Markdown/HTML 到既有 artifact。适用于需要在 Actions Summary 展示详细编译日志的课题。

[2026-09-26] Topic 04 的 PassRegistry 注册只声明可用工厂，build 才按名称构造并调度；禁用先于构造。IR 和汇编分阶段建管线，常量折叠与汇编常量加载合并不可混用；新增调用点须使用统一工厂，避免旧 ConstantFolder(program).run() 接口。

[2026-10-01] ONNX initializer 是否为标量应按秩为 0 判断，不能按元素数为 1 判断。保留 CNN 的 fc2.bias 为 (1,) 后，IR 指令数由 17 变 16，Topic 17 的峰值寄存器压力由 11 变 12；更新统计基线时须同时检查张量形状、真实汇编执行和零 spill，不能恢复错误标量化来迎合旧断言。专项测试通过后仍须覆盖现有 CNN 与寄存器分配回归。

[2026-10-02] 真实 Qwen3 小模型探测须保留 Q/K RMSNorm、全 head RoPE 和独立 head_dim，不能用 W1 手工 partial-RoPE 图替代。共享 IR 当前仅返回一个张量，逐层诊断可在 ONNX 副本追加 Reshape/Concat 打包观察点，并独立核对 ORT 具名输出和普通 logits-only 图；诊断图通过不能替代普通图验收。固定 CPU 导出环境、种子和输入，并分别验证 padding 隔离、因果性及故障非零退出。

[2026-10-14] W2 部署的 Python 依赖统一声明在 pyproject.toml 的 llm-deploy extra，通过 pip install -v -e ".[llm-deploy]" 一次安装；设计与测试文档每个 Bash 块只放一条 Python 命令，不要求手动安装依赖或设置 Shell 环境。工具路径由运行入口自动发现，QEMU 等系统工具单独注明前提；保留模型身份、精度阈值和输出目录约束。

[2026-10-14] W2 固定 Qwen3 资产位于 assets/qwen3-0.6b：完整 ONNX/权重分片及 Tokenizer 大文件由 Git LFS 管理，配置与许可证由普通 Git 管理；唯一哈希清单仍为 probes/w1_qwen3_export/manifest.json。W2 入口只读取本地资产做校验与测试，不提供 download/verify 模式参数或下载实现，CI 按门禁拉取并缓存 LFS；生成的随机模型和报告留在 output/，不得把 LFS pointer 当作真实资产。

[2026-10-18] W1/W2 的 pytest 对等测试放在 tests/llm-deploy/w1、w2，保留原 probes；公共 fixture 校验本地完整资产，并在 pytest 临时目录生成固定种子的两层模型和执行产物。每周仅一份 README，列测试项、内容和单条 Python 启动命令。该模型与工具链测试集须显式指定 tests/llm-deploy 运行，普通 tests 收集不自动包含它；保留各原始输入与比较标准，其中 W1 独立 NumPy 参考允许 FP64 中间值，不能误加 FP32 dtype 相等要求。
