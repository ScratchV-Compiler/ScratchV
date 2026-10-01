# ScratchV engineering memory

[2026-09-18] 课题 01 的解析成功和块数检查不能证明控制流执行正确；验收须核对比较契约、分支合流、循环变量更新与零次路径。当前 DSLInterpreter 的分支处理为空操作，不可作为控制流执行 oracle；未更新条件变量的旧 while 示例仅适合解析测试。

[2026-09-13] DSL benchmark 的详细日志按用例使用默认关闭的 details/summary 折叠，汇总表、失败检查和性能提示保留在外层；GitHub Markdown 与本地 HTML 均需支持，HTML 中的诊断源码必须转义。

[2026-09-13] DSL 错误标记缩进须按实际行号前缀的可见宽度计算，不能固定为 6；覆盖 9/10、99/100 位数边界及空格/tab、有色和无色输出，避免多位行号导致 caret 左移。ANSI 控制序列不计入宽度。

[2026-09-13] 维护者要求课题测试与 benchmark 接入原有 `.github/workflows/ci.yml` 的 test/benchmark jobs，不新建独立 pipeline；pytest tests/ 已自动发现 DSL 测试，报告复用现有 artifacts。适用于本仓库课题 PR 的 CI 接入。

[2026-09-13] DSL 诊断 benchmark 应用同一脚本和固定正确输入，在独立进程中导入基线与当前 checkout，并检查实际模块路径；同时比较源码与 IR 摘要。适用于课题 9 前端回归，避免 editable install 导致两侧都测到当前代码；错误输入的新增诊断能力单独验收，性能目标达标状态与功能通过状态分开报告。

[2026-09-19] 前端语义增强会有意改变固定 DSL 语料的 IR；基线 benchmark 应按用例显式列出允许变化，继续拒绝未列出的 IR 差异，并保留同语料、诊断正确与性能阈值检查。适用于多个前端课题共用同一 A/B benchmark 的 CI。

[2026-09-19] 课题专项 benchmark 应接入原有 CI job，摘要表常显，逐用例源码、IR 与汇编放入默认关闭的 details/summary，并同时产出 JSON/Markdown/HTML 到既有 artifact。适用于需要在 Actions Summary 展示详细编译日志的课题。

[2026-09-26] Topic 04 的 PassRegistry 注册只声明可用工厂，build 才按名称构造并调度；禁用先于构造。IR 和汇编分阶段建管线，常量折叠与汇编常量加载合并不可混用；新增调用点须使用统一工厂，避免旧 ConstantFolder(program).run() 接口。

[2026-10-01] ONNX initializer 是否为标量应按秩为 0 判断，不能按元素数为 1 判断。保留 CNN 的 fc2.bias 为 (1,) 后，IR 指令数由 17 变 16，Topic 17 的峰值寄存器压力由 11 变 12；更新统计基线时须同时检查张量形状、真实汇编执行和零 spill，不能恢复错误标量化来迎合旧断言。专项测试通过后仍须覆盖现有 CNN 与寄存器分配回归。

[2026-10-02] 真实 Qwen3 小模型探测须保留 Q/K RMSNorm、全 head RoPE 和独立 head_dim，不能用 W1 手工 partial-RoPE 图替代。共享 IR 当前仅返回一个张量，逐层诊断可在 ONNX 副本追加 Reshape/Concat 打包观察点，并独立核对 ORT 具名输出和普通 logits-only 图；诊断图通过不能替代普通图验收。固定 CPU 导出环境、种子和输入，并分别验证 padding 隔离、因果性及故障非零退出。
