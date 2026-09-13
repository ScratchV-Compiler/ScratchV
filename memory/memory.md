# ScratchV engineering memory

[2026-09-13] DSL 错误标记缩进须按实际行号前缀的可见宽度计算，不能固定为 6；覆盖 9/10、99/100 位数边界及空格/tab、有色和无色输出，避免多位行号导致 caret 左移。ANSI 控制序列不计入宽度。

[2026-09-13] 维护者要求课题测试与 benchmark 接入原有 `.github/workflows/ci.yml` 的 test/benchmark jobs，不新建独立 pipeline；pytest tests/ 已自动发现 DSL 测试，报告复用现有 artifacts。适用于本仓库课题 PR 的 CI 接入。

[2026-09-13] DSL 诊断 benchmark 应用同一脚本和固定正确输入，在独立进程中导入基线与当前 checkout，并检查实际模块路径；同时比较源码与 IR 摘要。适用于课题 9 前端回归，避免 editable install 导致两侧都测到当前代码；错误输入的新增诊断能力单独验收，性能目标达标状态与功能通过状态分开报告。
