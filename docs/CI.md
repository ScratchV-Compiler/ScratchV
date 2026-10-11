# PR 验收与 Actions 报告入口

普通 PR 由 **CI** 统一运行通用测试、编译器 benchmark、Topic 06、相关部署验收和 Linux 文档检查。各部分保留自己的 job、固定依赖和 artifact，在同一个 Actions 运行页查看；已有测试与数值门槛不因合并入口而减少。

**AI Code Review** 保持独立：它从基础分支读取配置，使用仓库密钥和评论权限。执行 PR 代码的 CI 只授予 `contents: read`，不继承这些权限。因此每次普通 PR 更新预期触发两个工作流，而不是把 AI Review 合并进测试环境。

| 查看位置 | 内容 |
| --- | --- |
| CI 的 `CI overview` | 每项检查的实际结果、是否在本次范围内；选中的任务失败、取消或跳过都会使总验收失败 |
| 同页部署 job summary | 本次小模型数值与完整前端解析数据，以及对应详细报告 |
| 同页 benchmark summary | 折叠的编译器与 Topic 06 详细报告，完整产物保留在 artifacts |
| 手动完整验证运行页 | 已交付阶段的完整模型实测指标和原始数组；不拿历史报告填充本次未运行项目 |

部署和文档路径条件集中在 `scripts/ci_scopes.json`。路由根据 PR base/head 的完整 merge-base diff 判断，删除和改名移出目录也会触发原范围；改动公共 CI 配置会检查所有已交付范围。范围判断失败不能默认为无需验收。相同 PR 的新提交取消旧 CI，避免重复占用 runner。

普通 PR 不运行完整 Qwen3 IR 或完整 QEMU 前向。保留 `LLM Deploy v1.0` 手动入口及各阶段已有完整验证选项；在包含 W4 的分支，W4 的可复用流程默认 `run_full: false`，只有显式完整调用才传 `true`。Nightly 的原有显式启用条件不变。范围外项目显示 **NOT RUN**，不代表通过。

同一运行页中的多个 job 不等于多个 workflow run；job 分开可保留环境隔离和失败定位。此前运行记录仍保留，新的合并入口不会删除历史日志与证据。若仓库另有按旧 check 名称设置的分支保护，维护者应核对新的 Checks 名称；`CI acceptance` 是统一的失败闭合验收入口，不能绕过其失败结果。
