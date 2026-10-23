# W5 Host 三轮文本对话

状态：Host 历史管理和官方模板一致性已实现；这里的验证不加载模型权重，不代表完整 QEMU 多轮语义验收。

## 接口和语义

`Conversation.from_directory(tokenizer_directory, system_prompt=None)` 从同一固定本地 snapshot 加载 tokenizer 与 `tokenizer_config.json` 中的官方 `chat_template`。生产运行只需 `tokenizers` 和 `Jinja2`，不依赖 Transformers 或 Torch。模板通过 Jinja2 的不可变沙箱渲染，固定 `add_generation_prompt=True`、`enable_thinking=False`，不手写替代 Qwen3 的模板。

`turn(user_text, forward=backend, max_new_tokens=..., cancelled=..., timeout_seconds=...)` 复用 Host `generate`，返回同一 `GenerationResult`。仅支持文本 system/user/assistant：system 只能在构造时提供，后续 user/assistant 顺序由成功的 turn 维护，不支持工具、图片或其他角色。`history` 是不可变快照；`reset()` 仅清除已完成回合，保留初始 system 文本。

每轮将完整历史重新套用官方模板并编码，不保留 KV cache，不暗中删除早期消息，也不截断到 256 tokens。超过容量会在调用 backend 前报错。原始完成模式的三条参考用例保持 raw prompt，不自动套用聊天模板。

一次 turn 只有成功返回且至少生成一个 token，才同时提交 user 和 assistant 消息。异常、取消、超时、零 token 预算，以及恰好占满 256-token 上下文且没有生成空间的情况，均不修改历史。EOS-only 回复可作为合法空 assistant 回复提交。预算或容量中止后的回复保留返回的文本，`stop_reason` 明确记录其终止原因。历史存储文本；下轮重新编码，未承诺历史 token IDs 原样拼接。

超时与取消沿用 Host 生成合同。该层的协作检查不能中断正在阻塞的 backend；QEMU adapter 负责运行中的取消、超时以及进程清理。一个 Conversation 实例不能并发或重入使用，活动 turn 中拒绝另一个 turn 或 reset。

## 相关验证

```bash
python -m pytest tests/test_conversation.py -q
```

固定本地 tokenizer 存在时，测试还会使用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 独立核对官方 Transformers 的 `apply_chat_template`，无网络下载、无模型前向。可通过 `SCRATCHV_QWEN3_TOKENIZER_DIR` 指定本地快照，默认是仓库 `assets/qwen3-0.6b/tokenizer`。

2026-10-23：40 项通过，无跳过。覆盖三轮精确历史顺序、非 thinking 格式、两会话隔离、reset、不可变快照、角色/文本校验、零预算、EOS-only、容量边界、异常原样传播、取消/超时回滚以及重入保护。其中官方 tokenizer 对照覆盖有/无 system 的各三轮，共六份渲染文本和 token 序列；assistant 文本在此模板测试中为预设值，未作完整模型生成。

默认单核、手动/自动 hart 选择及 padding 策略由已有 backend 保持，本模块不改变它们。
