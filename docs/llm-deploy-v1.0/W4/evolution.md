# W1–W4 的衔接与后续准备

## 当前共同合同

| 阶段 | 提供的基础 | 后续阶段须保持 |
|---|---|---|
| W1 | 固定 ONNX 资产、共享 IR、小图解释/目标执行及接口约定 | 模型身份、输入/输出规格、原有小模型调用兼容 |
| W2 | 完整解析、张量类型/shape、Tokenizer 元数据、输入/mask、greedy 和停止辅助函数 | INT64 token、FP32 mask、logits 宽度 151936、有效长度与 padding 规则 |
| W3 | 完整固定输入、ORT 参考、IR 精度、检查点映射、资源及证据约定 | 相同资产/case 与参考配置，IR 的独立误差门槛与来源记录 |
| W4 | 外置权重 ABI、完整地址规划、RV64 前向、目标精度与生命周期 | 数学策略显式、调用方容量/所有权、实际 guest 输出、原始数据可复核 |

完整模型输入为 `input_ids: INT64[1,256]` 和加性 `attention_mask: FP32[1,1,256,256]`，
输出为 `FP32[1,256,151936]`。指针次序来自 artifact，不由字典顺序或名字猜测。
mask 使用有效 key 与因果关系；padding query 仍可看到前面的有效 key。有效长度来自
未填充 token 数量，不能通过查找 pad ID 推断。greedy 取 `logits[0,valid_length-1]`，
不用固定最后一个位置；相同最大值取较小 ID。

模型 logits 宽度不等于 tokenizer 的可解码 ID 数。固定资产的 tokenizer 基础词表为
151643，包含 added tokens 的 ID 数为 151669；其余输出位置不能凭宽度假定为合法 token。
W5 必须显式检查采样 ID 能否解码，并与参考生成策略对齐，不能悄悄屏蔽这些 logits
或把无效 ID 替换为其他 token 后宣称输出等价。

后端接收共享 `Program` 和初始化张量，不依赖 ONNXParser 私有字段。图检查点映射是
诊断元数据，不能据名称替代运算实现。W1 小模型二参数 ABI 与 W4 五参数 ABI 并存，
外置内存接口详见[运行合同](runtime-contract.md)。

## W5 前置工作

1. 先用 W2 的输入、最后有效位置采样与停止函数搭建调用适配，明确新 token 追加、
   EOS、最大生成数和 L=256 容量边界。原 prompt 中的 EOS 不应被当作已生成的 EOS。
2. 当前 W4 runner 是单次前向：每次启动 guest、装载输入/权重、转储输出并退出。
   持久 guest、权重驻留、多次调用命令协议、取消/复位及 output 生命周期尚需单独设计。
   外置 C 函数可顺序复用 caller 内存，不代表运行器已实现持久会话。
3. 默认范围仍是无 KV cache。生成循环应先保持该语义，与同一参考逐 token 核对，再做
   复用与性能优化；不要在一次改动里同时引入 KV cache 和新的数值策略。
4. 增加固定 prompt 的 token/停止边界记录及端到端验证，按 W5 表验收 Paris、3 组 prompt、
   多轮与部署脚本。完整 logits 单例通过不自动证明文本生成已通过。

## 未来 MLIR/LLVM 边界

W4 保持 `IR → Tensor-C → Zig/LLVM → RV64`。不为准备工作引入 MLIR 依赖或通用双向
转换器。已有 `Value` 包含 dtype/shape；部分 builder 中间值可暂留默认 `shape=()`，
转换前必须完成类型推导/验证，不能将默认值直接认定为标量。

静态 ONNX 输入必须有明确维度；未知维度不能当作零。参数和初始化张量需要精确形状，
显式返回签名也必须与实际结果一致。另有一项共享 IR 语义需要后续统一：现有参数或
LOAD 结果的 `is_constant` 可作为提示，不能覆盖运行时绑定；原生指令选择路径对同名
常量标记的优先级与解释器、Tensor-C 不一致。该路径未参与 W4 Tensor-C 链路，未来
复用原生后端或接入转换器前，须先明确并回归测试常量标记与 SSA 定义的优先级。

未来可先定义静态纯张量子集的单向转换，明确保留：

- op 语义与属性、广播、轴顺序、reshape 规则、输入输出次序及错误条件；
- dtype、整数位宽、静态 shape 和常量原始位模式，不经十进制/FP64 改写；
- SSA、use-def、副作用与返回合同；不支持的控制流明确拒绝；
- 逻辑 tensor 类型与物理布局/offset/strides、地址空间、对齐和所有权的区别；
- 外部权重逻辑名称与实际 bundle 的绑定、版本和摘要；
- MatMul/归约顺序、显式 FMA 与非 fast-math 策略，不能仅用 FP32 dtype 表示数值等价；
- 目标 index/指针宽度、端序、ABI 与 checkpoint 的可观察性。

可评估 `linalg/tensor → bufferization → memref/loop → LLVM`，转换必须满足目标 dialect
及接口要求。相似 IR 格式不会自动适配 pass；任意低层优化也不能保证可逆恢复高层张量
语义。接口依据见 [Dialect conversion](https://mlir.llvm.org/docs/DialectConversion/)、
[Bufferization](https://mlir.llvm.org/docs/Bufferization/)。具体转换实现应带独立语义及数值
回归，并保留现有 ORT 与 RV64 验证链。
