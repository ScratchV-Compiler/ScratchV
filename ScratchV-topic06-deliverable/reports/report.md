# ScratchV DSL 编译器性能测试报告

## 测试概览

- Schema 版本: 1
- 运行模式: normal
- 类别筛选: null
- 名称筛选: null
- 生成时间: 2026-08-08 11:45:41
- 用例总数: 23
- 通过数量: 13
- 失败数量: 10
- 通过率: 56.5%
- 测试目录: `D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\tests_main`
- 汇编输出目录: `D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build`
- 性能基线文件: `D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\reports\benchmark_baseline.json`
- 性能退化阈值: 5.00%
- 单次编译超时: 30s
- 单次模拟超时: 5s

## 测试结果

| 用例 | 类别 | 状态 | 编译返回码 | 编译日志 | 模拟后端 | 指令数 | Benchmark 次数 | Benchmark 停止原因 | 平均指令数 | 95% 置信区间 | 最小 | 最大 | 编译耗时(s) | 模拟耗时(s) | 总耗时(s) | 基线 | 变化量 | 变化率(%) | 退化阈值(%) | 是否退化 | 预期输出 | TinyFive 输出 | 输出匹配 | 汇编文件 |
|---|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| add_relu_relu | activation | PASS | 0 | null | tinyfive | 7 | null | null | null | null | null | null | 0.0788 | 0.1846 | 0.2639 | null | null | null | null | null | 7 | 7 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_relu_relu.s |
| relu_add | activation | PASS | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0889 | 0.1463 | 0.2355 | null | null | null | null | null | 3 | 3 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_add.s |
| relu_only | activation | PASS | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0751 | 0.1408 | 0.2160 | null | null | null | null | null | 0 | 0 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_only.s |
| relu_twice | activation | PASS | 0 | null | tinyfive | 6 | null | null | null | null | null | null | 0.0783 | 0.1466 | 0.2251 | null | null | null | null | null | 4 | 4 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_twice.s |
| if_else | branch | FAIL | 0 | null | timeout | 0 | null | null | null | null | null | null | 0.0777 | 5.0203 | 5.0982 | null | null | null | null | null | 5 | None | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_else.s |
| if_relu | branch | FAIL | 0 | null | timeout | 0 | null | null | null | null | null | null | 0.0792 | 5.0179 | 5.0974 | null | null | null | null | null | 0 | None | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_relu.s |
| if_then | branch | FAIL | 0 | null | timeout | 0 | null | null | null | null | null | null | 0.0768 | 5.0203 | 5.0975 | null | null | null | null | null | 13 | None | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_then.s |
| add_chain | elementwise | PASS | 0 | null | tinyfive | 4 | null | null | null | null | null | null | 0.0753 | 0.1592 | 0.2349 | null | null | null | null | null | 9 | 9 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_chain.s |
| add_chain_3 | elementwise | PASS | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0741 | 0.1464 | 0.2209 | null | null | null | null | null | 14 | 14 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_chain_3.s |
| add_fan_in_4 | elementwise | PASS | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0710 | 0.1330 | 0.2042 | null | null | null | null | null | 10 | 10 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_fan_in_4.s |
| add_reuse | elementwise | PASS | 0 | null | tinyfive | 4 | null | null | null | null | null | null | 0.0700 | 0.1361 | 0.2062 | null | null | null | null | null | 10 | 10 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_reuse.s |
| vector_add | elementwise | PASS | 0 | null | tinyfive | 3 | null | null | null | null | null | null | 0.0700 | 0.1458 | 0.2161 | null | null | null | null | null | 5 | 5 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\vector_add.s |
| loop_add_4 | loop | PASS | 0 | null | tinyfive | 22 | null | null | null | null | null | null | 0.0794 | 0.1453 | 0.2250 | null | null | null | null | null | 5 | 5 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_add_4.s |
| loop_add_chain_4 | loop | PASS | 0 | null | tinyfive | 26 | null | null | null | null | null | null | 0.0772 | 0.1505 | 0.2280 | null | null | null | null | null | 9 | 9 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_add_chain_4.s |
| loop_relu_add_4 | loop | PASS | 0 | null | tinyfive | 30 | null | null | null | null | null | null | 0.0764 | 0.1367 | 0.2132 | null | null | null | null | null | 2 | 2 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_relu_add_4.s |
| dot_4 | reduction | FAIL | 0 | null | tinyfive | 3 | null | null | null | null | null | null | 0.0720 | 0.1365 | 0.2087 | null | null | null | null | null | 70 | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_4.s |
| dot_8 | reduction | FAIL | 0 | null | tinyfive | 3 | null | null | null | null | null | null | 0.0755 | 0.1409 | 0.2167 | null | null | null | null | null | 36 | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_8.s |
| dot_relu_4 | reduction | PASS | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0759 | 0.1373 | 0.2142 | null | null | null | null | null | 0 | 0 | True | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_relu_4.s |
| dot_relu_8 | reduction | FAIL | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0695 | 0.1336 | 0.2032 | null | null | null | null | null | 8 | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_relu_8.s |
| matmul_2x2 | tensor | FAIL | 0 | null | tinyfive | 3 | null | null | null | null | null | null | 0.0701 | 0.1332 | 0.2035 | null | null | null | null | null | [[19, 22], [43, 50]] | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_2x2.s |
| matmul_4x4 | tensor | FAIL | 0 | null | tinyfive | 3 | null | null | null | null | null | null | 0.0729 | 0.1335 | 0.2067 | null | null | null | null | null | [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_4x4.s |
| matmul_add_2x2 | tensor | FAIL | 0 | null | tinyfive | 4 | null | null | null | null | null | null | 0.0754 | 0.1598 | 0.2365 | null | null | null | null | null | [[20, 23], [44, 51]] | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_add_2x2.s |
| matmul_relu_2x2 | tensor | FAIL | 0 | null | tinyfive | 5 | null | null | null | null | null | null | 0.0865 | 0.1733 | 0.2609 | null | null | null | null | null | [[0, 2], [0, 4]] | 0 | False | D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_relu_2x2.s |

## 用例详情

### add_relu_relu

- 类别: activation
- 描述: Add input and bias, then apply ReLU twice.
- 预期输出 (scalar): 7
- TinyFive 输出: 7
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': -3, 't1': 10}
- 模拟后端: tinyfive
- 指令数: 7
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0788
- 模拟耗时(s): 0.1846
- 总耗时(s): 0.2639
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_relu_relu.s

### relu_add

- 类别: activation
- 描述: Add input and bias, then apply one ReLU.
- 预期输出 (scalar): 3
- TinyFive 输出: 3
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': -2, 't1': 5}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0889
- 模拟耗时(s): 0.1463
- 总耗时(s): 0.2355
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_add.s

### relu_only

- 类别: activation
- 描述: Apply ReLU directly to a single input value.
- 预期输出 (scalar): 0
- TinyFive 输出: 0
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': -5}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0751
- 模拟耗时(s): 0.1408
- 总耗时(s): 0.2160
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_only.s

### relu_twice

- 类别: activation
- 描述: Apply ReLU twice to the same activation path.
- 预期输出 (scalar): 4
- TinyFive 输出: 4
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 4}
- 模拟后端: tinyfive
- 指令数: 6
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0783
- 模拟耗时(s): 0.1466
- 总耗时(s): 0.2251
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\relu_twice.s

### if_else

- 类别: branch
- 描述: if/else branch returns subtraction result when flag is zero.
- 预期输出 (scalar): 5
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {'t1': 9, 't2': 4}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0777
- 模拟耗时(s): 5.0203
- 总耗时(s): 5.0982
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_else.s

### if_relu

- 类别: branch
- 描述: if/else branch combined with add and relu.
- 预期输出 (scalar): 0
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0792
- 模拟耗时(s): 5.0179
- 总耗时(s): 5.0974
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_relu.s

### if_then

- 类别: branch
- 描述: if/else branch returns add result when flag is non-zero.
- 预期输出 (scalar): 13
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {'t1': 9, 't2': 4}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0768
- 模拟耗时(s): 5.0203
- 总耗时(s): 5.0975
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\if_then.s

### add_chain

- 类别: elementwise
- 描述: Add a and b, then add c to the intermediate result.
- 预期输出 (scalar): 9
- TinyFive 输出: 9
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3, 't3': 4}
- 模拟后端: tinyfive
- 指令数: 4
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0753
- 模拟耗时(s): 0.1592
- 总耗时(s): 0.2349
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_chain.s

### add_chain_3

- 类别: elementwise
- 描述: Chain three add operations across four symbolic inputs.
- 预期输出 (scalar): 14
- TinyFive 输出: 14
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3, 't3': 4, 't5': 5}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0741
- 模拟耗时(s): 0.1464
- 总耗时(s): 0.2209
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_chain_3.s

### add_fan_in_4

- 类别: elementwise
- 描述: Compute two independent adds and then merge them with a final add.
- 预期输出 (scalar): 10
- TinyFive 输出: 10
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 1, 't1': 2, 't3': 3, 't4': 4}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0710
- 模拟耗时(s): 0.1330
- 总耗时(s): 0.2042
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_fan_in_4.s

### add_reuse

- 类别: elementwise
- 描述: Reuse the same intermediate add result on both operands of a second add.
- 预期输出 (scalar): 10
- TinyFive 输出: 10
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3}
- 模拟后端: tinyfive
- 指令数: 4
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0700
- 模拟耗时(s): 0.1361
- 总耗时(s): 0.2062
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\add_reuse.s

### vector_add

- 类别: elementwise
- 描述: Single add over two symbolic vector inputs.
- 预期输出 (scalar): 5
- TinyFive 输出: 5
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3}
- 模拟后端: tinyfive
- 指令数: 3
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0700
- 模拟耗时(s): 0.1458
- 总耗时(s): 0.2161
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\vector_add.s

### loop_add_4

- 类别: loop
- 描述: Run a four-iteration loop whose body computes one add; final returned value is the last loop-body result.
- 预期输出 (scalar): 5
- TinyFive 输出: 5
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3}
- 模拟后端: tinyfive
- 指令数: 22
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0794
- 模拟耗时(s): 0.1453
- 总耗时(s): 0.2250
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_add_4.s

### loop_add_chain_4

- 类别: loop
- 描述: Run a four-iteration loop whose body computes two chained adds; final returned value is the last loop-body result.
- 预期输出 (scalar): 9
- TinyFive 输出: 9
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': 2, 't1': 3, 't4': 4}
- 模拟后端: tinyfive
- 指令数: 26
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0772
- 模拟耗时(s): 0.1505
- 总耗时(s): 0.2280
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_add_chain_4.s

### loop_relu_add_4

- 类别: loop
- 描述: Run a four-iteration loop whose body computes add followed by ReLU; final returned value is the last loop-body result.
- 预期输出 (scalar): 2
- TinyFive 输出: 2
- 输出是否匹配: True
- TinyFive 初始寄存器: {'t0': -4, 't1': 6}
- 模拟后端: tinyfive
- 指令数: 30
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0764
- 模拟耗时(s): 0.1367
- 总耗时(s): 0.2132
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\loop_relu_add_4.s

### dot_4

- 类别: reduction
- 描述: Compute the dot product of two symbolic vectors of length 4.
- 预期输出 (scalar): 70
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 3
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0720
- 模拟耗时(s): 0.1365
- 总耗时(s): 0.2087
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_4.s

### dot_8

- 类别: reduction
- 描述: Compute the dot product of two symbolic vectors of length 8.
- 预期输出 (scalar): 36
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 3
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0755
- 模拟耗时(s): 0.1409
- 总耗时(s): 0.2167
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_8.s

### dot_relu_4

- 类别: reduction
- 描述: Compute a length-4 dot product and pass it through ReLU.
- 预期输出 (scalar): 0
- TinyFive 输出: 0
- 输出是否匹配: True
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0759
- 模拟耗时(s): 0.1373
- 总耗时(s): 0.2142
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_relu_4.s

### dot_relu_8

- 类别: reduction
- 描述: Compute a length-8 dot product and pass it through ReLU.
- 预期输出 (scalar): 8
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0695
- 模拟耗时(s): 0.1336
- 总耗时(s): 0.2032
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\dot_relu_8.s

### matmul_2x2

- 类别: tensor
- 描述: Compute a symbolic 2x2 by 2x2 matrix multiplication.
- 预期输出 (tensor): [[19, 22], [43, 50]]
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 3
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0701
- 模拟耗时(s): 0.1332
- 总耗时(s): 0.2035
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_2x2.s

### matmul_4x4

- 类别: tensor
- 描述: Compute a symbolic 4x4 by 4x4 matrix multiplication.
- 预期输出 (tensor): [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 3
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0729
- 模拟耗时(s): 0.1335
- 总耗时(s): 0.2067
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_4x4.s

### matmul_add_2x2

- 类别: tensor
- 描述: Compute a 2x2 matmul and then add a symbolic bias term.
- 预期输出 (tensor): [[20, 23], [44, 51]]
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 4
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0754
- 模拟耗时(s): 0.1598
- 总耗时(s): 0.2365
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_add_2x2.s

### matmul_relu_2x2

- 类别: tensor
- 描述: Compute a 2x2 matmul and then apply ReLU to its result.
- 预期输出 (tensor): [[0, 2], [0, 4]]
- TinyFive 输出: 0
- 输出是否匹配: False
- TinyFive 初始寄存器: {}
- 模拟后端: tinyfive
- 指令数: 5
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: null
- Benchmark 停止原因: null
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0865
- 模拟耗时(s): 0.1733
- 总耗时(s): 0.2609
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): null
- 是否性能退化: null
- 汇编文件: D:\PycharmProjects\ScratchV\ScratchV\ScratchV-topic06-deliverable\build\matmul_relu_2x2.s

