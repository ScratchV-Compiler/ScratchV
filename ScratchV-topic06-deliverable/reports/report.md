# ScratchV DSL 编译器性能测试报告

## 测试概览

- Schema 版本: 1
- 运行模式: benchmark
- 类别筛选: null
- 名称筛选: null
- 生成时间: 2026-09-02 16:40:25
- 用例总数: 23
- 通过数量: 13
- 失败数量: 10
- 通过率: 56.5%
- 测试目录: `tests_main`
- 汇编输出目录: `build`
- 性能基线文件: `reports/benchmark_baseline.json`
- 性能退化阈值: 5.00%
- 单次编译超时: 30s
- 单次模拟超时: 5s

## 测试结果

| 用例 | 类别 | 状态 | 编译返回码 | 编译日志 | 模拟后端 | 指令数 | Benchmark 次数 | Benchmark 停止原因 | 平均指令数 | 95% 置信区间 | 最小 | 最大 | 编译耗时(s) | 模拟耗时(s) | 总耗时(s) | 基线 | 变化量 | 变化率(%) | 退化阈值(%) | 是否退化 | 预期输出 | TinyFive 输出 | 输出匹配 | 汇编文件 |
|---|---|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| add_relu_relu | activation | PASS | 0 | null | tinyfive | 7 | 3 | null | 7.00 | ±0.00 | 7 | 7 | 0.0794 | 0.1433 | 0.6308 | 7.00 | 0.00 | 0.00 | 5.00 | False | 7 | 7 | True | build/add_relu_relu.s |
| relu_add | activation | PASS | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0770 | 0.1354 | 0.6199 | 5.00 | 0.00 | 0.00 | 5.00 | False | 3 | 3 | True | build/relu_add.s |
| relu_only | activation | PASS | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0751 | 0.1383 | 0.6184 | 5.00 | 0.00 | 0.00 | 5.00 | False | 0 | 0 | True | build/relu_only.s |
| relu_twice | activation | PASS | 0 | null | tinyfive | 6 | 3 | null | 6.00 | ±0.00 | 6 | 6 | 0.0804 | 0.1339 | 0.6124 | 6.00 | 0.00 | 0.00 | 5.00 | False | 4 | 4 | True | build/relu_twice.s |
| if_else | branch | FAIL | 0 | null | timeout | 0 | 0 | benchmark skipped: initial simulation timeout | null | null | null | null | 0.0816 | 5.0138 | 5.1004 | null | null | null | 5.00 | null | 5 | None | False | build/if_else.s |
| if_relu | branch | FAIL | 0 | null | timeout | 0 | 0 | benchmark skipped: initial simulation timeout | null | null | null | null | 0.0976 | 5.0081 | 5.1193 | null | null | null | 5.00 | null | 0 | None | False | build/if_relu.s |
| if_then | branch | FAIL | 0 | null | timeout | 0 | 0 | benchmark skipped: initial simulation timeout | null | null | null | null | 0.0855 | 5.0168 | 5.1040 | null | null | null | 5.00 | null | 13 | None | False | build/if_then.s |
| add_chain | elementwise | PASS | 0 | null | tinyfive | 4 | 3 | null | 4.00 | ±0.00 | 4 | 4 | 0.0875 | 0.1641 | 0.7484 | 4.00 | 0.00 | 0.00 | 5.00 | False | 9 | 9 | True | build/add_chain.s |
| add_chain_3 | elementwise | PASS | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0804 | 0.1379 | 0.6290 | 5.00 | 0.00 | 0.00 | 5.00 | False | 14 | 14 | True | build/add_chain_3.s |
| add_fan_in_4 | elementwise | PASS | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0771 | 0.1414 | 0.6389 | 5.00 | 0.00 | 0.00 | 5.00 | False | 10 | 10 | True | build/add_fan_in_4.s |
| add_reuse | elementwise | PASS | 0 | null | tinyfive | 4 | 3 | null | 4.00 | ±0.00 | 4 | 4 | 0.0916 | 0.1392 | 0.6641 | 4.00 | 0.00 | 0.00 | 5.00 | False | 10 | 10 | True | build/add_reuse.s |
| vector_add | elementwise | PASS | 0 | null | tinyfive | 3 | 3 | null | 3.00 | ±0.00 | 3 | 3 | 0.0820 | 0.1452 | 0.6593 | 3.00 | 0.00 | 0.00 | 5.00 | False | 5 | 5 | True | build/vector_add.s |
| loop_add_4 | loop | PASS | 0 | null | tinyfive | 22 | 3 | null | 22.00 | ±0.00 | 22 | 22 | 0.0813 | 0.1453 | 0.6405 | 22.00 | 0.00 | 0.00 | 5.00 | False | 5 | 5 | True | build/loop_add_4.s |
| loop_add_chain_4 | loop | PASS | 0 | null | tinyfive | 26 | 3 | null | 26.00 | ±0.00 | 26 | 26 | 0.0767 | 0.1412 | 0.6341 | 26.00 | 0.00 | 0.00 | 5.00 | False | 9 | 9 | True | build/loop_add_chain_4.s |
| loop_relu_add_4 | loop | PASS | 0 | null | tinyfive | 30 | 3 | null | 30.00 | ±0.00 | 30 | 30 | 0.0805 | 0.1378 | 0.6484 | 30.00 | 0.00 | 0.00 | 5.00 | False | 2 | 2 | True | build/loop_relu_add_4.s |
| dot_4 | reduction | FAIL | 0 | null | tinyfive | 3 | 3 | null | 3.00 | ±0.00 | 3 | 3 | 0.0774 | 0.1493 | 0.6318 | 3.00 | 0.00 | 0.00 | 5.00 | False | 70 | 0 | False | build/dot_4.s |
| dot_8 | reduction | FAIL | 0 | null | tinyfive | 3 | 3 | null | 3.00 | ±0.00 | 3 | 3 | 0.0847 | 0.1363 | 0.6288 | 3.00 | 0.00 | 0.00 | 5.00 | False | 36 | 0 | False | build/dot_8.s |
| dot_relu_4 | reduction | PASS | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0754 | 0.1405 | 0.6712 | 5.00 | 0.00 | 0.00 | 5.00 | False | 0 | 0 | True | build/dot_relu_4.s |
| dot_relu_8 | reduction | FAIL | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.0936 | 0.1599 | 0.7681 | 5.00 | 0.00 | 0.00 | 5.00 | False | 8 | 0 | False | build/dot_relu_8.s |
| matmul_2x2 | tensor | FAIL | 0 | null | tinyfive | 3 | 3 | null | 3.00 | ±0.00 | 3 | 3 | 0.0872 | 0.1736 | 0.7112 | 3.00 | 0.00 | 0.00 | 5.00 | False | [[19, 22], [43, 50]] | 0 | False | build/matmul_2x2.s |
| matmul_4x4 | tensor | FAIL | 0 | null | tinyfive | 3 | 3 | null | 3.00 | ±0.00 | 3 | 3 | 0.0896 | 0.1487 | 0.7480 | 3.00 | 0.00 | 0.00 | 5.00 | False | [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]] | 0 | False | build/matmul_4x4.s |
| matmul_add_2x2 | tensor | FAIL | 0 | null | tinyfive | 4 | 3 | null | 4.00 | ±0.00 | 4 | 4 | 0.0869 | 0.1599 | 0.6950 | 4.00 | 0.00 | 0.00 | 5.00 | False | [[20, 23], [44, 51]] | 0 | False | build/matmul_add_2x2.s |
| matmul_relu_2x2 | tensor | FAIL | 0 | null | tinyfive | 5 | 3 | null | 5.00 | ±0.00 | 5 | 5 | 0.1024 | 0.1636 | 0.7729 | 5.00 | 0.00 | 0.00 | 5.00 | False | [[0, 2], [0, 4]] | 0 | False | build/matmul_relu_2x2.s |

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 7.00
- 95% 置信区间: ±0.00
- 最小指令数: 7
- 最大指令数: 7
- 编译耗时(s): 0.0794
- 模拟耗时(s): 0.1433
- 总耗时(s): 0.6308
- 基线指令数: 7.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/add_relu_relu.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0770
- 模拟耗时(s): 0.1354
- 总耗时(s): 0.6199
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/relu_add.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0751
- 模拟耗时(s): 0.1383
- 总耗时(s): 0.6184
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/relu_only.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 6.00
- 95% 置信区间: ±0.00
- 最小指令数: 6
- 最大指令数: 6
- 编译耗时(s): 0.0804
- 模拟耗时(s): 0.1339
- 总耗时(s): 0.6124
- 基线指令数: 6.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/relu_twice.s

### if_else

- 类别: branch
- 描述: if/else branch returns subtraction result when flag is zero.
- 预期输出 (scalar): 5
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {'t1': 0, 't2': 9, 't3': 4}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: 0
- Benchmark 停止原因: benchmark skipped: initial simulation timeout
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0816
- 模拟耗时(s): 5.0138
- 总耗时(s): 5.1004
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): 5.00
- 是否性能退化: null
- 汇编文件: build/if_else.s

### if_relu

- 类别: branch
- 描述: if/else branch combined with add and relu.
- 预期输出 (scalar): 0
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {'t1': 1}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: 0
- Benchmark 停止原因: benchmark skipped: initial simulation timeout
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0976
- 模拟耗时(s): 5.0081
- 总耗时(s): 5.1193
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): 5.00
- 是否性能退化: null
- 汇编文件: build/if_relu.s

### if_then

- 类别: branch
- 描述: if/else branch returns add result when flag is non-zero.
- 预期输出 (scalar): 13
- TinyFive 输出: None
- 输出是否匹配: False
- TinyFive 初始寄存器: {'t1': 1, 't2': 9, 't3': 4}
- 模拟后端: timeout
- 指令数: 0
- 编译返回码: 0
- 编译是否超时: False
- 编译错误摘要: null
- 编译失败日志: null
- Benchmark 重复次数: 0
- Benchmark 停止原因: benchmark skipped: initial simulation timeout
- 平均指令数: null
- 95% 置信区间: null
- 最小指令数: null
- 最大指令数: null
- 编译耗时(s): 0.0855
- 模拟耗时(s): 5.0168
- 总耗时(s): 5.1040
- 基线指令数: null
- 性能变化量: null
- 性能变化率(%): null
- 性能退化阈值(%): 5.00
- 是否性能退化: null
- 汇编文件: build/if_then.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 4.00
- 95% 置信区间: ±0.00
- 最小指令数: 4
- 最大指令数: 4
- 编译耗时(s): 0.0875
- 模拟耗时(s): 0.1641
- 总耗时(s): 0.7484
- 基线指令数: 4.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/add_chain.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0804
- 模拟耗时(s): 0.1379
- 总耗时(s): 0.6290
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/add_chain_3.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0771
- 模拟耗时(s): 0.1414
- 总耗时(s): 0.6389
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/add_fan_in_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 4.00
- 95% 置信区间: ±0.00
- 最小指令数: 4
- 最大指令数: 4
- 编译耗时(s): 0.0916
- 模拟耗时(s): 0.1392
- 总耗时(s): 0.6641
- 基线指令数: 4.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/add_reuse.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 3.00
- 95% 置信区间: ±0.00
- 最小指令数: 3
- 最大指令数: 3
- 编译耗时(s): 0.0820
- 模拟耗时(s): 0.1452
- 总耗时(s): 0.6593
- 基线指令数: 3.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/vector_add.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 22.00
- 95% 置信区间: ±0.00
- 最小指令数: 22
- 最大指令数: 22
- 编译耗时(s): 0.0813
- 模拟耗时(s): 0.1453
- 总耗时(s): 0.6405
- 基线指令数: 22.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/loop_add_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 26.00
- 95% 置信区间: ±0.00
- 最小指令数: 26
- 最大指令数: 26
- 编译耗时(s): 0.0767
- 模拟耗时(s): 0.1412
- 总耗时(s): 0.6341
- 基线指令数: 26.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/loop_add_chain_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 30.00
- 95% 置信区间: ±0.00
- 最小指令数: 30
- 最大指令数: 30
- 编译耗时(s): 0.0805
- 模拟耗时(s): 0.1378
- 总耗时(s): 0.6484
- 基线指令数: 30.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/loop_relu_add_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 3.00
- 95% 置信区间: ±0.00
- 最小指令数: 3
- 最大指令数: 3
- 编译耗时(s): 0.0774
- 模拟耗时(s): 0.1493
- 总耗时(s): 0.6318
- 基线指令数: 3.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/dot_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 3.00
- 95% 置信区间: ±0.00
- 最小指令数: 3
- 最大指令数: 3
- 编译耗时(s): 0.0847
- 模拟耗时(s): 0.1363
- 总耗时(s): 0.6288
- 基线指令数: 3.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/dot_8.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0754
- 模拟耗时(s): 0.1405
- 总耗时(s): 0.6712
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/dot_relu_4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.0936
- 模拟耗时(s): 0.1599
- 总耗时(s): 0.7681
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/dot_relu_8.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 3.00
- 95% 置信区间: ±0.00
- 最小指令数: 3
- 最大指令数: 3
- 编译耗时(s): 0.0872
- 模拟耗时(s): 0.1736
- 总耗时(s): 0.7112
- 基线指令数: 3.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/matmul_2x2.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 3.00
- 95% 置信区间: ±0.00
- 最小指令数: 3
- 最大指令数: 3
- 编译耗时(s): 0.0896
- 模拟耗时(s): 0.1487
- 总耗时(s): 0.7480
- 基线指令数: 3.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/matmul_4x4.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 4.00
- 95% 置信区间: ±0.00
- 最小指令数: 4
- 最大指令数: 4
- 编译耗时(s): 0.0869
- 模拟耗时(s): 0.1599
- 总耗时(s): 0.6950
- 基线指令数: 4.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/matmul_add_2x2.s

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
- Benchmark 重复次数: 3
- Benchmark 停止原因: null
- 平均指令数: 5.00
- 95% 置信区间: ±0.00
- 最小指令数: 5
- 最大指令数: 5
- 编译耗时(s): 0.1024
- 模拟耗时(s): 0.1636
- 总耗时(s): 0.7729
- 基线指令数: 5.00
- 性能变化量: 0.00
- 性能变化率(%): 0.00
- 性能退化阈值(%): 5.00
- 是否性能退化: False
- 汇编文件: build/matmul_relu_2x2.s

