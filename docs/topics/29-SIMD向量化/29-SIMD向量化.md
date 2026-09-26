# 课题29：SIMD 向量化

> **难度**：高 | **类型**：项目实战 | **源文件**：待创建
> **状态**：⬜ 规划中

---

## 概述

SIMD（Single Instruction Multiple Data）向量化将标量运算转换为向量运算，一条指令同时处理多个数据。对 ScratchV RISC-V 后端而言，利用 RISC-V P-extension（SIMD 扩展）或 V-extension（向量扩展）可以一条指令同时做 2-4 个 MAC 操作，per-MAC 指令数从 ~12 降至 ~3-5，是从根本上缩小与 LLVM float32 路径差距的关键方向。

---

## 理解背景

待补充。

---

## 详细任务

待补充。

---

## 交付产物

待补充。

---

## 代码走读

待补充。

---

## 动手练习

待补充。

---

## 常见坑

待补充。

---

## 进阶阅读

- RISC-V V-extension 规范: [RISC-V Vector Extension](https://github.com/riscv/riscv-v-spec)
- 相关 topic: [课题19 — Standalone RISC-V 编译器](../19-Standalone-RISC-V编译器/19-Standalone-RISC-V编译器.md) | [课题28 — 扩展指令选择](../28-扩展指令选择/28-扩展指令选择.md)

---

## 12周每周目标

待补充。
