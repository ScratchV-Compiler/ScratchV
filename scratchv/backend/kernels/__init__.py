# -*- coding: utf-8 -*-
"""算子内核框架：把「形状在编译期已知、语义固定」的热点算子直接生成 RISC-V 汇编。

用法::

    python -m scratchv.backend.kernels --list
    python -m scratchv.backend.kernels --problem add-fp32 -o add_fp32.s

文档见同目录 ARCHITECTURE.md / DEVELOPMENT.md / USAGE.md / OPTIMIZATION.md。
"""
