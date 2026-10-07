# -*- coding: utf-8 -*-
"""命令行入口::

    python -m scratchv.backend.kernels --problem add-fp32 -o out.s
    python -m scratchv.backend.kernels --list
"""

import argparse
import os

from scratchv.backend.kernels.bodies import PROBLEMS
from scratchv.backend.kernels.pipeline import build_program


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='ScratchV 算子内核生成器')
    ap.add_argument('--problem', help='题名，见 --list')
    ap.add_argument('-o', '--output', default=None,
                    help='输出路径；省略则写到 ./<题名>.s；给 "-" 打到屏幕')
    ap.add_argument('--unroll', type=int, default=None,
                    help='展开因子（2 的幂）；省略则用实测表')
    ap.add_argument('--list', action='store_true', help='列出所有题名')
    args = ap.parse_args(argv)

    if args.list:
        for name, (body, dtype, target) in sorted(PROBLEMS.items()):
            print(f'{name:<18} body={body:<10} dtype={dtype:<5} target={target}')
        return 0

    if not args.problem:
        ap.error('要么给 --problem，要么给 --list')

    text = build_program(args.problem, args.unroll)

    # 省略 -o 时**落到当前目录**（与 gcc/clang 同一约定），而不是打到屏幕。
    # 打屏幕那条路（`-o -`）留着，但必须是显式的 —— 否则产物会散在终端里没有人接。
    if args.output == '-':
        print(text)
        return 0
    path = args.output or f'{args.problem}.s'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'已写出 {os.path.abspath(path)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
