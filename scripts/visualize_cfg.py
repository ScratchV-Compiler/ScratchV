#!/usr/bin/env python3
"""CFG 可视化工具 — 从 DSL/ONNX 生成控制流图 PNG/PDF/SVG。

用法:
    python3 scripts/visualize_cfg.py examples/cfg/if_else.dsl -o cfg.png
    python3 scripts/visualize_cfg.py examples/cfg/while_loop.dsl --show-loops
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="CFG 可视化工具")
    ap.add_argument("input", help="输入文件 (.dsl 或 .onnx)")
    ap.add_argument("-o", "--output", help="输出图片 (.png/.pdf/.svg)")
    ap.add_argument("--show-loops", action="store_true", help="打印循环信息")
    ap.add_argument("--eliminate-unreachable", action="store_true",
                    help="先消除不可达代码")
    ap.add_argument("--format", choices=["png","pdf","svg"], default="png")
    ap.add_argument("--extended", action="store_true",
                    help="使用 ExtendedDSLParser (if/else, while)")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Error: {args.input} not found", file=sys.stderr); return 1

    source = path.read_text(encoding="utf-8-sig")

    # 解析
    needs_extended = (
        args.extended
        or path.suffix == ".edsl"
        or bool(re.search(r"^\s*(?:if|while)\s*\(", source, re.MULTILINE))
    )
    if needs_extended:
        from scratchv.frontend.dsl_extended import ExtendedDSLParser
        program = ExtendedDSLParser().parse(source)
    elif path.suffix == ".onnx":
        from scratchv.frontend.onnx_parser import ONNXParser
        program = ONNXParser().parse(str(path))
    else:
        from scratchv.frontend.dsl_parser import DSLParser
        program = DSLParser().parse(source)

    # 构建 CFG
    from scratchv.ir.cfg import CFGBuilder
    builder = CFGBuilder()
    cfgs = builder.build(program)

    dot_parts = []
    for fname, cfg in cfgs.items():
        if args.eliminate_unreachable:
            removed = builder.eliminate_unreachable(cfg)
            if removed:
                print(f"[{fname}] removed: {removed}", file=sys.stderr)

        if args.show_loops:
            loops = builder.detect_nested_loops(cfg)
            for loop in loops:
                indent = "  " * loop.nesting_depth
                print(f"{indent}[{fname}] Loop: {loop.header} "
                      f"depth={loop.nesting_depth} body={sorted(loop.body)}",
                      file=sys.stderr)

        dot_parts.append(cfg.to_dot(
            loop_headers={l.header for l in builder.detect_loops(cfg)}))

    full_dot = "\n\n".join(dot_parts)

    if args.output:
        dot_path = Path(args.output).with_suffix(".dot")
        dot_path.write_text(full_dot, encoding="utf-8")
        print(f"DOT: {dot_path}", file=sys.stderr)
        try:
            subprocess.run(["dot", f"-T{args.format}", str(dot_path),
                           "-o", args.output], check=True)
            print(f"Image: {args.output}", file=sys.stderr)
        except FileNotFoundError:
            print("Install Graphviz: apt install graphviz", file=sys.stderr)
    else:
        print(full_dot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
