"""Scalar and RVV leaves for the W6 4×4 and 4×8 FP32 Attention timing probe."""
import json


def _tile_assembly(vector, columns):
    name = "w6_matched_rvv" if vector else "w6_matched_scalar"
    # RV64D psABI: a0 = Q[4][128], a1 = K[128][columns], a2 = out[4][columns].
    # Keep stack accesses identical in both leaves, outside the K loop.
    saved = ["fs0"] if columns == 4 else [f"fs{i}" for i in range(12)]
    frame = (8 * len(saved) + 15) // 16 * 16
    scalar_acc = [f"ft{i}" for i in range(8)] + [f"fa{i}" for i in range(8)]
    key = [f"ft{i}" for i in range(8, 12)]
    integer_acc = []
    if columns == 8:
        key += [f"fs{i}" for i in range(1, 5)]
        scalar_acc += [f"fs{i}" for i in range(5, 11)]
        # 32 FP registers have to accommodate eight keys, one Q, the
        # accumulators and one transfer register (fs11). Hold the remaining
        # ten FP32 bit patterns in caller-saved integer registers. FMV is an
        # exact bit transfer; the loop still reads Q and K just once.
        integer_acc = [f"t{i}" for i in range(1, 7)] + [f"a{i}" for i in range(3, 7)]
    lines = [
        f'.pushsection .text.{name},"ax",@progbits',
        ".option push",
        ".option arch,+v" if vector else ".option arch,-v",
        ".p2align 2",
        f".global {name}",
        f".type {name},@function",
        f"{name}:",
        f"addi sp,sp,-{frame}",
        *(f"fsd {reg},{8*i}(sp)" for i, reg in enumerate(saved)),
        "li t0,128",
    ]
    if vector:
        lines.append(f"vsetivli zero,{columns},e32,m1,ta,ma")
        lines.extend(f"vmv.v.i v{4+r},0" for r in range(4))
    else:
        lines.extend(f"fmv.w.x {acc},zero" for acc in scalar_acc)
        lines.extend(f"li {acc},0" for acc in integer_acc)

    lines.extend([f".global {name}_k_start", f"{name}_k_start:", f".L{name}_k:"])
    if vector:
        lines.append("vle32.v v0,(a1)")
    else:
        lines.extend(f"flw {reg},{4*c}(a1)" for c, reg in enumerate(key))
    for row in range(4):
        lines.append(f"flw fs0,{512*row}(a0)")
        if vector:
            lines.append(f"vfmacc.vf v{4+row},fs0,v0")
        else:
            for col, key_reg in enumerate(key):
                index = columns * row + col
                if index < len(scalar_acc):
                    acc = scalar_acc[index]
                    lines.append(f"fmadd.s {acc},fs0,{key_reg},{acc}")
                else:
                    reg = integer_acc[index - len(scalar_acc)]
                    lines.extend([f"fmv.w.x fs11,{reg}", f"fmadd.s fs11,fs0,{key_reg},fs11",
                                  f"fmv.x.w {reg},fs11"])
    lines.extend([
        "addi a0,a0,4",
        f"addi a1,a1,{4*columns}",
        "addi t0,t0,-1",
        f"bnez t0,.L{name}_k",
        f".global {name}_k_end",
        f"{name}_k_end:",
        "fmv.w.x fs0,zero",
    ])
    for row in range(4):
        if vector:
            lines.extend([f"vfadd.vf v{4+row},v{4+row},fs0", f"vse32.v v{4+row},(a2)"])
        else:
            for col in range(columns):
                index = columns * row + col
                if index < len(scalar_acc):
                    acc = scalar_acc[index]
                else:
                    acc = "fs11"
                    lines.append(f"fmv.w.x {acc},{integer_acc[index - len(scalar_acc)]}")
                lines.extend([f"fadd.s {acc},fs0,{acc}", f"fsw {acc},{4*col}(a2)"])
        if row < 3:
            lines.append(f"addi a2,a2,{4*columns}")
    lines.extend([
        *(f"fld {reg},{8*i}(sp)" for i, reg in enumerate(saved)),
        f"addi sp,sp,{frame}",
        "ret",
        f".size {name},.-{name}",
        ".option pop",
        ".popsection",
    ])
    return "\n".join(lines) + "\n"


def matched_kernel_source(columns=4):
    """Return two tile functions and a per-core vector-state initializer for one ELF.

    Each k reads all adjacent K columns, then Q for rows 0 through 3.
    Each output uses 128 ordered FP32 FMAs and the same final addition of +0.
    The caller packs the same K panel and assigns the same tiles to both leaves.
    """
    if type(columns) is not int or columns not in (4, 8):
        raise ValueError("The matched scalar/RVV experiment uses four or eight columns")
    assembly = _tile_assembly(False, columns) + _tile_assembly(True, columns)
    return f"""__asm__({json.dumps(assembly)});
extern void w6_matched_scalar(const float *q, const float *packed_k, float *out);
extern void w6_matched_rvv(const float *q, const float *packed_k, float *out);
static inline void w6_matched_enable(void) {{
    __asm__ volatile ("csrs mstatus, %0" :: "r"(3u << 9) : "memory");
}}
"""
