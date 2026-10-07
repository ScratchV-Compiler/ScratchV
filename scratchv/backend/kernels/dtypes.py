# -*- coding: utf-8 -*-
"""数值类型描述：把「这种数怎么算、用哪些寄存器算」变成数据。

**除助记符外，数据寄存器名也从这里取**——否则 body 里写死的 `ft0`
在整数路径下会变成 `lw ft0, ...` 这种汇编不过的代码。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DtypePolicy:
    """一种数值类型的全部静态事实。"""

    name: str
    load: str                    # 载入助记符
    store: str                   # 存储助记符
    add: str                     # 加法助记符
    acc: str                     # 累加器寄存器名
    tmp1: str                    # 临时寄存器 1
    tmp2: str                    # 临时寄存器 2
    prod: str                    # ★ 乘积寄存器 —— 必须与两个操作数都不同，
                                 #   否则分块时操作数会被乘积覆盖（见 mac_instrs）
    mac: tuple[str, ...]         # ★ 一次乘加（MAC）的指令序列；`{acc}`/`{t1}`/`{t2}` 由 body 填
    zero: tuple[str, ...]        # 累加器置零的模板；`{acc}` = 累加器，`{r}` = 调用方给的空闲整数寄存器
    bank: str                    # 数据寄存器取自哪个银行：'fp' 或 'int'
    comparison: str              # 结果怎么比对：'exact'（逐位）/'tolerance'（容差）


POLICIES: dict[str, DtypePolicy] = {
    'q16': DtypePolicy(
        name='q16', load='lw', store='sw', add='add',
        # 整数路径下累加器和临时值都是整数寄存器；用 s2/s3/s4，
        # 与三个 body 里用到的 t0-t6 / a3-a5 不相交。
        acc='s2', tmp1='s3', tmp2='s4', prod='s5',
        # ★ Q16.16 的乘积要**先右移 16 位**再累加（赛题语义 `Σ (A×B) >> 16`）。
        # 这是它与 f32 最本质的差别 —— "乘"在两边的形状不一样。
        # 内测比赛1 的交付版把它优化成 `mulh(a << 16, b)` + `add`（2 条），
        # 那是本 policy 的一个**变体**，不是另一个抽象层。
        mac=('mul {p}, {t1}, {t2}', 'srai {p}, {p}, 16', 'add {acc}, {acc}, {p}'),
        zero=('li {acc}, 0',),
        bank='int',
        comparison='exact',
    ),
    'f32': DtypePolicy(
        name='f32', load='flw', store='fsw', add='fadd.s',
        # 浮点路径下累加器和临时值都在浮点寄存器组，与整数寄存器天然不相交。
        acc='ft0', tmp1='ft1', tmp2='ft2', prod='ft3',
        # 浮点没有定点重标定，乘完直接累加。
        mac=('fmul.s {p}, {t1}, {t2}', 'fadd.s {acc}, {acc}, {p}'),
        zero=('li {r}, 0', 'fmv.w.x {acc}, {r}'),
        bank='fp',
        comparison='tolerance',          # 浮点不能逐位比，要看平台给的容差
    ),
}


def mac_instrs(dtype: DtypePolicy, acc: str | None = None, t1: str | None = None,
               t2: str | None = None, prod: str | None = None) -> list[str]:
    """把一次乘加的模板填上寄存器名。

    **`prod` 必须与 `t1`/`t2` 都不同。** 模板里 `{p}` 是乘积的落点：如果让它和
    某个操作数重合（早期版本就是 `fmul {t1}, {t1}, {t2}`），通用循环里看不出来
    （`t1` 每个 k 都重新载入），但**寄存器分块时操作数要在 nr 次乘加之间保持不变**，
    一覆盖就全错——而且是静默错，汇编器不报。
    """
    return [line.format(acc=acc or dtype.acc, t1=t1 or dtype.tmp1,
                        t2=t2 or dtype.tmp2, p=prod or dtype.prod)
            for line in dtype.mac]


def zero_acc(dtype: DtypePolicy, scratch: str) -> list[str]:
    """把累加器置零。

    ``scratch`` 必须是一个**调用方确定此刻空闲**的整数寄存器。
    为什么不能写死一个？DEVELOPMENT.md 第 3.3 节有一个真实的翻车例子
    （写死成 ``t6``，而 ``t6`` 在 matmul 里存着行步长）。
    """
    return [line.format(acc=dtype.acc, r=scratch) for line in dtype.zero]
