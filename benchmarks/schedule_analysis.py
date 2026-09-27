"""Control-flow liveness for the CNN scheduling report."""

from scratchv.backend._asm_parser import parse_line
from scratchv.backend.inst_scheduler import parse_instructions
from scratchv.backend.schedule_semantics import register_name


def cfg_liveness(source: str) -> dict:
    """Fixed-point instruction CFG; exit contract: memory output and preserved sp.

    An unrecognized register effect or indirect target makes this analysis N/A.
    AUIPC is address-dependent for scheduling but has known register effects.
    """
    insts = parse_instructions(source)
    labels = {}
    index = 0
    by_line = {inst.id: inst for inst in insts}
    for lineno, text in enumerate(source.splitlines()):
        parsed = parse_line(text, lineno)
        if parsed.label:
            if parsed.label in labels:
                return {"status": "not_modeled", "reason": "duplicate label"}
            labels[parsed.label] = index
        index += lineno in by_line
    successors, reads, writes = [], [], []
    for index, inst in enumerate(insts):
        use, define = set(inst.uses), set(inst.defines)
        if inst.effects.barrier_reason:
            if inst.opcode == "auipc" and len(inst.operands) == 2:
                reg = register_name(inst.operands[0])
                if reg is None:
                    return {"status": "not_modeled", "reason": "invalid AUIPC destination"}
                define = {reg} - {"x0"}
            else:
                return {"status": "not_modeled", "reason": f"line {inst.id + 1}: {inst.effects.barrier_reason}"}
        following = {index + 1} if index + 1 < len(insts) else set()
        if inst.effects.control in {"jump", "return"}:
            following = set()
        if inst.effects.target:
            if inst.effects.target not in labels:
                return {"status": "not_modeled", "reason": "unresolved branch target"}
            target = labels[inst.effects.target]
            if target < len(insts):
                following.add(target)
        elif inst.effects.control == "jump":
            return {"status": "not_modeled", "reason": "indirect jump"}
        successors.append(following)
        reads.append(use)
        writes.append(define)
    live_in = [set() for _ in insts]
    live_out = [set() for _ in insts]
    changed = True
    iterations = 0
    while changed:
        changed = False
        iterations += 1
        for i in reversed(range(len(insts))):
            out = set().union(*(live_in[j] for j in successors[i])) if successors[i] else {"x2"}
            entry = reads[i] | (out - writes[i])
            changed |= entry != live_in[i] or out != live_out[i]
            live_in[i], live_out[i] = entry, out
    return {"status": "completed", "exit_contract": "memory outputs; sp live; ret reads ra",
            "peak": max((len(s) for s in live_in + live_out), default=0), "iterations": iterations,
            "instructions": [{"line": inst.id + 1, "live_in": sorted(live_in[i]),
                              "live_out": sorted(live_out[i])} for i, inst in enumerate(insts)]}
