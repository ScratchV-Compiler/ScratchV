"""Single-issue teaching model shared by scheduling and order estimation.

These are local static estimates, not hardware cycles or the existing
five-stage PipelineCycleEstimator's stage occupancy counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .schedule_semantics import DIVIDE, LOADS, MULTIPLY, SchedInst, ScheduleError


@dataclass(frozen=True)
class Timing:
    latency: int = 1
    resource: str = "alu"
    occupancy: int = 1
    issue_occupancy: int = 1


@dataclass(frozen=True)
class ScheduleModel:
    # Conservative local model, not a complete model of any named CPU.
    # Integer multiply/divide latencies follow LLVM's Rocket scheduling table;
    # division movement is restricted separately in the dependency graph.
    name: str = "scratchv-conservative-v2"
    overrides: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.overrides)
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in values.values()
        ):
            raise ValueError("Scheduling latencies must be positive integers")
        object.__setattr__(self, "overrides", MappingProxyType(values))

    def sensitivity_model(self) -> ScheduleModel:
        """Check gains against a second plausible integer latency table.

        LLVM's Rocket and SiFive7 tables differ in multiply/divide latency.
        This deliberately retains our simplified issue/resource model; it is
        a sensitivity check, not a complete simulation of either processor.
        Explicit caller overrides take precedence in both estimates.
        """
        return ScheduleModel(
            name="scratchv-latency-sensitivity-v1",
            overrides={
                **{op: 3 for op in MULTIPLY},
                **{op: 66 for op in DIVIDE},
                **self.overrides,
            },
        )

    def timing(self, inst: SchedInst) -> Timing:
        if inst.effects.barrier_reason:
            raise ScheduleError(
                f"Cannot estimate {inst.opcode}: {inst.effects.barrier_reason}"
            )
        op = inst.opcode
        latency, resource, occupancy = 1, "alu", 1
        if inst.effects.memory != "none":
            latency, resource = (2 if op in LOADS else 1), "memory"
        elif op in MULTIPLY:
            latency, resource = 4, "multiply"
        elif op in DIVIDE:
            latency, resource, occupancy = 33, "divide", 33
        elif inst.terminator:
            resource = "branch"
        elif op.startswith("f"):
            resource = "float"
            double = op.endswith(".d") or op.startswith("fcvt.d.")
            stem = op.split(".")[0]
            if stem in {"fdiv", "fsqrt"}:
                latency = 16 if double else 12
                resource, occupancy = "float-divide", latency
            elif stem in {"fmul", "fmadd", "fmsub", "fnmadd", "fnmsub"}:
                latency = 5 if double else 4
            elif stem in {"fadd", "fsub", "fcvt"}:
                latency = 4 if double else 3
            elif stem in {"fmin", "fmax", "feq", "flt", "fle"}:
                latency = 3 if double else 2
        latency = self.overrides.get(op, latency)
        if resource in {"divide", "float-divide"}:
            occupancy = latency
        return Timing(latency, resource, occupancy, latency if op in DIVIDE else 1)


@dataclass(frozen=True)
class OrderEstimate:
    cycles: int
    stalls: int
    issue_cycles: tuple[int, ...]


def estimate_order(
    instructions: Sequence[SchedInst], model: ScheduleModel | None = None
) -> OrderEstimate:
    """Simulate this exact order, independently of graph/scheduler state."""
    model = model or ScheduleModel()
    ready: dict[str, int] = {}
    resources: dict[str, int] = {}
    next_issue = completion = stalls = issue_blocked_until = 0
    issues = []
    for inst in instructions:
        timing = model.timing(inst)
        issue = max(
            next_issue,
            issue_blocked_until,
            resources.get(timing.resource, 0),
            max((ready.get(reg, 0) for reg in inst.uses), default=0),
            # A younger write must not complete before an older write to the
            # same physical register on this conservative in-order model.
            max((ready.get(reg, 0) - timing.latency for reg in inst.defines), default=0),
        )
        stalls += issue - next_issue
        issues.append(issue)
        for reg in inst.defines:
            ready[reg] = issue + timing.latency
        resources[timing.resource] = issue + timing.occupancy
        completion = max(completion, issue + timing.latency)
        next_issue = issue + 1
        issue_blocked_until = issue + timing.issue_occupancy
    return OrderEstimate(completion, stalls, tuple(issues))
