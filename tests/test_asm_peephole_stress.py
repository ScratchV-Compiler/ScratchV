"""Stress tests: large inputs, repeated runs, time bounds."""

from __future__ import annotations

import statistics
import time

import pytest

from scratchv.backend.asm_peephole import AsmPeepholeOptimizer


def _gen_fusible_asm(n_pairs: int) -> str:
    lines = [".text", "stress:"]
    for i in range(n_pairs):
        reg = f"t{i % 8}"
        lines.append(f"  addi {reg}, {reg}, 1")
        lines.append(f"  addi {reg}, {reg}, 2")
    lines.append("  ret")
    return "\n".join(lines) + "\n"


@pytest.mark.stress
class TestPeepholeStress:
    """Large-scale and repeated optimization."""

    @pytest.mark.parametrize("n_pairs", [500, 2000, 5000])
    def test_large_fusion_completes(self, n_pairs: int):
        asm = _gen_fusible_asm(n_pairs)
        opt = AsmPeepholeOptimizer()
        t0 = time.perf_counter()
        result, changes = opt.optimize(asm)
        elapsed = time.perf_counter() - t0

        assert changes >= n_pairs
        assert "ret" in result
        assert elapsed < 60.0, f"too slow: {elapsed:.2f}s for {n_pairs} pairs"

    def test_repeated_optimize_deterministic(self):
        asm = _gen_fusible_asm(200)
        opt = AsmPeepholeOptimizer()
        results = [opt.optimize(asm)[0] for _ in range(5)]
        assert len(set(results)) == 1

    def test_max_iterations_safety(self):
        """Adversarial pattern: many chained addi need multiple passes."""
        lines = [".text", "chain:"]
        for _ in range(20):
            lines.append("  addi t0, t0, 1")
        lines.append("  ret")
        asm = "\n".join(lines)
        result, changes = AsmPeepholeOptimizer().optimize(asm)
        assert changes >= 1
        assert result.count("addi") < asm.count("addi")

    def test_throughput_baseline(self):
        """5000-pair input should stay under 2s on dev machine."""
        asm = _gen_fusible_asm(5000)
        times = []
        for _ in range(3):
            opt = AsmPeepholeOptimizer()
            t0 = time.perf_counter()
            opt.optimize(asm)
            times.append(time.perf_counter() - t0)
        median = statistics.median(times)
        assert median < 2.0, f"median {median:.3f}s exceeds 2s budget"

    def test_empty_and_whitespace_only(self):
        opt = AsmPeepholeOptimizer()
        for asm in ["", "\n\n", "   \n  \n"]:
            result, changes = opt.optimize(asm)
            assert changes == 0

    def test_very_long_label_preserved(self):
        label = "L_" + "x" * 200
        asm = f".text\n{label}:\n  addi t0, t0, 1\n  addi t0, t0, 1\n  ret\n"
        result, changes = AsmPeepholeOptimizer().optimize(asm)
        assert label + ":" in result
        assert changes >= 1

    def test_hex_fusion_large_batch(self):
        lines = [".text", "hex:"]
        for i in range(200):
            reg = f"t{i % 8}"
            lines.append(f"  addi {reg}, {reg}, 0x1")
            lines.append(f"  addi {reg}, {reg}, 0x2")
        lines.append("  ret")
        asm = "\n".join(lines)
        result, changes = AsmPeepholeOptimizer().optimize(asm)
        assert changes >= 200
        assert "(" not in result
        assert "3" in result

    def test_mid_labels_never_dropped_under_load(self):
        """Labels on the *second* window insn block fusion; none may vanish."""
        lines = [".text"]
        for i in range(50):
            lines.append("  addi t0, t0, 1")
            lines.append(f"L{i}:  addi t0, t0, 1")
        lines.append("  ret")
        asm = "\n".join(lines)
        result, _changes = AsmPeepholeOptimizer().optimize(asm)
        for i in range(50):
            assert f"L{i}:" in result
        # Unlabeled addi + labeled addi must not fuse (would move/drop L*).
        # Pattern: addi; L: addi — second has label → refuse.
        # Adjacent "L: addi; addi" *may* fuse while keeping L on the result.
        assert "L0:" in result
