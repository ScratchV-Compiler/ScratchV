"""Run complete standalone CNN binaries under QEMU user mode and compare state."""

import hashlib
from pathlib import Path
import resource
import shutil
import struct
import subprocess


def assemble_listing(source: Path, directory: Path) -> bytes:
    obj, binary = directory / "roundtrip.o", directory / "roundtrip.bin"
    subprocess.run(["clang", "--target=riscv32-unknown-elf", "-march=rv32im", "-mabi=ilp32",
                    "-mno-relax", "-c", str(source), "-o", str(obj)],
                   capture_output=True, check=True, timeout=30)
    subprocess.run(["ld.lld", "-m", "elf32lriscv", "--no-relax", "--image-base=0", "-Ttext=0", "-e", "0",
                    "--oformat=binary", str(obj), "-o", str(binary)],
                   capture_output=True, check=True, timeout=30)
    return binary.read_bytes()


def execute_pair(before: Path, after: Path, metadata: dict, directory: Path) -> dict:
    tools = {name: shutil.which(name) for name in ("clang", "ld.lld", "qemu-riscv32")}
    if not all(tools.values()):
        raise RuntimeError(f"CNN execution requires clang, ld.lld and qemu-riscv32: {tools}")
    sizes = {"registers": 128, "workspace": metadata["workspace_bytes"],
             "guard": 256, "output": metadata["output_elements"] * 4 + 256}
    samples = []
    for seed in (0, 1, 7):
        values = [0 if seed == 0 else ((i * 37 + seed * 17) % 65536) - 32768
                  for i in range(metadata["input_elements"])]
        input_bytes = struct.pack(f"<{len(values)}i", *values)
        input_path = directory / "input.bin"
        input_path.write_bytes(input_bytes)
        states = []
        for phase, binary in (("before", before), ("after", after)):
            asm = directory / f"execute-{phase}.s"
            elf = directory / f"execute-{phase}.elf"
            stores = "\n".join(f"  sw x{i}, {i * 4 - 128}(sp)" for i in range(32))
            total = sum(sizes.values())
            asm.write_text(f''' .option norvc
 .option norelax
 .text
 .globl _start
_start:
 la sp, workspace
 la a0, input_tensor
 la a1, output_tensor
 call cnn_entry
{stores}
 li a0, 1
 la a1, register_dump
 li a2, {total}
 li a7, 64
 ecall
 li t0, {total}
 bne a0, t0, failed
 li a0, 0
 li a7, 93
 ecall
failed:
 li a0, 1
 li a7, 93
 ecall
 .balign 4
cnn_entry:
 .incbin "{binary.as_posix()}"
 .data
 .balign 4
input_tensor:
 .incbin "{input_path.as_posix()}"
 .bss
 .balign 16
register_dump:
 .space 128
workspace:
 .space {sizes['workspace']}
 .space {sizes['guard']}
output_tensor:
 .space {sizes['output']}
''')
            subprocess.run([tools["clang"], "--target=riscv32-linux-gnu", "-march=rv32im", "-mabi=ilp32",
                            "-nostdlib", "-static", "-fuse-ld=lld", "-Wl,--no-relax", str(asm), "-o", str(elf)],
                           check=True, capture_output=True, timeout=30)
            result = subprocess.run([tools["qemu-riscv32"], str(elf)], capture_output=True, timeout=60,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            if result.returncode or len(result.stdout) != total:
                reason = (f"Full CNN {phase} failed: return={result.returncode}, "
                          f"bytes={len(result.stdout)}/{total}, stderr={result.stderr.decode(errors='replace')}")
                if phase == "before":
                    return {"status": "baseline_failed", "backend": "qemu-riscv32",
                            "reason": reason, "equivalence_verified": False,
                            "samples": samples, "metadata": metadata,
                            "failed_input": {"seed": seed, "sha256": hashlib.sha256(input_bytes).hexdigest()},
                            "baseline_returncode": result.returncode,
                            "qemu_version": subprocess.check_output([tools["qemu-riscv32"], "--version"], text=True).splitlines()[0]}
                raise RuntimeError(reason)
            offset = 0
            state = {}
            for name, size in sizes.items():
                data = result.stdout[offset:offset + size]
                state[name + "_sha256"] = hashlib.sha256(data).hexdigest()
                if name == "output":
                    state["output_q16"] = list(struct.unpack(f"<{metadata['output_elements']}i", data[:-256]))
                    if any(data[-256:]):
                        raise RuntimeError("CNN wrote beyond its output buffer")
                if name == "guard" and any(data):
                    raise RuntimeError("CNN wrote beyond its workspace")
                offset += size
            states.append(state)
        equal = states[0] == states[1]
        samples.append({"seed": seed, "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                        "before": states[0], "after": states[1], "equal": equal})
    return {"status": "passed" if all(s["equal"] for s in samples) else "failed",
            "backend": "qemu-riscv32", "scope": "complete standalone CNN; registers, workspace and output",
            "onnx_reference_checked": False, "samples": samples, "metadata": metadata,
            "tool_versions": {name: subprocess.check_output([exe, "--version"], text=True).splitlines()[0]
                              for name, exe in tools.items()}}
