"""Use an RVV-enabled QEMU virt CPU for all variants of the W6 Attention comparison."""
from dataclasses import replace
from pathlib import Path
import shlex


RVV_CPU = "rv64,v=true,vlen=256,elen=64,vext_spec=v1.0"


def rvv_toolchain(toolchain, directory):
    """Select the experiment's CPU without changing the production runner or compiler."""
    qemu = shlex.quote(toolchain.qemu)
    launcher = Path(directory).resolve() / "qemu-attention-rvv"
    # exec preserves the QEMU PID for RSS sampling, UART timing and failure cleanup.
    script = f"""#!/bin/sh
set -eu
qemu={qemu}
if [ "$#" = 1 ] && [ "$1" = --version ]; then
    exec "$qemu" --version
fi
if [ "$#" -lt 8 ] || [ "$1" != -machine ] || [ "$3" != -cpu ] || [ "$4" != rv64 ] ||
   [ "$5" != -accel ] || [ "$6" != tcg,thread=multi ] || [ "$7" != -smp ] || [ "$8" != 4 ]; then
    echo 'Expected the four-core QEMU virt experiment command' >&2
    exit 2
fi
machine=$2
shift 4
set -- -machine "$machine" -cpu {RVV_CPU} "$@"
printf '%s\\n' "$qemu" "$@" > qemu-argv.txt
exec "$qemu" "$@"
"""
    with launcher.open("x", encoding="utf-8") as stream:
        stream.write(script)
    launcher.chmod(0o755)
    return replace(toolchain, qemu=str(launcher))
