# Build from the repository root.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates python3.12 python3.12-venv qemu-system-misc libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

WORKDIR /opt/scratchv
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt ./
COPY LICENSES ./LICENSES
COPY scratchv ./scratchv
COPY scratchv_dag ./scratchv_dag
COPY benchmarks ./benchmarks
COPY scripts/run_inference.py ./scripts/run_inference.py
COPY probes ./probes

# Reuse the project's frozen Python/Zig deployment environment.
# Model/tokenizer assets and compiled artifacts are mounted at runtime.
RUN python -m pip install --no-cache-dir -e ".[llm-deploy]" \
    && python -c "from scratchv.runtime.riscv_tensor import discover_toolchain; discover_toolchain()" \
    && qemu-system-riscv64 --version

ENTRYPOINT ["python", "/opt/scratchv/scripts/run_inference.py"]
