#!/usr/bin/env bash
# Thin clean-container entry point; generation arguments belong to run_inference.py.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
engine="${SCRATCHV_CONTAINER_ENGINE:-docker}"
image="${SCRATCHV_W5_IMAGE:-scratchv-w5:local}"
base_image="${SCRATCHV_W5_BASE_IMAGE:-ubuntu:24.04}"
assets="${SCRATCHV_W5_ASSETS:-$repo_root/assets/qwen3-0.6b}"
output="${SCRATCHV_W5_OUTPUT:-$repo_root/output/w5-deployment}"

if ! command -v -- "$engine" >/dev/null 2>&1; then
    printf 'Container engine is unavailable: %s. Install/start it separately; no system configuration was changed.\n' "$engine" >&2
    exit 127
fi
if ! command -v setsid >/dev/null 2>&1; then
    printf 'Linux setsid is required to clean up cancelled container-client processes.\n' >&2
    exit 127
fi

# Own each preflight/build client process group while Bash waits. Killing only
# this script must not leave the client (or its helpers) building in background.
# The container engine owns daemon-side cancellation; a real-engine check is
# still required to verify that side of the protocol.
run_engine_stage() {
    local client_pid="" status attempt
    cancel_engine_stage() {
        local exit_status="$1" target="${client_pid:-${!:-}}"
        trap '' INT TERM
        if [[ -n "$target" ]]; then
            # The direct-PID fallback covers a signal before setsid takes effect.
            kill -TERM -- "-$target" 2>/dev/null || kill -TERM "$target" 2>/dev/null || true
            for ((attempt = 0; attempt < 50; attempt++)); do
                if ! kill -0 -- "-$target" 2>/dev/null; then
                    break
                fi
                sleep 0.1
            done
            kill -KILL -- "-$target" 2>/dev/null || true
            wait "$target" 2>/dev/null || true
        fi
        exit "$exit_status"
    }
    trap 'cancel_engine_stage 130' INT
    trap 'cancel_engine_stage 143' TERM
    setsid -- "$engine" "$@" &
    client_pid=$!
    if wait "$client_pid"; then
        status=0
    else
        status=$?
    fi
    trap - INT TERM
    return "$status"
}

if ! run_engine_stage info >/dev/null 2>&1; then
    printf 'Container engine is not ready: %s. No model execution was started.\n' "$engine" >&2
    exit 1
fi
if [[ ! -d "$assets" ]]; then
    printf 'Assets directory does not exist: %s\n' "$assets" >&2
    exit 2
fi
assets="$(cd -- "$assets" && pwd)"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd)"
# Docker/Podman --mount use comma-delimited fields.
if [[ "$assets" == *","* || "$output" == *","* ]]; then
    printf 'Container mount paths must not contain commas.\n' >&2
    exit 2
fi

run_engine_stage build \
    --file "$repo_root/deployment/w5/Dockerfile" \
    --build-arg "BASE_IMAGE=$base_image" \
    --tag "$image" "$repo_root"

# Numeric IDs keep report/build files owned by the invoking Linux developer.
# --init forwards signals/reaps children; the existing runner owns QEMU cleanup.
exec "$engine" run --rm --init \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --env SCRATCHV_ZIG_CACHE=/output/.zig-cache \
    --mount "type=bind,src=$assets,dst=/assets,readonly" \
    --mount "type=bind,src=$output,dst=/output" \
    "$image" --output-dir /output/run "$@"
