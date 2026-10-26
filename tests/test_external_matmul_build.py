"""Build dependency and failure coverage for selected external MatMul calls."""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from probes.w4_qwen3_full import host_diagnostic as host
from scratchv.backend.external_matmul import ExternalMatMul
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime import riscv_external, riscv_tensor
from scratchv.runtime.weight_bundle import write_weight_bundle


MATMUL_SOURCE = """
#include <stdint.h>
int sv_matmul_f32_v1(const float *a, const float *b, float *c,
                    uint64_t m, uint64_t n, uint64_t k) {
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (uint64_t q = 0; q < k; ++q)
                sum = sum + a[i*k+q] * b[q*n+j];
            c[i*n+j] = sum;
        }
    }
    return 0;
}
"""


def model(provider, *, target="host", storage="external"):
    a, b = Value("a", shape=(2, 3)), Value("b", shape=(3, 2))
    builder = IRBuilder()
    builder.new_function("main", [a])
    builder.new_block("entry")
    builder.program.global_values.append(b)
    builder.ret(builder.matmul(a, b))
    weights = np.arange(6, dtype=np.float32).reshape(3, 2) / 4
    artifact = TensorCCodegen(
        builder.program, {"b": weights}, constant_storage=storage,
        matmul_implementation="require_external", external_matmul=provider,
        matmul_target=target,
    ).generate()
    return artifact, weights


def build(kind, artifact, directory, **kwargs):
    if kind == "host":
        return host.build_library(artifact, directory, ["fixture-cc"], **kwargs)
    toolchain = riscv_tensor.RiscVToolchain(("fixture-clang",), "unused-qemu")
    if kind == "inline-rv64":
        return riscv_tensor.build_riscv_tensor(artifact, directory, toolchain, **kwargs)
    weights_dir = directory.with_name(directory.name + "-weights")
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, weights_dir)
    return riscv_external.build_external(artifact, weights_dir, directory, toolchain, **kwargs)


@pytest.mark.parametrize("kind", ["host", "inline-rv64", "external-rv64"])
def test_build_dependencies_resolve_in_callers_directory(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    dependencies = tmp_path / "deps"
    dependencies.mkdir()
    (dependencies / "provider.c").write_text(MATMUL_SOURCE)
    provider = ExternalMatMul(
        name="test-provider", sources=("deps/provider.c", "deps/early.a"), include_dirs=("deps",),
        link_flags=("-L", "deps", "deps/helper.a", "deps/object.o"), targets=("host", "riscv64"),
    )
    artifact, _ = model(provider, target="host" if kind == "host" else "riscv64",
                        storage="inline" if kind == "inline-rv64" else "external")
    commands = []

    def stop(command, **_):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, b"", b"controlled compile stop")

    module = {"host": host, "inline-rv64": riscv_tensor, "external-rv64": riscv_external}[kind]
    monkeypatch.setattr(module, "_run_process", stop)
    with pytest.raises(RuntimeError, match="controlled compile stop"):
        build(kind, artifact, tmp_path / "build", extra_sources=("deps/extra.c",),
              extra_cflags=("-DBUILD_OPTION=23", "-ffast-math"),
              extra_ldflags=("deps/extra.a",))
    command, = commands
    source = str(dependencies / "provider.c")
    extra_source = str(dependencies / "extra.c")
    archive = str(dependencies / "helper.a")
    assert source in command and extra_source in command
    assert "-I" + str(dependencies) in command
    assert command[command.index("-L") + 1] == str(dependencies)
    assert command.index(source) < command.index(archive)
    assert command.index(extra_source) < command.index(str(dependencies / "early.a"))
    assert command.index(str(dependencies / "object.o")) < command.index(str(dependencies / "early.a"))
    assert command.index(extra_source) < command.index(str(dependencies / "extra.a"))
    assert "-DBUILD_OPTION=23" in command
    assert max(i for i, flag in enumerate(command) if flag == "-fno-fast-math") > command.index("-ffast-math")
    assert command.index("-ffp-contract=off", command.index(archive)) > command.index(archive)


@pytest.mark.parametrize("kind", ["host", "inline-rv64", "external-rv64"])
def test_wrong_target_is_rejected_before_build_directory(tmp_path, kind):
    provider = ExternalMatMul(name="test-provider", targets=("host", "riscv64"))
    wrong_target = "riscv64" if kind == "host" else "host"
    artifact, _ = model(provider, target=wrong_target,
                        storage="inline" if kind == "inline-rv64" else "external")
    with pytest.raises(ValueError, match="build target"):
        build(kind, artifact, tmp_path / "build")
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("harts", [1, 2, 4])
def test_serial_external_call_can_link_with_builtin_worker_runtime(tmp_path, monkeypatch, harts):
    provider = ExternalMatMul(name="test-provider", targets=("riscv64",))
    artifact, _ = model(provider, target="riscv64")
    commands = []

    def stop(command, **_):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, b"", b"controlled compile stop")

    monkeypatch.setattr(riscv_external, "_run_process", stop)
    with pytest.raises(RuntimeError, match="controlled compile stop"):
        build("external-rv64", artifact, tmp_path / "build", harts=harts)
    command, = commands
    assert f"-DSCRATCHV_HARTS={harts}" in command


def test_serial_external_call_keeps_runtime_hart_validation(tmp_path):
    provider = ExternalMatMul(name="test-provider", targets=("riscv64",))
    artifact, _ = model(provider, target="riscv64")
    with pytest.raises(ValueError, match="harts must be"):
        build("external-rv64", artifact, tmp_path / "build", harts=3)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("flag", ["-Ofast", "-ffast-math", "-ffp-contract=fast", "-mllvm", "@unsafe.rsp"])
@pytest.mark.parametrize("location", ["provider", "extra"])
def test_link_flags_cannot_override_model_compilation(tmp_path, flag, location):
    provider = ExternalMatMul(
        name="test-provider", link_flags=(flag,) if location == "provider" else ())
    artifact, _ = model(provider)
    with pytest.raises(ValueError, match="compiler option"):
        build("host", artifact, tmp_path / "build",
              extra_ldflags=(flag,) if location == "extra" else ())
    assert not (tmp_path / "build").exists()


def host_compiler():
    try:
        return host.compiler_command()
    except FileNotFoundError:
        pytest.skip("Host C compiler is required")


@pytest.mark.parametrize("link_style", ["archive-path", "library-search-path"])
def test_host_links_static_library_and_extra_source(tmp_path, monkeypatch, link_style):
    compiler = host_compiler()
    ar = shutil.which("llvm-ar") or shutil.which("ar")
    if ar is None:
        pytest.skip("Static archive tool is required")
    monkeypatch.chdir(tmp_path)
    dependencies = tmp_path / "deps"
    dependencies.mkdir()
    source = dependencies / "provider.c"
    source.write_text(MATMUL_SOURCE.replace(
        "    return 0;", "    extern int build_option(void);\n    return build_option() == 23 ? 0 : 99;"))
    (dependencies / "config.h").write_text("#define EXPECTED_OPTION BUILD_OPTION\n")
    (dependencies / "extra.c").write_text(
        '#include "config.h"\nint build_option(void) { return EXPECTED_OPTION; }\n')
    obj, archive = dependencies / "provider.o", dependencies / "libprovider.a"
    subprocess.run([*compiler, "-c", "-fPIC", "-O2", "-fno-fast-math", "-ffp-contract=off",
                    str(source), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run([ar, "rcs", str(archive), str(obj)], check=True, capture_output=True)
    link_flags = (("deps/libprovider.a",) if link_style == "archive-path"
                  else ("-Ldeps", "-lprovider"))
    provider = ExternalMatMul(name="test-provider", include_dirs=("deps",), link_flags=link_flags)
    artifact, weights = model(provider)
    library, _ = host.build_library(
        artifact, tmp_path / "build", compiler,
        extra_sources=("deps/extra.c",), extra_cflags=("-DBUILD_OPTION=23",))
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 2
    buffers = host.prepare_buffers(artifact, {"a": a})
    np.testing.assert_array_equal(host.invoke_library(artifact, buffers, library), a @ weights)


@pytest.mark.skipif(sys.platform != "linux", reason="ELF shared-library path coverage")
@pytest.mark.parametrize("suffix", [".so", ".so.1", ".so.1.2"])
@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_host_links_shared_library_from_callers_directory(tmp_path, monkeypatch, suffix, absolute):
    compiler = host_compiler()
    monkeypatch.chdir(tmp_path)
    dependencies = tmp_path / "deps with spaces"
    dependencies.mkdir()
    source = dependencies / "provider.c"
    source.write_text(MATMUL_SOURCE)
    shared_library = dependencies / ("libprovider" + suffix)
    subprocess.run(
        [*compiler, "-shared", "-fPIC", "-O2", "-fno-fast-math", "-ffp-contract=off",
         str(source), "-o", str(shared_library)],
        check=True, capture_output=True,
    )
    library_path = shared_library if absolute else shared_library.relative_to(tmp_path)
    provider = ExternalMatMul(name="test-provider", link_flags=(str(library_path),))
    artifact, weights = model(provider)
    library, _ = host.build_library(artifact, tmp_path / "different build directory", compiler)
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 2
    buffers = host.prepare_buffers(artifact, {"a": a})
    np.testing.assert_array_equal(host.invoke_library(artifact, buffers, library), a @ weights)


def test_versioned_library_resolution_preserves_other_linker_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flags = ("deps/libprovider.so.debug", "deps/libprovider.so.1beta", "-lprovider",
             "-Wl,-rpath,$ORIGIN/deps")
    artifact, _ = model(ExternalMatMul(name="test-provider", link_flags=flags))
    _, _, resolved_flags, _ = riscv_tensor._external_matmul_build_args(artifact, "host")
    assert resolved_flags == list(flags)


@pytest.mark.parametrize("failure", ["missing-file", "missing-symbol"])
def test_host_dependency_failures_do_not_fall_back(tmp_path, failure):
    compiler = host_compiler()
    source = tmp_path / "provider.c"
    if failure == "missing-symbol":
        source.write_text("int unrelated_function(void) { return 0; }\n")
    provider = ExternalMatMul(name="test-provider", sources=(source,))
    artifact, _ = model(provider)
    with pytest.raises(RuntimeError, match="compiler failed"):
        host.build_library(artifact, tmp_path / "build", compiler)
    assert "sv_matmul_f32_v1(" in artifact.source


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_rv64_external_c_function_smoke(tmp_path):
    try:
        toolchain = riscv_tensor.discover_toolchain()
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(str(error))
    source = tmp_path / "provider.c"
    source.write_text(MATMUL_SOURCE)
    provider = ExternalMatMul(name="test-provider", sources=(source,), targets=("riscv64",))
    artifact, weights = model(provider, target="riscv64")
    weights_dir = tmp_path / "weights"
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, weights_dir)
    executable = riscv_external.build_external(
        artifact, weights_dir, tmp_path / "build", toolchain, timeout=180)
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 2
    output, report = riscv_external.run_external(
        executable, {"a": a}, tmp_path / "run", timeout=30)
    np.testing.assert_array_equal(output, a @ weights)
    assert report["passed"] and report["guest_completed"]


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("dependency,storage", [("object", "inline"), ("archive", "external")])
def test_rv64_precompiled_provider_executes(tmp_path, dependency, storage):
    # Explicit tool configuration must work, rather than silently skip a broken
    # cross compiler. Each case proves a different supported linker input.
    toolchain = riscv_tensor.discover_toolchain()
    source = tmp_path / "provider.c"
    source.write_text(MATMUL_SOURCE, encoding="utf-8")
    obj = tmp_path / "provider.o"
    target = (["-target", "riscv64-linux-musl"] if toolchain.is_zig
              else ["--target=riscv64-unknown-elf"])
    architecture = (["-mcpu=generic_rv64+m+a+f+d+c+zicsr+zifencei"] if toolchain.is_zig
                    else ["-march=rv64gc"])
    subprocess.run(
        [*toolchain.cc, *target, *architecture, "-mabi=lp64d", "-mcmodel=medany",
         "-msmall-data-limit=0", "-mno-relax", "-ffreestanding", "-fno-builtin",
         "-fno-stack-protector", "-fno-pie", "-fPIC" if toolchain.is_zig else "-fno-pic",
         "-O2", "-fno-fast-math", "-ffp-contract=off", "-fno-strict-aliasing",
         "-c", str(source), "-o", str(obj)],
        check=True, capture_output=True, timeout=180,
    )
    if dependency == "archive":
        archive = tmp_path / "libprovider.a"
        if toolchain.is_zig:
            ar = [toolchain.cc[0], "ar"]
        else:
            executable = shutil.which("llvm-ar") or shutil.which("ar")
            assert executable, "A static archive tool is required for explicit RV64 tests"
            ar = [executable]
        subprocess.run([*ar, "rcs", str(archive), str(obj)],
                       check=True, capture_output=True, timeout=30)
        provider = ExternalMatMul(
            name="precompiled-provider", link_flags=(str(archive),), targets=("riscv64",))
    else:
        provider = ExternalMatMul(
            name="precompiled-provider", sources=(obj,), targets=("riscv64",))
    artifact, weights = model(provider, target="riscv64", storage=storage)
    a = np.arange(6, dtype=np.float32).reshape(2, 3) / 2
    if storage == "inline":
        executable = riscv_tensor.build_riscv_tensor(
            artifact, tmp_path / "build", toolchain, timeout=180)
        output = riscv_tensor.run_riscv_tensor(
            executable, {"a": a}, tmp_path / "run", timeout=30).output
        assert str(obj) in executable.compile_command
        assert str(source) not in executable.compile_command
    else:
        weights_dir = tmp_path / "weights"
        write_weight_bundle(artifact.external_weights, artifact.external_initializers, weights_dir)
        executable = riscv_external.build_external(
            artifact, weights_dir, tmp_path / "build", toolchain, timeout=180)
        output, report = riscv_external.run_external(
            executable, {"a": a}, tmp_path / "run", timeout=30)
        assert report["passed"] and report["guest_completed"]
        assert str(archive) in executable.evidence["command"]
        assert str(source) not in executable.evidence["command"]
        assert str(obj) not in executable.evidence["command"]
    np.testing.assert_array_equal(output, a @ weights)



@pytest.mark.parametrize("measurements_match", [False, True])
@pytest.mark.parametrize("build_harts", [1, 2])
def test_measured_hart_mismatch_rejected_even_when_every_node_is_builtin(
        tmp_path, monkeypatch, measurements_match, build_harts):
    from scratchv.backend.external_matmul import MatMulMeasurement, MatMulPerformance

    selected = ExternalMatMul(name="test-provider", targets=("riscv64",),
                              numerics="provider_fp32", parallel_rows=True)
    a, b = Value("a", shape=(2, 3)), Value("b", shape=(3, 2))
    graph = IRBuilder()
    graph.new_function("main", [a])
    graph.new_block("entry")
    graph.program.global_values.append(b)
    graph.ret(graph.matmul(a, b))
    measurement = MatMulMeasurement(
        left_shape=a.shape, right_shape=b.shape, rhs_kind="constant",
        builtin_rhs_layout="column-panel", builtin_seconds=1.0, external_seconds=0.5)
    profile = MatMulPerformance(
        provider=selected, target="riscv64", environment="fixture QEMU; four harts", harts=4,
        measurements=(measurement,) if measurements_match else (),
        matmul_policy="blocked_fma", matmul_tile="2x4", matmul_rhs_layout="column-panel",
        kernel_calls=True, qemu_workspace_layout="none")
    artifact = TensorCCodegen(
        graph.program, {"b": np.arange(6, dtype=np.float32).reshape(3, 2) / 4},
        constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_tile="2x4", matmul_rhs_layout="column-panel",
        matmul_implementation="measured", external_matmul=selected,
        external_matmul_numerics="allow_provider_fp32", matmul_target="riscv64",
        matmul_performance=profile,
    ).generate()
    assert (artifact.external_matmul is not None) == measurements_match

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Hart mismatch must fail before the compiler is started")

    monkeypatch.setattr(riscv_external, "_run_process", unexpected_compile)
    directory = tmp_path / "build"
    with pytest.raises(ValueError, match="hart"):
        build("external-rv64", artifact, directory, harts=build_harts)
    assert not directory.exists()




@pytest.mark.parametrize("flag", ["-DSCRATCHV_HARTS=2", "-USCRATCHV_HARTS"])
@pytest.mark.parametrize("location", ["extra_cflags", "extra_ldflags"])
def test_measured_profile_rejects_worker_count_override_in_extra_flags(flag, location):
    from tests.test_tensor_c_external import matmul_experiment
    from tests.test_tensor_c_external_matmul import _measured_model, _performance, provider

    selected = provider(numerics="provider_fp32")
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    artifact = _measured_model(program, bindings, selected, _performance(selected))
    with pytest.raises(ValueError, match="hart"):
        riscv_tensor._external_matmul_build_args(artifact, "host", **{location: (flag,)})
