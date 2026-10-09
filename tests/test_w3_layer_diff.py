"""Recorded checkpoint comparison must fail closed on incomplete evidence."""

import io
import json
import zipfile

import numpy as np
import pytest

from probes.w3_layer_diff import run


@pytest.fixture
def evidence(tmp_path):
    entries = [dict(name=f"layer_{index}.hidden", shape=[1, 2], dtype="float32",
                    layer=index, checkpoint="hidden", sequence_axis=1) for index in range(2)]
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(dict(version=1, checkpoints=entries)), encoding="utf-8")
    arrays = {entry["name"]: np.array([[1, 2]], dtype="float32") for entry in entries}
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    np.savez(actual, **arrays)
    np.savez(reference, **arrays)
    return actual, reference, schema, arrays


def test_complete_compare_and_strict_tolerance(evidence):
    actual, reference, schema, arrays = evidence
    report = run.compare_evidence(actual, reference, schema)
    assert report["passed"] and report["status"] == "PASS"
    assert report["coverage"]["complete"]
    assert report["first_divergence"] is None
    arrays["layer_0.hidden"][0, 0] = 2
    arrays["layer_1.hidden"][0, 1] = 7
    np.savez(actual, **arrays)
    report = run.compare_evidence(actual, reference, schema, atol=1)
    assert report["status"] == "FAIL"
    assert report["first_divergence"] == "layer_0.hidden"
    assert report["worst_element"]["name"] == "layer_1.hidden"
    assert report["worst_element"]["worst_index"] == [0, 1]
    assert report["worst_element"]["actual_value"] == 7
    assert report["worst_element"]["expected_value"] == 2


@pytest.mark.parametrize("selection", [dict(layers=[0]), dict(checkpoints=["hidden"]),
                                        dict(layers=[0, 1]), dict(checkpoints=["layer_0.hidden"])])
def test_selection_is_always_partial(evidence, selection):
    report = run.compare_evidence(*evidence[:3], **selection)
    assert report["status"] == "PARTIAL"
    assert report["selected_passed"] and not report["passed"]
    assert not report["coverage"]["complete"]
    assert report["coverage"]["evidence_validated_checkpoints"] == 2


@pytest.mark.parametrize("selection", [dict(layers=[-1]), dict(layers=[2]), dict(layers=[True]),
                                        dict(layers=[0, 0]), dict(checkpoints=["missing"]),
                                        dict(checkpoints=["hidden", "hidden"]),
                                        dict(layers=[0], checkpoints=["layer_1.hidden"])])
def test_invalid_or_empty_selection_fails(evidence, selection):
    with pytest.raises(run.EvidenceError):
        run.compare_evidence(*evidence[:3], **selection)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "inf", "negative_inf", "object"])
def test_bad_arrays_fail_even_when_unselected(evidence, mutation):
    actual, reference, schema, arrays = evidence
    name = "layer_1.hidden"
    if mutation == "missing":
        arrays.pop(name)
    elif mutation == "extra":
        arrays["extra"] = arrays[name]
    elif mutation == "shape":
        arrays[name] = arrays[name].reshape(2)
    elif mutation == "dtype":
        arrays[name] = arrays[name].astype("float64")
    elif mutation == "object":
        arrays[name] = arrays[name].astype(object)
    else:
        arrays[name][0, 1] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[mutation]
    np.savez(actual, **arrays)
    with pytest.raises(run.EvidenceError):
        run.compare_evidence(actual, reference, schema, layers=[0])


def test_nonfinite_failure_preserves_index_and_json_safe_value(evidence):
    actual, reference, schema, arrays = evidence
    arrays["layer_0.hidden"][0, 1] = np.nan
    np.savez(actual, **arrays)
    with pytest.raises(run.EvidenceError) as caught:
        run.compare_evidence(actual, reference, schema)
    assert caught.value.checkpoint == "layer_0.hidden"
    assert caught.value.details["worst_index"] == [0, 1]
    assert caught.value.details["value"] == "NaN"
    assert caught.value.details["evidence_path"] == str(actual.resolve())


def test_forged_npy_shape_rejected_before_numpy_allocation(evidence, monkeypatch):
    actual, reference, schema, arrays = evidence
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(forged, dict(descr="<f4", fortran_order=False,
                                                     shape=(2**60,)))
    valid = io.BytesIO()
    np.save(valid, arrays["layer_1.hidden"])
    with zipfile.ZipFile(actual, "w") as archive:
        archive.writestr("layer_0.hidden.npy", forged.getvalue())
        archive.writestr("layer_1.hidden.npy", valid.getvalue())

    def no_array_load(*args, **kwargs):
        raise AssertionError("must reject the header before allocating")

    monkeypatch.setattr(run.np, "load", no_array_load)
    with pytest.raises(run.EvidenceError, match="NPY header"):
        run.compare_evidence(actual, reference, schema)


def test_error_overflow_is_reported_as_worst_checkpoint(evidence):
    actual, reference, schema, arrays = evidence
    document = json.loads(schema.read_text())
    for entry in document["checkpoints"]:
        entry["dtype"] = "float64"
    schema.write_text(json.dumps(document))
    expected = {name: value.astype("float64") for name, value in arrays.items()}
    expected["layer_1.hidden"][0, 1] = -1e308
    arrays = {name: value.copy() for name, value in expected.items()}
    arrays["layer_1.hidden"][0, 1] = 1e308
    np.savez(actual, **arrays)
    np.savez(reference, **expected)
    report = run.compare_evidence(actual, reference, schema)
    assert report["status"] == "FAIL"
    assert report["worst_element"]["name"] == "layer_1.hidden"
    assert report["worst_element"]["reason"] == "absolute error overflow"
    assert report["worst_element"]["worst_index"] == [0, 1]


def test_duplicate_npz_members_rejected(evidence):
    actual, reference, schema, arrays = evidence
    buffer = io.BytesIO()
    np.save(buffer, arrays["layer_0.hidden"])
    with zipfile.ZipFile(actual, "w") as archive:
        archive.writestr("layer_0.hidden.npy", buffer.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("layer_0.hidden.npy", buffer.getvalue())
        archive.writestr("layer_1.hidden.npy", buffer.getvalue())
    with pytest.raises(run.EvidenceError, match="duplicate ZIP"):
        run.compare_evidence(actual, reference, schema)


def test_ambiguous_numpy_member_aliases_rejected(evidence):
    actual, reference, schema, arrays = evidence
    document = json.loads(schema.read_text())
    document["checkpoints"][0]["name"] = "x"
    document["checkpoints"][1]["name"] = "x.npy"
    schema.write_text(json.dumps(document))
    np.savez(actual, **{"x": np.ones((1, 2), dtype="float32"),
                        "x.npy": np.zeros((1, 2), dtype="float32")})
    with pytest.raises(run.EvidenceError, match="ambiguous"):
        run.compare_evidence(actual, reference, schema)


@pytest.mark.parametrize("mutation", ["duplicate_name", "negative_shape", "bool_shape", "bad_axis", "bad_layer", "bad_dtype", "bool_version"])
def test_bad_schema_rejected(evidence, mutation):
    actual, reference, schema, arrays = evidence
    document = json.loads(schema.read_text())
    first = document["checkpoints"][0]
    if mutation == "duplicate_name":
        document["checkpoints"][1]["name"] = first["name"]
    elif mutation == "bool_version":
        document["version"] = True
    else:
        key, value = {"negative_shape": ("shape", [-1]), "bool_shape": ("shape", [True]),
                      "bad_axis": ("sequence_axis", 2), "bad_layer": ("layer", -1),
                      "bad_dtype": ("dtype", "object")}[mutation]
        first[key] = value
    schema.write_text(json.dumps(document))
    with pytest.raises(run.EvidenceError):
        run.compare_evidence(actual, reference, schema)


def test_duplicate_json_keys_and_reference_schema_mismatch(evidence, tmp_path):
    actual, reference, schema, arrays = evidence
    second = tmp_path / "reference_schema.json"
    document = json.loads(schema.read_text())
    document["checkpoints"].reverse()
    second.write_text(json.dumps(document))
    with pytest.raises(run.EvidenceError, match="schemas must match"):
        run.compare_evidence(actual, reference, schema, reference_schema_path=second)
    schema.write_text('{"version":1,"version":1,"checkpoints":[]}')
    with pytest.raises(run.EvidenceError, match="duplicate JSON"):
        run.load_schema(schema)


def test_size_limit_and_invalid_tolerance(evidence):
    with pytest.raises(run.EvidenceError, match="max_bytes"):
        run.compare_evidence(*evidence[:3], max_bytes=1)
    with pytest.raises(ValueError, match="atol"):
        run.compare_evidence(*evidence[:3], atol=float("nan"))


def _argv(evidence, out):
    return ["--actual", str(evidence[0]), "--reference", str(evidence[1]),
            "--schema", str(evidence[2]), "--out", str(out)]


@pytest.mark.parametrize("partial,expected", [(False, 0), (True, 2)])
def test_cli_exit_codes_and_reports(evidence, tmp_path, monkeypatch, partial, expected):
    monkeypatch.setattr(run, "source_evidence", lambda: {})
    out = tmp_path / "out"
    assert run.main(_argv(evidence, out) + (["--layer", "0"] if partial else [])) == expected
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == ("PARTIAL" if partial else "PASS")
    assert (out / "report.md").is_file() and (out / "report.html").is_file()
    assert report["artifacts"]["actual"]["sha256"]
    assert run.main(_argv(evidence, out)) == 1


def test_cli_invalid_evidence_and_publication_error_fail(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "source_evidence", lambda: {})
    actual, reference, schema, arrays = evidence
    arrays.pop("layer_1.hidden")
    np.savez(actual, **arrays)
    out = tmp_path / "out"
    assert run.main(_argv(evidence, out)) == 1
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "FAIL" and report["first_divergence"] == "layer_1.hidden"

    def fail_publication(*args):
        raise OSError("injected report writer failure")

    monkeypatch.setattr(run, "write_reports", fail_publication)
    assert run.main(_argv(evidence, tmp_path / "cannot-publish")) == 1
