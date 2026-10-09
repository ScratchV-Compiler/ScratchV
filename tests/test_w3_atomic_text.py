"""Bounded report publication retries, without hiding failed writes."""
import errno
from types import SimpleNamespace

import pytest

from probes import w3_common as common


def denied(winerror):
    error = PermissionError(errno.EACCES, "injected report replacement failure")
    if winerror is not None:
        error.winerror = winerror
    return error


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_report_lock_retries_atomic_replace(tmp_path, monkeypatch, winerror):
    path = tmp_path / "report.json"
    path.write_text("old")
    original_replace = common.os.replace
    attempts, sleeps = [], []

    def replace(source, destination):
        attempts.append(source)
        assert path.read_text() == "old"
        if len(attempts) <= 2:
            raise denied(winerror)
        original_replace(source, destination)

    monkeypatch.setattr(common, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    common.atomic_text(path, "new\n")
    assert path.read_text() == "new\n" and len(attempts) == 3
    assert sleeps == [0.01, 0.03]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("platform,winerror,expected_attempts", [
    ("win32", 5, 5), ("win32", 32, 5), ("win32", 33, 5),
    ("linux", 5, 1), ("linux", None, 1), ("win32", 2, 1), ("win32", None, 1),
])
def test_permanent_or_unrelated_permission_errors_still_fail(
        tmp_path, monkeypatch, platform, winerror, expected_attempts):
    path = tmp_path / "report.json"
    path.write_text("old")
    error, attempts, sleeps = denied(winerror), [], []

    def replace(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(common, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as raised:
        common.atomic_text(path, "new")
    assert raised.value is error
    assert len(attempts) == expected_attempts and len(sleeps) == expected_attempts - 1
    assert sum(sleeps) <= 0.45
    assert path.read_text() == "old" and list(tmp_path.iterdir()) == [path]


def test_cleanup_failure_does_not_replace_primary_publication_error(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    error = OSError(errno.ENOSPC, "injected full output volume")
    original_unlink = common.os.unlink

    def replace(*_):
        raise error

    def unlink(*_):
        raise PermissionError("injected pending file lock")

    monkeypatch.setattr(common.os, "replace", replace)
    monkeypatch.setattr(common.os, "unlink", unlink)
    try:
        with pytest.raises(OSError) as raised:
            common.atomic_text(path, "new")
        assert raised.value is error
        assert not path.exists()
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        # Remove only this test's observed pending file after restoring unlink.
        monkeypatch.setattr(common.os, "unlink", original_unlink)
        for pending in tmp_path.iterdir():
            pending.unlink()
