"""Tests for the torch-less console entry point (muscriptor/launcher.py)."""

import sys

import pytest

import muscriptor.launcher as launcher


def test_main_delegates_to_cli(monkeypatch):
    """With torch importable, main() just runs the real CLI."""
    called = []
    import muscriptor.main as main_mod

    monkeypatch.setattr(main_mod, "main", lambda: called.append(True))
    launcher.main()
    assert called == [True]


def test_reexec_command_pins_python_and_version(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["muscriptor", "serve", "--port", "1234"])
    monkeypatch.setenv(launcher._REEXEC_SPEC, "muscriptor==9.9.9")
    cmd = launcher._reexec_command()
    assert cmd[:5] == ["uv", "tool", "run", "--python", "3.12"]
    assert cmd[5:] == ["--from", "muscriptor==9.9.9", "muscriptor", "serve", "--port", "1234"]


def test_missing_torch_on_intel_mac_without_uv_explains(monkeypatch, capsys):
    """No uv on PATH: print the manual `uvx --python 3.12` fix and exit 1."""
    monkeypatch.setattr(launcher, "_is_intel_mac", lambda: True)
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit) as exc:
        launcher._handle_missing_torch()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "PyTorch has no Intel-mac wheels" in err
    assert "uvx --python 3.12 muscriptor" in err


def test_missing_torch_reexec_guard(monkeypatch, capsys):
    """The re-exec guard env var prevents exec loops even when uv exists."""
    monkeypatch.setattr(launcher, "_is_intel_mac", lambda: True)
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setenv(launcher._REEXEC_GUARD, "1")
    with pytest.raises(SystemExit):
        launcher._handle_missing_torch()
    assert "Re-launching" not in capsys.readouterr().err


def test_missing_torch_elsewhere_generic_message(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "_is_intel_mac", lambda: False)
    with pytest.raises(SystemExit) as exc:
        launcher._handle_missing_torch()
    assert exc.value.code == 1
    assert "PyTorch is not installed" in capsys.readouterr().err
