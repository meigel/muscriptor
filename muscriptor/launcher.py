"""Console entry point that survives environments without PyTorch.

PyTorch stopped publishing Intel-mac (darwin x86_64) wheels after 2.2.2, which
itself only supports Python <= 3.12. So that `uvx muscriptor` still resolves
on Intel macs with a newer interpreter, torch is simply not installed there
(see the dependency markers in pyproject.toml); this module detects the
missing torch and transparently re-launches the CLI under a supported Python
via uv — or explains what to do when it can't.

This module (and the package __init__) must stay importable without torch.
"""

import os
import platform
import shutil
import sys
from typing import NoReturn

# Last Python with Intel-mac torch wheels (torch 2.2.2).
_INTEL_MAC_PYTHON = "3.12"
_REEXEC_GUARD = "_MUSCRIPTOR_NO_REEXEC"
# Test hook: overrides the requirement re-exec installs (e.g. a local wheel).
_REEXEC_SPEC = "_MUSCRIPTOR_REEXEC_SPEC"


def _is_intel_mac() -> bool:
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def _requirement() -> str:
    override = os.environ.get(_REEXEC_SPEC)
    if override:
        return override
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"muscriptor=={version('muscriptor')}"
    except PackageNotFoundError:
        return "muscriptor"


def _reexec_command() -> list[str]:
    return [
        "uv",
        "tool",
        "run",
        "--python",
        _INTEL_MAC_PYTHON,
        "--from",
        _requirement(),
        "muscriptor",
        *sys.argv[1:],
    ]


def _handle_missing_torch() -> NoReturn:
    py = ".".join(str(v) for v in sys.version_info[:2])
    if _is_intel_mac() and sys.version_info >= (3, 13):
        print(
            f"[muscriptor] PyTorch has no Intel-mac wheels for Python {py} (the "
            f"last Intel-mac release, torch 2.2.2, supports Python <= {_INTEL_MAC_PYTHON}).",
            file=sys.stderr,
        )
        if os.environ.get(_REEXEC_GUARD) != "1" and shutil.which("uv"):
            cmd = _reexec_command()
            print(
                f"[muscriptor] Re-launching with Python {_INTEL_MAC_PYTHON}: "
                + " ".join(cmd),
                file=sys.stderr,
            )
            os.environ[_REEXEC_GUARD] = "1"
            os.execvp(cmd[0], cmd)
        print(
            "[muscriptor] Re-run with a supported interpreter, e.g.:\n"
            f"    uvx --python {_INTEL_MAC_PYTHON} muscriptor\n"
            f"or install muscriptor under Python 3.10-{_INTEL_MAC_PYTHON}.",
            file=sys.stderr,
        )
    else:
        print(
            "[muscriptor] PyTorch is not installed in this environment; "
            "reinstall muscriptor with its dependencies (e.g. pip install muscriptor).",
            file=sys.stderr,
        )
    raise SystemExit(1)


def main() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        _handle_missing_torch()

    from muscriptor.main import main as cli_main

    cli_main()
