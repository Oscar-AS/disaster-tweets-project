"""Utilities to auto-install missing Python packages at runtime."""

import importlib.util
import subprocess
import sys
from typing import Iterable, Tuple


def _is_installed(import_name: str) -> bool:
    """Return True if module can be imported, False otherwise."""
    return importlib.util.find_spec(import_name) is not None


def ensure_packages(packages: Iterable[Tuple[str, str]]) -> None:
    """
    Ensure required packages are installed.

    Args:
        packages: Iterable of (pip_name, import_name).
    """
    missing = [pip_name for pip_name, import_name in packages if not _is_installed(import_name)]
    if not missing:
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

