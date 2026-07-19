"""Fresh-interpreter tests for public package import order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_import(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def test_clustering_can_be_imported_before_optimization() -> None:
    result = _run_import(
        "from src.clustering import HERCAllocator; "
        "from src.optimization import HRPAllocator; "
        "assert HERCAllocator.__name__ == 'HERCAllocator'"
    )

    assert result.returncode == 0, result.stderr


def test_optimization_reexports_herc_without_import_cycle() -> None:
    result = _run_import(
        "from src.optimization import HERCAllocator, HERCOptimizer; "
        "assert HERCAllocator is HERCOptimizer"
    )

    assert result.returncode == 0, result.stderr
