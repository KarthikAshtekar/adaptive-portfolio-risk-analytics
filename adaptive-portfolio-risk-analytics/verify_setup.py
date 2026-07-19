"""Verify the current repository layout and core runtime dependencies."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_PATHS = (
    "README.md",
    "project_explainer.html",
    "pyproject.toml",
    "requirements.txt",
    "config/portfolio_config.yaml",
    "src/paths.py",
    "src/data_pipeline",
    "src/covariance",
    "src/clustering",
    "src/optimization",
    "src/backtesting",
    "src/analytics",
    "src/regime",
    "src/adaptive",
    "src/experiments",
    "src/validation",
    "src/selection",
    "src/sentiment",
    "src/dashboard/app.py",
    "tests",
    "docs/architecture/ARCHITECTURE.md",
    "docs/methodology/METHODOLOGY.md",
    "docs/PROJECT_AUDIT.md",
    "outputs/final_project_pack/INDEX.md",
)

CORE_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "plotly",
    "streamlit",
    "yfinance",
    "pytest",
    "ruff",
)


def verify_project_structure() -> list[str]:
    """Return required repository paths that are currently missing."""

    return [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).exists()]


def verify_dependencies() -> list[str]:
    """Return core import names that are unavailable in this environment."""

    return [name for name in CORE_IMPORTS if find_spec(name) is None]


def main() -> int:
    """Run setup checks and return a process status code."""

    missing_paths = verify_project_structure()
    missing_imports = verify_dependencies()

    print(f"Repository root: {PROJECT_ROOT}")
    print(f"Required paths: {len(REQUIRED_PATHS) - len(missing_paths)}/{len(REQUIRED_PATHS)}")
    print(f"Core imports: {len(CORE_IMPORTS) - len(missing_imports)}/{len(CORE_IMPORTS)}")

    if missing_paths:
        print("Missing paths:")
        for path in missing_paths:
            print(f"  - {path}")
    if missing_imports:
        print("Missing imports:")
        for name in missing_imports:
            print(f"  - {name}")

    if missing_paths or missing_imports:
        print("Setup verification failed.")
        return 1

    print("Setup verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
