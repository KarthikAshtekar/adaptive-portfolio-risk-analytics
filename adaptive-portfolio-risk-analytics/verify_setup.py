"""
Verification Checklist - Platform Initialization

Run this to verify the project structure was created correctly.
"""

import os
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger(__name__)


def verify_project_structure():
    """Verify all required directories and files exist."""

    project_root = Path(__file__).parent

    # Define required structure
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/interim",
        "data/external",
        "notebooks/01_data_exploration",
        "notebooks/02_covariance_analysis",
        "notebooks/03_clustering_hrp",
        "notebooks/04_regime_detection",
        "notebooks/05_nlp_sentiment",
        "notebooks/06_backtesting",
        "notebooks/07_visualizations",
        "src/data_pipeline",
        "src/covariance",
        "src/clustering",
        "src/regime_detection",
        "src/nlp",
        "src/optimization",
        "src/backtesting",
        "src/analytics",
        "src/dashboard/components",
        "outputs/weights",
        "outputs/reports",
        "outputs/metrics",
        "outputs/figures",
        "tests",
        "config",
        "docs/architecture",
        "docs/methodology",
        "docs/research_notes",
        "references/papers",
        "references/datasets",
    ]

    required_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "Makefile",
        "pytest.ini",
        "conftest.py",
        "main.py",
        "GETTING_STARTED.md",
        ".env.template",
        "PROJECT_STATUS.md",
        "src/__init__.py",
        "src/config.py",
        "src/logging_config.py",
        "src/types.py",
        "src/utils.py",
        "src/data_pipeline/__init__.py",
        "src/data_pipeline/ingest.py",
        "src/data_pipeline/preprocess.py",
        "src/data_pipeline/feature_engineering.py",
        "src/covariance/__init__.py",
        "src/clustering/__init__.py",
        "src/clustering/hrp.py",
        "src/clustering/herc.py",
        "src/regime_detection/__init__.py",
        "src/nlp/__init__.py",
        "src/optimization/__init__.py",
        "src/backtesting/__init__.py",
        "src/analytics/__init__.py",
        "src/dashboard/app.py",
        "src/dashboard/plots.py",
        "src/dashboard/components/__init__.py",
        "tests/__init__.py",
        "tests/test_hrp.py",
        "tests/test_herc.py",
        "tests/test_covariance.py",
        "tests/test_regime_detection.py",
        "tests/test_optimization.py",
        "config/portfolio_config.yaml",
        "docs/architecture/ARCHITECTURE.md",
        "docs/architecture/REFERENCES.md",
        "docs/methodology/METHODOLOGY.md",
        "docs/ROADMAP.md",
    ]

    logger.info("%s", "=" * 70)
    logger.info("%s", "PROJECT STRUCTURE VERIFICATION")
    logger.info("%s", "=" * 70)
    logger.info("")

    # Check directories
    logger.info("[*] Checking directories...")
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            logger.info("[+] %s", dir_path)
        else:
            logger.warning("[-] %s - MISSING", dir_path)
            missing_dirs.append(dir_path)

    logger.info("")

    # Check files
    logger.info("[*] Checking files...")
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            logger.info("[+] %s (%d bytes)", file_path, size)
        else:
            logger.warning("[-] %s - MISSING", file_path)
            missing_files.append(file_path)

    logger.info("")

    # Summary
    logger.info("%s", "=" * 70)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 70)
    logger.info("Total Directories: %d", len(required_dirs))
    logger.info("Directories Created: %d", len(required_dirs) - len(missing_dirs))
    logger.info("Directories Missing: %d", len(missing_dirs))
    logger.info("")
    logger.info("Total Files: %d", len(required_files))
    logger.info("Files Created: %d", len(required_files) - len(missing_files))
    logger.info("Files Missing: %d", len(missing_files))
    logger.info("")

    if not missing_dirs and not missing_files:
        logger.info("PROJECT STRUCTURE COMPLETE!")
        return True
    else:
        logger.warning("SOME FILES/DIRECTORIES ARE MISSING")
        if missing_dirs:
            logger.warning("Missing directories:")
            for d in missing_dirs:
                logger.warning("  - %s", d)
        if missing_files:
            logger.warning("Missing files:")
            for f in missing_files:
                logger.warning("  - %s", f)
        return False


def verify_dependencies():
    """Check if key dependencies are listed in requirements.txt."""
    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("DEPENDENCIES VERIFICATION")
    logger.info("%s", "=" * 70)

    requirements_file = Path(__file__).parent / "requirements.txt"

    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-portfolio",
        "riskfolio-lib",
        "statsmodels",
        "transformers",
        "torch",
        "yfinance",
        "plotly",
        "streamlit",
        "pytest",
        "black",
        "flake8",
        "mypy",
    ]

    if requirements_file.exists():
        with open(requirements_file, "r") as f:
            requirements = f.read().lower()

        logger.info("[*] Checking required packages...")
        found = 0
        for pkg in required_packages:
            if pkg.lower() in requirements:
                logger.info("[+] %s", pkg)
                found += 1
            else:
                logger.warning("[-] %s - NOT FOUND", pkg)

        logger.info("Found: %d/%d packages", found, len(required_packages))

        # Count total packages
        total_packages = len(
            [
                line
                for line in requirements.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
        )
        logger.info("Total packages in requirements.txt: %d", total_packages)
    else:
        logger.warning("[-] requirements.txt not found")


def print_quick_start():
    """Print quick start guide."""
    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("QUICK START")
    logger.info("%s", "=" * 70)
    logger.info("")
    logger.info("1. INSTALL DEPENDENCIES:")
    logger.info("   make install")
    logger.info("")
    logger.info("2. CONFIGURE ENVIRONMENT:")
    logger.info("   cp .env.template .env")
    logger.info("   # Edit .env with your API keys")
    logger.info("")
    logger.info("3. RUN TESTS:")
    logger.info("   make test")
    logger.info("")
    logger.info("4. START DASHBOARD:")
    logger.info("   make run-dashboard")
    logger.info("")
    logger.info("5. BUILD DOCUMENTATION:")
    logger.info("   make docs")
    logger.info("")
    logger.info("For more information, see:")
    logger.info("  - README.md")
    logger.info("  - GETTING_STARTED.md")
    logger.info("  - PROJECT_STATUS.md")
    logger.info("  - docs/architecture/ARCHITECTURE.md")
    logger.info("  - docs/methodology/METHODOLOGY.md")


def main():
    """Run all verifications."""
    success = verify_project_structure()
    verify_dependencies()
    print_quick_start()

    logger.info("")
    logger.info("%s", "=" * 70)
    if success:
        logger.info("PROJECT INITIALIZATION SUCCESSFUL!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Run: make install")
        logger.info("2. Configure: cp .env.template .env")
        logger.info("3. Test: make test")
    else:
        logger.warning("PROJECT INITIALIZATION HAS ISSUES")
        logger.warning("Please check the files listed above")
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()
