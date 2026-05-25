"""
Verification Checklist - Platform Initialization

Run this to verify the project structure was created correctly.
"""

import os
from pathlib import Path


def verify_project_structure():
    """Verify all required directories and files exist."""
    
    project_root = Path(__file__).parent
    
    # Define required structure
    required_dirs = [
        "data/raw", "data/processed", "data/interim", "data/external",
        "notebooks/01_data_exploration", "notebooks/02_covariance_analysis",
        "notebooks/03_clustering_hrp", "notebooks/04_regime_detection",
        "notebooks/05_nlp_sentiment", "notebooks/06_backtesting",
        "notebooks/07_visualizations",
        "src/data_pipeline", "src/covariance", "src/clustering",
        "src/regime_detection", "src/nlp", "src/optimization",
        "src/backtesting", "src/analytics", "src/dashboard/components",
        "outputs/weights", "outputs/reports", "outputs/metrics", "outputs/figures",
        "tests", "config", "docs/architecture", "docs/methodology",
        "docs/research_notes", "references/papers", "references/datasets",
    ]
    
    required_files = [
        "README.md", "LICENSE", "requirements.txt", "setup.py", ".gitignore",
        "Makefile", "pytest.ini", "conftest.py", "main.py", 
        "GETTING_STARTED.md", ".env.template", "PROJECT_STATUS.md",
        "src/__init__.py", "src/config.py", "src/logging_config.py",
        "src/types.py", "src/utils.py",
        "src/data_pipeline/__init__.py", "src/data_pipeline/ingest.py",
        "src/data_pipeline/preprocess.py", "src/data_pipeline/feature_engineering.py",
        "src/covariance/__init__.py",
        "src/clustering/__init__.py", "src/clustering/hrp.py", "src/clustering/herc.py",
        "src/regime_detection/__init__.py",
        "src/nlp/__init__.py",
        "src/optimization/__init__.py",
        "src/backtesting/__init__.py",
        "src/analytics/__init__.py",
        "src/dashboard/app.py", "src/dashboard/plots.py",
        "src/dashboard/components/__init__.py",
        "tests/__init__.py", "tests/test_hrp.py", "tests/test_herc.py",
        "tests/test_covariance.py", "tests/test_regime_detection.py",
        "tests/test_optimization.py",
        "config/portfolio_config.yaml",
        "docs/architecture/ARCHITECTURE.md", "docs/architecture/REFERENCES.md",
        "docs/methodology/METHODOLOGY.md", "docs/ROADMAP.md",
    ]
    
    print("=" * 70)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 70)
    print()
    
    # Check directories
    print("📂 Checking directories...")
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    print()
    
    # Check files
    print("📄 Checking files...")
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path:50} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Directories: {len(required_dirs)}")
    print(f"Directories Created: {len(required_dirs) - len(missing_dirs)}")
    print(f"Directories Missing: {len(missing_dirs)}")
    print()
    print(f"Total Files: {len(required_files)}")
    print(f"Files Created: {len(required_files) - len(missing_files)}")
    print(f"Files Missing: {len(missing_files)}")
    print()
    
    if not missing_dirs and not missing_files:
        print("✅ PROJECT STRUCTURE COMPLETE!")
        return True
    else:
        print("⚠️  SOME FILES/DIRECTORIES ARE MISSING")
        if missing_dirs:
            print("\nMissing directories:")
            for d in missing_dirs:
                print(f"  - {d}")
        if missing_files:
            print("\nMissing files:")
            for f in missing_files:
                print(f"  - {f}")
        return False


def verify_dependencies():
    """Check if key dependencies are listed in requirements.txt."""
    print()
    print("=" * 70)
    print("DEPENDENCIES VERIFICATION")
    print("=" * 70)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    required_packages = [
        "numpy", "pandas", "scipy", "scikit-learn",
        "scikit-portfolio", "riskfolio-lib",
        "statsmodels", "transformers", "torch",
        "yfinance", "plotly", "streamlit",
        "pytest", "black", "flake8", "mypy",
    ]
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            requirements = f.read().lower()
        
        print("\n📦 Checking required packages...")
        found = 0
        for pkg in required_packages:
            if pkg.lower() in requirements:
                print(f"  ✓ {pkg}")
                found += 1
            else:
                print(f"  ✗ {pkg} - NOT FOUND")
        
        print(f"\nFound: {found}/{len(required_packages)} packages")
        
        # Count total packages
        total_packages = len([line for line in requirements.split('\n') 
                            if line.strip() and not line.strip().startswith('#')])
        print(f"Total packages in requirements.txt: {total_packages}")
    else:
        print("✗ requirements.txt not found")


def print_quick_start():
    """Print quick start guide."""
    print()
    print("=" * 70)
    print("QUICK START")
    print("=" * 70)
    print("""
1. INSTALL DEPENDENCIES:
   make install
   
2. CONFIGURE ENVIRONMENT:
   cp .env.template .env
   # Edit .env with your API keys
   
3. RUN TESTS:
   make test
   
4. START DASHBOARD:
   make run-dashboard
   
5. BUILD DOCUMENTATION:
   make docs

For more information, see:
  - README.md
  - GETTING_STARTED.md
  - PROJECT_STATUS.md
  - docs/architecture/ARCHITECTURE.md
  - docs/methodology/METHODOLOGY.md
""")


def main():
    """Run all verifications."""
    success = verify_project_structure()
    verify_dependencies()
    print_quick_start()
    
    print("=" * 70)
    if success:
        print("✅ PROJECT INITIALIZATION SUCCESSFUL!")
        print()
        print("Next steps:")
        print("1. Run: make install")
        print("2. Configure: cp .env.template .env")
        print("3. Test: make test")
    else:
        print("⚠️  PROJECT INITIALIZATION HAS ISSUES")
        print("Please check the files listed above")
    print("=" * 70)


if __name__ == "__main__":
    main()
