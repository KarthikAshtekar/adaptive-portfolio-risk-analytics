# Contributing to Adaptive Portfolio Allocation and Risk Analytics

Welcome to the team! This guide will help you contribute effectively to our quantitative finance platform.

## 🚀 Quick Start

### 1. Fork and Clone
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/adaptive-portfolio-risk-analytics.git
cd adaptive-portfolio-risk-analytics

# Add upstream remote for syncing
git remote add upstream https://github.com/KarthikAshtekar/adaptive-portfolio-risk-analytics.git
```

### 2. Setup Development Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev tools: pytest, black, flake8, mypy

# Verify setup
python verify_setup.py
```

### 3. Create Feature Branch
```bash
# Update develop branch from upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Create feature branch from develop
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b bugfix/your-bug-fix
```

---

## 📝 Branching Strategy (Git Flow)

```
main (production)
├── develop (integration)
│   ├── feature/optimization-enhancements
│   ├── feature/nlp-sentiment-model
│   ├── bugfix/covariance-estimation
│   └── ...
└── release/v0.2.0
```

### Branch Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/short-description` | `feature/ledoit-wolf-implementation` |
| Bug Fix | `bugfix/short-description` | `bugfix/hrp-weight-calculation` |
| Hotfix | `hotfix/short-description` | `hotfix/critical-data-ingestion` |
| Release | `release/v0.X.X` | `release/v0.2.0` |
| Documentation | `docs/topic` | `docs/api-reference` |

---

## 💻 Development Workflow

### 1. Pick a Task
```bash
# Review TODO items in:
# - docs/ROADMAP.md (project-level planning)
# - docs/methodology/METHODOLOGY.md (technical details)
# - SOURCE CODE (look for # TODO: comments)
# - GitHub Issues (if using issue tracking)

# Example: Implementing LedoitWolfEstimator in src/covariance/__init__.py
```

### 2. Implement Your Changes

#### Code Style Requirements

**Type Hints** (mandatory)
```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def calculate_weights(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray,
    method: str = "equal"
) -> np.ndarray:
    """Calculate portfolio weights.
    
    Args:
        returns: Historical returns data (T x N)
        cov_matrix: Covariance matrix (N x N)
        method: Allocation method
    
    Returns:
        Portfolio weights (N,)
    
    Raises:
        ValueError: If inputs invalid
    """
    # Implementation...
    return weights
```

**Docstring Format** (Google-style)
```python
class MyOptimizer:
    """Optimize portfolio weights using custom method.
    
    This class implements a novel optimization approach that combines
    multiple techniques for robust portfolio construction.
    
    Attributes:
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance level
    
    Example:
        >>> optimizer = MyOptimizer(max_iterations=100)
        >>> weights = optimizer.optimize(returns, cov_matrix)
        >>> print(weights.shape)
        (50,)
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """Initialize optimizer.
        
        Args:
            max_iterations: Max iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
```

**Code Formatting** (automatic)
```bash
# Format code with Black (line length: 100)
make format
# or
black --line-length=100 src/

# Check linting with flake8
make lint
# or
flake8 src/ --max-line-length=100

# Type check with mypy
mypy src/

# Apply import sorting with isort
isort src/
```

### 3. Write Tests

#### Test File Location
```
tests/
├── test_optimization.py       # Test optimization module
├── test_covariance.py         # Test covariance methods
├── test_hrp.py                # Test HRP algorithm
└── conftest.py                # Shared fixtures
```

#### Test Example
```python
import pytest
import numpy as np
import pandas as pd
from src.optimization import EqualWeightOptimizer

@pytest.fixture
def sample_returns(sample_returns):
    """Use fixture from conftest.py"""
    return sample_returns

def test_equal_weight_optimizer_basic(sample_returns):
    """Test basic functionality of EqualWeightOptimizer."""
    optimizer = EqualWeightOptimizer()
    weights = optimizer.optimize(sample_returns)
    
    # Assertions
    assert weights.shape == (sample_returns.shape[1],)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0)

def test_equal_weight_optimizer_n_assets():
    """Test equal weight distribution for N assets."""
    n = 10
    returns = pd.DataFrame(
        np.random.randn(100, n),
        columns=[f"asset_{i}" for i in range(n)]
    )
    
    optimizer = EqualWeightOptimizer()
    weights = optimizer.optimize(returns)
    
    expected = np.ones(n) / n
    assert np.allclose(weights, expected)

def test_equal_weight_optimizer_invalid_input():
    """Test error handling for invalid inputs."""
    optimizer = EqualWeightOptimizer()
    
    with pytest.raises(ValueError):
        optimizer.optimize(pd.DataFrame())  # Empty dataframe
```

#### Running Tests
```bash
# Run all tests
make test
# or
pytest

# Run specific test file
pytest tests/test_optimization.py

# Run specific test function
pytest tests/test_optimization.py::test_equal_weight_optimizer_basic

# Run with coverage report
pytest --cov=src --cov-report=html

# View HTML coverage report
# (Check htmlcov/index.html)
```

### 4. Commit Changes

#### Commit Message Format
```
[module] Brief description (50 chars max)

Detailed explanation if needed (72 chars per line).
Can include:
- What was changed
- Why it was changed
- Related issue references

Example:
[covariance] Implement Ledoit-Wolf shrinkage estimator

- Added LedoitWolfEstimator class with dual optimized shrinkage
- Integrated with CovarianceEstimator factory pattern
- Includes unit tests for edge cases (rank deficiency, small N)
- Fixes issue #45: Ledoit-Wolf implementation gap

Fixes #45
```

```bash
# Commit with message
git add src/covariance/__init__.py tests/test_covariance.py
git commit -m "[covariance] Implement Ledoit-Wolf shrinkage estimator"

# If using full commit template:
git commit  # Opens editor for multi-line message
```

### 5. Keep Up-to-Date
```bash
# Fetch latest from upstream
git fetch upstream

# Rebase on latest develop (prefer rebase over merge)
git rebase upstream/develop

# If conflicts occur:
# 1. Resolve conflicts in files
# 2. Stage resolved files: git add <file>
# 3. Continue rebase: git rebase --continue
# 4. Push to your fork: git push -f origin feature/your-branch
```

### 6. Create Pull Request

#### Before Submitting
```bash
# Final checks
make format      # Auto-format code
make lint        # Check style
make test        # Run tests
python verify_setup.py  # Verify setup
```

#### PR Description Template
```markdown
## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] New feature (non-breaking)
- [ ] Bug fix (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Fixes #(issue number) or relates to #(issue number)

## Changes Made
- Implemented XYZ algorithm
- Added comprehensive test coverage (95%+ coverage)
- Updated documentation in docs/

## Testing
- [ ] Unit tests added/updated
- [ ] All tests passing (`pytest --cov`)
- [ ] Manual testing completed

## Documentation
- [ ] Updated relevant docstrings
- [ ] Updated README if needed
- [ ] Updated methodology docs if algorithm-related

## Checklist
- [ ] Code follows style guidelines (black, flake8, mypy)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No new warnings generated
- [ ] Tests added (pytest coverage > 80%)
```

#### Submitting PR
```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub, create Pull Request
# - Set base: develop (or main for hotfixes)
# - Set compare: your-feature-branch
# - Fill PR description from template
# - Request reviewers
```

---

## 🧪 Testing Requirements

### Minimum Coverage
- **Overall**: ≥ 80% code coverage
- **Critical Modules** (optimization, analytics, backtesting): ≥ 90%
- **New Features**: Must include tests, ≥ 85% coverage for that module

### Test Categories
```
✓ Unit Tests        - Test individual functions/classes
✓ Integration Tests - Test module interactions
✓ Edge Cases        - Boundary conditions, invalid inputs
✓ Performance Tests - Large datasets, time benchmarks (optional)
```

### Example: Testing an Optimizer
```python
# Unit test
def test_mean_variance_optimizer_solves():
    """Test solver works correctly."""
    # Implementation...

# Edge case test
def test_mean_variance_optimizer_singular_matrix():
    """Test handling of singular covariance matrix."""
    # Implementation...

# Integration test
def test_optimizer_with_regime_detection():
    """Test optimizer works with regime-detected data."""
    # Implementation...
```

---

## 📚 Code Organization Patterns

### Factory Pattern (for Optimizers)
```python
# In src/optimization/__init__.py
class OptimizerFactory:
    @staticmethod
    def create(method: str) -> PortfolioOptimizer:
        if method == "equal":
            return EqualWeightOptimizer()
        elif method == "mean_variance":
            return MeanVarianceOptimizer()
        else:
            raise ValueError(f"Unknown method: {method}")

# Usage:
optimizer = OptimizerFactory.create("mean_variance")
```

### Strategy Pattern (for Covariance Methods)
```python
# Each covariance method implements CovarianceEstimator
class CovarianceEstimator(ABC):
    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix."""
        pass

class LedoitWolfEstimator(CovarianceEstimator):
    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        # Implementation...
        pass
```

### Configuration Pattern
```python
# From src/config.py
from src.config import ConfigManager

config = ConfigManager()
config.set("optimization.method", "mean_variance")
config.set("optimization.max_iterations", 100)

method = config.get("optimization.method")  # Dot notation access
```

---

## 📖 Documentation Standards

### README Updates
When adding features, update:
- `README.md` - Feature overview section
- `docs/ROADMAP.md` - Update phase/status
- `docs/methodology/METHODOLOGY.md` - Technical details
- `FILE_TREE.md` - New files/directories

### Module Documentation
Every module should have:
1. **Module docstring** at top of file
2. **Class docstrings** with Attributes/Methods sections
3. **Method docstrings** with Args/Returns/Raises sections
4. **Inline comments** for complex logic
5. **TODO markers** for future work

Example:
```python
"""Portfolio optimization module.

This module provides multiple portfolio allocation methods:
- Equal weight allocation
- Mean-variance optimization
- Hierarchical risk parity
- Custom optimization via factory pattern

Usage:
    >>> from src.optimization import OptimizerFactory
    >>> optimizer = OptimizerFactory.create("mean_variance")
    >>> weights = optimizer.optimize(returns, cov_matrix)
"""
```

---

## 🔍 Code Review Checklist

Before submitting PR, verify:

- [ ] **Functionality**
  - [ ] Code solves the problem
  - [ ] Handles edge cases
  - [ ] Proper error handling

- [ ] **Code Quality**
  - [ ] Black formatted (100 char limit)
  - [ ] Flake8 passed (no style violations)
  - [ ] mypy passed (type hints correct)
  - [ ] isort applied (imports sorted)

- [ ] **Testing**
  - [ ] ≥80% coverage for module
  - [ ] All tests pass
  - [ ] Edge cases tested
  - [ ] No brittle/flaky tests

- [ ] **Documentation**
  - [ ] Docstrings complete (Google-style)
  - [ ] Type hints present
  - [ ] Comments for complex logic
  - [ ] README/docs updated if needed

- [ ] **Best Practices**
  - [ ] No hardcoded values (use config/constants)
  - [ ] Logging implemented for key operations
  - [ ] No dead code or commented sections
  - [ ] Follows SOLID principles

---

## 🚨 Common Issues & Solutions

### Issue: Merge Conflicts
```bash
# During rebase, conflicts occur:
# 1. Edit conflicted files, remove conflict markers
# 2. Stage resolved files
git add src/file.py

# 3. Continue rebase
git rebase --continue

# 4. Force push to your fork
git push -f origin feature/branch
```

### Issue: Tests Failing Locally But Passing in CI
```bash
# Ensure dependencies match:
pip install -r requirements.txt

# Clear cache:
rm -rf .pytest_cache
rm -rf __pycache__

# Run with verbose output:
pytest -v tests/
```

### Issue: Code Not Formatted After `black`
```bash
# Check for long lines:
black --line-length=100 --check src/

# Fix imports:
isort src/

# Verify no flake8 issues:
flake8 src/ --max-line-length=100
```

---

## 📞 Getting Help

1. **Setup Issues**: See `README.md`
2. **Architecture Questions**: See `docs/architecture/ARCHITECTURE.md`
3. **Algorithm Details**: See `docs/methodology/METHODOLOGY.md`
4. **Code Examples**: See `notebooks/` and existing test files
5. **GitHub Issues**: Create issue with detailed description

---

## 🎉 Thank You!

Your contributions are valuable! By following these guidelines, you help maintain code quality and make the platform better for the entire team.

**Happy coding!** 🚀
