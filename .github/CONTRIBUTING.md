# Contributing

## Development workflow
1. Create a branch from `main`:
   - `git checkout -b blackboxai/<short-description>`
2. Make changes.
3. Ensure formatting/linting/tests pass:
   - `pre-commit run --all-files` (recommended)
   - `make lint` and `make test` (inside `adaptive-portfolio-risk-analytics/`)
4. Open a PR and fill the pull request template.

## Code style / guardrails
- Python style: Black-compatible formatting (100 char line length).
- Import ordering: isort (Black profile).
- Linting: flake8.
- Mypy/type checks are allowed to run in CI as configured.

## Tests
- Add unit tests for new logic.
- Keep tests deterministic (no network calls, no randomness without seeding).

## Security
- Never commit API keys, tokens, or private datasets.
- Use `.env.template` and environment variables.

