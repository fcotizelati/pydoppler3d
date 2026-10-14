# Contributing

Install the development extras with `python -m pip install -e ".[dev]"`, then
run `pre-commit install`. Use the pinned Ruff version shared by precommit and CI.

Before proposing a change:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=pydoppler3d --cov-report=term-missing
python -m build
```

Scientific changes need a regression test of the relevant equation or observable,
preferably against an independently generated example. For operator changes,
check the adjoint identity, boundaries, exposure integration, and instrumental
response. Preserve signed observations and units. Report changes to conventions,
likelihoods, parameter meanings, or saved formats in CHANGELOG.md and the
scientific guide. Do not identify optimizer convergence with unique physical
recovery or calibrated significance.

Use a focused branch and pull request. Explain the problem, resulting behavior,
validation, and limitations. Keep native research inputs unchanged; write generated
products to a separate output directory. Include a small reproducer in bug reports
and redact private paths or data before publishing them.

The maintainer should enable required CI checks and review protection in GitHub
repository settings; those settings are not enforced by files in this checkout.
