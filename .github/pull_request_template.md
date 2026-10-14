## Summary

- 

## Validation

- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m pytest -q`
- [ ] `python -m build --sdist --wheel`

## Scientific Checks

- [ ] Physical sign/phase convention is unchanged or explicitly documented.
- [ ] Reconstruction changes include an independent synthetic or regression test.
- [ ] Signed observations, flux units, masks/covariance, and response are consistent.
- [ ] Convergence/fit acceptance and output provenance are explicit.
- [ ] Documentation caveats are updated for any new interpretation limits.
