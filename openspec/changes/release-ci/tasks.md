## 1. Smoke test and workflows

- [x] 1.1 Write tests/smoke_installed.py (D1); confirm it fails when run against the checkout
- [x] 1.2 Build a wheel from a copy in $TMPDIR; run the script core-only (venv with only the wheel) and with `--extras` (system-site-packages venv, wheel `--no-deps`) with `--expect-version` (D4); report output
- [x] 1.3 tests.yml `wheel` job; release.yml smoke step before upload (D2, D3); `--expect-version` from pyproject; timeouts, permissions, 4-cell matrix; YAML parses
- [x] 1.4 docs/api.md "Releasing" mentions the smoke step; CHANGELOG 1.20.0 (local); roadmap 1.20 in openspec/config.yaml; version 1.20.0
- [x] 1.5 Full suite passes; mypy clean; `openspec validate release-ci --strict` passes
