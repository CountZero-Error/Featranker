## 1. Planning baseline

- [x] 1.1 Archive `trust-the-consensus`, `rank-stability`, `shadow-significance`, `consensus-selector`, and `llm-interpretation`, in that order. Confirm that `openspec/specs/` holds their capabilities and that `openspec validate --all` passes.

## 2. Report object

- [x] 2.1 Write tests in `tests/test_report.py`:
  - dict compatibility (equality, json, pickle, isinstance) for single-split and resampled reports, with nested split reports as `Report`;
  - `summary()` equals `summarize`, and raises for resampled reports;
  - `to_frame()` columns and order for both kinds;
  - `plot()` returns an `Axes` with the groups as y tick labels, for both kinds and for an empty consensus (Agg backend);
  - an `ImportError` naming the extra when pandas or matplotlib cannot be imported (monkeypatched).

  Confirm they fail.
- [x] 2.2 Implement `featranker/report.py` and return `Report` from `rank_features`/`rank_resampled`. Export `Report`.

## 3. Public API and typing

- [x] 3.1 Write tests:
  - `__all__` is exactly the six names;
  - `__version__` matches the metadata;
  - the built wheel contains `py.typed` (build into tmp);
  - the extras match the spec.

  Confirm they fail.
- [x] 3.2 Add `__version__`, `py.typed` with package data, and the extras changes (remove `lightgbm`, `all` = xgboost and catboost, `plot` = matplotlib, test += types-PyYAML and matplotlib). Update the existing extras test.
- [x] 3.3 Fix the mypy errors, add the `[tool.mypy]` config, and confirm `mypy featranker` is clean.
- [x] 3.4 Add classifiers, project URLs, and keywords to `pyproject.toml`.

## 4. CI and release automation

- [x] 4.1 `tests.yml`: add a mypy step.
- [x] 4.2 Add `.github/workflows/release.yml`: build, twine check, a tag/version match check, and trusted publishing under the `pypi` environment. Validate the YAML syntax locally.

## 5. Docs

- [x] 5.1 Add `docs/api.md`: the public API list, the stability and deprecation policy, typing, and the release process (trusted publisher setup fields).
- [x] 5.2 `docs/usage.md`: a "Working with reports" section (`summary`, `to_frame`, `plot`, with a real example and a saved figure description), and the `plot` extra in the install notes.
- [x] 5.3 `docs/report.md`: the report is a `Report` (dict subclass). `docs/models.md`: remove the LightGBM row. `README.md`: install extras line, docs table entry for API, and the Python badge (now from classifiers). `docs/README.md`: index.
- [x] 5.4 Verify the doc claims and example output byte-for-byte, then render through the GitHub API and check anchors and escapes.

## 6. Release prep

- [x] 6.1 Bump to 1.0.0. Add a CHANGELOG entry that notes `Report`, the removed `lightgbm` extra, and that 0.5–0.9 ship together.
- [x] 6.2 Run the full suite on sklearn 1.5.1 and the latest, plus `mypy featranker`, plus `python -m build` and `twine check`.
