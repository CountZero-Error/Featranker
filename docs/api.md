# FEATRANKER ▍ API

<sub><a href="../README.md">◂ README</a> &nbsp;│&nbsp; <a href="README.md">Docs</a> &nbsp;│&nbsp; <a href="usage.md">Usage</a> &nbsp;│&nbsp; <a href="report.md">Report</a> &nbsp;│&nbsp; <a href="models.md">Models</a> &nbsp;│&nbsp; <b>API</b></sub>

## ▍ Public API

From 1.0, these are FeatRanker's public API:

| Kind | Public |
|---|---|
| Python names | Everything in `featranker.__all__`: `FeatureRanker`, `ConsensusSelector`, `Report`, `auto_groups`, `summarize`, and the deprecated `to_prompt` and `explain`. Also `featranker.__version__`. |
| Parameters and methods | The documented parameters, methods, and fitted attributes of those names (see [Usage](usage.md)). |
| Reports | The report keys and their meaning documented in [Report](report.md). |
| Command line | The `featranker` command and its flags (`featranker --help`). |

Everything else is internal and can change in any release: modules other than `featranker` itself, any name starting with `_`, the built-in model registry file's layout, progress-bar text, and the exact wording of warnings, errors, and `summarize()` messages (finding codes are public; messages are not).

## ▍ Stability policy

FeatRanker follows [semantic versioning](https://semver.org) from 1.0.0:

- **Major** (2.0): may remove or change public behavior.
- **Minor** (1.x): adds features. Reports may gain new keys; code that reads reports should ignore keys it does not know. Built-in models and their families may be added, which can change rankings; the change is listed in the release notes.
- **Patch** (1.x.y): fixes only.

Before something public is removed, it is deprecated with a `DeprecationWarning` for at least one minor release; a deprecation end users of a script or CLI should see by default uses `FutureWarning` instead. Results are reproducible within a release given the seeds described in [Usage](usage.md#parallel-models); they can differ between releases when estimators or the registry change.

## ▍ Deprecations

| Since | Deprecated | Removal | Note |
|---|---|---|---|
| 1.1 | `to_prompt()` and `explain()` | 2.0 | Each call emits a `FutureWarning`; behavior is otherwise unchanged through 1.x. The report [evidence](usage.md#evidence) added in 1.1 replaces the need for an LLM reading. |

## ▍ Typing

The package ships a `py.typed` marker, so type checkers such as mypy and Pyright read FeatRanker's annotations. `rank_features()`, `rank_resampled()`, and `ablate()` are annotated to return `Report`, so its helpers type-check. CI runs mypy on the package and on a small consumer file that uses the public API, on every push.

## ▍ Releasing

Releases are published by `.github/workflows/release.yml` when a version tag is pushed. Its actions are pinned to full commit SHAs (with the version in a comment); update a pin only to a reviewed upstream release. It runs the tests and mypy, checks that the tag equals `v` plus the version in `pyproject.toml`, builds and checks the package, and uploads it with PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token is stored anywhere.

One-time setup:

1. On pypi.org, open the `featranker` project → **Publishing** → add a GitHub publisher with owner `CountZero-Error`, repository `Featranker`, workflow `release.yml`, and environment `pypi`.
2. On GitHub, create the environment `pypi` under **Settings → Environments** (optionally require a reviewer, so every upload waits for approval).

Each release: bump `version` in `pyproject.toml`, update the changelog, commit, then `git tag vX.Y.Z && git push origin vX.Y.Z`.
