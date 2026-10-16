# report-object Specification

## Purpose
Give users a result object with convenient reading, tabulating, and plotting, without breaking any code that treats reports as plain dicts.
## Requirements
### Requirement: Reports are dict-compatible Report objects
`rank_features()` and `rank_resampled()` SHALL return a `featranker.Report`, which is a subclass of `dict` with exactly the keys and values those methods returned in 0.9. A `Report` SHALL:
- compare equal to a dict with the same contents;
- serialize with `json.dumps(report, allow_nan=False)`;
- pickle and unpickle to an equal `Report`;
- pass `isinstance(report, dict)`.

Each split's report inside a resampled report SHALL also be a `Report`.

#### Scenario: Drop-in compatible
- **WHEN** a report from `rank_features()` is compared with `dict(report)`, serialized, and pickled
- **THEN** it equals the dict, serializes to the same JSON as the dict, and round-trips through pickle as an equal `Report`

### Requirement: summary
`Report.summary()` SHALL return `summarize(self)`. For a resampled report it SHALL raise the same `ValueError` as `summarize`.

#### Scenario: Same findings
- **WHEN** `report.summary()` is called on a single-split report
- **THEN** it equals `summarize(report)`

### Requirement: to_frame
`Report.to_frame()` SHALL return a pandas DataFrame:
- for a single-split report, one row per consensus row, in consensus order, with the consensus fields as columns and `family_ranks` omitted;
- for a resampled report, one row per stability row, with `split_ranks` omitted.

If pandas is not installed, it SHALL raise `ImportError` naming `pandas`.

#### Scenario: Consensus frame
- **WHEN** `to_frame()` is called on a single-split report
- **THEN** its `feature_group` column equals the consensus `feature_group` order, and it has a `median_rank` column

### Requirement: plot
`Report.plot(ax=None)` SHALL draw on `ax`, or on a new figure's axes, and return the `Axes`.
- For a single-split report: one row per consensus group, top group first, with one point per voting family rank and a marker at `median_rank`.
- For a resampled report: one row per stability group, with a line from `min_rank` to `max_rank` and a marker at `median_rank`.
- An empty consensus or stability table SHALL produce an `Axes` with a "no consensus" message and no error.

If matplotlib is not installed, it SHALL raise `ImportError` naming `featranker[plot]`.

#### Scenario: Plot returns axes
- **WHEN** `report.plot()` is called with the Agg backend
- **THEN** a matplotlib `Axes` is returned, whose y tick labels are the groups in consensus order

