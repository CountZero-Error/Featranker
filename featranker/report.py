"""
Report: the dict FeatRanker returns, plus reading, table, and plot helpers.
"""

from typing import Any, Dict, List, Tuple

from .summary import summarize


class Report(dict):
    """
    A FeatRanker report: a plain ``dict`` with three helpers.

    Indexing, ``json.dumps``, pickling and dict equality work as for any
    dict. ``summary()`` lists the findings, ``to_frame()`` returns the ranking
    table and ``plot()`` draws it. pandas and matplotlib are imported only
    when those helpers are called. The keys are described in docs/report.md.
    """

    def summary(self) -> List[Dict[str, Any]]:
        """
        Return the ``summarize()`` findings for this report.

        Returns
        -------
        list of dict
            One dict per finding, with ``code``, ``severity``, ``message``,
            ``evidence``, ``basis`` and ``scoring``. Works for single-split
            and resampled reports.
        """
        return summarize(self)

    def _rows(self) -> Tuple[str, List[Dict[str, Any]]]:
        if self.get("evaluation_mode") == "resampled":
            return "stability", list(self.get("stability", []))
        return "consensus", list(self.get("consensus", []))

    def to_frame(self) -> Any:
        """
        Return the ranking table as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            The consensus table, or for a resampled report the stability
            table, one row per feature group. The nested per-family or
            per-split rank columns are left out.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError(
                "[!] Report.to_frame() needs pandas: pip install pandas"
            ) from error
        kind, rows = self._rows()
        nested = "split_ranks" if kind == "stability" else "family_ranks"
        return pd.DataFrame([{k: v for k, v in row.items() if k != nested} for row in rows])

    def plot(self, ax: Any = None) -> Any:
        """
        Plot the ranking and return the matplotlib Axes.

        Single-split report: each voting family's rank per group, with the
        median marked. Resampled report: each group's range of places across
        splits, with the median marked. The most important group is on top.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. ``None`` creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ImportError
            If matplotlib is not installed (``pip install 'featranker[plot]'``).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "[!] Report.plot() needs matplotlib: pip install 'featranker[plot]'"
            ) from error
        kind, rows = self._rows()
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 1 + 0.4 * max(len(rows), 3)))
        if not rows:
            ax.text(0.5, 0.5, "No consensus to plot", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            return ax

        positions = list(range(len(rows)))
        medians = [row["median_rank"] for row in rows]
        if kind == "consensus":
            for position, row in zip(positions, rows):
                family_ranks = list(row.get("family_ranks", {}).values())
                ax.scatter(family_ranks, [position] * len(family_ranks), color="0.6",
                           s=20, zorder=2, label="family rank" if position == 0 else None)
            ax.scatter(medians, positions, marker="|", s=300, color="C0", zorder=3,
                       label="median rank")
        else:
            ax.hlines(positions, [row["min_rank"] for row in rows],
                      [row["max_rank"] for row in rows], color="0.6",
                      label="place range across splits")
            ax.scatter(medians, positions, color="C0", zorder=3, label="median place")
        ax.set_yticks(positions)
        ax.set_yticklabels([row["feature_group"] for row in rows])
        ax.yaxis.set_inverted(True)  # most important on top; idempotent on reused axes
        ax.set_xlabel("rank (1 = most important)")
        ax.legend(loc="best", fontsize="small")
        return ax
