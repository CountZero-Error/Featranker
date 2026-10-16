"""
Report: the dict FeatRanker returns, plus reading, table, and plot helpers.
"""

from typing import Any, Dict, List, Tuple

from .summary import summarize


class Report(dict):
    """
    A FeatRanker report.

    It is a plain dict (indexing, json.dumps, pickling and dict equality all
    work as before) with three helpers: summary(), to_frame(), and plot().
    pandas and matplotlib are imported only when those helpers are called.
    """

    def summary(self) -> List[Dict[str, Any]]:
        """The summarize() findings (single-split reports only)."""
        return summarize(self)

    def _rows(self) -> Tuple[str, List[Dict[str, Any]]]:
        if self.get("evaluation_mode") == "resampled":
            return "stability", list(self.get("stability", []))
        return "consensus", list(self.get("consensus", []))

    def to_frame(self) -> Any:
        """The consensus (or, for resampled reports, stability) table as a DataFrame."""
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

        Single-split: each voting family's rank per group, median marked.
        Resampled: each group's place range across splits, median marked.
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
