"""FeatRanker package."""

import importlib.metadata

from .evidence import auto_groups
from .importance import FeatureRanker
from .report import Report
from .selection import ConsensusSelector
from .summary import summarize

__all__ = ["ConsensusSelector", "FeatureRanker", "Report", "auto_groups", "summarize"]

try:
    __version__: "str | None" = importlib.metadata.version("featranker")
except importlib.metadata.PackageNotFoundError:  # running from a source tree
    __version__ = None
