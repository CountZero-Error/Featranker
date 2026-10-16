"""FeatRanker package."""

import importlib.metadata

from .importance import FeatureRanker
from .interpret import explain, to_prompt
from .report import Report
from .selection import ConsensusSelector
from .summary import summarize

__all__ = [
    "ConsensusSelector", "FeatureRanker", "Report", "explain", "summarize", "to_prompt",
]

try:
    __version__: "str | None" = importlib.metadata.version("featranker")
except importlib.metadata.PackageNotFoundError:  # running from a source tree
    __version__ = None
