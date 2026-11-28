"""Acceptance check for review-17 round 2 (lead-written; builders must not edit)."""
import numpy as np
import pandas as pd
import pytest

from featranker.importance import _check_index


def test_a_real_label_cannot_collide_with_the_missing_key():
    labels = [np.nan, ("<missing>", "float"), 3]
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.Index(labels, dtype=object, tupleize_cols=False))
    swapped = pd.Index([("<missing>", "float"), np.nan, 3], dtype=object, tupleize_cols=False)
    with pytest.raises(ValueError, match="index"):
        _check_index(X, pd.Series([2.0, 1.0, 3.0], index=swapped), "y")
