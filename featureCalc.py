from typing import Dict, Sequence

from sklearn.datasets import load_diabetes, load_iris


class prepFeature:
    """
    Feature-preparation class for FeatureRanker.

    Override ``_calc_features`` to supply your own data.  The method must
    return a dict whose keys are feature names (strings) mapped to
    equal-length lists of numeric values, plus a special ``"label"`` key:

        {
            "feature1": [value1, value2, ...],
            "feature2": [value1, value2, ...],
            ...
            "label":    [label1, label2, ...],
        }

    You may define multiple prep classes in this file and select one at
    runtime with the ``--prep-class`` CLI flag.
    """

    def __init__(self) -> None:
        pass

    def _calc_features(self) -> Dict[str, list]:
        """
        Compute and return the feature dict.

        Replace this placeholder with your own preprocessing logic.
        """
        # TODO: implement your feature pipeline and return the result.
        return features

    # -----------------------------------------------------------------
    # Example implementations (uncomment ONE to try it out)
    # -----------------------------------------------------------------

    # --- Classification example using the sklearn Iris dataset ---
    # def _calc_features(self) -> Dict[str, list]:
    #     data = load_iris()
    #     X = data.data
    #     y = data.target
    #
    #     features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
    #     features["label"] = y.tolist()
    #     return features

    # --- Regression example using the sklearn Diabetes dataset ---
    # def _calc_features(self) -> Dict[str, list]:
    #     data = load_diabetes()
    #     X = data.data
    #     y = data.target
    #
    #     features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
    #     features["label"] = y.tolist()
    #     return features
