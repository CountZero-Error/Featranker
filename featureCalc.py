from typing import Dict, Sequence

from sklearn.datasets import load_diabetes, load_iris


class prepFeature:
    def __init__(self) -> None:
        """
        Preprocess your features here and return it.
        Return features format:
            {
                'feature1': [value1, value2, ...],
                'feature2': [value1, value2, ...],
                ...
                'label': [label1, label2, ...],
            }
        """

        # Users should override _calc_features to return the expected dict.
        # You may define multiple prepFeature-style classes in this file
        # and select one via the CLI flag --prep-class.
        self.features: Dict[str, Sequence[float]] = self._calc_features()

    def _calc_features(self):
        """
        Write your custom function here
        """
        # Placeholder implementation; override in a subclass or edit this file
        # to preprocess and return your training data.

        return features

    """
    Example implements
    """
    # === Example for classifiaction using sklearn iris dataset ===
    # def _calc_features(self):
    #     data = load_iris()
    #     X = data.data
    #     y = data.target

    #     features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
    #     features["label"] = y.tolist()
    #     return features

    # === Example for regression using sklearn diabetes dataset ===
    # def _calc_features(self):
    #     data = load_diabetes()
    #     X = data.data
    #     y = data.target

    #     features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
    #     features["label"] = y.tolist()
    #     return features
