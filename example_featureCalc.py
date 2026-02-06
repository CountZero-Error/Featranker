"""
Example data-prep file.

Copy the class you want into `featureRanker/featureCalc.py` and keep the
same class name. Then run the CLI with `--prep-class <ClassName>`.
"""

from sklearn.datasets import load_diabetes, load_iris

from featureRanker.featureCalc import prepFeature


class IrisPrepFeature(prepFeature):
    def _calc_features(self):
        data = load_iris()
        X = data.data
        y = data.target

        features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
        features["label"] = y.tolist()
        return features


class DiabetesPrepFeature(prepFeature):
    def _calc_features(self):
        data = load_diabetes()
        X = data.data
        y = data.target

        features = {data.feature_names[i]: X[:, i].tolist() for i in range(X.shape[1])}
        features["label"] = y.tolist()
        return features


if __name__ == "__main__":
    iris = IrisPrepFeature().features
    diabetes = DiabetesPrepFeature().features

    print("Iris features:", len(iris) - 1, "columns", "| samples:", len(iris["label"]))
    print(
        "Diabetes features:",
        len(diabetes) - 1,
        "columns",
        "| samples:",
        len(diabetes["label"]),
    )
