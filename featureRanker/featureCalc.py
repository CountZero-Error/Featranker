from typing import Dict, Sequence


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
        return self.features
