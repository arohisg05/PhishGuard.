"""
PhishGuard Pure-Python Inference Engine
---------------------------------------
An offline, pure-Python Random Forest evaluator that executes predictions
directly from exported decision tree topologies (models/phishguard_trees.json).

Runtime requirements:
- Python 3.10+ standard library only.
- ZERO dependencies on NumPy, pandas, or scikit-learn.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Sequence, Tuple, Union


class PhishGuardEngine:
    """Pure-Python Random Forest classification engine for PhishGuard."""

    EXPECTED_FEATURE_COUNT = 49
    EXPECTED_CLASSES = (0, 1)  # 0: Phishing, 1: Legitimate

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize and load the exported Random Forest tree structures.

        Args:
            model_path: Optional explicit path to phishguard_trees.json.
                        Defaults to looking in models/phishguard_trees.json.
        """
        if model_path is None:
            # Default to ../models/phishguard_trees.json relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "phishguard_trees.json")

        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"PhishGuard tree model file not found at: {self.model_path}"
            )

        with open(self.model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.model_name = data.get("model_name", "PhishGuard_RandomForest_Reduced")
        self.scikit_learn_version = data.get("scikit_learn_version", "unknown")
        self.classes: Tuple[int, ...] = tuple(data.get("classes", [0, 1]))
        self.feature_names: Tuple[str, ...] = tuple(data.get("feature_names", []))
        self.n_features = len(self.feature_names)

        if self.n_features != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Model error: Expected {self.EXPECTED_FEATURE_COUNT} features, "
                f"found {self.n_features} in {self.model_path}"
            )

        # Mapping for O(1) dictionary feature ordering
        self._feature_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.feature_names)
        }
        self._feature_set = set(self.feature_names)

        self.trees: List[Dict[str, Any]] = data.get("trees", [])
        self.n_estimators = len(self.trees)
        if self.n_estimators == 0:
            raise ValueError("Model error: No decision trees found in model file.")

    # -------------------------------------------------------------------------
    # Input Validation & Normalization
    # -------------------------------------------------------------------------

    def validate_and_order_features(
        self, sample: Union[Sequence[Union[int, float]], Dict[str, Any]]
    ) -> List[float]:
        """Validate input features and return an ordered list of 49 floats.

        Supports:
            1. Dict[str, number]: Must have all 49 features. No missing or extra keys.
            2. Sequence[number]: Must have exactly 49 numbers in the original order.

        Raises:
            TypeError: If input structure is not a dict or sequence.
            ValueError: If features are missing, extra, non-numeric, or NaN/Inf.
        """
        ordered_values: List[float] = [0.0] * self.EXPECTED_FEATURE_COUNT

        if isinstance(sample, dict):
            sample_keys = set(sample.keys())

            # Check for missing features
            missing = self._feature_set - sample_keys
            if missing:
                missing_sorted = sorted(list(missing))
                raise ValueError(
                    f"Input validation error: Missing {len(missing)} required feature(s): {missing_sorted}"
                )

            # Check for unexpected extra features
            extra = sample_keys - self._feature_set
            if extra:
                extra_sorted = sorted(list(extra))
                raise ValueError(
                    f"Input validation error: Unexpected extra feature(s): {extra_sorted}"
                )

            # Populate ordered vector
            for name, val in sample.items():
                idx = self._feature_to_idx[name]
                ordered_values[idx] = self._validate_numeric(val, feature_name=name)

        elif isinstance(sample, (list, tuple)):
            if len(sample) != self.EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"Input validation error: Expected {self.EXPECTED_FEATURE_COUNT} features in sequence, "
                    f"but received {len(sample)}."
                )

            for idx, val in enumerate(sample):
                feat_name = self.feature_names[idx]
                ordered_values[idx] = self._validate_numeric(val, feature_name=feat_name)

        else:
            raise TypeError(
                f"Input validation error: Sample must be a dictionary or list/tuple of {self.EXPECTED_FEATURE_COUNT} numbers, "
                f"got {type(sample).__name__}."
            )

        return ordered_values

    @staticmethod
    def _validate_numeric(val: Any, feature_name: str) -> float:
        """Validate that a single feature value is a valid finite float."""
        if val is None or isinstance(val, bool):
            raise ValueError(
                f"Invalid value for feature '{feature_name}': cannot be {val!r} (must be numeric)."
            )
        try:
            f_val = float(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid non-numeric value for feature '{feature_name}': {val!r}"
            ) from exc

        if math.isnan(f_val):
            raise ValueError(
                f"Invalid NaN value encountered for feature '{feature_name}'."
            )
        if math.isinf(f_val):
            raise ValueError(
                f"Invalid Infinite value encountered for feature '{feature_name}'."
            )

        return f_val

    # -------------------------------------------------------------------------
    # Inference Algorithms
    # -------------------------------------------------------------------------

    def _traverse_tree(self, tree: Dict[str, Any], x: List[float]) -> List[float]:
        """Traverse a single binary decision tree from root to leaf node.

        Returns:
            Normalized class probability distribution [P(phishing), P(legitimate)].
        """
        children_left: List[int] = tree["children_left"]
        children_right: List[int] = tree["children_right"]
        feature_indices: List[int] = tree["feature"]
        thresholds: List[float] = tree["threshold"]
        node_values = tree["value"]

        node = 0
        # While current node is an internal split node (not a leaf)
        while children_left[node] != -1:
            feat_idx = feature_indices[node]
            split_th = thresholds[node]
            if x[feat_idx] <= split_th:
                node = children_left[node]
            else:
                node = children_right[node]

        # Leaf node reached: extract class sample distribution
        val = node_values[node]
        if isinstance(val[0], list):
            val = val[0]

        total_weight = val[0] + val[1]
        if total_weight <= 0.0:
            return [0.5, 0.5]

        return [val[0] / total_weight, val[1] / total_weight]

    def predict_proba_sample(
        self, sample: Union[Sequence[Union[int, float]], Dict[str, Any]]
    ) -> List[float]:
        """Predict class probabilities for a single sample.

        Returns:
            [P(class_0=Phishing), P(class_1=Legitimate)]
        """
        x = self.validate_and_order_features(sample)
        sum_p0 = 0.0
        sum_p1 = 0.0

        for tree in self.trees:
            t_p0, t_p1 = self._traverse_tree(tree, x)
            sum_p0 += t_p0
            sum_p1 += t_p1

        return [sum_p0 / self.n_estimators, sum_p1 / self.n_estimators]

    def predict_sample(
        self, sample: Union[Sequence[Union[int, float]], Dict[str, Any]]
    ) -> int:
        """Predict binary class for a single sample.

        Returns:
            0 for Phishing, 1 for Legitimate.
        """
        p0, p1 = self.predict_proba_sample(sample)
        # scikit-learn argmax tie-breaker: if tied, argmax picks index 0
        return 1 if p1 > p0 else 0

    def predict_proba(
        self, samples: Union[Sequence[Union[int, float]], Dict[str, Any], Sequence[Any]]
    ) -> Union[List[float], List[List[float]]]:
        """Predict class probabilities for a single sample or batch of samples."""
        if isinstance(samples, dict):
            return self.predict_proba_sample(samples)
        if isinstance(samples, (list, tuple)):
            if len(samples) > 0 and isinstance(samples[0], (dict, list, tuple)):
                return [self.predict_proba_sample(s) for s in samples]
            return self.predict_proba_sample(samples)
        raise TypeError(f"Unsupported input type: {type(samples).__name__}")

    def predict(
        self, samples: Union[Sequence[Union[int, float]], Dict[str, Any], Sequence[Any]]
    ) -> Union[int, List[int]]:
        """Predict binary class label(s) for a single sample or batch of samples."""
        if isinstance(samples, dict):
            return self.predict_sample(samples)
        if isinstance(samples, (list, tuple)):
            if len(samples) > 0 and isinstance(samples[0], (dict, list, tuple)):
                return [self.predict_sample(s) for s in samples]
            return self.predict_sample(samples)
        raise TypeError(f"Unsupported input type: {type(samples).__name__}")
