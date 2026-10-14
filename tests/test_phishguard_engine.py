"""
Automated Verification & Unit Tests for PhishGuard Engine
---------------------------------------------------------
Validates mathematical equivalence against original Colab scikit-learn outputs
and exercises comprehensive input validation edge cases.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

# Ensure the project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phishguard_engine import PhishGuardEngine


class TestPhishGuardEngine(unittest.TestCase):
    """Test suite for PhishGuard pure-Python inference engine."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load engine and benchmark data once for all test cases."""
        cls.models_dir = os.path.join(BASE_DIR, "models")
        cls.trees_path = os.path.join(cls.models_dir, "phishguard_trees.json")
        cls.bench_path = os.path.join(cls.models_dir, "phishguard_benchmark.json")

        cls.engine = PhishGuardEngine(cls.trees_path)

        with open(cls.bench_path, "r", encoding="utf-8") as f:
            cls.bench_data = json.load(f)

    def test_no_forbidden_imports(self) -> None:
        """Verify runtime engine does not import numpy, pandas, or sklearn."""
        forbidden = ["numpy", "pandas", "scipy", "sklearn"]
        for mod in forbidden:
            self.assertNotIn(
                mod,
                sys.modules,
                f"Engine violated pure-Python constraint: '{mod}' is imported in sys.modules!",
            )

    def test_model_metadata(self) -> None:
        """Verify model configuration: 100 trees, 49 features, classes [0, 1]."""
        self.assertEqual(self.engine.n_estimators, 100)
        self.assertEqual(self.engine.n_features, 49)
        self.assertEqual(self.engine.classes, (0, 1))
        self.assertEqual(len(self.engine.feature_names), 49)

    def test_benchmark_predictions_and_probabilities(self) -> None:
        """Compare all 10 benchmark samples against original scikit-learn outputs.

        Requires:
        - 10/10 exact class label matches.
        - Probability differences strictly below 1e-7.
        """
        samples = self.bench_data.get("samples", [])
        self.assertEqual(len(samples), 10, "Expected 10 benchmark samples.")

        print("\n" + "=" * 80)
        print("RUNNING BENCHMARK VERIFICATION (10 TEST SAMPLES)")
        print("=" * 80)
        header = f"{'ID':<3} | {'True':<4} | {'Colab Pred':<10} | {'Engine Pred':<11} | {'Colab Probs':<16} | {'Engine Probs':<16} | {'Max Diff':<10} | {'Status'}"
        print(header)
        print("-" * 80)

        for s in samples:
            sample_id = s["sample_id"]
            true_label = s["true_label"]
            orig_pred = s["original_prediction"]
            orig_probs = s["original_probabilities"]  # [prob_0, prob_1]

            # 1. Test via ordered list
            x_list = s["features_ordered_list"]
            calc_pred_list = self.engine.predict_sample(x_list)
            calc_probs_list = self.engine.predict_proba_sample(x_list)

            # 2. Test via dictionary mapping
            x_dict = s["features_dict"]
            calc_pred_dict = self.engine.predict_sample(x_dict)
            calc_probs_dict = self.engine.predict_proba_sample(x_dict)

            # Verify list and dict produce identical results
            self.assertEqual(calc_pred_list, calc_pred_dict)
            self.assertAlmostEqual(calc_probs_list[0], calc_probs_dict[0], places=9)
            self.assertAlmostEqual(calc_probs_list[1], calc_probs_dict[1], places=9)

            # Verify against original Colab outputs
            diff_0 = abs(calc_probs_list[0] - orig_probs[0])
            diff_1 = abs(calc_probs_list[1] - orig_probs[1])
            max_diff = max(diff_0, diff_1)

            self.assertEqual(
                calc_pred_list,
                orig_pred,
                f"Sample {sample_id} class mismatch: expected {orig_pred}, got {calc_pred_list}",
            )
            self.assertLess(
                max_diff,
                1e-7,
                f"Sample {sample_id} probability diff {max_diff} exceeded tolerance 1e-7",
            )

            colab_str = f"[{orig_probs[0]:.2f}, {orig_probs[1]:.2f}]"
            engine_str = f"[{calc_probs_list[0]:.2f}, {calc_probs_list[1]:.2f}]"
            row = f"{sample_id:<3} | {true_label:<4} | {orig_pred:<10} | {calc_pred_list:<11} | {colab_str:<16} | {engine_str:<16} | {max_diff:<10.2e} | PASS"
            print(row)

        print("=" * 80)
        print("ALL 10 BENCHMARK SAMPLES MATCHED ORIGINAL MODEL WITH ZERO DEVIATION!\n")

    def test_batch_inference(self) -> None:
        """Verify predict() and predict_proba() handle batch inputs."""
        samples = self.bench_data["samples"]
        batch_list = [s["features_ordered_list"] for s in samples]
        batch_dict = [s["features_dict"] for s in samples]

        preds_list = self.engine.predict(batch_list)
        preds_dict = self.engine.predict(batch_dict)
        probas_list = self.engine.predict_proba(batch_list)

        self.assertEqual(len(preds_list), 10)
        self.assertEqual(preds_list, preds_dict)
        self.assertEqual(len(probas_list), 10)
        for i, s in enumerate(samples):
            self.assertEqual(preds_list[i], s["original_prediction"])

    # -------------------------------------------------------------------------
    # Input Validation Edge Cases
    # -------------------------------------------------------------------------

    def test_validation_missing_feature(self) -> None:
        """Missing keys in dictionary input must raise ValueError."""
        valid_sample = dict(self.bench_data["samples"][0]["features_dict"])
        del valid_sample["URLLength"]
        with self.assertRaises(ValueError) as ctx:
            self.engine.predict_sample(valid_sample)
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("URLLength", str(ctx.exception))

    def test_validation_extra_feature(self) -> None:
        """Extra unexpected keys in dictionary input must raise ValueError."""
        valid_sample = dict(self.bench_data["samples"][0]["features_dict"])
        valid_sample["UnknownFeature123"] = 42.0
        with self.assertRaises(ValueError) as ctx:
            self.engine.predict_sample(valid_sample)
        self.assertIn("Unexpected extra", str(ctx.exception))
        self.assertIn("UnknownFeature123", str(ctx.exception))

    def test_validation_wrong_list_length(self) -> None:
        """List with fewer or more than 49 features must raise ValueError."""
        valid_list = list(self.bench_data["samples"][0]["features_ordered_list"])
        with self.assertRaises(ValueError):
            self.engine.predict_sample(valid_list[:-1])  # 48 features
        with self.assertRaises(ValueError):
            self.engine.predict_sample(valid_list + [1.0])  # 50 features

    def test_validation_non_numeric_values(self) -> None:
        """String, None, or boolean values must raise ValueError."""
        valid_sample = dict(self.bench_data["samples"][0]["features_dict"])

        valid_sample["URLLength"] = "bad_string"
        with self.assertRaises(ValueError):
            self.engine.predict_sample(valid_sample)

        valid_sample["URLLength"] = None
        with self.assertRaises(ValueError):
            self.engine.predict_sample(valid_sample)

        valid_sample["URLLength"] = True
        with self.assertRaises(ValueError):
            self.engine.predict_sample(valid_sample)

    def test_validation_nan_and_inf(self) -> None:
        """NaN and Infinite values must raise ValueError."""
        valid_sample = dict(self.bench_data["samples"][0]["features_dict"])

        valid_sample["LineOfCode"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            self.engine.predict_sample(valid_sample)
        self.assertIn("NaN", str(ctx.exception))

        valid_sample["LineOfCode"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            self.engine.predict_sample(valid_sample)
        self.assertIn("Infinite", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
