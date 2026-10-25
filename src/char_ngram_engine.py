"""
PhishGuard - Experimental Character n-gram TF-IDF Inference Engine
------------------------------------------------------------------
Standalone, zero-dependency pure-Python inference engine for Model 4:
Subword 3-to-5 character n-gram TF-IDF classifier for offline URL detection.

Runtime requirements:
- Python 3.10+ standard library (math, re, json, collections).
- ZERO dependencies on NumPy, pandas, or scikit-learn.
- Fully compatible with Windows Smart App Control (SAC).

Target Class Mapping:
- Class 0 = Phishing
- Class 1 = Legitimate
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


class CharNgramEngine:
    """Pure-Python inference engine for Character n-gram TF-IDF URL classification."""

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.model_name: str = data.get("model_name", "PhishGuard_CharNgram_TFIDF")
        self.vocab: Dict[str, int] = data["vocab"]
        self.idf: List[float] = data["idf"]
        self.weights: List[float] = data["weights"]
        self.bias: float = float(data.get("bias", 0.0))
        self.threshold: float = float(data.get("threshold", 0.55))
        self.min_n: int = int(data.get("min_n", 3))
        self.max_n: int = int(data.get("max_n", 5))
        self.vocab_size: int = len(self.vocab)

        # Reverse vocabulary for explanation
        self.inv_vocab: Dict[int, str] = {idx: ng for ng, idx in self.vocab.items()}

    def extract_ngrams(self, raw_url: str) -> List[str]:
        """Extract word-boundary anchored character 3-to-5 n-grams."""
        url_lower = raw_url.strip().lower()
        # Split into alphanumeric tokens
        tokens = re.split(r"[^a-zA-Z0-9]+", url_lower)
        ngrams: List[str] = []

        for tok in tokens:
            if not tok:
                continue
            # Anchor word boundaries with $
            w = "$" + tok + "$"
            w_len = len(w)
            for n in range(self.min_n, min(self.max_n + 1, w_len + 1)):
                for i in range(w_len - n + 1):
                    ngrams.append(w[i : i + n])

        return ngrams

    def transform_url(self, raw_url: str) -> Dict[int, float]:
        """Convert a raw URL into an L2-normalized sparse TF-IDF dictionary."""
        ngrams = self.extract_ngrams(raw_url)
        tf = Counter(ngrams)

        sparse_vec: Dict[int, float] = {}
        norm_sq = 0.0

        for ng, count in tf.items():
            if ng in self.vocab:
                idx = self.vocab[ng]
                val = (1.0 + math.log(count)) * self.idf[idx]
                sparse_vec[idx] = val
                norm_sq += val ** 2

        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            for idx in sparse_vec:
                sparse_vec[idx] /= norm

        return sparse_vec

    def predict_proba_sample(self, raw_url: str) -> List[float]:
        """Predict class probabilities: [P(Phishing), P(Legitimate)]."""
        sparse_vec = self.transform_url(raw_url)
        z = sum(self.weights[idx] * val for idx, val in sparse_vec.items()) + self.bias
        # Numeric clipping
        z = max(min(z, 20.0), -20.0)
        p_phish = 1.0 / (1.0 + math.exp(-z))
        p_legit = 1.0 - p_phish
        return [round(p_phish, 5), round(p_legit, 5)]

    def predict_sample(self, raw_url: str) -> int:
        """Predict binary class: 0 = Phishing, 1 = Legitimate.
        Uses validation-tuned threshold (default: 0.55).
        """
        probs = self.predict_proba_sample(raw_url)
        # Class 0 if P(phish) >= threshold, else Class 1
        return 0 if probs[0] >= self.threshold else 1

    def explain_sample(self, raw_url: str, top_k: int = 5) -> Dict[str, Any]:
        """Identify the top subwords pushing prediction toward Phishing or Legitimate."""
        sparse_vec = self.transform_url(raw_url)
        token_contributions: List[Tuple[str, float]] = []

        for idx, val in sparse_vec.items():
            contrib = self.weights[idx] * val
            token_contributions.append((self.inv_vocab[idx], round(contrib, 4)))

        # Positive contribution = pushes toward Phishing (label 0)
        phish_tokens = sorted([t for t in token_contributions if t[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
        # Negative contribution = pushes toward Legitimate (label 1)
        legit_tokens = sorted([t for t in token_contributions if t[1] < 0], key=lambda x: x[1])[:top_k]

        probs = self.predict_proba_sample(raw_url)
        pred = self.predict_sample(raw_url)

        return {
            "url": raw_url,
            "prediction": "Phishing" if pred == 0 else "Legitimate",
            "prob_phishing": probs[0],
            "prob_legitimate": probs[1],
            "bias_log_odds": round(self.bias, 4),
            "top_phishing_subwords": phish_tokens,
            "top_legitimate_subwords": legit_tokens,
        }
