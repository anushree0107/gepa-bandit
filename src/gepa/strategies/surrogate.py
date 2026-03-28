"""Surrogate model for predicting prompt performance.

Provides a computationally cheap MLP regressor (backed by scikit-learn) to
estimate the performance of prompt mutations *before* running expensive LLM
validation.  This is the core enabler of the bandit strategy described in
``bandit-gepa-prompt.txt``.

Public API:
    :class:`SurrogateModel` — fit / predict / predict_with_ucb
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor


@dataclass
class SurrogateModel:
    """MLP surrogate that maps (prompt, question) text → predicted score.

    Inputs are encoded via TF-IDF over the concatenated ``prompt [SEP] question``
    strings.  The model is re-trained each round on all accumulated data using
    ``warm_start=True`` so the previous weights serve as initialisation.

    Attributes:
        vectorizer: TF-IDF feature extractor.
        model: scikit-learn ``MLPRegressor``.
        training_buffer: List of ``(text, score)`` tuples accumulated over rounds.
        is_fitted: Whether the model has been fitted at least once.
    """

    vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(max_features=500)
    )
    model: MLPRegressor = field(
        default_factory=lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            warm_start=True,
        )
    )
    training_buffer: list[tuple[str, float]] = field(default_factory=list)
    is_fitted: bool = False

    # ------------------------------------------------------------------
    # Data accumulation
    # ------------------------------------------------------------------

    def add_data(self, prompt: str, questions: list[str], scores: list[float]) -> None:
        """Append ``(prompt+question, score)`` pairs to the training buffer."""
        for q, s in zip(questions, scores):
            self.training_buffer.append((f"{prompt} [SEP] {q}", s))

    def add_data_single(self, prompt: str, question: str, score: float) -> None:
        """Append a single ``(prompt+question, score)`` to the training buffer."""
        self.training_buffer.append((f"{prompt} [SEP] {question}", score))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> float:
        """(Re-)train the surrogate on all accumulated data.

        Returns:
            Mean squared error on the training set (the "Error / Loss" metric).
            Returns ``inf`` when there is insufficient data (< 5 samples).
        """
        if len(self.training_buffer) < 5:
            return float("inf")

        texts = [t[0] for t in self.training_buffer]
        targets = [t[1] for t in self.training_buffer]

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, targets)
        self.is_fitted = True

        # Compute training MSE
        preds = self.model.predict(X)
        mse = sum((a - p) ** 2 for a, p in zip(targets, preds)) / len(targets)
        return float(mse)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, prompt: str, questions: list[str]) -> list[float]:
        """Predict per-question scores for *prompt*.

        Returns a uniform prior of ``0.5`` when the model has not been fitted.
        """
        if not self.is_fitted:
            return [0.5] * len(questions)
        texts = [f"{prompt} [SEP] {q}" for q in questions]
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        return [max(0.0, min(1.0, float(p))) for p in preds]

    def predict_mean(self, prompt: str, questions: list[str]) -> float:
        """Return the mean predicted score for *prompt* across *questions*."""
        preds = self.predict(prompt, questions)
        return sum(preds) / len(preds) if preds else 0.5

    def predict_with_ucb(
        self,
        prompt: str,
        questions: list[str],
        exploration: float = 1.0,
        total_candidates: int = 1,
    ) -> tuple[float, float, float]:
        """Predict score with an Upper Confidence Bound bonus.

        UCB = predicted_score + exploration * sqrt(log(total_candidates) / (1 + buffer_size))

        Returns:
            ``(predicted_mean, ucb_bonus, ucb_score)``
        """
        predicted = self.predict_mean(prompt, questions)
        denom = 1 + len(self.training_buffer)
        bonus = exploration * math.sqrt(math.log(max(total_candidates, 2)) / denom)
        return predicted, bonus, predicted + bonus
