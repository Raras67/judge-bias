"""Metrics for HaluEval and SimpleQA - bootstrap CI."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def halueval_metrics(y_true, y_pred):
    """Binary metrics: 1 = hallucination (YES), 0 = factual (NO)."""
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "yes_ratio": (sum(y_pred) / len(y_pred)) if y_pred else 0.0,
    }


def simpleqa_metrics(verdicts: Sequence[str]):
    """
    verdicts: CORRECT | INCORRECT | NOT_ATTEMPTED (other → ignored in counts
    except they still sit in N if you pass them through; prefer cleaning upstream).
    """
    verdicts = [str(v).strip().upper() for v in verdicts]
    n = len(verdicts)
    if n == 0:
        return {
            "accuracy": 0.0,
            "attempted_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "abstention_rate": 0.0,
            "f_score": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
            "n_not_attempted": 0,
        }

    n_correct = sum(1 for v in verdicts if v == "CORRECT")
    n_incorrect = sum(1 for v in verdicts if v == "INCORRECT")
    n_not_attempted = sum(1 for v in verdicts if v == "NOT_ATTEMPTED")
    n_attempted = n_correct + n_incorrect

    accuracy = n_correct / n
    attempted_acc = n_correct / n_attempted if n_attempted else 0.0
    hallucination_rate = n_incorrect / n
    abstention_rate = n_not_attempted / n

    if accuracy + attempted_acc > 0:
        f_score = 2 * accuracy * attempted_acc / (accuracy + attempted_acc)
    else:
        f_score = 0.0

    return {
        "accuracy": accuracy,
        "attempted_accuracy": attempted_acc,
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "f_score": f_score,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_not_attempted,
    }


def bootstrap_ci(
    scores: Iterable[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) for 0/1 scores."""
    rng = np.random.default_rng(seed)
    scores_arr = np.asarray(list(scores), dtype=float)
    if len(scores_arr) == 0:
        return 0.0, 0.0, 0.0
    means: List[float] = []
    n = len(scores_arr)
    for _ in range(n_boot):
        sample = rng.choice(scores_arr, size=n, replace=True)
        means.append(float(sample.mean()))
    means_sorted = np.sort(means)
    lo = means_sorted[int((alpha / 2) * n_boot)]
    hi = means_sorted[int((1 - alpha / 2) * n_boot) - 1]
    return float(scores_arr.mean()), float(lo), float(hi)


def simpleqa_metrics_with_ci(verdicts: Sequence[str], n_boot: int = 1000):
    """SimpleQA metrics + 95% bootstrap CI on accuracy."""
    verdicts = [str(v).strip().upper() for v in verdicts]
    base = simpleqa_metrics(verdicts)
    correct_flags = [1.0 if v == "CORRECT" else 0.0 for v in verdicts]
    mean, lo, hi = bootstrap_ci(correct_flags, n_boot=n_boot)
    base["accuracy_mean"] = round(mean, 4)
    base["accuracy_ci95_low"] = round(lo, 4)
    base["accuracy_ci95_high"] = round(hi, 4)
    return base
