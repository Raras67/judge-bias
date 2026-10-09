import numpy as np

def bootstrap_ci(scores, n_boot=1000, alpha=0.05, seed=42):
    """
    scores: 0/1 array (correct indicator per example).
    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    means = []
    n = len(scores)
    for _ in range(n_boot):
        sample = rng.choice(scores, size=n, replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return float(scores.mean()), float(lo), float(hi)


def simpleqa_metrics_with_ci(verdicts, n_boot=1000):
    base = simpleqa_metrics(verdicts)
    correct_flags = [1.0 if v == "CORRECT" else 0.0 for v in verdicts]
    mean, lo, hi = bootstrap_ci(correct_flags, n_boot=n_boot)
    base["accuracy_ci95"] = (round(lo, 4), round(hi, 4))
    base["accuracy_mean"] = round(mean, 4)
    return base


def simpleqa_metrics(verdicts):
    """
    verdicts: list of strings in {CORRECT, INCORRECT, NOT_ATTEMPTED}.
    Returns dict with accuracy, attempted_accuracy, hallucination_rate,
    abstention_rate, f_score.
    """
    n = len(verdicts)
    if n == 0:
        return {k: 0.0 for k in
                ["accuracy", "attempted_accuracy", "hallucination_rate",
                 "abstention_rate", "f_score"]}

    n_correct = sum(1 for v in verdicts if v == "CORRECT")
    n_incorrect = sum(1 for v in verdicts if v == "INCORRECT")
    n_not_attempted = sum(1 for v in verdicts if v == "NOT_ATTEMPTED")
    n_attempted = n_correct + n_incorrect

    accuracy = n_correct / n
    attempted_acc = n_correct / n_attempted if n_attempted else 0.0
    hallucination_rate = n_incorrect / n
    abstention_rate = n_not_attempted / n

    # F-score: harmonic mean of accuracy and attempted_accuracy
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
