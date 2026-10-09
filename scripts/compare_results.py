"""
Aggregate HaluEval + SimpleQA results into comparison tables.
Usage:
    PYTHONPATH=. python scripts/compare_results.py
    # atau dari root portfo:
    python scripts/compare_results.py
"""
import sys
from pathlib import Path

# Pastikan root project ada di sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

try:
    from src.metrics import halueval_metrics
except ImportError:
    # fallback sederhana kalau metrics belum ada / beda nama
    def halueval_metrics(y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_true = list(y_true)
        y_pred = list(y_pred)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "yes_ratio": sum(y_pred) / len(y_pred) if y_pred else 0,
        }

RESULTS = ROOT / "results"

def load_halueval_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "prediction" not in df.columns or "ground_truth" not in df.columns:
        print(f"[SKIP] {csv_path.name} → kolom tidak sesuai")
        return None
    valid = df[df["prediction"] != -1]
    if len(valid) == 0:
        return None
    m = halueval_metrics(valid["ground_truth"], valid["prediction"])
    m = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in m.items()}
    m["unparseable"] = int((df["prediction"] == -1).sum())
    m["n_rows"] = len(df)
    m["model"] = df["model"].iloc[0] if "model" in df.columns else csv_path.stem
    m["file"] = csv_path.name
    return m

def simpleqa_metrics(df: pd.DataFrame):
    # Cari kolom verdict
    col = None
    for c in ["verdict", "prediction", "judge_verdict", "label", "result"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return None

    s = df[col].astype(str).str.upper().str.strip()
    n = len(s)
    n_correct = (s == "CORRECT").sum()
    n_incorrect = (s == "INCORRECT").sum()
    n_not = (s == "NOT_ATTEMPTED").sum()
    n_unp = ((s == "UNPARSEABLE") | (~s.isin(["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]))).sum()

    attempted = n_correct + n_incorrect
    acc = n_correct / n if n else 0.0
    attempted_acc = n_correct / attempted if attempted else 0.0
    halluc = n_incorrect / n if n else 0.0
    abstain = n_not / n if n else 0.0
    f = (2 * acc * (1 - abstain)) / (acc + (1 - abstain)) if (acc + (1 - abstain)) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "attempted_accuracy": round(attempted_acc, 4),
        "hallucination_rate": round(halluc, 4),
        "abstention_rate": round(abstain, 4),
        "f_score": round(f, 4),
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "n_not_attempted": int(n_not),
        "unparseable": int(n_unp),
        "n": n,
    }

def main():
    print("=" * 70)
    print("HALUEVAL COMPARISON (100 samples)")
    print("=" * 70)

    halu_files = sorted(RESULTS.glob("halueval_*_100.csv"))
    rows = []
    for f in halu_files:
        m = load_halueval_metrics(f)
        if m:
            rows.append(m)

    if rows:
        hdf = pd.DataFrame(rows)
        preferred = ["model", "accuracy", "precision", "recall", "f1", "yes_ratio", "unparseable", "n_rows", "file"]
        cols = [c for c in preferred if c in hdf.columns]
        print(hdf[cols].sort_values("f1", ascending=False).to_string(index=False))
        out = RESULTS / "summary_halueval.csv"
        hdf.to_csv(out, index=False)
        print(f"\nSaved → {out}")
    else:
        print("Tidak ada file HaluEval *_100.csv yang valid.")

    print("\n" + "=" * 70)
    print("SIMPLEQA COMPARISON (100 samples)")
    print("=" * 70)

    sq_files = sorted(RESULTS.glob("simpleqa_*_100.csv"))
    rows = []
    for f in sq_files:
        df = pd.read_csv(f)
        m = simpleqa_metrics(df)
        if m is None:
            print(f"[SKIP] {f.name} → tidak ketemu kolom verdict")
            continue
        m["file"] = f.name
        m["run"] = f.stem
        rows.append(m)

    if rows:
        sdf = pd.DataFrame(rows)
        print(sdf.sort_values("accuracy", ascending=False).to_string(index=False))
        out = RESULTS / "summary_simpleqa.csv"
        sdf.to_csv(out, index=False)
        print(f"\nSaved → {out}")
    else:
        print("Tidak ada file SimpleQA *_100.csv yang valid.")

if __name__ == "__main__":
    main()
