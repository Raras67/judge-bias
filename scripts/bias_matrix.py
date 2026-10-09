"""
Build Answer × Judge accuracy matrix from SimpleQA result CSVs.

Expected filenames (flexible):
  simpleqa_<answer>_ans_<judge>_judge_100.csv
  simpleqa_<answer>_ans_<judge>_judge_100.csv
  or any *simpleqa*.csv with columns that include verdict + optional model fields

Usage:
  PYTHONPATH=. python scripts/bias_matrix.py
  PYTHONPATH=. python scripts/bias_matrix.py --results results --out results/bias_matrix.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_verdict_column(df: pd.DataFrame) -> str | None:
    for c in ["verdict", "prediction", "judge_verdict", "label", "result"]:
        if c in df.columns:
            return c
    return None


def parse_answer_judge(name: str) -> tuple[str, str]:
    """
    Parse stem like:
      simpleqa_claude_ans_gemini_judge_100
      simpleqa_muse_ans_muse_judge_100
      simpleqa_gemini_ans_muse-spark_judge_100
    """
    stem = Path(name).stem
    m = re.search(
        r"simpleqa_(.+?)_ans_(.+?)_judge",
        stem,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2)
    # fallback
    return stem, "unknown"


def accuracy_from_df(df: pd.DataFrame) -> float | None:
    col = find_verdict_column(df)
    if col is None:
        return None
    s = df[col].astype(str).str.upper().str.strip()
    n = len(s)
    if n == 0:
        return None
    return float((s == "CORRECT").sum() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "bias_matrix.csv")
    args = ap.parse_args()

    files = sorted(args.results.glob("simpleqa_*.csv"))
    if not files:
        print(f"No simpleqa_*.csv under {args.results}")
        sys.exit(1)

    records = []
    for f in files:
        if f.name.startswith("summary"):
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] {f.name}: {e}")
            continue
        acc = accuracy_from_df(df)
        if acc is None:
            print(f"[skip] {f.name}: no verdict column")
            continue
        answer, judge = parse_answer_judge(f.name)
        records.append(
            {
                "answer_model": answer,
                "judge_model": judge,
                "accuracy": round(acc, 4),
                "n": len(df),
                "file": f.name,
            }
        )

    long_df = pd.DataFrame(records)
    if long_df.empty:
        print("No valid runs parsed.")
        sys.exit(1)

    matrix = long_df.pivot_table(
        index="answer_model",
        columns="judge_model",
        values="accuracy",
        aggfunc="mean",
    )
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1)

    print("=" * 70)
    print("SIMPLEQA BIAS MATRIX (accuracy)  rows=answer  cols=judge")
    print("=" * 70)
    print(matrix.to_string())
    print()

    # Self-preference diagonal gap when both exist
    print("Self-preference check (answer == judge), if present:")
    for _, r in long_df.iterrows():
        if r["answer_model"].replace("-", "") in r["judge_model"].replace("-", "") or r[
            "answer_model"
        ] == r["judge_model"]:
            print(f"  {r['answer_model']} judged by {r['judge_model']}: {r['accuracy']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out)
    long_df.to_csv(args.out.with_name("bias_matrix_long.csv"), index=False)
    print(f"\nSaved matrix → {args.out}")
    print(f"Saved long   → {args.out.with_name('bias_matrix_long.csv')}")


if __name__ == "__main__":
    main()
