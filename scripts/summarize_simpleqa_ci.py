"""Recompute SimpleQA summary with bootstrap CI from raw CSVs."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import simpleqa_metrics_with_ci

RESULTS = ROOT / "results"
rows = []
for f in sorted(RESULTS.glob("simpleqa_*_100.csv")):
    df = pd.read_csv(f)
    col = None
    for c in ["verdict", "prediction", "judge_verdict", "label", "result"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        print(f"[skip] {f.name}")
        continue
    verdicts = df[col].astype(str).str.upper().tolist()
    m = simpleqa_metrics_with_ci(verdicts)
    m["file"] = f.name
    m["run"] = f.stem
    rows.append(m)

out = pd.DataFrame(rows)
out_path = RESULTS / "summary_simpleqa.csv"
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print(f"\nSaved → {out_path}")
