"""Group SimpleQA failures: INCORRECT vs NOT_ATTEMPTED vs UNPARSEABLE."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "output"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted((ROOT / "results").glob("simpleqa_*_100.csv")):
    df = pd.read_csv(f)
    col = "verdict" if "verdict" in df.columns else "prediction"
    g = df[col].astype(str).str.upper().value_counts().to_dict()
    rows.append({"file": f.name, **g})

summary = pd.DataFrame(rows).fillna(0)
summary.to_csv(OUT / "error_taxonomy_summary.csv", index=False)
print(summary)
print(f"Wrote {OUT / 'error_taxonomy_summary.csv'}")
