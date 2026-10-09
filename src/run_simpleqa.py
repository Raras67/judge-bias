"""
Run SimpleQA evaluation using an OpenAI-compatible router.

Cross-judge pattern: one model answers, a DIFFERENT model grades.
This avoids self-preference bias.

Usage:
    # Gemini answers, Grok judges
    python -m src.run_simpleqa \
        --answer-model "google/gemini-2.5-flash" \
        --judge-model  "x-ai/grok-4-fast" \
        --n-samples 200 \
        --out results/simpleqa_gemini.csv

    # Grok answers, Gemini judges
    python -m src.run_simpleqa \
        --answer-model "x-ai/grok-4-fast" \
        --judge-model  "google/gemini-2.5-flash" \
        --n-samples 200 \
        --out results/simpleqa_grok.csv
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
from src.config import ENV_PATH
from dotenv import load_dotenv
load_dotenv(dotenv_path=ENV_PATH, override=False)
from tqdm import tqdm

from src.datasets import load_simpleqa
from src.prompts import SIMPLEQA_ANSWER_PROMPT, SIMPLEQA_JUDGE_PROMPT
from src.clients import get_client


def parse_verdict(text: str) -> str:
    """Extract CORRECT / INCORRECT / NOT_ATTEMPTED from judge output."""
    t = text.strip().upper()
    # Order matters: NOT_ATTEMPTED first to avoid matching "ATTEMPTED" substring
    for label in ["NOT_ATTEMPTED", "INCORRECT", "CORRECT"]:
        if label in t:
            return label
    m = re.search(r"\b(CORRECT|INCORRECT|NOT_ATTEMPTED)\b", t)
    if m:
        return m.group(1)
    return "UNPARSEABLE"


def evaluate(
    answer_model: str,
    judge_model: str,
    n_samples: int,
    out_path: str,
    sleep_between: float = 0.0,
):
    ds = load_simpleqa(max_samples=n_samples)
    client = get_client()

    rows = []
    for i, row in enumerate(tqdm(ds, desc=f"simpleqa::{answer_model}")):
        problem = row["problem"]
        ground_truth = row["answer"]

        # --- Step 1: model answers ---
        try:
            model_answer = client.generate(
                model=answer_model,
                prompt=SIMPLEQA_ANSWER_PROMPT.format(problem=problem),
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as e:
            model_answer = f"ERROR: {e}"

        # --- Step 2: judge grades ---
        try:
            raw_verdict = client.generate(
                model=judge_model,
                prompt=SIMPLEQA_JUDGE_PROMPT.format(
                    problem=problem,
                    ground_truth=ground_truth,
                    model_answer=model_answer,
                ),
                temperature=0.0,
                max_tokens=8,
            )
            verdict = parse_verdict(raw_verdict)
        except Exception as e:
            raw_verdict = f"ERROR: {e}"
            verdict = "UNPARSEABLE"

        # Metadata (defensive: some mirrors wrap it differently)
        topic = None
        try:
            meta = row.get("metadata") or {}
            if isinstance(meta, dict):
                topic = meta.get("topic")
        except Exception:
            pass

        rows.append({
            "sample_id": i,
            "problem": problem[:200],
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "verdict": verdict,
            "judge_raw": raw_verdict,
            "topic": topic,
            "answer_model": answer_model,
            "judge_model": judge_model,
        })

        if sleep_between:
            time.sleep(sleep_between)

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-model", required=True,
                        help="Model ID that answers (e.g. 'google/gemini-2.5-flash')")
    parser.add_argument("--judge-model", required=True,
                        help="Model ID that grades (should differ from answer-model)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between calls")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.answer_model == args.judge_model:
        print("[warn] answer-model == judge-model. "
              "Cross-judge is strongly recommended to avoid self-preference bias.")

    df = evaluate(
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        n_samples=args.n_samples,
        out_path=args.out,
        sleep_between=args.sleep,
    )

    valid = df[df["verdict"] != "UNPARSEABLE"]
    if len(valid):
        from src.metrics import simpleqa_metrics
        m = simpleqa_metrics(valid["verdict"].tolist())
        print(f"\n=== SimpleQA / answer={args.answer_model} / judge={args.judge_model} ===")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        unparseable = (df["verdict"] == "UNPARSEABLE").sum()
        print(f"  unparseable: {unparseable}/{len(df)}")
    print(f"\nSaved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
