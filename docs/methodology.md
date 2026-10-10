# Methodology — LLM-as-Judge Bias (SimpleQA)

## Goal
Measure how much reported factual accuracy on SimpleQA depends on the **judging model**, including self-preference and cross-judge asymmetry.

## Dataset
- SimpleQA (loader in `src/datasets.py`)
- Evaluation uses the **first N rows** (default N=100) so every answer×judge pair sees the same questions

## Protocol
1. **Answer stage:** model A generates an answer for each question (`temperature=0` recommended).
2. **Judge stage:** model B labels the answer as `CORRECT`, `INCORRECT`, or `NOT_ATTEMPTED` given the gold target (see `SIMPLEQA_JUDGE_PROMPT`).
3. Optionally A = B (**self-judge**) to probe self-preference.
4. Aggregate metrics per run; build Answer × Judge accuracy matrix.

## Metrics
| Metric | Definition |
|--------|------------|
| Accuracy | CORRECT / N |
| Attempted accuracy | CORRECT / (CORRECT + INCORRECT) |
| Hallucination rate | INCORRECT / N |
| Abstention rate | NOT_ATTEMPTED / N |
| F-score | Harmonic mean of accuracy and attempted accuracy |
| Unparseable | Labels outside the three canonical verdicts |
| Bootstrap CI | Resample 0/1 correct flags (see `simpleqa_metrics_with_ci`) |

## Design choices
- Fixed question prefix across runs (fair comparison)
- Explicit CLI flags: `--answer-model`, `--judge-model`
- OpenAI-compatible router client with retries
- Raw per-run CSVs may be local-only; summaries and matrix can be committed

## Limitations
- N=100 — small gaps may not be statistically significant
- Matrix may be incomplete (not all answer×judge pairs)
- No human gold labels in-repo; scores are judge-relative
- Single judge prompt (prompt sensitivity not ablated)
- Some historical runs (e.g. Qwen) may reflect provider issues
