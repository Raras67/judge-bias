# LLM-as-Judge Bias Analysis

Cross-model evaluation on **SimpleQA** to measure how much factual scores depend on the **judging model**, including self-preference effects.

## Motivation
Single-model “LLM-as-judge” pipelines can hide bias: a model may score its own answers higher, or systematically prefer certain answer styles. This repo runs a controlled **answer × judge** matrix.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
python -m src.run_simpleqa \
  --answer-model "claude-sonnet-5" \
  --judge-model "gemini-3.7-flash" \
  --n-samples 100 \
  --out results/simpleqa_claude_ans_gemini_judge_100.csv
```


## Metrics

| Metric | Definition |
|----------|----------|
| Accuracy | CORRECT / N |
| Attempted accuracy | CORRECT / (CORRECT + INCORRECT) |
| Hallucination rate | INCORRECT / N |
| Abstention rate | NOT_ATTEMPTED / N |
| F-score | Harmonic mean of accuracy and attempted accuracy |
| Unparseable | Labels outside `{CORRECT, INCORRECT, NOT_ATTEMPTED}` |

All primary runs use **N = 100 questions** (fixed prefix for fair comparison).

---

## Results

### Accuracy Matrix (Answer × Judge)

Rows = answer model, columns = judge model. Values = accuracy.

| Answer Model | Gemini 3.7 Flash | Muse-spark | Grok 4.6 | Claude Sonnet 5 | Qwen* |
|--------------|------------------|------------|----------|-----------------|-------|
| Claude Sonnet 5 | 0.65 | 0.59 | N/A | N/A | N/A |
| Muse-spark | 0.63 | 0.66 | N/A | 0.59 | 0.60 |
| Grok 4.6 | 0.61 | 0.61 | N/A | N/A | N/A |
| Gemini 3.7 Flash | 0.60 | 0.59 | 0.57 | N/A | N/A |
| Qwen* | N/A | 0.62 | N/A | N/A | N/A |


Full numeric summary: `results/summary_simpleqa.csv`

---

## Ranking Under a Fixed Judge

### Judge = Gemini 3.7 Flash

| Rank | Answer Model | Accuracy | Attempted Acc | Hallucination | Abstention |
|------|-------------|----------|---------------|---------------|------------|
| 1 | Claude Sonnet 5 | 0.65 | 0.79 | 0.17 | 0.17 |
| 2 | Muse-spark | 0.63 | 0.79 | 0.17 | 0.20 |
| 3 | Grok 4.6 | 0.61 | 0.73 | 0.23 | 0.16 |
| 4 | Gemini 3.7 Flash (self) | 0.60 | 0.69 | 0.27 | 0.13 |

### Judge = Muse-spark

| Rank | Answer Model | Accuracy | Attempted Acc | Hallucination | Abstention |
|------|-------------|----------|---------------|---------------|------------|
| 1 | Muse-spark (self) | 0.66 | 0.79 | 0.18 | 0.15 |
| 2 | Grok 4.6 | 0.61 | 0.74 | 0.21 | 0.17 |
| 3 | Claude Sonnet 5 | 0.59 | 0.75 | 0.20 | 0.19 |
| 4 | Gemini 3.7 Flash | 0.59 | 0.73 | 0.22 | 0.18 |


## Findings

Self-preference (Muse): highest score when Muse judges Muse (0.66).
No clear Gemini self-preference: Gemini scoring itself is not top; higher hallucination rate on own answers.
Rankings flip by judge: Claude leads under Gemini-judge; Muse leads under Muse-judge.
Practical rule: do not rank answer models with a single (especially self) judge; use a fixed external judge or multi-judge panel.

## Metrics

Accuracy — overall correctness per judge labels  
Attempted accuracy — correctness excluding NOT_ATTEMPTED  
Hallucination rate — INCORRECT / N  
Abstention rate — NOT_ATTEMPTED / N



## Analysis

### F1 — Judge Identity Dominates Ranking

Swapping the judge (Gemini vs Muse) shifts an answer model's accuracy by up to ~6 points and changes the leaderboard.

- Claude ranks first under **Gemini-as-judge**.
- Muse ranks first under **Muse-as-judge**.

**Takeaway:** A single LLM judge is insufficient for a robust factual ranking.

---

### F2 — Self-Preference Is Model-Dependent

#### Muse

Muse exhibits the strongest self-preference:

- Highest global score when **Muse judges Muse** (`0.66`).
- Higher than when Muse judges Claude or Gemini.

#### Gemini

Gemini shows no evidence of self-preference:

- Self-judge produces the highest hallucination rate (`0.27`).
- Self-judge produces the lowest attempted accuracy (`0.69`) in the Gemini-judge panel.

This suggests a possible **self-critique pattern** rather than self-favoritism.

---

### F3 — Accuracy Alone Is Misleading

Gemini-as-answer often abstains less but makes more incorrect attempts.

For meaningful evaluation, always report:

| Metric | Importance |
|----------|----------|
| Accuracy | Overall correctness |
| Attempted Accuracy | Correctness when attempting |
| Hallucination Rate | Frequency of incorrect answers |
| Abstention Rate | Frequency of declining to answer |

**Recommendation:** Never compare models using accuracy alone.

---

### F4 — Cross-Judge Asymmetry

Judging behavior is not symmetric.

| Pair | Accuracy |
|--------|----------|
| Gemini judged by Grok | 0.57 |
| Grok judged by Gemini | 0.61 |

**Observation:** The evaluation outcome depends on both the answer model and the judge model.

**Takeaway:** "Who judges whom" materially affects rankings.

---

### F5 — Practical Recommendations

For internal benchmarking and leaderboards:

1. Use one fixed external judge for all answer models.
2. Prefer a multi-judge panel using:
   - Mean score aggregation, or
   - Majority voting.
3. Avoid self-judge as the sole metric.
4. With approximately `n = 100` questions:
   - Treat gaps of **3–6 percentage points** as indicative rather than definitive.
   - Prefer bootstrap confidence intervals (CIs) before claiming statistical significance.

---

### F6 — Limitations

| Limitation | Description |
|------------|-------------|
| Sample size | `n = 100` fixed-prefix questions |
| Incomplete matrix | Not all answer × judge combinations are available |
| Judge dependency | Labels are entirely judge-generated |
| No human reference | No human-annotated gold subset is included |
| Historical Qwen runs | Results may reflect degraded provider conditions |
| Prompt sensitivity | Single judge prompt; sensitivity was not ablated |

---

## Design Choices

| Design Choice | Description |
|---------------|-------------|
| Fixed dataset slice | Same first-N SimpleQA items used for all runs |
| Fair comparison | Identical question set across models |
| Explicit separation | Answer model and judge model selected independently |
| Deterministic settings | Temperature `0` used for answer and judge where supported |
| Structured verdicts | Standardized evaluation schema |

### Verdict Schema

| Label | Meaning |
|--------|----------|
| `CORRECT` | Factually correct answer |
| `INCORRECT` | Incorrect or hallucinated answer |
| `NOT_ATTEMPTED` | Abstained from answering |
| `UNPARSEABLE` | Output could not be mapped to a valid label |

---

## Key Takeaway

The benchmark indicates that **judge selection significantly influences observed model rankings**. Self-evaluation behavior differs across models, and accuracy alone does not adequately capture performance. For reliable comparisons, use a fixed external judge or a multi-judge evaluation framework while reporting accuracy, attempted accuracy, hallucination rate, and abstention rate together.
