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


