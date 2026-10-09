"""Prompt templates for HaluEval and SimpleQA evaluation."""

HALUEVAL_JUDGE_PROMPT = """You are a response judge. Given a knowledge passage, a question, and a response, determine whether the response contains non-factual or hallucinated information.

#Knowledge#: {knowledge}

#Question#: {question}

#Response#: {response}

Does the response contain hallucinated information? Answer ONLY with YES or NO.

#Your Judgment#:"""


SIMPLEQA_ANSWER_PROMPT = """Answer the following question concisely. If you are not sure, say "I don't know."

Question: {problem}

Answer:"""


SIMPLEQA_JUDGE_PROMPT = """You are grading a model's answer against a ground-truth answer.

Question: {problem}
Ground truth: {ground_truth}
Model answer: {model_answer}

Classify the model answer into exactly one category:
- CORRECT: the answer matches the ground truth (allow minor phrasing differences)
- INCORRECT: the answer contradicts the ground truth
- NOT_ATTEMPTED: the model declined, said "I don't know", or gave no factual answer

Respond with ONLY one word: CORRECT, INCORRECT, or NOT_ATTEMPTED."""
