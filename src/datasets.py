"""Dataset loaders for HaluEval and SimpleQA."""
from datasets import load_dataset
from typing import Optional

def load_halueval(config: str = "qa", max_samples: Optional[int] = None):
    """
    Load HaluEval dataset.
    Available configs: qa, dialogue, summarization, general
    Split yang tersedia di HF: "data"
    """
    ds = load_dataset("pminervini/HaluEval", config, split="data")
    
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    return ds


def load_simpleqa(max_samples: int = 200):
    """
    Load SimpleQA (OpenEvals mirror on HF).

    Each row contains:
      - metadata: {answer_type, topic, urls}
      - problem: str
      - answer: str  (ground truth)
    """
    ds = load_dataset("OpenEvals/SimpleQA", "default", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds
