# src/config.py
"""Central path resolution for the project.

Resolves paths relative to this file so scripts work regardless of cwd.
"""
from pathlib import Path

# src/config.py -> parent = src/ -> parent = portfo/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"

__all__ = ["PROJECT_ROOT", "ENV_PATH", "RESULTS_DIR", "CONFIGS_DIR"]
