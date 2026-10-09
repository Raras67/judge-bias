"""
Unified OpenAI-compatible client.

Works with ANY router exposing /v1/chat/completions in OpenAI format:
CleanAPIs, OpenRouter, Together, Groq, Fireworks, xAI, local vLLM/Ollama.
"""
import random
import os
import time
from typing import Optional, List

from openai import OpenAI
from dotenv import load_dotenv

# Import the config that resolves paths relative to THIS file.
from src.config import ENV_PATH, PROJECT_ROOT

# Load .env from the project root explicitly, not from cwd.
# override=False means real environment vars win over .env values.
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _debug_env_status() -> str:
    return (
        f"[config] PROJECT_ROOT={PROJECT_ROOT}\n"
        f"[config] ENV_PATH={ENV_PATH} (exists={ENV_PATH.exists()})\n"
        f"[config] ROUTER_BASE_URL={'SET' if os.getenv('ROUTER_BASE_URL') else 'MISSING'}\n"
        f"[config] ROUTER_API_KEY={'SET' if os.getenv('ROUTER_API_KEY') else 'MISSING'}"
    )


class RouterClient:
    """Thin wrapper around OpenAI SDK pointed at a configurable base_url."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or os.getenv("ROUTER_BASE_URL")
        self.api_key = api_key or os.getenv("ROUTER_API_KEY")

        if not self.base_url or not self.api_key:
            # Print full context so user sees exactly what failed
            raise ValueError(
                "RouterClient misconfigured.\n"
                f"{_debug_env_status()}\n"
                "Fix: ensure .env exists at the path above and contains "
                "ROUTER_BASE_URL and ROUTER_API_KEY."
            )

        default_headers = {}
        referer = http_referer or os.getenv("ROUTER_HTTP_REFERER")
        title = x_title or os.getenv("ROUTER_X_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-Title"] = title

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
            default_headers=default_headers or None,
        )

  def generate(self, model, prompt, temperature=0.0, max_tokens=256, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[retry] model={model} attempt={attempt+1}/{max_attempts} "
                  f"sleep={sleep_s:.1f}s err={e}")
            time.sleep(sleep_s)

    raise RuntimeError(
        f"RouterClient failed after {max_attempts} retries for model={model}: {last_error}"
    )

    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception as e:
            print(f"[warn] Could not list models: {e}")
            return []


def get_client() -> RouterClient:
    return RouterClient()


def get_client_legacy(provider: str = None, model: str = None):
    return get_client()
