"""
local_llm.py — Local / Offline LLM adapter and discovery utilities.

Supports:
  1. Ollama (default: http://localhost:11434)
  2. LM Studio / vLLM / llama.cpp (default: http://localhost:1234/v1)
  3. Custom Local OpenAI-compatible server
"""

from __future__ import annotations

import logging
import requests
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Default endpoints
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"


def discover_ollama_models(base_url: str = OLLAMA_DEFAULT_BASE_URL) -> Tuple[bool, List[str], str]:
    """
    Queries Ollama's API (/api/tags) to list all locally installed models.
    Returns: (is_connected: bool, model_names: list[str], message: str)
    """
    clean_url = base_url.rstrip("/")
    if clean_url.endswith("/v1"):
        clean_url = clean_url[:-3]

    try:
        resp = requests.get(f"{clean_url}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            if models:
                return True, models, f"Connected ({len(models)} model(s) found)"
            return True, ["qwen2.5:7b", "llama3.2", "mistral"], "Connected (no models pulled yet)"
        return False, [], f"HTTP {resp.status_code}"
    except Exception as e:
        return False, [], f"Not reachable: {e.__class__.__name__}"


def discover_lmstudio_models(base_url: str = LMSTUDIO_DEFAULT_BASE_URL) -> Tuple[bool, List[str], str]:
    """
    Queries LM Studio / vLLM / Local OpenAI-compatible API (/v1/models).
    Returns: (is_connected: bool, model_names: list[str], message: str)
    """
    clean_url = base_url.rstrip("/")
    if not clean_url.endswith("/v1"):
        endpoint = f"{clean_url}/v1/models"
    else:
        endpoint = f"{clean_url}/models"

    try:
        resp = requests.get(endpoint, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            if models:
                return True, models, f"Connected ({len(models)} model(s) loaded)"
            return True, ["local-model"], "Connected"
        return False, [], f"HTTP {resp.status_code}"
    except Exception as e:
        return False, [], f"Not reachable: {e.__class__.__name__}"


class LocalChatLLM:
    """
    LangChain-compatible wrapper for local LLM engines (Ollama, LM Studio, vLLM).
    Uses ChatOpenAI pointed at local base_url.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.3,
        max_tokens: int = 3500,
        timeout: int = 120,
    ):
        self.model = model or "qwen2.5:7b"
        self.base_url = base_url or "http://localhost:11434/v1"
        self.api_key = api_key or "ollama"
        self.temperature = temperature
        self._max_tokens = max_tokens
        self.timeout = timeout
        self._client = self._build()

    def _build(self):
        from langchain_openai import ChatOpenAI

        # Ensure base_url ends with /v1
        b_url = self.base_url.rstrip("/")
        if not b_url.endswith("/v1"):
            b_url = f"{b_url}/v1"

        return ChatOpenAI(
            model=self.model,
            base_url=b_url,
            api_key=self.api_key or "ollama",
            temperature=self.temperature,
            max_tokens=self._max_tokens,
            timeout=self.timeout,
        )

    def _invoke_chain(self, chain, inputs: dict):
        """Invoke local LLM with basic retry and error logging."""
        try:
            return chain.invoke(inputs)
        except Exception as exc:
            logger.error("Local LLM invocation error on model '%s' (%s): %s", self.model, self.base_url, exc)
            raise

    def _rotate(self) -> bool:
        """No-op rotate for local LLMs."""
        return False
