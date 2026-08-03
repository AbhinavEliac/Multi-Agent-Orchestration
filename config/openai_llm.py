"""
openai_llm.py — OpenAI GPT-4 LLM wrapper.

Provides an OpenAIChatLLM class whose interface matches RotatingChatGroq
so every agent can use it transparently through the shared registry.

Supported models (override via OPENAI_MODEL in .env):
  gpt-4o        — default, best quality, 128K context
  gpt-4-turbo   — faster, slightly cheaper
  gpt-4         — original GPT-4
  gpt-3.5-turbo — fallback / cost saving

Token limits are set per-call via max_tokens, same as Groq.
OpenAI does not have a per-minute rotation requirement — it uses
standard 429 rate-limit handling with exponential back-off.
"""

import logging
import time

from langchain_openai import ChatOpenAI
from openai import RateLimitError, APIStatusError

from config import settings as _settings

logger = logging.getLogger(__name__)

# Default model — can be overridden in .env as OPENAI_MODEL
_DEFAULT_MODEL = "gpt-4o"
_ACTIVE_MODEL  = getattr(_settings, "OPENAI_MODEL", None) or _DEFAULT_MODEL

# Available models in priority order (used if the primary hits a hard cap)
_MODEL_POOL = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]


class OpenAIChatLLM:
    """
    OpenAI LLM wrapper that mirrors the RotatingChatGroq interface.

    Agents call it exactly the same way:
        self.llm = make_llm(max_tokens=700)          # from llm_registry
        response = invoke_with_retry(prompt, self.llm, inputs)

    Internally exposes:
        self._client   — the raw ChatOpenAI instance (used by invoke_with_retry)
        self._invoke_chain(chain, inputs) — rate-limit aware invocation
    """

    def __init__(self, max_tokens: int = 700, model_override: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self._max_tokens  = max_tokens
        self._active_model = model_override or _ACTIVE_MODEL
        self._api_key = api_key or _settings.OPENAI_API_KEY
        self._base_url = base_url
        self._client: ChatOpenAI = self._build()

    def _build(self) -> ChatOpenAI:
        kwargs = {
            "model": self._active_model,
            "temperature": 0.3,
            "max_tokens": self._max_tokens,
            "max_retries": 2,
            "api_key": self._api_key,
            "timeout": 60,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return ChatOpenAI(**kwargs)

    def _invoke_chain(self, chain, inputs: dict):
        """
        Invoke a pre-built chain with OpenAI-specific rate-limit handling.

        429 RateLimitError → exponential back-off up to 120 s then re-raise.
        413 / context-length errors → try falling back to a smaller model.
        5xx errors → short sleep then re-raise.
        """
        backoff = 10.0
        max_wait = 120.0

        for attempt in range(6):
            try:
                return chain.invoke(inputs)

            except RateLimitError as exc:
                if attempt >= 5:
                    raise
                wait = min(backoff * (2 ** attempt), max_wait)
                logger.warning(
                    "OpenAI RateLimitError (attempt %d) — waiting %.0fs.",
                    attempt + 1, wait,
                )
                time.sleep(wait)

            except APIStatusError as exc:
                # Context length exceeded — try a model with larger context
                if exc.status_code in (400, 413):
                    next_model = self._fallback_model()
                    if next_model:
                        logger.warning(
                            "OpenAI %d on %s — falling back to %s.",
                            exc.status_code, self._active_model, next_model,
                        )
                        self._active_model = next_model
                        self._client = self._build()
                        chain = chain.first | self._client  # type: ignore[attr-defined]
                        continue
                if exc.status_code >= 500:
                    time.sleep(10)
                raise

        raise RuntimeError("OpenAI request failed after maximum retries.")

    def _fallback_model(self) -> str | None:
        """Return the next model in the pool after the currently active one."""
        try:
            idx = _MODEL_POOL.index(self._active_model)
            return _MODEL_POOL[idx + 1] if idx + 1 < len(_MODEL_POOL) else None
        except ValueError:
            return _MODEL_POOL[0]
