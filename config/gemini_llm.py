"""
gemini_llm.py — Google Gemini LLM wrapper.

Uses langchain-google-genai which is the official LangChain integration
for Google's Gemini API.  Interface mirrors RotatingChatGroq so all agents
use it transparently through llm_registry.make_llm().

Free tier (as of 2026):
  gemini-2.0-flash  — 15 RPM, 250 000 TPM, 1 500 RPD  (no credit card)
  gemini-1.5-flash  — 15 RPM, 250 000 TPM, 1 500 RPD  (fallback)

Install:
  pip install langchain-google-genai

Key must be set in .env:
  GEMINI_API_KEY=<your key from https://aistudio.google.com/app/apikey>
  GEMINI_MODEL=gemini-2.0-flash   # optional — this is the default
"""

from __future__ import annotations

import logging
import time

from config import settings as _settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL  = "gemini-2.0-flash"
_FALLBACK_CHAIN = ["gemini-2.0-flash-lite", "gemini-1.5-flash"]
_ACTIVE_MODEL   = getattr(_settings, "GEMINI_MODEL", None) or _DEFAULT_MODEL


class GeminiChatLLM:
    """
    Google Gemini wrapper matching the RotatingChatGroq / OpenAIChatLLM interface.

    Exposes:
        self._client          — raw LangChain ChatGoogleGenerativeAI instance
        self._invoke_chain()  — rate-limit-aware invocation
    """

    def __init__(self, max_tokens: int = 700):
        self._max_tokens    = max_tokens
        self._active_model  = _ACTIVE_MODEL
        self._client        = self._build()

    def _build(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self._active_model,
            google_api_key=_settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=self._max_tokens,
            # Disable safety blocks that would reject blog content
            convert_system_message_to_human=True,
            timeout=60,
        )

    def _invoke_chain(self, chain, inputs: dict):
        """
        Invoke with Gemini-specific error handling.

        429 / ResourceExhausted  → wait then retry, then try next fallback model.
        Model not found / invalid → immediately try next fallback in chain.
        5xx server errors        → short sleep then re-raise.
        """
        backoff = 15.0
        for attempt in range(5):
            try:
                return chain.invoke(inputs)

            except Exception as exc:
                msg = str(exc).lower()

                is_rate = (
                    "429" in msg
                    or "resource_exhausted" in msg
                    or "quota" in msg
                    or "rate limit" in msg
                )
                if is_rate:
                    is_daily = (
                        "limit: 0" in msg
                        or "per day" in msg
                        or "daily" in msg
                    )
                    if is_daily:
                        # Try next fallback model, but if no more models, fail immediately
                        if self._try_next_fallback():
                            chain = chain.first | self._client  # type: ignore
                            continue
                        raise RuntimeError(
                            "All Gemini models have exhausted their daily quota. "
                            "Please upgrade your plan or try again tomorrow."
                        ) from exc

                    if attempt >= 4:
                        raise
                    # Try next fallback model before sleeping
                    if self._try_next_fallback():
                        chain = chain.first | self._client  # type: ignore
                        continue
                    wait = min(backoff * (2 ** attempt), 90)
                    logger.warning(
                        "Gemini quota exhausted on all models (attempt %d) — waiting %.0fs.",
                        attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue

                is_model_err = (
                    "not found" in msg
                    or "invalid model" in msg
                    or "model_not_found" in msg
                )
                if is_model_err:
                    if self._try_next_fallback():
                        chain = chain.first | self._client  # type: ignore
                        continue

                if "500" in msg or "503" in msg or "internal" in msg:
                    time.sleep(10)

                raise

        raise RuntimeError("Gemini request failed after maximum retries on all fallback models.")

    def _try_next_fallback(self) -> bool:
        """Switch to the next model in the fallback chain. Returns True if switched."""
        try:
            idx = _FALLBACK_CHAIN.index(self._active_model)
            next_idx = idx + 1
        except ValueError:
            next_idx = 0

        if next_idx >= len(_FALLBACK_CHAIN):
            return False

        next_model = _FALLBACK_CHAIN[next_idx]
        logger.warning("Gemini switching from %s → %s.", self._active_model, next_model)
        self._active_model = next_model
        self._client = self._build()
        return True
