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

_DEFAULT_MODEL  = "gemini-3.5-flash-lite"
_FALLBACK_CHAIN = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

_raw_gemini_model = getattr(_settings, "GEMINI_MODEL", None)
if _raw_gemini_model and _raw_gemini_model.strip() in (
    "gemini-2.0-flash", "gemini-2.0-flash-exp", "models/gemini-2.0-flash",
    "gemini-2.5-flash-lite", "models/gemini-2.5-flash-lite"
):
    logger.warning("Deprecated Gemini model '%s' specified. Migrating to '%s'.", _raw_gemini_model, _DEFAULT_MODEL)
    _ACTIVE_MODEL = _DEFAULT_MODEL
else:
    _ACTIVE_MODEL = _raw_gemini_model or _DEFAULT_MODEL


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
        model_name = self._active_model
        if model_name.startswith("models/"):
            model_name = model_name[len("models/"):]
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=_settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=self._max_tokens,
            # Disable safety blocks that would reject blog content
            convert_system_message_to_human=True,
            timeout=60,
        )

    def _invoke_chain(self, chain, inputs: dict):
        """
        Invoke with Gemini-specific error handling and dynamic model fallback.

        429 / ResourceExhausted   → try next fallback model in chain, then backoff.
        404 / Model not found     → immediately switch to next fallback in chain.
        5xx server errors         → short sleep then try next fallback.
        """
        backoff = 10.0
        prompt_template = None
        if hasattr(chain, "steps") and len(chain.steps) > 0:
            prompt_template = chain.steps[0]

        for attempt in range(len(_FALLBACK_CHAIN) * 2):
            try:
                if prompt_template is not None:
                    runnable = prompt_template | self._client
                else:
                    runnable = chain
                return runnable.invoke(inputs)

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
                        if self._try_next_fallback():
                            continue
                        raise RuntimeError(
                            "All Gemini models have exhausted their daily quota. "
                            "Please upgrade your plan or try again tomorrow."
                        ) from exc

                    if self._try_next_fallback():
                        continue
                    wait = min(backoff * (2 ** (attempt % 3)), 60)
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
                    or "404" in msg
                    or "no longer available" in msg
                )
                if is_model_err:
                    if self._try_next_fallback():
                        continue

                if "500" in msg or "503" in msg or "internal" in msg:
                    time.sleep(5)
                    if self._try_next_fallback():
                        continue

                raise

        raise RuntimeError("Gemini request failed after maximum retries on all fallback models.")

    def _try_next_fallback(self) -> bool:
        """Switch to the next model in the fallback chain. Returns True if switched."""
        try:
            curr = self._active_model
            if curr.startswith("models/"):
                curr = curr[len("models/"):]
            clean_chain = [m[len("models/"):] if m.startswith("models/") else m for m in _FALLBACK_CHAIN]
            if curr in clean_chain:
                next_idx = clean_chain.index(curr) + 1
            else:
                next_idx = 0
        except Exception:
            next_idx = 0

        if next_idx >= len(_FALLBACK_CHAIN):
            return False

        next_model = _FALLBACK_CHAIN[next_idx]
        logger.warning("Gemini switching from %s → %s.", self._active_model, next_model)
        self._active_model = next_model
        self._client = self._build()
        return True

    def _rotate(self) -> bool:
        """Alias for _try_next_fallback for uniform streaming rotation."""
        return self._try_next_fallback()

