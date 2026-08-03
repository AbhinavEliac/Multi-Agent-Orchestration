"""
LLM factory with automatic model rotation.

Rotation pool — models with context sufficient for long outputs.
Free-tier TPM limits (per model):
  llama-3.3-70b-versatile : ~6 000 TPM  (use for analysis agents ≤700 output)
  llama-3.1-8b-instant    : ~30 000 TPM (same)

IMPORTANT: aggregator and optimizer request 4 096 output tokens.
With any Groq free-tier model that has an 8 000 TPM window, input must
stay under 3 904 tokens. These two agents use the prose provider
(Gemini / OpenAI) set by the user, NOT Groq, so they are unaffected by
Groq TPM limits. All force_groq=True agents output ≤2 000 tokens and
have input budgets well under 6 000 tokens total.
"""

import logging
import re
import time

from langchain_groq import ChatGroq
from groq import RateLimitError, APIStatusError

from config import settings as _settings

logger = logging.getLogger(__name__)

# ── Rotation pool ──────────────────────────────────────────────────────────
_MODEL_POOL: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Per-model TPM caps (approximate free-tier limits).
# Used to decide whether a 413 should rotate (model genuinely too small)
# or sleep-and-retry (hit per-minute window on a capable model).
_MODEL_TPM: dict[str, int] = {
    "llama-3.3-70b-versatile": 6_000,
    "llama-3.1-8b-instant":    30_000,
}

_configured   = _settings.MODEL
_active_model: str = _configured if _configured in _MODEL_POOL else _MODEL_POOL[0]
_exhausted: set[str] = set()


def _next_model() -> str | None:
    for m in _MODEL_POOL:
        if m not in _exhausted:
            return m
    return None


def _is_tpd(msg: str) -> bool:
    m = msg.lower()
    return "tokens per day" in m or "tpd" in m or "requests per day" in m or "rpd" in m


def _is_tpm(msg: str) -> bool:
    m = msg.lower()
    return "tokens per minute" in m or "tpm" in m or "requests per minute" in m or "rpm" in m



def _parse_wait(msg: str) -> float | None:
    match = re.search(
        r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?",
        msg, re.IGNORECASE,
    )
    if not match or not any(match.groups()):
        return None
    return (
        float(match.group(1) or 0) * 3600
        + float(match.group(2) or 0) * 60
        + float(match.group(3) or 0)
        + 1.0
    )


class RotatingChatGroq:
    """
    Wraps ChatGroq with transparent model rotation on TPD exhaustion.

    Usage in agents:
        self.llm  = make_llm(max_tokens=700)
        ...
        response  = invoke_with_retry(self.prompt, self.llm, inputs)

    invoke_with_retry rebuilds `prompt | self.llm._client` on every call so
    the chain always references the currently active model.
    """

    def __init__(self, max_tokens: int = 700):
        self._max_tokens = max_tokens
        self._client: ChatGroq = self._build()

    def _build(self) -> ChatGroq:
        return ChatGroq(
            model=_active_model,
            temperature=0.3,
            max_tokens=self._max_tokens,
            max_retries=2,
            timeout=60,
        )

    def _rotate(self) -> bool:
        global _active_model, _exhausted
        _exhausted.add(_active_model)
        logger.warning("Model %s TPD exhausted — rotating.", _active_model)
        nxt = _next_model()
        if nxt is None:
            return False
        _active_model = nxt
        self._client = self._build()
        logger.info("Active model → %s", _active_model)
        return True

    def _invoke_chain(self, chain, inputs: dict):
        """
        Invoke a pre-built chain with rotation + retry logic.
        Called by invoke_with_retry after it builds chain = prompt | self._client.
        """
        tpm_attempt = 0
        tpm_base = 15.0

        while True:
            try:
                return chain.invoke(inputs)

            except RateLimitError as exc:
                msg = str(exc)

                if _is_tpd(msg):
                    rotated = self._rotate()
                    if not rotated:
                        from config.llm_registry import get_provider, _is_valid_gemini_key, _is_valid_openai_key
                        provider = get_provider()
                        if provider == "groq":
                            if _is_valid_gemini_key(_settings.GEMINI_API_KEY):
                                provider = "gemini"
                            elif _is_valid_openai_key(_settings.OPENAI_API_KEY):
                                provider = "openai"
                        if provider != "groq":
                            logger.warning("Groq daily quota exhausted. Falling back dynamically to '%s'.", provider)
                            if provider == "openai":
                                from config.openai_llm import OpenAIChatLLM
                                fallback_llm = OpenAIChatLLM(max_tokens=self._max_tokens)
                            else: # gemini
                                from config.gemini_llm import GeminiChatLLM
                                fallback_llm = GeminiChatLLM(max_tokens=self._max_tokens)
                            self._client = fallback_llm._client
                            self._invoke_chain = fallback_llm._invoke_chain
                            raise _RotationOccurred() from exc
                        raise RuntimeError(
                            "All Groq models have exhausted their daily quota. "
                            "Try again tomorrow or upgrade your plan."
                        ) from exc
                    # Rebuild the chain with the new model and retry
                    chain = chain.first | self._client  # type: ignore[attr-defined]
                    # Simpler: let invoke_with_retry handle the rebuild on next call
                    # by raising a sentinel that it catches
                    raise _RotationOccurred() from exc

                elif _is_tpm(msg):
                    tpm_attempt += 1
                    if tpm_attempt > 3:
                        logger.warning("Max TPM retries (3) exceeded for model %s. Attempting model rotation or provider fallback.", _active_model)
                        rotated = self._rotate()
                        if not rotated:
                            from config.llm_registry import get_provider, _is_valid_gemini_key, _is_valid_openai_key
                            provider = get_provider()
                            if provider == "groq":
                                if _is_valid_gemini_key(_settings.GEMINI_API_KEY):
                                    provider = "gemini"
                                elif _is_valid_openai_key(_settings.OPENAI_API_KEY):
                                    provider = "openai"
                            if provider != "groq":
                                logger.warning("Groq TPM limit reached on all models. Falling back dynamically to '%s'.", provider)
                                if provider == "openai":
                                    from config.openai_llm import OpenAIChatLLM
                                    fallback_llm = OpenAIChatLLM(max_tokens=self._max_tokens)
                                else: # gemini
                                    from config.gemini_llm import GeminiChatLLM
                                    fallback_llm = GeminiChatLLM(max_tokens=self._max_tokens)
                                self._client = fallback_llm._client
                                self._invoke_chain = fallback_llm._invoke_chain
                                raise _RotationOccurred() from exc
                            raise RuntimeError(
                                "Request size or rate exceeds Groq free tier model limits. "
                                "Please select Gemini or OpenAI as your LLM Provider in settings."
                            ) from exc
                        raise _RotationOccurred() from exc

                    wait = _parse_wait(msg) or min(tpm_base * (2 ** (tpm_attempt - 1)), 120)
                    logger.warning("TPM limit (attempt %d/3) — waiting %.1fs.", tpm_attempt, wait)
                    time.sleep(wait)
                    # chain is still valid, retry same model

                else:
                    raise

            except APIStatusError as exc:
                if exc.status_code == 413:
                    msg = str(exc)
                    logger.warning("HTTP 413 on %s: %s. Attempting model rotation or provider fallback.", _active_model, msg)
                    rotated = self._rotate()
                    if not rotated:
                        from config.llm_registry import get_provider, _is_valid_gemini_key, _is_valid_openai_key
                        provider = get_provider()
                        if provider == "groq":
                            if _is_valid_gemini_key(_settings.GEMINI_API_KEY):
                                provider = "gemini"
                            elif _is_valid_openai_key(_settings.OPENAI_API_KEY):
                                provider = "openai"
                        if provider != "groq":
                            logger.warning("Groq 413 limit reached on all models. Falling back dynamically to '%s'.", provider)
                            if provider == "openai":
                                from config.openai_llm import OpenAIChatLLM
                                fallback_llm = OpenAIChatLLM(max_tokens=self._max_tokens)
                            else: # gemini
                                from config.gemini_llm import GeminiChatLLM
                                fallback_llm = GeminiChatLLM(max_tokens=self._max_tokens)
                            self._client = fallback_llm._client
                            self._invoke_chain = fallback_llm._invoke_chain
                            raise _RotationOccurred() from exc
                        raise RuntimeError(
                            "Request size exceeds the 8,000 TPM limit of Groq free tier models. "
                            "Please select Gemini or OpenAI as your LLM Provider in settings."
                        ) from exc
                    raise _RotationOccurred() from exc

                if exc.status_code < 500:
                    raise
                time.sleep(10)
                raise


class _RotationOccurred(Exception):
    """Internal sentinel: rotation happened, caller should rebuild chain and retry."""


def make_llm(max_tokens: int = 700) -> RotatingChatGroq:
    """
    Return a RotatingChatGroq with the given output token budget.

    Budgets by role:
      specialist agents (language/facts/structure/seo/geo) : 700
      aggregator (editorial brief)                         : 800
      evaluator  (JSON scores)                             : 500
      planner / learner (JSON)                             : 600
      researcher (research brief)                          : 2000
      optimizer  (full article)                            : 3000
    """
    return RotatingChatGroq(max_tokens=max_tokens)


# Shared default for agents that `from config.llm import llm`
llm = make_llm(max_tokens=700)
