"""
llm_registry.py — single import point for ALL agents.

Mixed-provider routing
──────────────────────
The UI-selected provider (Groq / Gemini / OpenAI) is the PROSE provider.
It is used only for aggregator and optimizer — the two agents that write
long-form article text and benefit most from a high-quality model.

All other agents (evaluators, specialists, researchers, learner, planner)
always run on Groq regardless of the UI selection.  This has two benefits:

  1. Token cost: Gemini/OpenAI free-tier quota is preserved for prose only.
     A typical run uses ~9 000 tokens on the prose provider instead of
     ~175 000 if every agent used it.

  2. Speed: Groq is faster for short structured outputs (JSON scores,
     review notes, research briefs) than Gemini Flash or GPT-4o.

Usage in agents:
  # Prose agent — uses whatever provider the user selected
  llm = make_llm(max_tokens=4096)

  # Analysis agent — always Groq, fast + free
  llm = make_llm(max_tokens=700, force_groq=True)
"""

from __future__ import annotations

import threading
import os
import logging
import time

_lock     = threading.Lock()
_provider = "groq"           # changed by set_provider() from Streamlit


# ── Token budgets ──────────────────────────────────────────────────────────────
_GROQ_MAX_TOKENS: dict[str, int] = {
    "small":  600,
    "medium": 1200,
    "large":  4096,
}
_OPENAI_MODEL: dict[str, str] = {
    "small":  "gpt-4o-mini",
    "medium": "gpt-4o-mini",
    "large":  "gpt-4o",
}
_OPENAI_MAX_TOKENS: dict[str, int] = {
    "small":  600,
    "medium": 1200,
    "large":  4096,
}


# ── Provider management ────────────────────────────────────────────────────────

_custom_model_name = ""
_custom_api_key = ""
_custom_base_url = ""


def set_custom_llm_settings(model_name: str, api_key: str, base_url: str) -> None:
    global _custom_model_name, _custom_api_key, _custom_base_url
    with _lock:
        _custom_model_name = model_name
        _custom_api_key = api_key
        _custom_base_url = base_url


def set_provider(name: str) -> None:
    """
    Switch the active prose provider globally.
    Called once from prepare_blog before the graph runs.
    Also clears OPENAI_API_KEY from the environment when not using OpenAI
    so LangChain internals don't auto-validate it and hit quota.
    """
    global _provider
    allowed = {"groq", "openai", "gemini", "custom"}
    name = name.lower().strip()
    if name not in allowed:
        raise ValueError(f"Unknown LLM provider '{name}'. Choose from: {allowed}")
    with _lock:
        _provider = name
    if name not in ("openai", "custom"):
        os.environ.pop("OPENAI_API_KEY", None)


def _is_valid_gemini_key(key: str | None) -> bool:
    if not key: return False
    k = key.strip()
    return k.startswith("AIzaSy") and len(k) >= 30

def _is_valid_openai_key(key: str | None) -> bool:
    if not key: return False
    k = key.strip()
    return k.startswith("sk-") and len(k) >= 30

def get_provider() -> str:
    with _lock:
        return _provider


# ── LLM factory ───────────────────────────────────────────────────────────────

def make_llm(max_tokens: int | None = None, size: str = "large", force_groq: bool = False):
    """
    Factory used by every agent.

    Args:
        max_tokens:  Explicit token budget. Overrides size preset if provided.
        size:        "small" | "medium" | "large" — role-based token preset.
        force_groq:  If True, always return a RotatingChatGroq instance
                     regardless of the globally selected provider.
                     Use this for every agent that does NOT write long-form
                     prose (evaluators, specialists, researchers, etc.)
                     to preserve Gemini/OpenAI quota for the aggregator
                     and optimizer only.

    Returns:
        RotatingChatGroq | OpenAIChatLLM | GeminiChatLLM
    """
    with _lock:
        provider = _provider

    size = size if size in ("small", "medium", "large") else "large"

    # Analysis agents always use Groq — fast, free, no quota pressure
    use_groq = force_groq or provider == "groq"
    if use_groq:
        import logging
        from config.llm import _MODEL_POOL, _exhausted
        from config import settings
        
        fallback_provider = provider
        if fallback_provider == "groq":
            if _is_valid_gemini_key(settings.GEMINI_API_KEY):
                fallback_provider = "gemini"
            elif _is_valid_openai_key(settings.OPENAI_API_KEY):
                fallback_provider = "openai"
                
        if len(_exhausted) >= len(_MODEL_POOL) and fallback_provider != "groq":
            logging.getLogger(__name__).warning("Groq is fully exhausted. Falling back to '%s' pre-emptively.", fallback_provider)
            use_groq = False
            provider = fallback_provider

    if use_groq:
        from config.llm import RotatingChatGroq
        tokens = max_tokens if max_tokens is not None else _GROQ_MAX_TOKENS[size]
        return RotatingChatGroq(max_tokens=tokens)

    if provider == "custom":
        from config.openai_llm import OpenAIChatLLM
        tokens = max_tokens if max_tokens is not None else _OPENAI_MAX_TOKENS[size]
        with _lock:
            model = _custom_model_name
            api_key = _custom_api_key
            base_url = _custom_base_url
        return OpenAIChatLLM(max_tokens=tokens, model_override=model, api_key=api_key, base_url=base_url)

    if provider == "openai":
        from config.openai_llm import OpenAIChatLLM
        tokens = max_tokens if max_tokens is not None else _OPENAI_MAX_TOKENS[size]
        model  = _OPENAI_MODEL[size]
        return OpenAIChatLLM(max_tokens=tokens, model_override=model)

    # gemini — prose provider
    from config.gemini_llm import GeminiChatLLM
    tokens = max_tokens if max_tokens is not None else _GROQ_MAX_TOKENS[size]
    return GeminiChatLLM(max_tokens=tokens)


# ── Streaming invoke ───────────────────────────────────────────────────────────

def stream_invoke(prompt, llm, inputs: dict, chunk_callback) -> str:
    """
    Streaming invoke used by aggregator and optimizer.
    Handles retries, model rotation, and dynamic provider fallbacks for RotatingChatGroq.
    """
    from utilis.retry import invoke_with_retry
    try:
        from config.llm import _RotationOccurred
    except ImportError:
        _RotationOccurred = type("_RotationOccurred", (Exception,), {})
    try:
        from groq import RateLimitError, APIStatusError as GroqAPIStatusError
    except ImportError:
        RateLimitError = GroqAPIStatusError = None

    tpm_attempt = 0
    tpm_base = 15.0

    while True:
        try:
            chain     = prompt | llm._client
            full_text = ""
            for chunk in chain.stream(inputs):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    chunk_callback(token)
                    full_text += token
            try:
                from utilis.token_counter import add_tokens
                p_tok = len(str(inputs)) // 4
                c_tok = len(full_text) // 4
                add_tokens(p_tok, c_tok)
            except Exception:
                pass
            return full_text

        except Exception as exc:
            err_msg = str(exc).lower()

            if isinstance(exc, _RotationOccurred):
                continue

            if hasattr(llm, "_rotate") and hasattr(llm, "_invoke_chain"):
                if RateLimitError and isinstance(exc, RateLimitError):
                    from config.llm import _is_tpd, _is_tpm, _parse_wait
                    if _is_tpd(err_msg):
                        rotated = llm._rotate()
                        if not rotated:
                            provider = get_provider()
                            if provider == "groq":
                                from config import settings
                                if settings.GEMINI_API_KEY:
                                    provider = "gemini"
                                elif settings.OPENAI_API_KEY:
                                    provider = "openai"
                            if provider != "groq":
                                logging.getLogger(__name__).warning("Groq daily quota exhausted. Falling back dynamically to '%s'.", provider)
                                if provider == "openai":
                                    from config.openai_llm import OpenAIChatLLM
                                    fallback_llm = OpenAIChatLLM(max_tokens=getattr(llm, "_max_tokens", 3500))
                                else:
                                    from config.gemini_llm import GeminiChatLLM
                                    fallback_llm = GeminiChatLLM(max_tokens=getattr(llm, "_max_tokens", 3500))
                                llm._client = fallback_llm._client
                                llm._invoke_chain = fallback_llm._invoke_chain
                                continue
                            raise RuntimeError("All Groq models have exhausted their daily quota. Try again tomorrow or upgrade your plan.") from exc
                        continue
                    elif _is_tpm(err_msg):
                        tpm_attempt += 1
                        wait = _parse_wait(err_msg) or min(tpm_base * (2 ** (tpm_attempt - 1)), 120)
                        logging.getLogger(__name__).warning("TPM limit in streaming — waiting %.1fs.", wait)
                        time.sleep(wait)
                        continue

                if GroqAPIStatusError and isinstance(exc, GroqAPIStatusError) and getattr(exc, "status_code", None) == 413:
                    logging.getLogger(__name__).warning("413 on Groq in streaming. Attempting model rotation or provider fallback.")
                    rotated = llm._rotate()
                    if not rotated:
                        provider = get_provider()
                        if provider == "groq":
                            from config import settings
                            if settings.GEMINI_API_KEY:
                                provider = "gemini"
                            elif settings.OPENAI_API_KEY:
                                provider = "openai"
                        if provider != "groq":
                            logging.getLogger(__name__).warning("413 on Groq: all Groq models exhausted/too small. Falling back dynamically to '%s'.", provider)
                            if provider == "openai":
                                from config.openai_llm import OpenAIChatLLM
                                fallback_llm = OpenAIChatLLM(max_tokens=getattr(llm, "_max_tokens", 3500))
                            else:
                                from config.gemini_llm import GeminiChatLLM
                                fallback_llm = GeminiChatLLM(max_tokens=getattr(llm, "_max_tokens", 3500))
                            llm._client = fallback_llm._client
                            llm._invoke_chain = fallback_llm._invoke_chain
                            continue
                        raise RuntimeError("Request size exceeds the 8,000 TPM limit of Groq free tier models. Please select Gemini or OpenAI as your LLM Provider in settings.") from exc
                    continue

            # Fall back for streaming-not-supported type errors
            if any(k in err_msg for k in ("stream", "not support", "chunk")):
                response = invoke_with_retry(prompt, llm, inputs)
                text     = response.content if hasattr(response, "content") else str(response)
                chunk_callback(text)
                return text

            raise
