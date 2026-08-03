"""
invoke_with_retry — the single call-site for all agent LLM calls.

Works with both RotatingChatGroq (config/llm.py) and OpenAIChatLLM
(config/openai_llm.py). Both expose the same interface:
  - ._client          : raw LangChain chat model (used to build the chain)
  - ._invoke_chain()  : rate-limit-aware invocation

Rebuilds the chain on every call so model rotation is always reflected.
Handles the _RotationOccurred sentinel from RotatingChatGroq by rebuilding
the chain with the new active model and retrying.
"""

import logging

logger = logging.getLogger(__name__)

_MAX_ROTATIONS = 3  # maximum model switches before giving up


def invoke_with_retry(prompt, llm, inputs: dict):
    """
    Build a fresh chain from prompt + llm._client and invoke it.

    Compatible with any LLM wrapper that exposes:
      - llm._client          (the raw LangChain chat model)
      - llm._invoke_chain()  (rate-limit + rotation aware invoke)

    On TPD/rotation exhaustion (Groq):
      - RotatingChatGroq raises _RotationOccurred
      - invoke_with_retry catches it, rebuilds the chain, retries on
        the new model — up to _MAX_ROTATIONS times

    On TPM exhaustion (Groq):
      - RotatingChatGroq sleeps internally and retries — nothing to do here

    On 429 rate-limit (OpenAI):
      - OpenAIChatLLM handles exponential back-off internally

    Args:
        prompt:   LangChain ChatPromptTemplate
        llm:      RotatingChatGroq or OpenAIChatLLM instance from make_llm()
        inputs:   Dict of template variables

    Returns:
        LangChain AIMessage
    """
    # Import lazily to avoid circular imports; _RotationOccurred only exists
    # in the Groq path, so we guard with a fallback sentinel.
    try:
        from config.llm import _RotationOccurred
    except ImportError:
        _RotationOccurred = type("_RotationOccurred", (Exception,), {})

    for rotation_attempt in range(_MAX_ROTATIONS + 1):
        chain = prompt | llm._client
        try:
            response = llm._invoke_chain(chain, inputs)
            try:
                from utilis.token_counter import add_tokens
                usage = getattr(response, "usage_metadata", None)
                if isinstance(usage, dict):
                    p_tok = usage.get("input_tokens", 0)
                    c_tok = usage.get("output_tokens", 0)
                    add_tokens(p_tok, c_tok)
                else:
                    p_tok = len(str(inputs)) // 4
                    c_tok = len(getattr(response, "content", "")) // 4
                    add_tokens(p_tok, c_tok)
            except Exception:
                pass
            return response
        except _RotationOccurred:
            if rotation_attempt >= _MAX_ROTATIONS:
                raise RuntimeError(
                    "All Groq models exhausted their daily quota. "
                    "Try again tomorrow or upgrade your plan."
                )
            logger.info(
                "Model rotated (attempt %d/%d). Rebuilding chain.",
                rotation_attempt + 1, _MAX_ROTATIONS,
            )
            # Loop: chain will be rebuilt with llm._client pointing
            # to the new active model at the top of the next iteration.
