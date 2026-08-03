"""
embeddings.py — unified embedding provider routing.

Supported providers (set EMBEDDING_PROVIDER in .env):
  openai      — OpenAI text-embedding-3-large / text-embedding-3-small / ada-002
                Requires: OPENAI_API_KEY
                Dimensions: 3072 (3-large), 1536 (3-small / ada-002)

  gemini      — Google text-embedding-004  (free tier, no credit card)
                Requires: GEMINI_API_KEY
                Dimensions: 768
                Free limits: 1500 RPD, 100 RPM

  huggingface — Local sentence-transformers (no API key, runs on CPU)
                Default model: sentence-transformers/all-MiniLM-L6-v2
                Dimensions: 384
                Cost: free, but slower on first run (downloads model)

EMBEDDING_PROVIDER=auto (default):
  Detects from EMBEDDING_MODEL name:
    text-embedding-*  → openai
    models/text-embedding-* or gemini-embedding-* → gemini
    anything else     → huggingface

IMPORTANT — Pinecone index dimension must match:
  If you change provider/model, the existing Pinecone index must be
  recreated with the correct dimension. The system auto-recreates it
  if the dimension doesn't match.

Recommended .env for each provider:
  # OpenAI (paid)
  EMBEDDING_PROVIDER=openai
  EMBEDDING_MODEL=text-embedding-3-large
  EMBEDDING_DIMENSION=3072

  # Gemini (free)
  EMBEDDING_PROVIDER=gemini
  EMBEDDING_MODEL=models/text-embedding-004
  EMBEDDING_DIMENSION=768

  # HuggingFace (free, local)
  EMBEDDING_PROVIDER=huggingface
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  EMBEDDING_DIMENSION=384
"""

from __future__ import annotations
from config import settings

# ── Provider auto-detection ───────────────────────────────────────────────────
_OPENAI_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
}

_GEMINI_MODELS = {
    "models/gemini-embedding-001",
    "models/gemini-embedding-2-preview",
    "models/gemini-embedding-2",
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
    "gemini-embedding-2",
    # Legacy v1beta names (no longer valid but kept for detection)
    "models/text-embedding-004",
    "text-embedding-004",
    "gemini-embedding-exp-03-07",
}


def _detect_provider(model: str) -> str:
    """Infer provider from model name when EMBEDDING_PROVIDER=auto."""
    m = model.lower()
    if m in {x.lower() for x in _OPENAI_MODELS} or m.startswith("text-embedding-3") or m == "text-embedding-ada-002":
        return "openai"
    if (m in {x.lower() for x in _GEMINI_MODELS}
            or "gemini-embedding" in m
            or m.startswith("models/gemini")):
        return "gemini"
    return "huggingface"


_raw_provider = (settings.EMBEDDING_PROVIDER or "auto").strip().lower()
_model        = settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"

if _raw_provider == "auto":
    _provider = _detect_provider(_model)
else:
    _provider = _raw_provider


# ── Build the embeddings object ───────────────────────────────────────────────

if _provider == "openai":
    if not getattr(settings, "OPENAI_API_KEY", None):
        raise ValueError(
            "EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY in .env. "
            "Switch to EMBEDDING_PROVIDER=gemini or huggingface to avoid this."
        )
    import os
    from langchain_openai import OpenAIEmbeddings
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(
        model=_model,
        openai_api_key=settings.OPENAI_API_KEY,
    )

elif _provider == "gemini":
    if not getattr(settings, "GEMINI_API_KEY", None):
        raise ValueError(
            "EMBEDDING_PROVIDER=gemini requires GEMINI_API_KEY in .env. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    # Ensure the model has the models/ prefix — required by the API
    _gem_model = _model if _model.startswith("models/") else f"models/{_model.split('/')[-1]}"
    embeddings = GoogleGenerativeAIEmbeddings(
        model=_gem_model,
        google_api_key=settings.GEMINI_API_KEY,
    )

else:
    # HuggingFace local — no API key needed
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=_model)
