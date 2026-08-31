from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    found = find_dotenv()
    if found:
        ENV_PATH = Path(found)

load_dotenv(ENV_PATH, override=True)


def _get_env(*names, default=None):
    load_dotenv(ENV_PATH, override=True)
    for name in names:
        value = os.getenv(name)

        if value is not None and value != "":
            val = str(value).strip().strip('"').strip("'")
            if val.startswith("-") and not val.startswith("-sk-"):
                val = val.lstrip("-")
            return val

    try:
        if ENV_PATH.exists():
            with open(ENV_PATH, "r", encoding="utf-8") as env_file:
                for line in env_file:
                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)

                    if key.strip() in names:
                        val = value.strip().strip('"').strip("'")
                        if val.startswith("-") and not val.startswith("-sk-"):
                            val = val.lstrip("-")
                        return val
    except OSError:
        pass

    return default

class Settings:

    @property
    def GROQ_API_KEY(self):
        return _get_env("GROQ_API_KEY")

    @property
    def OPENAI_API_KEY(self):
        return _get_env("OPENAI_API_KEY")

    @property
    def PINECONE_API_KEY(self):
        return _get_env("PINECONE_API_KEY")

    @property
    def PINECONE_INDEX_NAME(self):
        return _get_env("PINECONE_INDEX_NAME")

    @property
    def TAVILY_API_KEY(self):
        return _get_env("TAVILY_API_KEY")

    @property
    def FIRECRAWL_API_KEY(self):
        return _get_env("FIRECRAWL_API_KEY")

    @property
    def NEMOTRON_MODEL(self):
        return _get_env("NEMOTRON_MODEL", default="NEMOTRON MODEL")

    @property
    def NEMOTRON_API_KEY(self):
        return _get_env("NEMOTRON_API_KEY")

    @property
    def NEMOTRON_BASE_URL(self):
        return _get_env("NEMOTRON_BASE_URL", default="https://openrouter.ai/api/v1")

    MODEL = _get_env("GROQ_MODEL", "MODEL", default="llama-3.1-8b-instant")

    OPENAI_MODEL = _get_env("OPENAI_MODEL", default="gpt-4o")

    GEMINI_API_KEY = _get_env("GEMINI_API_KEY")

    GEMINI_MODEL = _get_env("GEMINI_MODEL", default="gemini-2.0-flash")

    EMBEDDING_MODEL = _get_env(
        "EMBEDDING_MODEL",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    # EMBEDDING_PROVIDER controls which embedding backend is used.
    # Supported values: "openai" | "gemini" | "huggingface"
    # Default: auto-detected from EMBEDDING_MODEL name.
    EMBEDDING_PROVIDER = _get_env("EMBEDDING_PROVIDER", default="auto")

    EMBEDDING_DIMENSION = int(_get_env("EMBEDDING_DIMENSION", default="384"))

    CHUNK_SIZE = int(_get_env("CHUNK_SIZE", default="1000"))

    CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", default="200"))

    TOP_K = int(_get_env("TOP_K", default="5"))

settings = Settings()
