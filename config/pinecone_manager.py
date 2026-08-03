"""
pinecone_manager.py

Creates or retrieves the Pinecone index.

Auto-recreation: if an existing index was built with a different dimension
(e.g. 3072 for OpenAI but now using Gemini at 768), the old index is deleted
and a new one is created with the correct dimension. All previously ingested
vectors are lost — the next ingest_blog() call will repopulate it.
"""

import logging

from pinecone import Pinecone, ServerlessSpec

from config import settings

logger = logging.getLogger(__name__)

pc         = Pinecone(api_key=settings.PINECONE_API_KEY)
INDEX_NAME = settings.PINECONE_INDEX_NAME
DIMENSION  = settings.EMBEDDING_DIMENSION


def get_or_create_index():
    existing = {i.name: i for i in pc.list_indexes()}

    if INDEX_NAME in existing:
        index_info = existing[INDEX_NAME]
        # Check dimension match — access via dict-style or attribute
        try:
            stored_dim = (
                index_info.dimension                          # pinecone-client ≥ 3
                if hasattr(index_info, "dimension")
                else index_info["dimension"]                  # older client
            )
        except Exception:
            stored_dim = None

        if stored_dim is not None and int(stored_dim) != int(DIMENSION):
            logger.warning(
                "Pinecone index '%s' has dimension %d but current embedding "
                "model requires %d. Deleting and recreating the index.",
                INDEX_NAME, stored_dim, DIMENSION,
            )
            pc.delete_index(INDEX_NAME)
        else:
            return pc.Index(INDEX_NAME)

    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return pc.Index(INDEX_NAME)
