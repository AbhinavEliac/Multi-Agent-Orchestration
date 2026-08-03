"""
ingestion.py
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import settings

from langchain_core.documents import Document

from langchain_pinecone import PineconeVectorStore

from config.embeddings import embeddings

from config.pinecone_manager import get_or_create_index


index = get_or_create_index()


vectorstore = PineconeVectorStore(

    index=index,

    embedding=embeddings

)


def ingest_blog(
    chunks,
    url
):
    if not chunks:
        return
    docs = []

    for i, chunk in enumerate(chunks):

        docs.append(

            Document(

                page_content=chunk,

                metadata={

                    "url": url,

                    "chunk": i

                }

            )

        )

    vectorstore.add_documents(docs)