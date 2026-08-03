"""
retriever.py
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_pinecone import PineconeVectorStore

from config.embeddings import embeddings

from config.pinecone_manager import get_or_create_index

from config import settings


index = get_or_create_index()


vectorstore = PineconeVectorStore(

    index=index,

    embedding=embeddings

)


retriever = vectorstore.as_retriever(

    search_kwargs={

        "k": settings.TOP_K

    }

)


def retrieve(query):

    return retriever.invoke(query)