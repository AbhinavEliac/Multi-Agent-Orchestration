"""
chunker.py
"""

from pathlib import Path
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import settings


splitter = RecursiveCharacterTextSplitter(

    chunk_size=settings.CHUNK_SIZE,

    chunk_overlap=settings.CHUNK_OVERLAP,

    separators=[

        "\n\n",

        "\n",

        ". ",

        " ",

        ""

    ]

)


def create_chunks(text):

    return splitter.split_text(text)
