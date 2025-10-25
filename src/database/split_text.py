import os
import sys
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50

def split_text(documents: List[Document]) -> List[Document]:
    print("Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    print(f"Initial chunks: {len(chunks)}")

    valid_chunks = [
        c for c in chunks 
        if c.page_content.strip() and len(c.page_content) > MIN_CHUNK_LENGTH
    ]
    print(f"Valid chunks: {len(valid_chunks)}\n")

    if not valid_chunks:
        print("Error: No valid chunks generated")
        sys.exit(1)

    return valid_chunks