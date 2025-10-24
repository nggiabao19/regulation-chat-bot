"""
Database creation script for RAG-based CTU Regulations chatbot.
Creates a vector database from PDF document using OCR and text cleaning.
"""
import os
import sys
import shutil
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from pdf2image import convert_from_path

from get_embedding_function import get_embedding_function
from load_documents_from_scanned_pdf import load_documents_from_scanned_pdf
from clean_text_with_llm import clean_text_with_llm
from split_text import split_text

# Suppress unnecessary warnings and logging
logging.getLogger("langchain").setLevel(logging.ERROR)

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/quy-che-hoc-vu-ctu.pdf"

BATCH_SIZE = 200

# LLM configuration (for OCR text cleaning)
LLM_MODEL = "llama3"

def main() -> None:
    """
    Main execution flow for database creation.
    Performs OCR, text cleaning, chunking, and vector database creation.
    """
    print("\nCreating RAG Database...\n")

    documents = load_documents_from_scanned_pdf()
    documents = clean_text_with_llm(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

    print("\n=== Database Creation Complete ===\n")


def save_to_chroma(chunks: List[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    # Creating Chroma database in batches
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    total_chunks = len(chunks)
    total_batches = (total_chunks - 1) // BATCH_SIZE + 1

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        try:
            db.add_documents(batch)
        except Exception as e:
            print(f"Error adding batch {batch_num}: {e}")
            continue

    db.persist()
    print(f"Database created: {len(chunks)} chunks saved to {CHROMA_PATH}\n")


if __name__ == "__main__":
    main()
