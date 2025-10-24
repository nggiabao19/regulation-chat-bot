"""
Database creation script for RAG-based CTU Academic Regulations chatbot.
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
import pytesseract


# Suppress unnecessary warnings and logging
logging.getLogger("langchain").setLevel(logging.ERROR)

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/quy-che-hoc-vu-ctu.pdf"


# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50
BATCH_SIZE = 200

# LLM configuration (for OCR text cleaning)
LLM_MODEL = "llama3"


def main() -> None:
    """
    Main execution flow for database creation.
    Performs OCR, text cleaning, chunking, and vector database creation.
    """
    print("\n=== Creating RAG Database ===\n")

    documents = load_documents_from_scanned_pdf()
    documents = clean_text_with_llm(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

    print("\n=== Database Creation Complete ===\n")


def load_documents_from_scanned_pdf() -> List[Document]:
    """
    Extract text from scanned PDF using OCR.
    
    Returns:
        List[Document]: List of documents with extracted text and metadata
        
    Raises:
        SystemExit: If PDF cannot be found or processed
    """
    print(f"Loading PDF: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        sys.exit(1)

    try:
        images = convert_from_path(DATA_PATH)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Hint: Install poppler-utils with 'sudo apt install poppler-utils'")
        sys.exit(1)

    documents = []
    print(f"Processing OCR for {len(images)} pages...")

    for i, image in enumerate(images):
        page_num = i + 1
        try:
            text = pytesseract.image_to_string(image, lang="vie").strip()
            if len(text) < 10:
                print(f"Warning: Page {page_num} has very little text.")
                continue
                
            metadata = {"source": DATA_PATH, "page": page_num}
            documents.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"OCR failed for page {page_num}: {e}")

    if not documents:
        print("Error: No pages could be processed")
        sys.exit(1)

    print(f"OCR complete: {len(documents)} pages processed\n")
    return documents


def clean_text_with_llm(documents: List[Document]) -> List[Document]:
    """
    Clean OCR text using local LLM model.
    
    Args:
        documents: List of documents containing OCR text
        
    Returns:
        List[Document]: Documents with cleaned text
    """
    print("Cleaning OCR text with LLM...")

    try:
        llm = OllamaLLM(model=LLM_MODEL)
    except Exception as e:
        print(f"Error connecting to Ollama or loading {LLM_MODEL}: {e}")
        return documents

    cleaned_docs = []
    for doc in documents:
        prompt = f"""
        Please clean up this Vietnamese text while preserving the original meaning:
        ---
        {doc.page_content}
        ---
        Rules:
        - Fix spelling and formatting
        - Maintain original content
        - No new information
        """
        try:
            response = llm.invoke(prompt).strip()
            cleaned_docs.append(Document(page_content=response, metadata=doc.metadata))
            print(f"Cleaned page {doc.metadata['page']}")
        except Exception as e:
            print(f"Error cleaning page {doc.metadata['page']}: {e}")
            cleaned_docs.append(doc)

    print(f"Text cleaning complete: {len(cleaned_docs)} pages processed\n")
    return cleaned_docs


def split_text(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for vector storage.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List[Document]: List of text chunks with metadata preserved
        
    Raises:
        SystemExit: If no valid chunks are generated
    """
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
