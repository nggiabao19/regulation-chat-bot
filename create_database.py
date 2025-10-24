"""
Database creation script for RAG-based CTU Regulations chatbot.
Creates a vector database from PDF document using OCR and text cleaning.
"""
import os
import shutil
from typing import List
import argparse

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function
from load_documents_from_scanned_pdf import load_documents_from_scanned_pdf
from clean_text_with_llm import clean_text_with_llm
from split_text import split_text

# Constants
CHROMA_PATH = "chroma"
BATCH_SIZE = 200
DATA_PATH="data/quy-che-hoc-vu-ctu.pdf"

def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    """Main execution flow for database creation."""
    print("\nCreating RAG Database...\n")
    documents = load_documents_from_scanned_pdf(DATA_PATH)
    documents = clean_text_with_llm(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

    print("\nDatabase Creation Complete\n")

def save_to_chroma(chunks: List[Document]) -> None:
    # Create Chroma database in batches
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
    
    total_batches = (len(chunks) - 1) // BATCH_SIZE + 1
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        try:
            db.add_documents(chunks[i : i + BATCH_SIZE])
        except Exception as e:
            print(f"Error adding batch {batch_num}: {e}")

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/ABC.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
