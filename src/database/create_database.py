"""
Database creation script for RAG-based CTU Regulations chatbot.
Creates a vector database from PDF document using OCR and text cleaning.
"""
import os
import sys
import shutil
from typing import List
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.utils.get_embedding_function import get_embedding_function
from src.database.load_documents_from_scanned_pdf import load_documents_from_scanned_pdf, process_single_pdf
from src.database.clean_text_with_llm import clean_text_with_llm
from src.database.split_text import split_text

# Constants
CHROMA_PATH = "chroma"
BATCH_SIZE = 200
DATA_PATH = "data"  # Can be a folder or a single PDF file

def main():
    parser = argparse.ArgumentParser(description="Create or update RAG database from PDF documents")
    parser.add_argument("--reset", action="store_true", help="Reset the database before adding documents")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to PDF file or folder containing PDFs")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Resetting Database\n")
        clear_database()
    
    print("Creating/Updating RAG Database...\n")
    
    # Get list of PDF files to process
    pdf_files = get_pdf_files(args.data)
    
    # Filter out PDFs that are already in database
    new_pdf_files = filter_new_files(pdf_files)
    
    if not new_pdf_files:
        print("✅ All PDF files are already in the database. Nothing to add.\n")
        return
    
    print(f"📝 Processing {len(new_pdf_files)} new PDF file(s)...\n")
    
    # Only process new files
    documents = load_documents_from_pdf_list(new_pdf_files)
    documents = clean_text_with_llm(documents)
    chunks = split_text(documents)
    add_to_chroma(chunks)

    print("\nDatabase Update Complete\n")


def get_pdf_files(data_path: str) -> List[str]:
    """
    Get list of PDF files from path.
    """
    pdf_files = []
    
    if os.path.isdir(data_path):
        pdf_files = [
            os.path.abspath(os.path.join(data_path, f))
            for f in os.listdir(data_path)
            if f.lower().endswith('.pdf')
        ]
    elif os.path.isfile(data_path) and data_path.lower().endswith('.pdf'):
        pdf_files = [os.path.abspath(data_path)]
    
    return sorted(pdf_files)


def filter_new_files(pdf_files: List[str]) -> List[str]:
    """
    Filter out PDF files that are already in the database.
    """
    if not os.path.exists(CHROMA_PATH):
        print("No existing database found. All files will be processed.\n")
        return pdf_files
    
    # Load existing database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    # Get all existing sources from database
    existing_items = db.get(include=["metadatas"])
    existing_sources = set()
    
    for metadata in existing_items["metadatas"]:
        if "source" in metadata:
            existing_sources.add(os.path.abspath(metadata["source"]))
    
    print(f"Existing database contains {len(existing_sources)} PDF file(s)")
    print(f"Found {len(pdf_files)} PDF file(s) in folder")
    
    # Filter new files
    new_files = [f for f in pdf_files if f not in existing_sources]
    
    if new_files:
        print(f"New files to process: {len(new_files)}")
        for f in new_files:
            print(f"  - {os.path.basename(f)}")
        print()
    
    return new_files


def load_documents_from_pdf_list(pdf_files: List[str]) -> List[Document]:
    """
    Load documents from a list of PDF files.
    
    Args:
        pdf_files: List of PDF file paths
        
    Returns:
        List[Document]: All documents from the PDF files
    """
    all_documents = []
    for pdf_file in pdf_files:
        documents = process_single_pdf(pdf_file)
        all_documents.extend(documents)
    
    return all_documents

def add_to_chroma(chunks: List[Document]) -> None:
    """
    Add documents to Chroma database.
    Files are already filtered, so all chunks are new.
    """
    if not chunks:
        print("No chunks to add.\n")
        return
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Load or create Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )
    
    print(f"Adding {len(chunks_with_ids)} chunks to database...\n")
    
    # Add documents in batches
    total_batches = (len(chunks_with_ids) - 1) // BATCH_SIZE + 1
    
    for i in range(0, len(chunks_with_ids), BATCH_SIZE):
        batch = chunks_with_ids[i : i + BATCH_SIZE]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        try:
            db.add_documents(batch, ids=batch_ids)
        except Exception as e:
            print(f"Error adding batch {batch_num}: {e}")
    
    db.persist()
    print(f"\nSuccessfully added {len(chunks_with_ids)} chunks to database")


def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Calculate unique IDs for each chunk based on source file, page, and chunk index.
    ID format: "source_file.pdf:page_number:chunk_index"
    
    Args:
        chunks: List of document chunks
        
    Returns:
        List[Document]: Chunks with ID added to metadata
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If same page as last chunk, increment index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Create unique chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add ID to chunk metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database() -> None:
    """Remove the entire Chroma database directory."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Database cleared: {CHROMA_PATH}\n")

if __name__ == "__main__":
    main()
