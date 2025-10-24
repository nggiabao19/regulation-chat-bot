import os
import sys
from typing import List
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract


def load_documents_from_scanned_pdf(data_path: str) -> List[Document]:
    """
    Load and process PDF files using OCR.
    
    Args:
        data_path: Path to a PDF file or directory containing PDF files
        
    Returns:
        List[Document]: Documents with OCR-extracted text and metadata
    """
    all_documents = []
    
    # Check if path is a directory or file
    if os.path.isdir(data_path):
        pdf_files = [
            os.path.join(data_path, f) 
            for f in os.listdir(data_path) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            print(f"Error: No PDF files found in {data_path}")
            sys.exit(1)
            
        print(f"Found {len(pdf_files)} PDF file(s) in {data_path}\n")
        
        for pdf_file in pdf_files:
            documents = process_single_pdf(pdf_file)
            all_documents.extend(documents)
            
    elif os.path.isfile(data_path):
        if not data_path.lower().endswith('.pdf'):
            print(f"Error: {data_path} is not a PDF file")
            sys.exit(1)
        all_documents = process_single_pdf(data_path)
    else:
        print(f"Error: Path not found: {data_path}")
        sys.exit(1)
    
    if not all_documents:
        print("Error: No documents could be processed")
        sys.exit(1)
        
    print(f"\nTotal: {len(all_documents)} pages processed from all PDFs\n")
    return all_documents


def process_single_pdf(pdf_path: str) -> List[Document]:
    """
    Process a single PDF file with OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List[Document]: Documents extracted from the PDF
    """
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Hint: Install poppler-utils with 'sudo apt install poppler-utils'")
        return []

    documents = []
    print(f"  OCR processing {len(images)} pages...")

    for i, image in enumerate(images):
        page_num = i + 1
        try:
            text = pytesseract.image_to_string(image, lang="vie").strip()
            if len(text) < 10:
                print(f"  Warning: Page {page_num} has very little text")
                continue
                
            metadata = {"source": pdf_path, "page": page_num}
            documents.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"  OCR failed for page {page_num}: {e}")

    print(f"  Completed: {len(documents)} pages extracted\n")
    return documents