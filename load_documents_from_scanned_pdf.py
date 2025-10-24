import os
import sys
from typing import List
from langchain_core.documents import Document
from pdf2image import convert_from_path
import pytesseract

def load_documents_from_scanned_pdf(DATA_PATH) -> List[Document]:
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