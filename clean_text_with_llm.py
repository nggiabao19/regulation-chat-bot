from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List
LLM_MODEL = "llama3"

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