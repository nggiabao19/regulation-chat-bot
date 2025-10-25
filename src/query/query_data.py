"""
CTU-Chatbot: A QA system for CTU academic regulations using RAG pattern.
"""

import os
import sys
import re
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Suppress unnecessary warnings and logging
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = os.environ["GRPC_CPP_VERBOSITY"] = "NONE"

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CONTEXT_LENGTH = 3000

PROMPT_TEMPLATE = """
Bạn là trợ lý ảo CTU-Chatbot, được cung cấp một đoạn trích từ Quy chế học vụ.
Hãy:
1. Đọc kỹ nội dung trong NGỮ CẢNH
2. SUY LUẬN để trả lời câu hỏi của sinh viên thật chính xác
3. Trả lời NGẮN GỌN, BẰNG TIẾNG VIỆT, không chế thêm thông tin ngoài quy chế
4. Nếu không thấy thông tin trong ngữ cảnh, hãy nói bạn không biết theo phong cách hài hước

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

SUY LUẬN & TRẢ LỜI:
"""

def shorten_context(text: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """
    Truncate text to max_length while preserving complete sentences.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated text ending at a sentence boundary
    """
    if len(text) <= max_length:
        return text
        
    # Find last sentence boundary before max_length
    truncated = text[:max_length]
    last_boundary = max(
        truncated.rfind("."),
        truncated.rfind("!"),
        truncated.rfind("?"),
        truncated.rfind("\n")
    )
    return text[:last_boundary + 1] if last_boundary > 0 else truncated

def find_relevant_snippet(results: List[Tuple], question: str, min_length: int = 40) -> Optional[Tuple]:
    """
    Find most relevant snippet from search results that matches the question.
    
    Args:
        results: List of (document, score) tuples from vector search
        question: User query string
        min_length: Minimum snippet length to consider
        
    Returns:
        Tuple of (snippet, source, page, score) if found, None otherwise
    """
    # Extract keywords from question
    words = [w for w in re.findall(r"\w+", question.lower()) if len(w) >= 2]
    if not words:
        return None

    # Check each result document
    for doc, score in results:
        sentences = re.split(r'(?<=[.!?。\n])\s+', doc.page_content)
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < min_length:
                continue
                
            hits = sum(1 for w in words if w in sent_clean.lower())
            if hits >= max(1, len(words)//2):
                source = doc.metadata.get("source", "N/A")
                source = "Quy chế học vụ" if "quy-che-hoc-vu" in str(source).lower() \
                    else os.path.basename(str(source))
                return (
                    sent_clean,
                    source, 
                    doc.metadata.get("page", "N/A"),
                    score
                )
    return None

def main():
    query_text = input("Nhập câu hỏi của bạn: ").strip()
    if not query_text:
        print("Vui lòng nhập câu hỏi hợp lệ.")
        return

    # embedding (local)
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # load chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # lấy top-k (tăng k giúp tìm snippet tốt hơn)
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    if not results:
        print("CTU-Chatbot: Mình chưa rõ thông tin này trong quy chế học vụ.")
        return

    # Try to find most relevant snippet
    relevant = find_relevant_snippet(results, query_text, min_length=40)
    
    if relevant:
        snippet, source, page, score = relevant
        print("\n--- KẾT QUẢ TRÍCH DẪN ---")
        print(f"Trích dẫn (nguyên văn):\n\"{snippet}\"")
        print(f"Nguồn: {source}, Trang {page} (Độ liên quan: {score:.3f})")

        # Process snippet with LLM
        prompt = f"""
        Bạn là trợ lý ảo CTU-Chatbot.
        Dưới đây là đoạn trích từ Quy chế học vụ:
        "{snippet}"

        Hãy trả lời câu hỏi sau của sinh viên dựa trên đoạn trích:
        "{query_text}"

        Yêu cầu:
        - Trả lời ngắn gọn, bằng tiếng Việt
        - Chỉ dùng thông tin trong đoạn trích
        - Nếu không đủ thông tin, hãy nói "Đoạn trích không có thông tin cụ thể về vấn đề này"
        """
        
        try:
            model = OllamaLLM(model="llama3", temperature=0.0, max_new_tokens=128)
            response_text = model.invoke(prompt).strip()
            print("\nCTU-Chatbot:", response_text)
        except Exception as e:
            print(f"\nLỗi khi gọi LLM: {e}")
            print("Đoạn trích nguyên văn:", snippet)
        
        return


    # Use LLM as fallback when no specific snippet is found
    print("\n--- KẾT QUẢ TỔNG HỢP ---")
    context_text = "\n\n---\n\n".join(
        doc.page_content for doc, _ in results
    )
    context_text = shorten_context(context_text)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    try:
        model = OllamaLLM(model="llama3", temperature=0.0, max_new_tokens=256)
        print(f"\nCTU-Chatbot: {model.invoke(prompt).strip()}\n")

        print("--- Nguồn tham khảo ---")
        for doc, score in results[:3]:
            source = doc.metadata.get('source', '')
            source = "Quy chế học vụ" if "quy-che-hoc-vu" in str(source).lower() \
                else os.path.basename(str(source))
            page = doc.metadata.get('page', 'N/A')
            print(f"- {source}, Trang {page} (Độ liên quan: {score:.3f})")

    except Exception as e:
        print(f"\nLỗi khi xử lý: {e}")

if __name__ == "__main__":
    main()