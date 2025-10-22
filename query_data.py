# query_data.py (cải tiến: ưu tiên snippet chính xác + ép trả tiếng Việt)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
import warnings
import logging
import sys
import re

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_VERBOSITY"] = "NONE"

load_dotenv()

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
Bạn là trợ lý ảo CTU-Chatbot, được cung cấp một đoạn trích từ Quy chế học vụ.
Hãy:
Đọc kỹ nội dung trong NGỮ CẢNH.
SUY LUẬN để trả lời câu hỏi của sinh viên thật chính xác.
Trả lời NGẮN GỌN, BẰNG TIẾNG VIỆT, không chế thêm thông tin ngoài quy chế.
Nếu không thấy thông tin trong ngữ cảnh, hãy nói bạn không biết theo phong cách hài hước"

---

NGỮ CẢNH:
{context}

---

CÂU HỎI: {question}

---

SUY LUẬN & TRẢ LỜI:
"""


# giới hạn context (kí tự)
def shorten_context(context: str, max_chars: int = 3000) -> str:
    return context[:max_chars]

# tìm snippet chính xác (rất cơ bản): tách câu và kiểm tra từ khoá xuất hiện đủ
def find_exact_snippet(results, question, min_hits=1):
    """
    results: list of (doc, score)
    question: string
    Trả về (snippet, source, page, score) nếu tìm thấy, else None
    """
    # lấy từ khóa: loại bỏ stopwords đơn giản và ký tự đặc biệt
    q = question.lower()
    # split on non-word, keep words length>=2
    words = [w for w in re.findall(r"\w+", q) if len(w) >= 2]
    if not words:
        return None

    # ưu tiên doc có score cao
    for doc, score in results:
        text = doc.page_content
        # chia câu (đơn giản)
        sentences = re.split(r'(?<=[.!?。\n])\s+', text)
        for sent in sentences:
            sent_lower = sent.lower()
            hits = sum(1 for w in words if w in sent_lower)
            # nếu câu chứa nhiều từ khoá (tùy bạn điều chỉnh ngưỡng)
            if hits >= max(1, len(words)//2):
                snippet = sent.strip()
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("page", "N/A")
                return snippet, source, page, score
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
    results = db.similarity_search_with_relevance_scores(query_text, k=8)

    # nếu không có kết quả tốt
    if not results:
        print("CTU-Chatbot: Mình chưa rõ thông tin này trong quy chế học vụ.")
        return

    # cố tìm snippet chính xác trong các chunk
    exact = find_exact_snippet(results, query_text)
    if exact:
        snippet, source, page, score = exact
        # map source name
        if "quy-che-hoc-vu" in str(source).lower():
            source_name = "Quy chế học vụ"
        else:
            source_name = os.path.basename(str(source))
        print("\n--- KẾT QUẢ DÒ TĨNH (snippet match) ---")
        print(f"Trích dẫn (nguyên văn):\n\"{snippet}\"")
        print(f"Nguồn: {source_name}, Trang {page} (Độ liên quan: {score:.3f})")
        # tóm tắt ngắn gọn: (nên trả bằng tiếng Việt -> cố gắng rút gọn)
        # đơn giản: trả câu hỏi nếu snippet có chứa đáp án
        print("\nCTU-Chatbot: ", end="")
        # nếu snippet có số kỳ/tuần/năm, in luôn
        m = re.search(r'\b(\d+)\b', snippet)
        if m:
            print(f"{m.group(1)} (theo Quy chế học vụ).")
        else:
            # fallback: in snippet + một câu tóm tắt
            print(f"{snippet}")
        return

    # không tìm snippet -> dùng LLM local như fallback
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    context_text = shorten_context(context_text)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    try:
        # Khởi tạo Ollama LLM (hoặc model local khác bạn đã cài)
        # Đặt temperature=0 để giảm sáng tạo
        model = OllamaLLM(model="llama3", temperature=0.0, max_new_tokens=256)
        # gọi model: tùy phiên bản langchain-ollama, có thể là model.generate(...)
        # chúng ta thử dùng .generate để lấy text an toàn
        gen = model.generate([prompt])
        # Lấy text
        # cấu trúc trả về có thể khác nhau; ta cố gắng trích text ra theo chuẩn chung
        response_text = ""
        try:
            response_text = gen.generations[0][0].text.strip()
        except Exception:
            # fallback: nếu .generate không có field như trên, thử gọi như hàm
            try:
                response_text = model(prompt).strip()
            except Exception:
                response_text = "[Lỗi khi lấy phản hồi từ LLM]"

        # in kết quả
        print(f"\nCTU-Chatbot: {response_text}\n")

        print("--- Nguồn thông tin (top chunks) ---")
        for doc, score in results[:3]:
            source = doc.metadata.get('source', '')
            if "quy-che-hoc-vu" in source.lower():
                source_name = "Quy chế học vụ"
            else:
                source_name = os.path.basename(source)
            page = doc.metadata.get('page', 'N/A')
            print(f"- {source_name}, Trang {page} (Độ liên quan: {score:.3f})")

    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi gọi LLM: {e}")

if __name__ == "__main__":
    main()
