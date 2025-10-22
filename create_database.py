# create_database_local.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from pdf2image import convert_from_path
import pytesseract
import os
import shutil
import sys
import warnings
import logging

# --- Cấu hình & Hằng số ---
warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)

CHROMA_PATH = "chroma"
DATA_PATH = "data/quy-che-hoc-vu-ctu.pdf"

# Embedding local
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # hoặc "cuda"

# Chia đoạn văn
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 50
BATCH_SIZE = 200

# Mô hình local để làm sạch OCR (qua Ollama)
LLM_MODEL = "llama3"
# --- Kết thúc cấu hình ---


def main():
    print("\n=== BẮT ĐẦU TẠO DATABASE CHO RAG ===\n")

    # 1️⃣ OCR PDF scan → text
    documents = load_documents_from_scanned_pdf()

    # 2️⃣ Làm sạch text bằng LLaMA3 local
    documents = clean_text_with_llm(documents)

    # 3️⃣ Chia nhỏ văn bản
    chunks = split_text(documents)

    # 4️⃣ Lưu vào Chroma
    save_to_chroma(chunks)

    print("\n=== HOÀN TẤT TẠO DATABASE ===\n")


def load_documents_from_scanned_pdf() -> list[Document]:
    """OCR PDF scan thành text."""
    print(f"→ Đang tải file PDF: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"[LỖI] Không tìm thấy file tại: {DATA_PATH}")
        sys.exit(1)

    try:
        images = convert_from_path(DATA_PATH)
    except Exception as e:
        print(f"[LỖI] Lỗi khi chuyển PDF sang ảnh: {e}")
        print("💡 Cài poppler-utils: sudo apt install poppler-utils")
        sys.exit(1)

    documents = []
    print(f"→ Đang OCR {len(images)} trang PDF...")

    for i, image in enumerate(images):
        page_num = i + 1
        try:
            text = pytesseract.image_to_string(image, lang="vie").strip()
            if len(text) < 10:
                print(f"⚠️  Trang {page_num}: OCR rất ít ký tự.")
            metadata = {"source": DATA_PATH, "page": page_num}
            documents.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            print(f"[LỖI] OCR lỗi ở trang {page_num}: {e}")

    if not documents:
        print("[LỖI] Không đọc được trang nào.")
        sys.exit(1)

    print(f"✅ OCR hoàn tất: {len(documents)} trang đã xử lý.\n")
    return documents


def clean_text_with_llm(documents: list[Document]) -> list[Document]:
    """
    Dùng mô hình local LLaMA3 để hiệu chỉnh text OCR.
    """
    print("→ Đang hiệu chỉnh văn bản OCR bằng mô hình local (LLaMA3)...")

    try:
        llm = OllamaLLM(model=LLM_MODEL)
    except Exception as e:
        print(f"[LỖI] Không kết nối được Ollama hoặc model {LLM_MODEL}: {e}")
        return documents

    cleaned_docs = []
    for i, doc in enumerate(documents):
        prompt = f"""
        Hãy chỉnh sửa văn bản tiếng Việt sau cho rõ ràng, đúng chính tả,
        giữ nguyên nội dung gốc (không thêm thông tin mới):
        ---
        {doc.page_content}
        ---
        """
        try:
            response = llm.invoke(prompt).strip()
            cleaned_docs.append(Document(page_content=response, metadata=doc.metadata))
            print(f"✅ Trang {doc.metadata['page']} đã làm sạch.")
        except Exception as e:
            print(f"⚠️  Lỗi khi làm sạch trang {doc.metadata['page']}: {e}")
            cleaned_docs.append(doc)

    print(f"✅ Hoàn tất làm sạch {len(cleaned_docs)} trang.\n")
    return cleaned_docs


def split_text(documents: list[Document]) -> list[Document]:
    """Chia văn bản thành các chunk nhỏ."""
    print("→ Đang chia văn bản thành các đoạn nhỏ...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    print(f"→ Tổng số chunk trước khi lọc: {len(chunks)}")

    valid_chunks = [
        c for c in chunks if c.page_content.strip() and len(c.page_content) > MIN_CHUNK_LENGTH
    ]
    print(f"✅ Sau khi lọc: {len(valid_chunks)} chunks hợp lệ.\n")

    if not valid_chunks:
        print("[LỖI] Không có chunk hợp lệ.")
        sys.exit(1)

    return valid_chunks


def save_to_chroma(chunks: list[Document]):
    """Tạo vector database bằng Chroma."""
    print("→ Chuẩn bị lưu vào Chroma DB...")

    if os.path.exists(CHROMA_PATH):
        print(f"🗑️  Xóa database cũ: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    print(f"→ Tải model embedding: {EMBEDDING_MODEL}")
    try:
        embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"[LỖI] Không thể tải model embedding: {e}")
        sys.exit(1)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    total_chunks = len(chunks)
    total_batches = (total_chunks - 1) // BATCH_SIZE + 1

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"→ Thêm lô {i//BATCH_SIZE + 1}/{total_batches}...")
        try:
            db.add_documents(batch)
        except Exception as e:
            print(f"[LỖI] Khi thêm batch: {e}")
            continue

    db.persist()
    print(f"✅ Đã lưu {len(chunks)} chunks vào {CHROMA_PATH}.\n")


if __name__ == "__main__":
    main()
