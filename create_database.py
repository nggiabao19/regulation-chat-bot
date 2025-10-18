# create_database.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # Thay thế GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import time
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books/quy-che-hoc-vu-ctu.pdf"

def main():
    generate_data_store()

def generate_data_store():
    print("--- BẮT ĐẦU QUÁ TRÌNH ---")
    documents = load_documents_from_scanned_pdf()
    
    if not documents:
        print("\n[LỖI NGHIÊM TRỌNG] Không đọc được bất kỳ trang nào từ file PDF. Vui lòng kiểm tra lại đường dẫn file và đảm bảo file không bị lỗi.")
        return

    chunks = split_text(documents)

    if not chunks:
        print("\n[LỖI NGHIÊM TRỌNG] Sau khi xử lý, không còn lại chunk văn bản hợp lệ nào. Điều này có nghĩa là OCR không nhận dạng được chữ từ file PDF của bạn. Có thể file scan quá mờ hoặc có vấn đề với Tesseract.")
        return

    save_to_chroma(chunks)
    print("--- HOÀN TẤT ---")

def load_documents_from_scanned_pdf():
    print(f"Đang tải file PDF từ: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"[LỖI] Không tìm thấy file tại đường dẫn: {DATA_PATH}")
        return []
        
    try:
        images = convert_from_path(DATA_PATH)
    except Exception as e:
        print(f"[LỖI] Gặp sự cố khi chuyển đổi PDF thành ảnh: {e}")
        print("Gợi ý: Có thể bạn chưa cài đặt 'poppler-utils'. Chạy lệnh: sudo apt install poppler-utils")
        return []

    documents = []
    
    for i, image in enumerate(images):
        page_number = i + 1
        try:
            text = pytesseract.image_to_string(image, lang='vie')
            
            if i == 0:
                print("\n--- Văn bản trích xuất được từ trang 1 (để kiểm tra) ---")
                print(text[:500])
                print("----------------------------------------------------------\n")

            metadata = {"source": DATA_PATH, "page": page_number}
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        except Exception as e:
            print(f"[LỖI] Gặp sự cố khi OCR trang {page_number}: {e}")
            continue
            
    print(f"Đã xử lý xong {len(documents)} trang từ PDF.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Giảm chunk_size để tối ưu hóa
        chunk_overlap=50,  # Giảm overlap
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia {len(documents)} trang thành {len(chunks)} chunks ban đầu.")

    filtered_chunks = [chunk for chunk in chunks if chunk.page_content.strip() and len(chunk.page_content) > 50]
    print(f"Sau khi lọc bỏ các chunks rỗng hoặc quá ngắn, còn lại: {len(filtered_chunks)} chunks.")
    return filtered_chunks

def save_to_chroma(chunks: list[Document]):
    print("Chuẩn bị lưu vào Chroma DB...")
    if os.path.exists(CHROMA_PATH):
        print("Xóa cơ sở dữ liệu Chroma cũ...")
        shutil.rmtree(CHROMA_PATH)

    try:
        # Khởi tạo HuggingFaceEmbeddings với mô hình sentence-transformers
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Mô hình nhẹ, phù hợp cho tài nguyên hạn chế
            model_kwargs={'device': 'cpu'},  # Chạy trên CPU (thay bằng 'cuda' nếu có GPU)
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"[LỖI KHỞI TẠO MODEL] Gặp lỗi khi thiết lập model embedding: {e}")
        return

    # Tạo cơ sở dữ liệu Chroma
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    batch_size = 200  # Tăng batch size để giảm số lần xử lý
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Đang thêm lô {i//batch_size + 1}/{(len(chunks) - 1)//batch_size + 1} vào DB...")
        try:
            db.add_documents(batch)
            time.sleep(0.5)  # Giảm thời gian nghỉ vì không phụ thuộc vào API
        except Exception as e:
            print(f"[LỖI] Gặp lỗi khi thêm một lô vào DB. Lỗi: {e}")
            continue

    db.persist()
    print(f"Đã lưu thành công {len(chunks)} chunks vào {CHROMA_PATH}.")

if __name__ == "__main__":
    main()