# CTU Regulation Chatbot 

Hệ thống chatbot RAG (Retrieval Augmented Generation) để trả lời câu hỏi về Quy chế học vụ CTU.

## Cấu trúc Project

```
RAG-regulation-chatbot/
├── src/                          # Source code chính
│   ├── __init__.py
│   ├── database/                 # Module xử lý database
│   │   ├── __init__.py
│   │   ├── create_database.py   # Tạo/cập nhật vector database
│   │   ├── load_documents_from_scanned_pdf.py  # OCR PDF files
│   │   ├── clean_text_with_llm.py              # Làm sạch text với LLM
│   │   └── split_text.py                       # Chia nhỏ documents
│   ├── query/                    # Module xử lý truy vấn
│   │   ├── __init__.py
│   │   └── query_data.py        # Chatbot query interface
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── get_embedding_function.py  # Embedding functions
├── scripts/                      # Utility scripts
│   └── check_pdf.py             # Kiểm tra PDF files
├── data/                         # Thư mục chứa PDF files
├── chroma/                       # Vector database storage
├── create_db.py                  # Script tạo database (wrapper)
├── query.py                      # Script chạy chatbot (wrapper)
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
└── README.md                     # Documentation

```

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd RAG-regulation-chatbot
```

### 2. Tạo virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cài đặt Ollama và model
```bash
# Cài đặt Ollama từ https://ollama.ai
# Sau đó tải model:
ollama pull llama3
```

### 5. Cài đặt poppler-utils (cho OCR)
```bash
sudo apt install poppler-utils tesseract-ocr tesseract-ocr-vie
```

## Sử dụng

### Tạo/Cập nhật Database

**Lần đầu tạo database:**
```bash
python create_db.py
```

**Thêm file mới vào database:**
1. Copy file PDF vào thư mục `data/`
2. Chạy: `python create_db.py`
3. Script sẽ tự động phát hiện và chỉ xử lý file mới

**Reset toàn bộ database:**
```bash
python create_db.py --reset
```

**Chỉ định thư mục khác:**
```bash
python create_db.py --data path/to/pdfs
```

### Chạy Chatbot

```bash
python query.py
```

Sau đó nhập câu hỏi của bạn. Ví dụ:
```
Nhập câu hỏi của bạn: Điều kiện để được xét tốt nghiệp là gì?
```

## Cấu hình

Tạo file `.env` với nội dung:
```env
# Không cần API key nếu dùng model local
CHROMA_PATH=chroma
```

## Workflow

### 1. Xử lý tài liệu mới
```
PDF files → OCR → Clean text → Split chunks → Embeddings → Vector DB
```

### 2. Truy vấn
```
User question → Embedding → Vector search → Retrieve context → LLM → Answer
```

## Tính năng

- **Xử lý PDF scan**: OCR tự động với pytesseract  
- **Làm sạch text**: Sử dụng LLM để cải thiện chất lượng OCR  
- **Kiểm tra trùng lặp**: Tự động bỏ qua file đã có trong database  
- **RAG pattern**: Trả lời dựa trên ngữ cảnh từ tài liệu  
- **Local LLM**: Sử dụng Ollama, không cần API key  
- **Embedding local**: Sử dụng HuggingFace embeddings  

## Công nghệ sử dụng 

- **LangChain**: Framework RAG
- **Chroma**: Vector database
- **Ollama + LLaMA3**: Local LLM
- **HuggingFace**: Sentence embeddings
- **Pytesseract**: OCR engine
- **PDF2Image + Poppler**: PDF processing

## Performance

- Thời gian OCR: ~2-3s/trang
- Thời gian clean LLM: ~5-10s/trang
- Thời gian query: ~2-5s
- Chỉ xử lý file mới → Tiết kiệm thời gian đáng kể

## Credits & Acknowledgments

Project này được phát triển dựa trên kiến thức học được từ kênh YouTube [**@pixegami**](https://www.youtube.com/@pixegami).

Cảm ơn Pixegami đã chia sẻ những tutorial chất lượng về RAG và LangChain!

**Các cải tiến so với tutorial gốc:**
- Hỗ trợ xử lý PDF scan với OCR
- Làm sạch text bằng LLM
- Xử lý nhiều PDF files trong folder

## Đóng góp

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Authors

- Nguyễn Gia Bảo (nggiabao19@gmail.com)

## 🔗 Links

- [Ollama](https://ollama.ai)
- [LangChain](https://python.langchain.com)
- [ChromaDB](https://www.trychroma.com)
