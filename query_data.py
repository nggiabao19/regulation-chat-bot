# query_data.py
import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Thay thế GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_VERBOSITY"] = "NONE"

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    query_text = input("Nhập câu hỏi của bạn: ").strip()
    if not query_text:
        print("Vui lòng nhập câu hỏi hợp lệ.")
        return
    
    # Khởi tạo HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Kết nối tới cơ sở dữ liệu Chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Tìm kiếm tương tự
    results = db.similarity_search_with_relevance_scores(query_text, k=3)  # Giảm k để tối ưu
    
    if len(results) == 0 or results[0][1] < 0.8:  # Tăng ngưỡng relevance
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    print("\n--- Ngữ cảnh được tìm thấy ---")
    print(context_text)
    print("---------------------------------\n")

    # Gọi LLM (vẫn sử dụng Gemini)
    model = GoogleGenerativeAI(model="gemini-2.5-flash")
    response = model.invoke(prompt)

    print("--- Câu trả lời ---")
    print(response)
    print("-------------------\n")
    
    print("--- Nguồn thông tin ---")
    for doc, score in results:
        print(f"Nguồn: {doc.metadata.get('source', 'N/A')}, Trang: {doc.metadata.get('page', 'N/A')}, Độ liên quan: {score:.4f}")
    print("----------------------\n")

if __name__ == "__main__":
    main()