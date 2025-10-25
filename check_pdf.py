from pypdf import PdfReader
import os
DATA_CHECK_PATH = "data_to_check"  # Thay đổi đường dẫn này đến thư mục hoặc tệp PDF bạn muốn kiểm tra
def check_pdf_text_extraction(file_path):
    """Kiểm tra xem PDF có thể trích xuất văn bản thành công hay không."""
    try:
        reader = PdfReader(file_path)
        
        # Thử trích xuất văn bản từ trang đầu tiên
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text()
            
            # Kiểm tra xem có đủ văn bản được trích xuất không
            if len(first_page_text.strip()) > 50: # Đặt ngưỡng 50 ký tự
                print(f"Tệp '{os.path.basename(file_path)}' là Dạng Văn Bản (Text-Based).")
                print(f"   (Đã trích xuất được {len(first_page_text.strip())} ký tự.)")
                return True
            else:
                print(f"Tệp '{os.path.basename(file_path)}' có vẻ là Dạng Ảnh Quét (Scanned Image) hoặc nội dung rỗng.")
                return False
        else:
            print(f"Tệp '{os.path.basename(file_path)}' không có trang nào.")
            return False

    except Exception as e:
        print(f"Lỗi khi đọc tệp PDF: {e}")
        return False

check_pdf_text_extraction(DATA_CHECK_PATH)
