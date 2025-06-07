import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
import time

# Cấu hình kết nối MySQL
db_config = {
    'host': 'localhost',
    'port': '3306',  # từ Docker
    'database': 'db_capstone',
    'user': 'root',
    'password': '123456789'
}

# Lưu trữ kết nối hiện tại
_current_connection = None

def close_all_connections():
    """Đóng tất cả kết nối database hiện tại"""
    global _current_connection
    if _current_connection is not None and _current_connection.is_connected():
        try:
            _current_connection.close()
            print("✅ Đã đóng kết nối database cũ")
        except Error as e:
            print(f"⚠️ Lỗi khi đóng kết nối cũ: {e}")
        finally:
            _current_connection = None

def refresh_connection():
    """Tạo kết nối mới đến database"""
    global _current_connection
    close_all_connections()
    
    # Đợi một chút để đảm bảo kết nối cũ đã đóng hoàn toàn
    time.sleep(0.5)
    
    try:
        _current_connection = mysql.connector.connect(**db_config)
        if _current_connection.is_connected():
            print("✅ Đã tạo kết nối database mới!")
            return True
    except Error as e:
        print(f"❌ Lỗi khi tạo kết nối mới: {e}")
        _current_connection = None
        return False

@contextmanager
def get_db_connection(force_new=False):
    """Context manager để quản lý kết nối database
    
    Args:
        force_new (bool): Nếu True, tạo kết nối mới
    
    Sử dụng:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM table")
        result = cursor.fetchall()
    """
    global _current_connection
    
    # Nếu yêu cầu kết nối mới hoặc không có kết nối hoặc kết nối đã đóng
    if force_new or _current_connection is None or not _current_connection.is_connected():
        if not refresh_connection():
            raise Error("Không thể kết nối đến database")
    
    try:
        yield _current_connection
    except Error as e:
        print(f"❌ Lỗi database: {e}")
        # Nếu có lỗi, thử tạo kết nối mới
        if not refresh_connection():
            raise
        # Thử lại một lần nữa với kết nối mới
        yield _current_connection

def execute_query(query, params=None, fetch=True, force_new=False):
    """Thực thi câu query và trả về kết quả
    
    Args:
        query (str): Câu query SQL
        params (tuple, optional): Tham số cho câu query
        fetch (bool): True nếu cần lấy kết quả, False nếu là INSERT/UPDATE
        force_new (bool): True nếu muốn tạo kết nối mới
    
    Returns:
        list: Kết quả query nếu fetch=True
        int: Số dòng bị ảnh hưởng nếu fetch=False
    """
    with get_db_connection(force_new=force_new) as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())
            if fetch:
                result = cursor.fetchall()
            else:
                conn.commit()
                result = cursor.rowcount
            return result
        except Error as e:
            print(f"❌ Lỗi thực thi query: {e}")
            raise
        finally:
            cursor.close()

def execute_many(query, params_list, force_new=False):
    """Thực thi nhiều câu query cùng lúc
    
    Args:
        query (str): Câu query SQL
        params_list (list): Danh sách các tham số
        force_new (bool): True nếu muốn tạo kết nối mới
    
    Returns:
        int: Số dòng bị ảnh hưởng
    """
    with get_db_connection(force_new=force_new) as conn:
        cursor = conn.cursor()
        try:
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
        except Error as e:
            print(f"❌ Lỗi thực thi nhiều query: {e}")
            raise
        finally:
            cursor.close()

# Đóng tất cả kết nối khi import module
close_all_connections()
    
