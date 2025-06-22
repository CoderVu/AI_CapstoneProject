from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from werkzeug.utils import secure_filename
import requests
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2
from PIL import Image
import io
from functools import lru_cache
import json
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from datetime import datetime
import base64
import csv
from io import StringIO

app = Flask(__name__)

# Enable CORS for all origins
CORS(app, supports_credentials=True, origins="*", allow_headers="*")
# CORS(app, supports_credentials=True, origins=["http://localhost:3000"], allow_headers="*")
# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
# Khởi tạo StandardScaler và OneHotEncoder
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# Biến toàn cục để lưu mô hình và survey
dl_model = None
survey = None

# Global variables for feature extraction status
extraction_status = {
    "is_running": False,
    "total_images": 0,
    "processed_count": 0,
    "failed_count": 0,
    "current_image": "",
    "error": None
}
extraction_lock = threading.Lock()

# Global variable for configurable vector dimensions
VECTOR_DIMENSIONS_CONFIG = 2048

# Global variables for models
_models = {}
_model_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for image processing
MIN_IMAGE_SIZE = 224  # Minimum size for feature extraction
MAX_IMAGE_SIZE = 1024  # Maximum size to prevent memory issues
QUALITY_THRESHOLD = 0.5  # Lowered quality threshold for images

# Global variables for performance tracking
performance_metrics = {
    "extraction_times": [],  # List of extraction times in seconds
    "search_times": [],      # List of search times in milliseconds
    "accuracy_scores": [],   # List of accuracy scores
    "total_images_processed": 0,
    "total_searches_performed": 0,
    "last_updated": None
}

performance_lock = threading.Lock()

def record_extraction_time(extraction_time_seconds):
    """Ghi lại thời gian trích xuất đặc trưng"""
    global performance_metrics
    with performance_lock:
        performance_metrics["extraction_times"].append(extraction_time_seconds)
        performance_metrics["total_images_processed"] += 1
        performance_metrics["last_updated"] = datetime.now()
        
        # Giữ lại tối đa 1000 mẫu để tránh memory leak
        if len(performance_metrics["extraction_times"]) > 1000:
            performance_metrics["extraction_times"] = performance_metrics["extraction_times"][-1000:]

def record_search_time(search_time_ms):
    """Ghi lại thời gian tìm kiếm"""
    global performance_metrics
    with performance_lock:
        performance_metrics["search_times"].append(search_time_ms)
        performance_metrics["total_searches_performed"] += 1
        performance_metrics["last_updated"] = datetime.now()
        
        # Giữ lại tối đa 1000 mẫu
        if len(performance_metrics["search_times"]) > 1000:
            performance_metrics["search_times"] = performance_metrics["search_times"][-1000:]

def record_accuracy_score(accuracy):
    """Ghi lại độ chính xác"""
    global performance_metrics
    with performance_lock:
        performance_metrics["accuracy_scores"].append(accuracy)
        performance_metrics["last_updated"] = datetime.now()
        
        # Giữ lại tối đa 1000 mẫu
        if len(performance_metrics["accuracy_scores"]) > 1000:
            performance_metrics["accuracy_scores"] = performance_metrics["accuracy_scores"][-1000:]

def get_performance_statistics():
    """Lấy thống kê hiệu năng thực tế"""
    global performance_metrics
    with performance_lock:
        stats = {
            "extraction_performance": {
                "avg_time_per_image": 0,
                "min_time": 0,
                "max_time": 0,
                "total_images": performance_metrics["total_images_processed"]
            },
            "search_performance": {
                "avg_time_per_search": 0,
                "min_time": 0,
                "max_time": 0,
                "total_searches": performance_metrics["total_searches_performed"]
            },
            "accuracy": {
                "avg_accuracy": 0,
                "min_accuracy": 0,
                "max_accuracy": 0,
                "total_measurements": 0
            },
            "last_updated": performance_metrics["last_updated"]
        }
        
        # Tính toán thống kê trích xuất
        if performance_metrics["extraction_times"]:
            times = performance_metrics["extraction_times"]
            stats["extraction_performance"]["avg_time_per_image"] = sum(times) / len(times)
            stats["extraction_performance"]["min_time"] = min(times)
            stats["extraction_performance"]["max_time"] = max(times)
        
        # Tính toán thống kê tìm kiếm
        if performance_metrics["search_times"]:
            times = performance_metrics["search_times"]
            stats["search_performance"]["avg_time_per_search"] = sum(times) / len(times)
            stats["search_performance"]["min_time"] = min(times)
            stats["search_performance"]["max_time"] = max(times)
        
        # Tính toán thống kê độ chính xác
        if performance_metrics["accuracy_scores"]:
            scores = performance_metrics["accuracy_scores"]
            stats["accuracy"]["avg_accuracy"] = sum(scores) / len(scores)
            stats["accuracy"]["min_accuracy"] = min(scores)
            stats["accuracy"]["max_accuracy"] = max(scores)
            stats["accuracy"]["total_measurements"] = len(scores)
        
        return stats

def create_performance_chart():
    """Tạo biểu đồ hiệu năng từ dữ liệu thực tế"""
    try:
        # Tạo thư mục chart nếu chưa có
        os.makedirs('chart', exist_ok=True)
        
        # Lấy thống kê hiệu năng thực tế
        stats = get_performance_statistics()
        
        # Chuẩn bị dữ liệu cho biểu đồ
        metrics = []
        values = []
        colors = []
        
        # Hiệu năng trích xuất (giây/ảnh)
        if stats["extraction_performance"]["total_images"] > 0:
            metrics.append('Hiệu năng\n(giây/ảnh)')
            values.append(round(stats["extraction_performance"]["avg_time_per_image"], 2))
            colors.append('#FF6B6B')
        else:
            metrics.append('Hiệu năng\n(giây/ảnh)')
            values.append(0)  # Chưa có dữ liệu
            colors.append('#FF6B6B')
        
        # Tốc độ tìm kiếm (ms/ảnh)
        if stats["search_performance"]["total_searches"] > 0:
            metrics.append('Tốc độ tìm kiếm\n(ms/ảnh)')
            values.append(round(stats["search_performance"]["avg_time_per_search"], 2))
            colors.append('#4ECDC4')
        else:
            metrics.append('Tốc độ tìm kiếm\n(ms/ảnh)')
            values.append(0)  # Chưa có dữ liệu
            colors.append('#4ECDC4')
        
        # Độ chính xác (%)
        if stats["accuracy"]["total_measurements"] > 0:
            metrics.append('Độ chính xác\n(%)')
            values.append(round(stats["accuracy"]["avg_accuracy"] * 100, 1))
            colors.append('#45B7D1')
        else:
            metrics.append('Độ chính xác\n(%)')
            values.append(0)  # Chưa có dữ liệu
            colors.append('#45B7D1')
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 8))
        bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Thêm giá trị trên mỗi cột
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.05),
                    f'{value}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.title('Hiệu Năng Hệ Thống AI Image Similarity (Dữ Liệu Thực Tế)', fontsize=18, fontweight='bold', pad=20)
        plt.ylabel('Giá trị', fontsize=14)
        plt.ylim(0, max(values) * 1.3 if max(values) > 0 else 10)
        
        # Thêm grid
        plt.grid(True, alpha=0.3, axis='y')
        
        # Thêm thông tin chi tiết từ dữ liệu thực tế
        detail_text = []
        if stats["extraction_performance"]["total_images"] > 0:
            detail_text.append(f'• Hiệu năng: {stats["extraction_performance"]["avg_time_per_image"]:.2f}s/ảnh (từ {stats["extraction_performance"]["total_images"]} ảnh)')
        else:
            detail_text.append('• Hiệu năng: Chưa có dữ liệu')
            
        if stats["search_performance"]["total_searches"] > 0:
            detail_text.append(f'• Tốc độ tìm kiếm: {stats["search_performance"]["avg_time_per_search"]:.2f}ms (từ {stats["search_performance"]["total_searches"]} lần tìm)')
        else:
            detail_text.append('• Tốc độ tìm kiếm: Chưa có dữ liệu')
            
        if stats["accuracy"]["total_measurements"] > 0:
            detail_text.append(f'• Độ chính xác: {stats["accuracy"]["avg_accuracy"]*100:.1f}% (từ {stats["accuracy"]["total_measurements"]} đo)')
        else:
            detail_text.append('• Độ chính xác: Chưa có dữ liệu')
        
        detail_text.append(f'• Cập nhật lần cuối: {stats["last_updated"].strftime("%Y-%m-%d %H:%M:%S") if stats["last_updated"] else "Chưa có"}')
        
        plt.figtext(0.5, 0.02, '\n'.join(detail_text),
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Lưu biểu đồ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chart/performance_chart_real_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Đã tạo biểu đồ hiệu năng từ dữ liệu thực tế: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ hiệu năng: {str(e)}")
        return None

def create_data_augmentation_flowchart():
    """Tạo sơ đồ quy trình tăng cường dữ liệu"""
    try:
        # Tạo thư mục chart nếu chưa có
        os.makedirs('chart', exist_ok=True)
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Định nghĩa các bước trong quy trình
        steps = [
            'Ảnh gốc',
            'Xoay ảnh\n(±20°)',
            'Dịch chuyển\n(±20%)',
            'Lật ngang',
            'Thay đổi độ sáng\n(±20%)',
            'Ảnh tăng cường'
        ]
        
        # Vị trí các bước
        x_positions = [0, 2, 4, 6, 8, 10]
        y_position = 0
        
        # Vẽ các bước
        for i, (step, x) in enumerate(zip(steps, x_positions)):
            # Màu sắc khác nhau cho từng bước
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            # Vẽ box
            box = plt.Rectangle((x-0.8, y_position-0.4), 1.6, 0.8, 
                              facecolor=colors[i], edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(box)
            
            # Thêm text
            ax.text(x, y_position, step, ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Vẽ mũi tên
            if i < len(steps) - 1:
                arrow = plt.arrow(x+0.8, y_position, 0.4, 0, head_width=0.1, head_length=0.1, 
                                fc='black', ec='black', linewidth=2)
        
        # Thêm tiêu đề
        ax.set_title('Sơ Đồ Quy Trình Tăng Cường Dữ Liệu (Data Augmentation)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Thêm thông tin chi tiết
        ax.text(5, -1.5, 
               'Quy trình tăng cường dữ liệu giúp tạo ra nhiều biến thể của ảnh gốc,\n'
               'cải thiện độ chính xác và khả năng tổng quát hóa của mô hình ResNet50.',
               ha='center', fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Cấu hình trục
        ax.set_xlim(-1, 11)
        ax.set_ylim(-2, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Lưu biểu đồ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chart/data_augmentation_flowchart_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Đã tạo sơ đồ tăng cường dữ liệu: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo sơ đồ tăng cường dữ liệu: {str(e)}")
        return None

def create_vector_dimensions_comparison():
    """Tạo biểu đồ so sánh các chiều vector"""
    try:
        # Tạo thư mục chart nếu chưa có
        os.makedirs('chart', exist_ok=True)
        
        # Dữ liệu từ README
        dimensions = [512, 1024, 1536, 2048, 4096]
        memory_savings = [75, 50, 25, 0, -100]  # %
        accuracy = [87, 92, 97, 100, 100]  # %
        speed = [95, 85, 70, 60, 40]  # % (relative speed)
        
        # Tạo subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Biểu đồ 1: Tiết kiệm bộ nhớ
        bars1 = ax1.bar([str(d) for d in dimensions], memory_savings, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('Tiết Kiệm Bộ Nhớ (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Tiết kiệm (%)')
        ax1.grid(True, alpha=0.3)
        for bar, value in zip(bars1, memory_savings):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{value}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Biểu đồ 2: Độ chính xác
        bars2 = ax2.bar([str(d) for d in dimensions], accuracy, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax2.set_title('Độ Chính Xác (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Độ chính xác (%)')
        ax2.set_ylim(80, 105)
        ax2.grid(True, alpha=0.3)
        for bar, value in zip(bars2, accuracy):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        # Biểu đồ 3: Tốc độ
        bars3 = ax3.bar([str(d) for d in dimensions], speed, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax3.set_title('Tốc Độ Tương Đối (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Tốc độ (%)')
        ax3.grid(True, alpha=0.3)
        for bar, value in zip(bars3, speed):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}%', ha='center', va='bottom', fontweight='bold')
        
        # Tiêu đề chung
        fig.suptitle('So Sánh Hiệu Suất Các Chiều Vector ResNet50', fontsize=18, fontweight='bold')
        
        # Lưu biểu đồ
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chart/vector_dimensions_comparison_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Đã tạo biểu đồ so sánh chiều vector: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ so sánh chiều vector: {str(e)}")
        return None

def auto_generate_charts():
    """Tự động tạo tất cả biểu đồ"""
    try:
        charts_created = []
        
        # Tạo biểu đồ hiệu năng
        perf_chart = create_performance_chart()
        if perf_chart:
            charts_created.append(perf_chart)
        
        # Tạo sơ đồ tăng cường dữ liệu
        aug_chart = create_data_augmentation_flowchart()
        if aug_chart:
            charts_created.append(aug_chart)
        
        # Tạo biểu đồ so sánh chiều vector
        dim_chart = create_vector_dimensions_comparison()
        if dim_chart:
            charts_created.append(dim_chart)
        
        logger.info(f"Đã tạo {len(charts_created)} biểu đồ: {charts_created}")
        return charts_created
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ tự động: {str(e)}")
        return []

def get_model_for_dimensions(dimensions):
    """Get or create model for specified dimensions using only ResNet50"""
    global _models
    
    with _model_lock:
        if dimensions not in _models:
            logger.info(f"Creating new model for {dimensions} dimensions")
            # Always use ResNet50 as base model
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False
            
            if dimensions == 2048:
                # Default ResNet50 with GlobalMaxPooling2D (2048 dimensions)
                model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
                logger.info("Created ResNet50 with GlobalMaxPooling2D for 2048 dimensions")
            elif dimensions == 4096:
                # ResNet50 with additional Dense layer to expand to 4096
                model = tf.keras.Sequential([
                    base_model, 
                    GlobalMaxPooling2D(),
                    tf.keras.layers.Dense(4096, activation='relu')
                ])
                logger.info("Created ResNet50 with Dense(4096) for 4096 dimensions")
            elif dimensions in [512, 1024, 1536]:
                # ResNet50 with Dense layer to reduce dimensions
                model = tf.keras.Sequential([
                    base_model, 
                    GlobalMaxPooling2D(),
                    tf.keras.layers.Dense(dimensions, activation='relu')
                ])
                logger.info(f"Created ResNet50 with Dense({dimensions}) for {dimensions} dimensions")
            else:
                # Default to ResNet50 with GlobalMaxPooling2D
                model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
                logger.info(f"Created default ResNet50 with GlobalMaxPooling2D for {dimensions} dimensions")
            
            # Test the model output shape
            test_input = tf.random.normal((1, 224, 224, 3))
            test_output = model(test_input)
            logger.info(f"Model test output shape: {test_output.shape}")
            
            _models[dimensions] = model
            logger.info(f"Created ResNet50 model for {dimensions} dimensions")
        else:
            logger.info(f"Using cached model for {dimensions} dimensions")
        
        return _models[dimensions]

# Initialize default model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
_models[2048] = model  # Store default model

# Thêm biến toàn cục để lưu cache
_features_cache = {
    'data': None,
    'last_update': None,
    'lock': threading.Lock()
}

# Thời gian cache hết hạn (30 phút)
CACHE_EXPIRY = 30 * 60  # seconds

# Định nghĩa mô hình deep learning
def build_model(input_dim):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output: relevance score (0 to 1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Azure Blob Storage URL
AZURE_URL = "https://dbimage.blob.core.windows.net/images"
API_URL = os.getenv("API_URL", "http://localhost:8080/api/v1/public")

def preprocess_image(image_data):
    """Enhanced image preprocessing for better feature extraction"""
    try:
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply basic enhancement
        img = enhance_image(img)

        # Resize while maintaining aspect ratio
        h, w = img.shape[:2]
        if min(h, w) < MIN_IMAGE_SIZE:
            scale = MIN_IMAGE_SIZE / min(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)
        elif max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

        return img, None
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, str(e)

def enhance_image(img):
    """Apply basic image enhancement techniques"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 9.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return img

@lru_cache(maxsize=1000)
def extract_features(img_array_bytes):
    """
    Extract features from image array with caching.
    Args:
        img_array_bytes: bytes representation of the image array
    Returns:
        Normalized feature vector
    """
    try:
        # Convert bytes back to numpy array
        img = np.frombuffer(img_array_bytes, dtype=np.uint8).reshape(-1, 224, 224, 3)
        
        # Convert to tensor and normalize
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_tensor = preprocess_input(img_tensor)
        
        # Get the appropriate model for current dimensions
        current_model = get_model_for_dimensions(VECTOR_DIMENSIONS_CONFIG)
        logger.info(f"Using model for {VECTOR_DIMENSIONS_CONFIG} dimensions")
        
        # Extract features using the model
        features = current_model(img_tensor)
        logger.info(f"Raw features shape: {features.shape}")
        
        # Convert to numpy and normalize
        features = features.numpy()
        features = features / np.linalg.norm(features)
        logger.info(f"Normalized features shape: {features.shape}")
        
        # Check if dimensions match expected - FIXED LOGIC
        expected_dimensions = VECTOR_DIMENSIONS_CONFIG
        actual_dimensions = features.shape[-1] if len(features.shape) > 1 else features.size
        
        logger.info(f"Expected dimensions: {expected_dimensions}, Actual dimensions: {actual_dimensions}")
        
        if actual_dimensions != expected_dimensions:
            logger.error(f"Model output dimensions {actual_dimensions} don't match expected {expected_dimensions}")
            return None
        
        flattened_features = features.flatten()  # Return 1D array
        logger.info(f"Final flattened features shape: {flattened_features.shape}")
        return flattened_features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def get_vector_dimensions():
    """Get the expected vector dimensions for the current model"""
    # ResNet50 with GlobalMaxPooling2D produces 2048-dimensional features
    return VECTOR_DIMENSIONS_CONFIG

def send_vector_features(image_id, features):
    """
    Send vector features to the API for storage.
    Args:
        image_id: ID of the image
        features: numpy array of features
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert features to string format
        features_str = ','.join(map(str, features))
        
        # Get vector dimensions
        vector_dimensions = len(features)
        
        # Send update request to API
        update_response = requests.put(
            f"{API_URL}/image/update/vector_feature",
            json={
                "id": image_id,
                "vectorFeatures": features_str,
                "vectorDimensions": vector_dimensions
            }
        )
        
        if update_response.status_code == 200:
            logger.info(f"Successfully updated vector features for image {image_id} with {vector_dimensions} dimensions")
            return True
        else:
            logger.error(f"Failed to update vector features: {update_response.status_code}")
            logger.error(f"Response: {update_response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending vector features: {str(e)}")
        return False

def augment_and_save_image(img_path, image_id, n_aug=5):
    """
    Augment image and save both original and augmented images to database
    Args:
        img_path: Path to original image
        image_id: ID of original image
        n_aug: Number of augmented images to generate
    Returns:
        List of (augmented_image_id, features) tuples
    """
    try:
        # Load and preprocess original image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Configure augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )
        
        aug_iter = datagen.flow(img_array, batch_size=1)
        aug_results = []
        
        # Process each augmented image
        for i in range(n_aug):
            try:
                # Generate augmented image
                aug_img = next(aug_iter)[0].astype('uint8')
                
                # Convert to bytes for feature extraction
                aug_img_bytes = aug_img.tobytes()
                
                # Extract features
                features = extract_features(aug_img_bytes)
                if features is None or not validate_features(features):
                    logger.warning(f"Invalid features for augmented image {i} of {image_id}")
                    continue
                
                # Create unique ID for augmented image
                aug_image_id = f"{image_id}_aug_{i}"
                
                # Convert features to string
                features_str = ','.join(map(str, features))
                
                # Save to database
                update_response = requests.put(
                    f"{API_URL}/image/update/vector_feature",
                    json={
                        "id": aug_image_id,
                        "vectorFeatures": features_str,
                        "originalImageId": image_id,
                        "isAugmented": True
                    }
                )
                
                if update_response.status_code == 200:
                    logger.info(f"Successfully saved augmented image {i} for {image_id}")
                    aug_results.append((aug_image_id, features))
                else:
                    logger.error(f"Failed to save augmented image {i} for {image_id}")
                    
            except Exception as e:
                logger.error(f"Error processing augmented image {i} for {image_id}: {str(e)}")
                continue
                
        return aug_results
        
    except Exception as e:
        logger.error(f"Error in augment_and_save_image for {image_id}: {str(e)}")
        return []

def process_single_image(image_url, image_id):
    """Process a single image with enhanced error handling and validation"""
    try:
        start_time = time.time()  # Bắt đầu đo thời gian
        
        # Download image
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            logger.error(f"Failed to download image: {response.status_code}")
            return False, f"Failed to download image: {response.status_code}"

        # Save original image temporarily
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{image_id}_original.jpg")
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        try:
            # Preprocess image
            img, error = preprocess_image(response.content)
            if error:
                logger.error(f"Error preprocessing image: {error}")
                return False, error

            # Extract features for original image
            features = extract_features(img.tobytes())
            if features is None:
                logger.error("Feature extraction failed")
                return False, "Feature extraction failed"

            # Validate features
            if not validate_features(features):
                logger.error("Invalid feature vector")
                return False, "Invalid feature vector"

            # Send original image features to server
            success = send_vector_features(image_id, features)
            if not success:
                logger.error("Failed to save features to server")
                return False, "Failed to save features to server"

            # Generate and save augmented images
            aug_results = augment_and_save_image(temp_path, image_id)
            logger.info(f"Generated {len(aug_results)} augmented images for {image_id}")

            # Tính thời gian xử lý và ghi lại
            extraction_time = time.time() - start_time
            record_extraction_time(extraction_time)
            logger.info(f"Successfully processed image {image_url} for image {image_id} in {extraction_time:.2f} seconds")
            
            return True, None
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing image {image_url}: {str(e)}")
        return False, str(e)

def validate_features(features):
    """Validate extracted features"""
    try:
        # Check if features are valid
        if features is None or not isinstance(features, np.ndarray):
            logger.error("Features is None or not a numpy array")
            return False
            
        # Check feature dimensions
        if features.size == 0:
            logger.error("Empty feature vector")
            return False
            
        # Reshape if necessary (should be 1D array)
        if len(features.shape) > 1:
            features = features.flatten()
            
        # Check if vector has expected dimensions from config
        expected_dimensions = get_vector_dimensions()
        if features.size != expected_dimensions:
            logger.error(f"Invalid feature dimensions: {features.size} != {expected_dimensions}")
            return False
            
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.error("Features contain NaN or infinite values")
            return False
            
        # Check feature norm
        norm = np.linalg.norm(features)
        if norm == 0:
            logger.error("Zero vector")
            return False
            
        # Check if vector is already normalized
        if not np.isclose(norm, 1.0, atol=1e-6):
            logger.debug(f"Vector not normalized (norm={norm}), will normalize later")
            
        return True
    except Exception as e:
        logger.error(f"Error validating features: {str(e)}")
        return False

def reduce_dimensions(features, target_dim=512):
    """
    Reduce feature dimensions using PCA if needed
    Args:
        features: numpy array of features (n_samples, n_features) or (n_features,)
        target_dim: target dimension (default: 512)
    Returns:
        Reduced features
    """
    try:
        # Ensure features is 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        original_dim = features.shape[1]
        
        if original_dim == target_dim:
            return features
            
        logger.info(f"Applying PCA to reduce dimensions from {original_dim} to {target_dim}")
        
        pca = PCA(n_components=target_dim)
        reduced_features = pca.fit_transform(features)
        
        # Calculate explained variance ratio
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance ratio: {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%)")
        
        # Normalize the reduced features
        norms = np.linalg.norm(reduced_features, axis=1)
        reduced_features = reduced_features / norms[:, np.newaxis]
        
        logger.info(f"Successfully reduced features from {original_dim} to {target_dim} dimensions")
        return reduced_features
    except Exception as e:
        logger.error(f"Error reducing dimensions: {str(e)}")
        return None

def background_extraction(process_all=False):
    """Enhanced background extraction process"""
    global extraction_status, extraction_thread, stop_extraction
    
    try:
        extraction_status = {
            'is_running': True,
            'total_images': 0,
            'processed_count': 0,
            'failed_count': 0,
            'current_image': '',
            'error': None,
            'progress': 0
        }
        
        # Get all images
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            raise Exception("Failed to fetch images")
            
        data = response.json()
        # Check if the response has a 'data' field
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []
            
        if not process_all:
            images = [img for img in images if not img.get('vectorFeatures')]
            
        extraction_status['total_images'] = len(images)
        
        # Process images in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for img in images:
                if stop_extraction:
                    break
                    
                # Get the image URL and product ID from the correct fields
                image_url = img.get('url') or img.get('imageUrl')
                image_id = img.get('id')
                
                if not image_url or not image_id:
                    logger.warning(f"Skipping image with missing URL or image ID: {img}")
                    extraction_status['failed_count'] += 1
                    continue
                    
                extraction_status['current_image'] = image_url
                future = executor.submit(
                    process_single_image,
                    image_url,
                    image_id
                )
                futures.append(future)
                
            # Process results as they complete
            for future in futures:
                if stop_extraction:
                    break
                    
                success, error = future.result()
                if success:
                    extraction_status['processed_count'] += 1
                else:
                    extraction_status['failed_count'] += 1
                    logger.error(f"Failed to process image: {error}")
                    
                # Update progress
                total = extraction_status['processed_count'] + extraction_status['failed_count']
                extraction_status['progress'] = (total / extraction_status['total_images']) * 100 if extraction_status['total_images'] > 0 else 0
                
    except Exception as e:
        logger.error(f"Extraction process failed: {str(e)}")
        extraction_status['error'] = str(e)
    finally:
        extraction_status['is_running'] = False
        extraction_thread = None
        stop_extraction = False

@app.route("/")
def home():
    return jsonify({"message": "Welcome to AI Image Similarity API!"})

def get_cached_features():
    """
    Lấy dữ liệu vector features từ cache hoặc API
    Returns:
        tuple: (features, filenames, valid_indices) hoặc (None, None, None) nếu có lỗi
    """
    global _features_cache
    
    current_time = time.time()
    
    with _features_cache['lock']:
        # Kiểm tra cache có hợp lệ không
        if (_features_cache['data'] is not None and 
            _features_cache['last_update'] is not None and 
            current_time - _features_cache['last_update'] < CACHE_EXPIRY):
            logger.info("Sử dụng dữ liệu từ cache")
            return _features_cache['data']
            
        # Cache hết hạn hoặc chưa có, lấy dữ liệu mới
        logger.info("Cache hết hạn hoặc chưa có, lấy dữ liệu mới từ API")
        features, filenames, valid_indices = get_all_image_features()
        
        if features is not None:
            _features_cache['data'] = (features, filenames, valid_indices)
            _features_cache['last_update'] = current_time
            logger.info("Đã cập nhật cache với dữ liệu mới")
            
        return features, filenames, valid_indices

def invalidate_features_cache():
    """Xóa cache để force lấy dữ liệu mới"""
    global _features_cache
    with _features_cache['lock']:
        _features_cache['data'] = None
        _features_cache['last_update'] = None
        logger.info("Đã xóa cache")

@app.route('/api/find_similar', methods=['POST'])
def find_similar_images():
    """Find similar images based on a query image URL or uploaded file"""
    try:
        search_start_time = time.time()  # Bắt đầu đo thời gian tìm kiếm
        
        # Lấy features từ cache
        features, filenames, valid_indices = get_cached_features()
        if features is None or filenames is None:
            return jsonify({"error": "No valid images found in database"}), 400

        # Process query image
        if request.is_json:
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({"error": "No URL provided. Please send a JSON payload with a 'url' field."}), 400

            image_url = data['url']
            if not image_url:
                return jsonify({"error": "Empty URL provided."}), 400

            # Extract features directly from URL
            query_features = extract_features_from_url(image_url)
            if query_features is None:
                return jsonify({"error": "Failed to extract features from query image"}), 400

        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            # Save file temporarily
            temp_dir = 'temp'
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)

            try:
                # Extract features from saved file
                query_features = extract_features_from_url(temp_path)
                if query_features is None:
                    return jsonify({"error": "Failed to extract features from uploaded image"}), 400
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        else:
            return jsonify({"error": "No URL or file provided. Please send a JSON payload with a 'url' field or upload a file."}), 400

        # Calculate cosine similarity with all images
        similarities = np.dot(features, query_features.T).flatten()
        
        # Get top 10 similar images
        top_indices = np.argsort(similarities)[::-1][:10]
        
        similar_images = [
            {
                "url": filenames[idx],
                "similarity": float(similarities[idx]),
                "index": int(valid_indices[idx])
            }
            for idx in top_indices if similarities[idx] > 0
        ]

        # Tính thời gian tìm kiếm và ghi lại
        search_time_ms = (time.time() - search_start_time) * 1000  # Chuyển sang milliseconds
        record_search_time(search_time_ms)
        
        # Tính độ chính xác dựa trên similarity scores
        if similar_images:
            avg_similarity = sum(img["similarity"] for img in similar_images) / len(similar_images)
            record_accuracy_score(avg_similarity)
        
        logger.info(f"Search completed in {search_time_ms:.2f}ms with {len(similar_images)} results")

        return jsonify({
            "similar_images": similar_images,
            "search_time_ms": round(search_time_ms, 2),
            "total_images_searched": len(features)
        }), 200

    except Exception as e:
        logger.error(f"Error in find_similar_images route: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/status", methods=["GET"])
def get_extraction_status():
    """Lấy trạng thái hiện tại của quá trình trích xuất đặc trưng"""
    with extraction_lock:
        return jsonify({
            "is_running": extraction_status["is_running"],
            "total_images": extraction_status["total_images"],
            "processed_count": extraction_status["processed_count"],
            "failed_count": extraction_status["failed_count"],
            "current_image": extraction_status["current_image"],
            "error": extraction_status["error"],
            "progress": (extraction_status["processed_count"] + extraction_status["failed_count"]) / 
                       extraction_status["total_images"] * 100 if extraction_status["total_images"] > 0 else 0
        })

@app.route("/api/extraction/start", methods=["POST"])
def start_extraction():
    """Bắt đầu quá trình trích xuất đặc trưng"""
    global extraction_status
    
    with extraction_lock:
        if extraction_status["is_running"]:
            return jsonify({"error": "Đang có quá trình trích xuất đang chạy"}), 400
    
    data = request.get_json() or {}
    process_all = data.get("process_all", False)
    
    # Chạy trong thread riêng
    thread = threading.Thread(target=background_extraction, args=(process_all,))
    thread.daemon = True
    thread.start()
    
    response_data = {"message": "Đã bắt đầu quá trình trích xuất đặc trưng"}
    
    # Tự động tạo biểu đồ
    response_data = add_chart_generation_to_response(response_data, 'extraction_start')
    
    return jsonify(response_data)

@app.route("/api/extraction/stop", methods=["POST"])
def stop_extraction():
    """Dừng quá trình trích xuất đặc trưng"""
    global extraction_status
    
    with extraction_lock:
        if not extraction_status["is_running"]:
            return jsonify({"error": "Không có quá trình trích xuất nào đang chạy"}), 400
        extraction_status["is_running"] = False
    
    return jsonify({"message": "Đã gửi yêu cầu dừng quá trình trích xuất"})

@app.route("/api/extraction/stats", methods=["GET"])
def get_extraction_stats():
    """Lấy thống kê về trạng thái trích xuất đặc trưng của tất cả ảnh"""
    try:
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            return jsonify({"error": "Không thể lấy danh sách ảnh từ API"}), 500

        data = response.json()
        if "data" not in data:
            return jsonify({"error": "Dữ liệu API không hợp lệ"}), 500

        images = data["data"]
        total_images = len(images)
        images_with_features = sum(1 for img in images if img.get("vectorFeatures") and img.get("vectorFeatures").strip())
        
        # Thống kê theo product
        product_stats = {}
        for img in images:
            product_id = img.get("productId")
            if product_id:
                if product_id not in product_stats:
                    product_stats[product_id] = {"total": 0, "with_features": 0}
                product_stats[product_id]["total"] += 1
                if img.get("vectorFeatures") and img.get("vectorFeatures").strip():
                    product_stats[product_id]["with_features"] += 1

        response_data = {
            "total_images": total_images,
            "images_with_features": images_with_features,
            "images_without_features": total_images - images_with_features,
            "completion_percentage": (images_with_features / total_images * 100) if total_images > 0 else 0,
            "product_stats": product_stats
        }
        
        # Tự động tạo biểu đồ
        response_data = add_chart_generation_to_response(response_data, 'extraction_stats')

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_all_image_features():
    """
    Fetch all image features from the API and validate them.
    Returns a tuple of (features, filenames, valid_indices) where:
    - features: numpy array of normalized feature vectors
    - filenames: list of image URLs
    - valid_indices: list of indices of valid features
    """
    try:
        # Fetch all images from API
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            logger.error(f"Failed to fetch images: {response.status_code}")
            return None, None, None

        data = response.json()
        logger.info(f"API Response structure: {type(data)}")
        
        # Handle different API response structures
        if isinstance(data, dict):
            if 'data' in data:
                images = data['data']
                logger.info(f"Found {len(images)} images in 'data' field")
            else:
                logger.error("API response is a dict but has no 'data' field")
                return None, None, None
        elif isinstance(data, list):
            images = data
            logger.info(f"Found {len(images)} images in direct list")
        else:
            logger.error(f"Unexpected API response type: {type(data)}")
            return None, None, None

        if not images:
            logger.warning("No images found in the database")
            return None, None, None

        # Initialize lists to store valid features and filenames
        valid_features = []
        valid_filenames = []
        valid_indices = []

        # Process each image
        for idx, img in enumerate(images):
            try:
                # Check if image has vector features
                vector_features = img.get('vectorFeatures') or img.get('vector_features')
                if not vector_features:
                    logger.warning(f"Image {idx} has no vector features")
                    continue

                # Get image URL from 'path' field
                image_url = img.get('path')
                if not image_url:
                    logger.warning(f"Image {idx} has no path")
                    continue

                # Convert string representation of vector to numpy array
                try:
                    # Parse comma-separated string (as used in feature_extractor.py)
                    vector = np.array([float(x.strip()) for x in vector_features.split(',')])
                    
                    # Log vector shape for debugging
                    logger.debug(f"Vector shape for image {image_url}: {vector.shape}")

                    # Validate vector
                    if not validate_features(vector):
                        logger.warning(f"Invalid feature vector for image {image_url}")
                        continue

                    # Vector should already be normalized (as done in feature_extractor.py)
                    # But we'll check and normalize again just to be safe
                    norm = np.linalg.norm(vector)
                    if norm > 0 and not np.isclose(norm, 1.0, atol=1e-6):
                        vector = vector / norm

                    # Add to valid features
                    valid_features.append(vector)
                    valid_filenames.append(image_url)
                    valid_indices.append(idx)
                    logger.debug(f"Successfully processed image {idx}: {image_url}")

                except Exception as e:
                    logger.error(f"Error processing vector for image {image_url}: {str(e)}")
                    continue

            except Exception as e:
                logger.error(f"Error processing image {idx}: {str(e)}")
                continue

        if not valid_features:
            logger.warning("No valid features found after processing all images")
            return None, None, None

        # Convert to numpy arrays
        features = np.array(valid_features)
        filenames = np.array(valid_filenames)
        valid_indices = np.array(valid_indices)

        logger.info(f"Successfully loaded {len(valid_features)} valid feature vectors")
        return features, filenames, valid_indices

    except Exception as e:
        logger.error(f"Error in get_all_image_features: {str(e)}")
        return None, None, None

def find_similar_images(query_image, top_k=5):
    """
    Find similar images using cosine similarity.
    Args:
        query_image: URL or file path of the query image
        top_k: Number of similar images to return
    Returns:
        List of similar images with their similarity scores
    """
    try:
        # Get all features
        features, filenames, valid_indices = get_all_image_features()
        if features is None or filenames is None:
            return []

        # Extract features from query image
        query_features = extract_features_from_url(query_image)
        if query_features is None:
            return []

        # Calculate cosine similarity
        similarities = np.dot(features, query_features.T).flatten()

        # Get top k similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                results.append({
                    'url': filenames[idx],
                    'similarity': float(similarities[idx]),
                    'index': int(valid_indices[idx])
                })

        return results

    except Exception as e:
        logger.error(f"Error in find_similar_images: {str(e)}")
        return []

def extract_features_from_url(image_path_or_url):
    """
    Extract features from an image URL or local file path.
    Args:
        image_path_or_url: URL of the image or local file path
    Returns:
        Normalized feature vector or None if extraction fails
    """
    try:
        extraction_start_time = time.time()  # Bắt đầu đo thời gian
        
        # Check if it's a URL or local file path
        if image_path_or_url.startswith(('http://', 'https://')):
            # Download image from URL
            response = requests.get(image_path_or_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to download image: {response.status_code}")
                return None
            img_data = response.content
        else:
            # Read local file
            if not os.path.exists(image_path_or_url):
                logger.error(f"Local file not found: {image_path_or_url}")
                return None
            with open(image_path_or_url, 'rb') as f:
                img_data = f.read()

        # Preprocess image
        img, error = preprocess_image(img_data)
        if error:
            logger.error(f"Error preprocessing image: {error}")
            return None

        # Resize image to model input size
        img = cv2.resize(img, (224, 224))
        
        # Convert image to bytes for caching
        img_bytes = img.tobytes()
        
        # Extract features using cached function
        features = extract_features(img_bytes)
        if features is None:
            return None

        # Validate features
        if not validate_features(features):
            logger.error("Invalid features extracted")
            return None

        # Ghi lại thời gian trích xuất cho ảnh đơn lẻ
        extraction_time = time.time() - extraction_start_time
        record_extraction_time(extraction_time)
        logger.info(f"Feature extraction completed in {extraction_time:.2f} seconds for {image_path_or_url}")

        return features

    except Exception as e:
        logger.error(f"Error extracting features from {image_path_or_url}: {str(e)}")
        return None

def preprocess_for_model(img):
    """
    Preprocess image for model input.
    Args:
        img: numpy array of image
    Returns:
        Preprocessed tensor
    """
    try:
        # Resize to model input size
        img = cv2.resize(img, (224, 224))
        
        # Convert to tensor and normalize
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_tensor = preprocess_input(img_tensor)
        
        # Add batch dimension
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image for model: {str(e)}")
        return None

# Add new endpoint to get processing details
@app.route('/api/extraction/details', methods=['GET'])
def get_extraction_details():
    """Get detailed processing information for an image"""
    try:
        image_url = request.args.get('imageUrl')
        if not image_url:
            return jsonify({'error': 'Image URL is required'}), 400

        # Get image details from API
        response = requests.get(f"{API_URL}/images", params={'url': image_url})
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image details'}), 500

        image_data = response.json()
        if not image_data:
            return jsonify({'error': 'Image not found'}), 404

        # Get processing details
        details = {
            'qualityScore': image_data.get('quality_score', 0),
            'processingTime': image_data.get('processing_time', 0),
            'imageSize': {
                'width': image_data.get('width', 0),
                'height': image_data.get('height', 0)
            },
            'featureDimensions': len(image_data.get('vector_features', [])) if image_data.get('vector_features') else 0
        }

        return jsonify(details)

    except Exception as e:
        logger.error(f"Error getting extraction details: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/extraction/update_all", methods=["POST"])
def update_all_vector_features():
    """Update vector features for all images in the database"""
    try:
        # Get all images from API
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch images from API"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        if not images:
            return jsonify({"error": "No images found in database"}), 400

        # Process each image
        success_count = 0
        failed_count = 0
        failed_images = []

        for img in images:
            try:
                # Get image URL and ID
                image_url = img.get('url') or img.get('imageUrl')
                image_id = img.get('id')

                if not image_url or not image_id:
                    logger.warning(f"Skipping image with missing URL or image ID: {img}")
                    failed_count += 1
                    failed_images.append({"id": image_id, "error": "Missing URL or ID"})
                    continue

                # Extract features
                features = extract_features_from_url(image_url)
                if features is None:
                    logger.error(f"Failed to extract features for image {image_url}")
                    failed_count += 1
                    failed_images.append({"id": image_id, "error": "Feature extraction failed"})
                    continue

                # Convert features to string format
                features_str = ','.join(map(str, features))

                # Update vector features in API
                update_response = requests.put(
                    f"{API_URL}/image/update/vector_feature",
                    json={
                        "id": image_id,
                        "vectorFeatures": features_str
                    }
                )

                if update_response.status_code == 200:
                    success_count += 1
                    logger.info(f"Successfully updated features for image {image_id}")
                else:
                    failed_count += 1
                    failed_images.append({
                        "id": image_id,
                        "error": f"API update failed: {update_response.status_code}"
                    })
                    logger.error(f"Failed to update features for image {image_id}: {update_response.status_code}")

            except Exception as e:
                failed_count += 1
                failed_images.append({
                    "id": image_id if 'image_id' in locals() else "unknown",
                    "error": str(e)
                })
                logger.error(f"Error processing image: {str(e)}")

        # Sau khi cập nhật xong, xóa cache
        invalidate_features_cache()
        logger.info("Đã cập nhật xong tất cả vector features và xóa cache")

        return jsonify({
            "message": "Vector feature update completed",
            "total_images": len(images),
            "success_count": success_count,
            "failed_count": failed_count,
            "failed_images": failed_images
        }), 200

    except Exception as e:
        logger.error(f"Error in update_all_vector_features: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/process_new", methods=["POST"])
def process_new_images():
    """Xử lý và trích xuất đặc trưng cho ảnh mới từ shop"""
    try:
        logger.info("=== BẮT ĐẦU XỬ LÝ ẢNH MỚI ===")
        
        # Lấy danh sách ảnh từ API
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            logger.error(f"Không thể lấy danh sách ảnh: {response.status_code}")
            return jsonify({"error": "Không thể lấy danh sách ảnh từ API"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        if not images:
            logger.warning("Không có ảnh nào trong hệ thống")
            return jsonify({"message": "Không có ảnh nào cần xử lý"}), 200

        # Lọc ra những ảnh chưa có vector features
        images_to_process = [
            img for img in images 
            if not img.get('vectorFeatures') or not img.get('vectorFeatures').strip()
        ]

        logger.info(f"Tổng số ảnh: {len(images)}")
        logger.info(f"Số ảnh cần xử lý: {len(images_to_process)}")

        if not images_to_process:
            logger.info("Không có ảnh nào cần xử lý")
            return jsonify({"message": "Không có ảnh nào cần xử lý"}), 200

        # Xử lý từng ảnh
        results = {
            "total": len(images_to_process),
            "success": 0,
            "failed": 0,
            "failed_images": []
        }

        for img in images_to_process:
            try:
                image_id = img.get('id')
                image_path = img.get('path')
                
                if not image_id or not image_path:
                    logger.warning(f"Ảnh thiếu thông tin: ID={image_id}, Path={image_path}")
                    results["failed"] += 1
                    results["failed_images"].append({
                        "id": image_id,
                        "error": "Thiếu thông tin ảnh"
                    })
                    continue

                logger.info(f"Đang xử lý ảnh: {image_path} (ID: {image_id})")

                # Trích xuất đặc trưng
                features = extract_features_from_url(image_path)
                if features is None:
                    logger.error(f"Không thể trích xuất đặc trưng cho ảnh {image_id}")
                    results["failed"] += 1
                    results["failed_images"].append({
                        "id": image_id,
                        "error": "Không thể trích xuất đặc trưng"
                    })
                    continue

                # Chuyển đổi features thành string
                features_str = ','.join(map(str, features))

                # Cập nhật vector features
                update_response = requests.put(
                    f"{API_URL}/image/update/vector_feature",
                    json={
                        "id": image_id,
                        "vectorFeatures": features_str
                    }
                )

                if update_response.status_code == 200:
                    logger.info(f"Đã cập nhật thành công vector features cho ảnh {image_id}")
                    results["success"] += 1
                else:
                    logger.error(f"Không thể cập nhật vector features: {update_response.status_code}")
                    results["failed"] += 1
                    results["failed_images"].append({
                        "id": image_id,
                        "error": f"Lỗi cập nhật: {update_response.status_code}"
                    })

            except Exception as e:
                logger.error(f"Lỗi khi xử lý ảnh {image_id}: {str(e)}")
                results["failed"] += 1
                results["failed_images"].append({
                    "id": image_id,
                    "error": str(e)
                })

        # Sau khi xử lý xong, xóa cache
        invalidate_features_cache()
        logger.info("Đã xử lý xong ảnh mới và xóa cache")

        logger.info("=== KẾT THÚC XỬ LÝ ẢNH MỚI ===")
        logger.info(f"Kết quả: {results['success']} thành công, {results['failed']} thất bại")

        return jsonify({
            "message": "Đã xử lý xong ảnh mới",
            "results": results
        }), 200

    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/update_single", methods=["POST"])
def update_single_vector_feature():
    """Cập nhật vector features cho một ảnh"""
    try:
        logger.info("=== BẮT ĐẦU CẬP NHẬT VECTOR FEATURES CHO ẢNH ĐƠN LẺ ===")
        data = request.get_json()
        logger.info(f"Dữ liệu nhận được: {data}")
        
        if not data:
            logger.error("Không có dữ liệu trong request")
            return jsonify({"error": "Không có dữ liệu"}), 400
            
        # Kiểm tra cả path và id
        if 'path' not in data or 'id' not in data:
            logger.error("Thiếu thông tin ảnh trong request")
            return jsonify({"error": "Cần cung cấp cả path và id của ảnh"}), 400

        image_path = data['path']
        image_id = data['id']
        force_update = data.get('forceUpdate', False)  # Lấy flag forceUpdate, mặc định là False
        logger.info(f"Đang xử lý ảnh: {image_path} (ID: {image_id}, forceUpdate: {force_update})")

        # Lấy tất cả ảnh từ API và tìm ảnh cần xử lý
        try:
            logger.info("Lấy danh sách ảnh từ API public")
            response = requests.get(f"{API_URL}/images")
            
            if response.status_code != 200:
                logger.error(f"Lỗi khi lấy danh sách ảnh từ API public: {response.status_code}")
                logger.error(f"Response text: {response.text}")
                return jsonify({"error": "Không thể lấy danh sách ảnh từ hệ thống"}), 500

            response_data = response.json()
            if not isinstance(response_data, dict) or 'data' not in response_data:
                logger.error(f"Format response không hợp lệ: {response_data}")
                return jsonify({"error": "Format dữ liệu không hợp lệ"}), 500

            images = response_data['data']
            logger.info(f"Tổng số ảnh từ API: {len(images)}")

            # Tìm ảnh cần xử lý
            image_data = next((img for img in images if img.get('id') == image_id), None)
            if not image_data:
                logger.error(f"Không tìm thấy ảnh với ID {image_id} trong danh sách")
                return jsonify({"error": "Không tìm thấy ảnh trong hệ thống"}), 404

            vector_features = image_data.get('vectorFeatures')
            expected_dimensions = get_vector_dimensions()
            
            # Log chi tiết về vector features
            logger.info(f"Vector features của ảnh {image_id}:")
            logger.info(f"- Có vector features: {bool(vector_features)}")
            logger.info(f"- Độ dài vector: {len(vector_features) if vector_features else 0}")
            logger.info(f"- Vector features: {vector_features[:100] + '...' if vector_features and len(vector_features) > 100 else vector_features}")

            # Kiểm tra kỹ hơn về vector features và chiều vector
            has_valid_features = (
                vector_features and 
                isinstance(vector_features, str) and 
                vector_features.strip() and 
                len(vector_features.split(',')) == expected_dimensions  # Đảm bảo vector có đúng số chiều
            )

            # Nếu có vector features hợp lệ và không yêu cầu force update thì bỏ qua
            if has_valid_features and not force_update:
                logger.info(f"Ảnh {image_id} đã có vector features hợp lệ, bỏ qua")
                return jsonify({
                    "message": "Ảnh đã có vector features",
                    "id": image_id,
                    "path": image_path,
                    "status": "skipped",
                    "vector_length": len(vector_features.split(',')),
                    "expected_dimensions": expected_dimensions,
                    "is_consistent": True
                }), 200
            else:
                if has_valid_features:
                    logger.info(f"Ảnh {image_id} đã có vector features nhưng sẽ được cập nhật lại do forceUpdate=True")
                else:
                    logger.info(f"Ảnh {image_id} chưa có vector features hợp lệ, tiếp tục xử lý")
                
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra thông tin ảnh: {str(e)}")
            logger.error(f"Chi tiết lỗi: {str(e)}")
            return jsonify({"error": "Lỗi khi kiểm tra thông tin ảnh từ hệ thống"}), 500

        # Trích xuất đặc trưng
        logger.info("Bắt đầu trích xuất đặc trưng")
        features = extract_features_from_url(image_path)
        if features is None:
            logger.error(f"Không thể trích xuất đặc trưng cho ảnh {image_id}")
            return jsonify({"error": "Không thể trích xuất đặc trưng"}), 400

        # Kiểm tra chiều vector sau khi trích xuất - CHỈ KIỂM TRA KHI KHÔNG PHẢI FORCE UPDATE
        actual_dimensions = len(features)
        if not force_update and actual_dimensions != expected_dimensions:
            logger.error(f"Chiều vector không đúng: {actual_dimensions} != {expected_dimensions}")
            return jsonify({
                "error": f"Chiều vector không đúng: {actual_dimensions} != {expected_dimensions}",
                "actual_dimensions": actual_dimensions,
                "expected_dimensions": expected_dimensions
            }), 400

        # Chuyển đổi features thành string
        features_str = ','.join(map(str, features))
        logger.info(f"Đã trích xuất và chuẩn hóa đặc trưng thành công, độ dài vector: {len(features)}")

        # Cập nhật vector features
        try:
            logger.info(f"Gửi yêu cầu cập nhật vector features cho ảnh {image_id}")
            update_response = requests.put(
                f"{API_URL}/image/update/vector_feature",
                json={
                    "id": image_id,
                    "vectorFeatures": features_str,
                    "vectorDimensions": actual_dimensions
                }
            )
            
            if update_response.status_code == 200:
                # Sau khi cập nhật thành công, xóa cache
                invalidate_features_cache()
                logger.info(f"Đã cập nhật thành công vector features cho ảnh {image_id} và xóa cache")
                return jsonify({
                    "message": "Cập nhật vector features thành công",
                    "id": image_id,
                    "path": image_path,
                    "status": "success",
                    "vector_length": len(features),
                    "expected_dimensions": expected_dimensions,
                    "is_consistent": True
                }), 200
            else:
                logger.error(f"Lỗi khi cập nhật vector features: {update_response.status_code}")
                logger.error(f"Chi tiết lỗi: {update_response.text}")
                return jsonify({
                    "error": f"Lỗi khi cập nhật vector features: {update_response.status_code}",
                    "details": update_response.text,
                    "status": "error"
                }), update_response.status_code
                
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật vector features: {str(e)}")
            logger.error(f"Chi tiết lỗi: {str(e)}")
            return jsonify({
                "error": "Lỗi khi cập nhật vector features",
                "status": "error"
            }), 500

    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        logger.error(f"Chi tiết lỗi: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/api/extraction/product_images", methods=["GET"])
def get_product_images():
    """Get all images for a specific product"""
    try:
        product_id = request.args.get('productId')
        if not product_id:
            return jsonify({"error": "Missing productId parameter"}), 400

        # Get images from API
        response = requests.get(f"{API_URL}/images", params={"productId": product_id})
        if response.status_code != 200:
            logger.error(f"Failed to fetch images: {response.status_code}")
            return jsonify({"error": "Failed to fetch images"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        # Format response
        formatted_images = []
        for img in images:
            formatted_images.append({
                'id': img.get('id'),
                'url': img.get('url') or img.get('imageUrl'),
                'vectorFeatures': img.get('vectorFeatures'),
                'productId': img.get('productId')
            })

        return jsonify({
            "productId": product_id,
            "images": formatted_images
        }), 200

    except Exception as e:
        logger.error(f"Error getting product images: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/products", methods=["GET"])
def get_products():
    """Get all products with their image statistics"""
    try:
        # Get all images from API
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            logger.error(f"Failed to fetch images: {response.status_code}")
            return jsonify({"error": "Failed to fetch images"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        # Group images by product
        product_stats = {}
        for img in images:
            product_id = img.get('productId')
            if not product_id:
                continue

            if product_id not in product_stats:
                product_stats[product_id] = {
                    'id': product_id,
                    'productName': img.get('productName', ''),
                    'total_images': 0,
                    'with_features': 0,
                    'without_features': 0,
                    'images': []
                }

            # Add image info with both URL and path
            image_info = {
                'id': img.get('id'),
                'url': img.get('url') or img.get('imageUrl'),
                'path': img.get('path'),
                'vectorFeatures': img.get('vectorFeatures'),
                'productName': img.get('productName', '')
            }
            product_stats[product_id]['images'].append(image_info)

            product_stats[product_id]['total_images'] += 1
            if img.get('vectorFeatures'):
                product_stats[product_id]['with_features'] += 1
            else:
                product_stats[product_id]['without_features'] += 1

            # Update product name if not set
            if not product_stats[product_id]['productName'] and img.get('productName'):
                product_stats[product_id]['productName'] = img.get('productName')

        # Convert to list and sort by total images
        products = list(product_stats.values())
        products.sort(key=lambda x: x['total_images'], reverse=True)

        # Add completion percentage for each product
        for product in products:
            product['completion_percentage'] = (
                (product['with_features'] / product['total_images'] * 100)
                if product['total_images'] > 0 else 0
            )

        return jsonify({
            "products": products,
            "total_products": len(products),
            "total_images": sum(p['total_images'] for p in products),
            "total_with_features": sum(p['with_features'] for p in products),
            "total_without_features": sum(p['without_features'] for p in products),
            "overall_completion": (
                sum(p['with_features'] for p in products) / 
                sum(p['total_images'] for p in products) * 100
            ) if sum(p['total_images'] for p in products) > 0 else 0
        }), 200

    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/clear_cache", methods=["POST"])
def clear_features_cache():
    """Xóa cache để force lấy dữ liệu mới"""
    try:
        invalidate_features_cache()
        return jsonify({"message": "Cache đã được xóa thành công"}), 200
    except Exception as e:
        logger.error(f"Lỗi khi xóa cache: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/vector_dimensions", methods=["GET"])
def get_vector_dimensions_info():
    """Get information about vector dimensions across all images"""
    try:
        # Get all images from API
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch images from API"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        if not images:
            return jsonify({"error": "No images found in database"}), 400

        # Analyze vector dimensions
        dimension_stats = {}
        images_with_features = []
        images_without_features = []
        inconsistent_dimensions = []

        expected_dimensions = get_vector_dimensions()

        for img in images:
            vector_features = img.get('vectorFeatures')
            vector_dimensions = img.get('vectorDimensions')
            
            if vector_features and vector_features.strip():
                # Count dimensions
                actual_dimensions = len(vector_features.split(','))
                
                if actual_dimensions not in dimension_stats:
                    dimension_stats[actual_dimensions] = 0
                dimension_stats[actual_dimensions] += 1
                
                images_with_features.append({
                    'id': img.get('id'),
                    'path': img.get('path'),
                    'productId': img.get('productId'),
                    'actual_dimensions': actual_dimensions,
                    'expected_dimensions': expected_dimensions,
                    'is_consistent': actual_dimensions == expected_dimensions
                })
                
                # Check for inconsistency
                if actual_dimensions != expected_dimensions:
                    inconsistent_dimensions.append({
                        'id': img.get('id'),
                        'path': img.get('path'),
                        'productId': img.get('productId'),
                        'actual_dimensions': actual_dimensions,
                        'expected_dimensions': expected_dimensions
                    })
            else:
                images_without_features.append({
                    'id': img.get('id'),
                    'path': img.get('path'),
                    'productId': img.get('productId')
                })

        # Group by product
        product_dimensions = {}
        for img in images_with_features:
            product_id = img['productId']
            if product_id not in product_dimensions:
                product_dimensions[product_id] = {
                    'productId': product_id,
                    'total_images': 0,
                    'images_with_features': 0,
                    'dimensions': set(),
                    'is_consistent': True
                }
            
            product_dimensions[product_id]['total_images'] += 1
            product_dimensions[product_id]['images_with_features'] += 1
            product_dimensions[product_id]['dimensions'].add(img['actual_dimensions'])
            
            if not img['is_consistent']:
                product_dimensions[product_id]['is_consistent'] = False

        # Convert sets to lists for JSON serialization
        for product in product_dimensions.values():
            product['dimensions'] = list(product['dimensions'])

        return jsonify({
            "expected_dimensions": expected_dimensions,
            "dimension_stats": dimension_stats,
            "total_images": len(images),
            "images_with_features": len(images_with_features),
            "images_without_features": len(images_without_features),
            "inconsistent_dimensions": inconsistent_dimensions,
            "product_dimensions": list(product_dimensions.values()),
            "is_system_consistent": len(inconsistent_dimensions) == 0 and len(images_with_features) > 0
        }), 200

    except Exception as e:
        logger.error(f"Error getting vector dimensions info: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/config", methods=["GET", "POST"])
def configure_extraction():
    """Configure extraction settings and get current configuration"""
    global VECTOR_DIMENSIONS_CONFIG
    
    if request.method == "GET":
        # Get current configuration
        current_dimensions = VECTOR_DIMENSIONS_CONFIG
        
        # Always use ResNet50 for all dimensions
        model_type = "ResNet50"
        
        response_data = {
            "vector_dimensions": current_dimensions,
            "model_type": model_type,
            "available_dimensions": [512, 1024, 1536, 2048, 4096],
            "description": "Sử dụng ResNet50 với Dense layer để giảm hoặc mở rộng chiều vector"
        }
        
        # Tự động tạo biểu đồ
        response_data = add_chart_generation_to_response(response_data, 'config')
        
        return jsonify(response_data), 200
    
    elif request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No configuration data provided"}), 400
            
            new_dimensions = data.get("vector_dimensions")
            if not new_dimensions:
                return jsonify({"error": "vector_dimensions is required"}), 400
            
            # Validate dimensions
            available_dimensions = [512, 1024, 1536, 2048, 4096]
            if new_dimensions not in available_dimensions:
                return jsonify({
                    "error": f"Invalid vector dimensions. Must be one of: {available_dimensions}"
                }), 400
            
            # Check if dimensions changed
            dimensions_changed = VECTOR_DIMENSIONS_CONFIG != new_dimensions
            
            # Update configuration
            old_dimensions = VECTOR_DIMENSIONS_CONFIG
            VECTOR_DIMENSIONS_CONFIG = new_dimensions
            
            logger.info(f"Vector dimensions changed from {old_dimensions} to {new_dimensions}")
            
            # Clear cache when dimensions change
            if dimensions_changed:
                invalidate_features_cache()
                # Clear model cache to force recreation of model for new dimensions
                global _models
                with _model_lock:
                    _models.clear()
                logger.info("Cache and model cache cleared due to dimension change")
            
            # Check if there are any existing features
            response = requests.get(f"{API_URL}/images")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    images = data['data']
                else:
                    images = data if isinstance(data, list) else []
                
                existing_features_count = sum(1 for img in images if img.get('vectorFeatures') and img.get('vectorFeatures').strip())
                
                response_data = {
                    "message": "Configuration updated successfully",
                    "old_dimensions": old_dimensions,
                    "new_dimensions": new_dimensions,
                    "dimensions_changed": dimensions_changed,
                    "existing_features_count": existing_features_count,
                    "needs_re_extraction": dimensions_changed and existing_features_count > 0,
                    "available_dimensions": available_dimensions
                }
                
                # Tự động tạo biểu đồ
                response_data = add_chart_generation_to_response(response_data, 'config')
                
                return jsonify(response_data), 200
            else:
                response_data = {
                    "message": "Configuration updated successfully",
                    "old_dimensions": old_dimensions,
                    "new_dimensions": new_dimensions,
                    "dimensions_changed": dimensions_changed,
                    "warning": "Could not check existing features"
                }
                
                # Tự động tạo biểu đồ
                response_data = add_chart_generation_to_response(response_data, 'config')
                
                return jsonify(response_data), 200
                
        except Exception as e:
            logger.error(f"Error configuring extraction: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/api/extraction/fix_dimensions", methods=["POST"])
def fix_inconsistent_dimensions():
    """Fix images with inconsistent vector dimensions"""
    try:
        logger.info("=== BẮT ĐẦU SỬA CHỮA CHIỀU VECTOR KHÔNG NHẤT QUÁN ===")
        
        # Get vector dimensions info
        response = requests.get(f"{API_URL}/images")
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch images from API"}), 500

        data = response.json()
        if isinstance(data, dict) and 'data' in data:
            images = data['data']
        else:
            images = data if isinstance(data, list) else []

        if not images:
            return jsonify({"error": "No images found in database"}), 400

        expected_dimensions = get_vector_dimensions()
        inconsistent_images = []
        
        # Find images with inconsistent dimensions
        for img in images:
            vector_features = img.get('vectorFeatures')
            if vector_features and vector_features.strip():
                actual_dimensions = len(vector_features.split(','))
                if actual_dimensions != expected_dimensions:
                    inconsistent_images.append(img)

        if not inconsistent_images:
            return jsonify({
                "message": "Không có ảnh nào có chiều vector không nhất quán",
                "fixed_count": 0,
                "total_checked": len(images)
            }), 200

        logger.info(f"Tìm thấy {len(inconsistent_images)} ảnh có chiều vector không nhất quán")

        # Fix each inconsistent image
        success_count = 0
        failed_count = 0
        failed_images = []

        for img in inconsistent_images:
            try:
                image_id = img.get('id')
                image_path = img.get('path')
                
                if not image_id or not image_path:
                    logger.warning(f"Ảnh thiếu thông tin: ID={image_id}, Path={image_path}")
                    failed_count += 1
                    failed_images.append({
                        "id": image_id,
                        "error": "Thiếu thông tin ảnh"
                    })
                    continue

                logger.info(f"Đang sửa chữa ảnh: {image_path} (ID: {image_id})")

                # Extract features
                features = extract_features_from_url(image_path)
                if features is None:
                    logger.error(f"Không thể trích xuất đặc trưng cho ảnh {image_id}")
                    failed_count += 1
                    failed_images.append({
                        "id": image_id,
                        "error": "Không thể trích xuất đặc trưng"
                    })
                    continue

                # Check dimensions again
                actual_dimensions = len(features)
                if actual_dimensions != expected_dimensions:
                    logger.error(f"Chiều vector vẫn không đúng sau khi trích xuất: {actual_dimensions} != {expected_dimensions}")
                    failed_count += 1
                    failed_images.append({
                        "id": image_id,
                        "error": f"Chiều vector không đúng: {actual_dimensions} != {expected_dimensions}"
                    })
                    continue

                # Convert features to string
                features_str = ','.join(map(str, features))

                # Update vector features
                update_response = requests.put(
                    f"{API_URL}/image/update/vector_feature",
                    json={
                        "id": image_id,
                        "vectorFeatures": features_str,
                        "vectorDimensions": actual_dimensions
                    }
                )

                if update_response.status_code == 200:
                    logger.info(f"Đã sửa chữa thành công vector features cho ảnh {image_id}")
                    success_count += 1
                else:
                    logger.error(f"Không thể cập nhật vector features: {update_response.status_code}")
                    failed_count += 1
                    failed_images.append({
                        "id": image_id,
                        "error": f"Lỗi cập nhật: {update_response.status_code}"
                    })

            except Exception as e:
                logger.error(f"Lỗi khi sửa chữa ảnh {image_id}: {str(e)}")
                failed_count += 1
                failed_images.append({
                    "id": image_id,
                    "error": str(e)
                })

        # Invalidate cache
        invalidate_features_cache()
        logger.info("Đã sửa chữa xong và xóa cache")

        logger.info("=== KẾT THÚC SỬA CHỮA CHIỀU VECTOR KHÔNG NHẤT QUÁN ===")
        logger.info(f"Kết quả: {success_count} thành công, {failed_count} thất bại")

        return jsonify({
            "message": "Đã sửa chữa xong chiều vector không nhất quán",
            "total_checked": len(images),
            "inconsistent_found": len(inconsistent_images),
            "fixed_count": success_count,
            "failed_count": failed_count,
            "failed_images": failed_images
        }), 200

    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Chart generation API endpoints
@app.route("/api/charts/generate", methods=["POST"])
def generate_charts():
    """Tạo tất cả biểu đồ"""
    try:
        data = request.get_json() or {}
        chart_types = data.get('chart_types', ['all'])  # ['performance', 'augmentation', 'dimensions', 'all']
        
        charts_created = []
        
        if 'all' in chart_types or 'performance' in chart_types:
            perf_chart = create_performance_chart()
            if perf_chart:
                charts_created.append({"type": "performance", "file": perf_chart})
        
        if 'all' in chart_types or 'augmentation' in chart_types:
            aug_chart = create_data_augmentation_flowchart()
            if aug_chart:
                charts_created.append({"type": "augmentation", "file": aug_chart})
        
        if 'all' in chart_types or 'dimensions' in chart_types:
            dim_chart = create_vector_dimensions_comparison()
            if dim_chart:
                charts_created.append({"type": "dimensions", "file": dim_chart})
        
        return jsonify({
            "message": "Đã tạo biểu đồ thành công",
            "charts_created": charts_created,
            "total_charts": len(charts_created)
        }), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/charts/performance", methods=["GET"])
def get_performance_chart():
    """Tạo và trả về biểu đồ hiệu năng"""
    try:
        chart_file = create_performance_chart()
        if chart_file and os.path.exists(chart_file):
            # Đọc file ảnh và trả về base64
            with open(chart_file, 'rb') as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            return jsonify({
                "message": "Biểu đồ hiệu năng đã được tạo",
                "chart_file": chart_file,
                "base64_image": base64_image,
                "image_type": "image/png"
            }), 200
        else:
            return jsonify({"error": "Không thể tạo biểu đồ hiệu năng"}), 500
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ hiệu năng: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/charts/augmentation", methods=["GET"])
def get_augmentation_chart():
    """Tạo và trả về sơ đồ tăng cường dữ liệu"""
    try:
        chart_file = create_data_augmentation_flowchart()
        if chart_file and os.path.exists(chart_file):
            # Đọc file ảnh và trả về base64
            with open(chart_file, 'rb') as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            return jsonify({
                "message": "Sơ đồ tăng cường dữ liệu đã được tạo",
                "chart_file": chart_file,
                "base64_image": base64_image,
                "image_type": "image/png"
            }), 200
        else:
            return jsonify({"error": "Không thể tạo sơ đồ tăng cường dữ liệu"}), 500
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo sơ đồ tăng cường dữ liệu: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/charts/dimensions", methods=["GET"])
def get_dimensions_chart():
    """Tạo và trả về biểu đồ so sánh chiều vector"""
    try:
        chart_file = create_vector_dimensions_comparison()
        if chart_file and os.path.exists(chart_file):
            # Đọc file ảnh và trả về base64
            with open(chart_file, 'rb') as f:
                image_data = f.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            return jsonify({
                "message": "Biểu đồ so sánh chiều vector đã được tạo",
                "chart_file": chart_file,
                "base64_image": base64_image,
                "image_type": "image/png"
            }), 200
        else:
            return jsonify({"error": "Không thể tạo biểu đồ so sánh chiều vector"}), 500
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ so sánh chiều vector: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/charts/list", methods=["GET"])
def list_charts():
    """Liệt kê tất cả biểu đồ đã tạo"""
    try:
        chart_dir = 'chart'
        if not os.path.exists(chart_dir):
            return jsonify({"charts": [], "message": "Thư mục chart chưa tồn tại"}), 200
        
        charts = []
        for filename in os.listdir(chart_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(chart_dir, filename)
                file_stats = os.stat(file_path)
                charts.append({
                    "filename": filename,
                    "file_path": file_path,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "created_time": datetime.fromtimestamp(file_stats.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    "modified_time": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Sắp xếp theo thời gian tạo mới nhất
        charts.sort(key=lambda x: x['created_time'], reverse=True)
        
        return jsonify({
            "charts": charts,
            "total_charts": len(charts),
            "chart_directory": os.path.abspath(chart_dir)
        }), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê biểu đồ: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Modify existing endpoints to auto-generate charts
def add_chart_generation_to_response(response_data, endpoint_name):
    """Thêm thông tin biểu đồ vào response"""
    try:
        # Tạo biểu đồ tự động cho một số endpoint quan trọng
        if endpoint_name in ['extraction_stats', 'extraction_start', 'config']:
            charts_created = auto_generate_charts()
            if charts_created:
                response_data['charts_generated'] = charts_created
                response_data['chart_message'] = f"Đã tự động tạo {len(charts_created)} biểu đồ"
        
        return response_data
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ tự động: {str(e)}")
        return response_data

@app.route("/api/performance/stats", methods=["GET"])
def get_performance_stats():
    """Lấy thống kê hiệu năng thực tế"""
    try:
        stats = get_performance_statistics()
        
        # Tự động tạo biểu đồ hiệu năng
        chart_file = create_performance_chart()
        
        response_data = {
            "performance_statistics": stats,
            "chart_generated": chart_file is not None,
            "chart_file": chart_file
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê hiệu năng: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/performance/reset", methods=["POST"])
def reset_performance_data():
    """Reset dữ liệu hiệu năng"""
    try:
        global performance_metrics
        with performance_lock:
            performance_metrics = {
                "extraction_times": [],
                "search_times": [],
                "accuracy_scores": [],
                "total_images_processed": 0,
                "total_searches_performed": 0,
                "last_updated": None
            }
        
        logger.info("Đã reset dữ liệu hiệu năng")
        return jsonify({"message": "Đã reset dữ liệu hiệu năng thành công"}), 200
        
    except Exception as e:
        logger.error(f"Lỗi khi reset dữ liệu hiệu năng: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/performance/export", methods=["GET"])
def export_performance_data():
    """Xuất dữ liệu hiệu năng dưới dạng CSV"""
    try:
        import csv
        from io import StringIO
        
        stats = get_performance_statistics()
        
        # Tạo CSV data
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Metric', 'Value', 'Unit', 'Description'])
        
        # Extraction performance
        writer.writerow(['Average Extraction Time', 
                        f"{stats['extraction_performance']['avg_time_per_image']:.3f}", 
                        'seconds', 
                        f"Based on {stats['extraction_performance']['total_images']} images"])
        writer.writerow(['Min Extraction Time', 
                        f"{stats['extraction_performance']['min_time']:.3f}", 
                        'seconds', 
                        'Fastest extraction'])
        writer.writerow(['Max Extraction Time', 
                        f"{stats['extraction_performance']['max_time']:.3f}", 
                        'seconds', 
                        'Slowest extraction'])
        
        # Search performance
        writer.writerow(['Average Search Time', 
                        f"{stats['search_performance']['avg_time_per_search']:.3f}", 
                        'milliseconds', 
                        f"Based on {stats['search_performance']['total_searches']} searches"])
        writer.writerow(['Min Search Time', 
                        f"{stats['search_performance']['min_time']:.3f}", 
                        'milliseconds', 
                        'Fastest search'])
        writer.writerow(['Max Search Time', 
                        f"{stats['search_performance']['max_time']:.3f}", 
                        'milliseconds', 
                        'Slowest search'])
        
        # Accuracy
        writer.writerow(['Average Accuracy', 
                        f"{stats['accuracy']['avg_accuracy']*100:.2f}", 
                        'percent', 
                        f"Based on {stats['accuracy']['total_measurements']} measurements"])
        writer.writerow(['Min Accuracy', 
                        f"{stats['accuracy']['min_accuracy']*100:.2f}", 
                        'percent', 
                        'Lowest accuracy'])
        writer.writerow(['Max Accuracy', 
                        f"{stats['accuracy']['max_accuracy']*100:.2f}", 
                        'percent', 
                        'Highest accuracy'])
        
        # Metadata
        writer.writerow(['Last Updated', 
                        stats['last_updated'].strftime("%Y-%m-%d %H:%M:%S") if stats['last_updated'] else 'N/A', 
                        '', 
                        'Last performance measurement'])
        
        csv_data = output.getvalue()
        output.close()
        
        # Tạo response với CSV data
        response = app.response_class(
            response=csv_data,
            status=200,
            mimetype='text/csv'
        )
        response.headers["Content-Disposition"] = "attachment; filename=performance_stats.csv"
        
        return response
        
    except Exception as e:
        logger.error(f"Lỗi khi xuất dữ liệu hiệu năng: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
