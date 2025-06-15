import os

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "db_capstone")
DB_USERNAME = os.getenv("DB_USERNAME", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456789")

# Flask Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model Configuration   
MIN_IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 1024
QUALITY_THRESHOLD = 0.5
FEATURE_DIMENSIONS = 2048  # ResNet50 with GlobalMaxPooling2D

# Cache Configuration
CACHE_EXPIRY = 30 * 60  # 30 minutes in seconds
CACHE_MAX_SIZE = 1000

# Threading Configuration
MAX_WORKERS = 4

# Image Processing Configuration
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Azure Blob Storage URL
AZURE_URL = "https://dbimage.blob.core.windows.net/images" 