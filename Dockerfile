FROM python:3.10

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements và cài đặt Python packages
COPY requirements.txt ./

# Cài đặt và cấu hình pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorflow-cpu==2.15.0 && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir scikit-learn==1.3.2 && \
    pip install --no-cache-dir flask==2.3.3 && \
    pip install --no-cache-dir flask-cors==4.0.0 && \
    pip install --no-cache-dir Pillow==10.0.1 && \
    pip install --no-cache-dir requests==2.31.0

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Tạo thư mục temp cho ảnh tạm nếu chưa có
RUN mkdir -p temp

# Kiểm tra cài đặt
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)" && \
    python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

# Expose port Flask
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
# Chay file train 
#