FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements và cài đặt Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Tạo thư mục temp cho ảnh tạm nếu chưa có
RUN mkdir -p temp

# Expose port Flask
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app2.py"]
# Chay file train 
