# AI Image Similarity API - ResNet50 Only

## Tổng quan
Hệ thống trích xuất đặc trưng ảnh sử dụng **ResNet50** với các chiều vector có thể cấu hình.

## Mô hình sử dụng
- **ResNet50**: Mô hình cơ sở duy nhất
- **GlobalMaxPooling2D**: Trích xuất 2048 chiều mặc định
- **Dense Layer**: Giảm hoặc mở rộng chiều vector
- **PCA**: Không sử dụng (thay thế bằng Dense layer)

## Các chiều vector được hỗ trợ

### 1. **512 chiều** (Tiết kiệm 75% bộ nhớ)
- **Cách thức**: ResNet50 + GlobalMaxPooling2D + Dense(512)
- **Độ chính xác**: 85-90% so với 2048 chiều
- **Tốc độ**: Rất nhanh
- **Bộ nhớ**: Tiết kiệm 75%

### 2. **1024 chiều** (Tiết kiệm 50% bộ nhớ)
- **Cách thức**: ResNet50 + GlobalMaxPooling2D + Dense(1024)
- **Độ chính xác**: 90-95% so với 2048 chiều
- **Tốc độ**: Nhanh
- **Bộ nhớ**: Tiết kiệm 50%

### 3. **1536 chiều** (Tiết kiệm 25% bộ nhớ)
- **Cách thức**: ResNet50 + GlobalMaxPooling2D + Dense(1536)
- **Độ chính xác**: 95-98% so với 2048 chiều
- **Tốc độ**: Trung bình
- **Bộ nhớ**: Tiết kiệm 25%

### 4. **2048 chiều** (Mặc định)
- **Cách thức**: ResNet50 + GlobalMaxPooling2D
- **Độ chính xác**: 100% (gốc)
- **Tốc độ**: Trung bình
- **Bộ nhớ**: Không tiết kiệm

### 5. **4096 chiều** (Mở rộng)
- **Cách thức**: ResNet50 + GlobalMaxPooling2D + Dense(4096)
- **Độ chính xác**: Cao nhất
- **Tốc độ**: Chậm hơn
- **Bộ nhớ**: Tăng 100%

## So sánh hiệu suất

| Chiều vector | Tiết kiệm bộ nhớ | Độ chính xác | Tốc độ | Cách thức |
|-------------|------------------|-------------|--------|-----------|
| 512 | 75% | 85-90% | Rất nhanh | Dense(512) |
| 1024 | 50% | 90-95% | Nhanh | Dense(1024) |
| 1536 | 25% | 95-98% | Trung bình | Dense(1536) |
| 2048 | 0% | 100% | Trung bình | GlobalMaxPooling2D |
| 4096 | -100% | 100%+ | Chậm | Dense(4096) |

## Cách sử dụng

### 1. Lấy cấu hình hiện tại
```bash
GET /api/extraction/config
```

### 2. Thay đổi chiều vector
```bash
POST /api/extraction/config
Content-Type: application/json

{
    "vector_dimensions": 512
}
```

### 3. Trích xuất đặc trưng
```bash
POST /api/extraction/start
```

## Khuyến nghị sử dụng

### Cho ứng dụng production:
- **512 chiều**: Khi cần tốc độ cao, bộ nhớ ít
- **1024 chiều**: Cân bằng tốt giữa hiệu suất và bộ nhớ
- **2048 chiều**: Mặc định, phù hợp cho hầu hết trường hợp

### Cho nghiên cứu/development:
- **1536 chiều**: Khi cần độ chính xác cao hơn
- **4096 chiều**: Khi cần độ chính xác tối đa

## Lưu ý quan trọng

1. **Dense Layer Reduction**: Sử dụng Dense layer để giảm chiều thay vì PCA
2. **Cache Management**: Khi thay đổi chiều vector, cache sẽ được xóa tự động
3. **Model Consistency**: Tất cả ảnh sẽ sử dụng cùng một model ResNet50
4. **Performance**: Dense layer giảm chiều nhanh và ổn định hơn PCA
5. **Single Sample**: Hỗ trợ xử lý từng ảnh riêng lẻ mà không cần batch

## API Endpoints

### Cấu hình
- `GET /api/extraction/config` - Lấy cấu hình hiện tại
- `POST /api/extraction/config` - Thay đổi cấu hình

### Trích xuất
- `POST /api/extraction/start` - Bắt đầu trích xuất
- `GET /api/extraction/status` - Trạng thái trích xuất
- `GET /api/extraction/stats` - Thống kê trích xuất

### Tìm kiếm
- `POST /api/find_similar` - Tìm ảnh tương tự

## Cài đặt và chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server
python app.py
```

Server sẽ chạy tại `http://localhost:5000` 