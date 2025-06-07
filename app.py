from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
#CORS(app, supports_credentials=True, origins=["https://polite-plant-004c99b1e.6.azurestaticapps.net"], allow_headers="*")
CORS(app, supports_credentials=True, origins=["http://localhost:3000"], allow_headers="*")
# Load model ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

# Load embeddings từ file (Kiểm tra dữ liệu hợp lệ)
FEATURES_FILE = r"D:\DUT\AI\flask-ai-app\app\resnet\features.pkl"
FILENAMES_FILE = r"D:\DUT\AI\flask-ai-app\app\resnet\filenames.pkl"
PCA_FILE = r"D:\DUT\AI\flask-ai-app\app\resnet\pca.pkl"
NN_MODEL_FILE = r"D:\DUT\AI\flask-ai-app\app\resnet\nn_model.pkl"

if all(os.path.exists(f) for f in [FEATURES_FILE, FILENAMES_FILE, PCA_FILE, NN_MODEL_FILE]):
    try:
        features_dict = pickle.load(open(FEATURES_FILE, "rb"))
        filenames = np.array(list(features_dict.keys()))
        feature_list = np.array(list(features_dict.values()))
        pca = pickle.load(open(PCA_FILE, "rb"))
        nn_model = pickle.load(open(NN_MODEL_FILE, "rb"))
    except Exception as e:
        print(f"❌ ERROR: Cannot load data files: {e}")
        features_dict, filenames, feature_list, pca, nn_model = {}, np.array([]), np.array([]), None, None
else:
    print("⚠️ WARNING: Required data files not found.")
    features_dict, filenames, feature_list, pca, nn_model = {}, np.array([]), np.array([]), None, None

def extract_features(img_path, model):
    """Trích xuất đặc trưng từ ảnh bằng ResNet50"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to AI Image Similarity API!"})

AZURE_URL = "https://dbimage.blob.core.windows.net/images"

@app.route('/api/find_similar', methods=['POST'])
def find_similar_images():
    """Nhận ảnh (file hoặc url), tìm ảnh tương tự và trả về danh sách"""
    try:
        temp_path = None
        # Nếu gửi JSON có url
        if request.is_json:
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({"error": "No URL provided. Please send a JSON payload with a 'url' field."}), 400
            image_url = data['url']
            if not image_url:
                return jsonify({"error": "Empty URL provided."}), 400
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                return jsonify({"error": f"Failed to fetch image from URL: {image_url}"}), 400
            temp_path = os.path.join('temp', 'query_image.jpg')
            os.makedirs('temp', exist_ok=True)
            with open(temp_path, 'wb') as f:
                f.write(response.content)
        # Nếu gửi file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            temp_path = os.path.join('temp', secure_filename(file.filename))
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
        else:
            return jsonify({"error": "No URL or file provided. Please send a JSON payload with a 'url' field or upload a file."}), 400

        # Kiểm tra nếu không có dữ liệu embeddings
        if len(feature_list) == 0:
            return jsonify({"error": "Feature data is empty. Ensure embeddings are generated."}), 500

        # Trích xuất đặc trưng và tính toán độ tương đồng
        query_features = extract_features(temp_path, model).reshape(1, -1)
        similarities = cosine_similarity(query_features, feature_list)[0]
        # Rank và lấy 10 ảnh tương tự nhất
        top_indices = np.argsort(similarities)[::-1][:10]
        # Trả về danh sách ảnh tương tự với URL đầy đủ
        similar_images = [
            {
                "filename": filenames[idx],
                "url": f"{AZURE_URL}/{filenames[idx]}",
                "similarity": float(similarities[idx])
            }
            for idx in top_indices
        ]
        return jsonify({"similar_images": similar_images}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
