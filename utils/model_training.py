import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from utils.feature_extraction import extract_features

# Khai báo đường dẫn dataset
dataset_path = "data"
X, y = [], []

print("👉 Bắt đầu trích xuất đặc trưng từ ảnh...")

# Duyệt qua từng thư mục (mỗi thư mục là một class)
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                # Ảnh gốc
                features = extract_features(img_path, augment=False)
                X.append(features)
                y.append(label)

                # Ảnh tăng cường
                features_aug = extract_features(img_path, augment=True)
                X.append(features_aug)
                y.append(label)

            except Exception as e:
                print(f"❌ Lỗi khi xử lý ảnh {img_path}: {e}")

# Chuyển đổi sang NumPy array
X = np.array(X)
y = np.array(y)

# Chuẩn hóa nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Huấn luyện mô hình Random Forest
print("🌲 Đang huấn luyện mô hình Random Forest...")
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Lưu mô hình và encoder
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Mô hình và label encoder đã được lưu.")