import numpy as np
from keras.api.preprocessing import image
from keras.api.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from models.mobilenet_model import load_mobilenet_model

# Load mô hình MobileNetV2 đã pretrained
feature_extractor = load_mobilenet_model()

# Khởi tạo ImageDataGenerator để tăng cường dữ liệu
datagen = ImageDataGenerator(
    rotation_range=10,        # Xoay ±10 độ
    width_shift_range=0.1,    # Dịch trái/phải 10%
    height_shift_range=0.1,   # Dịch lên/xuống 10%
    shear_range=0.1,          # Biến dạng nghiêng
    zoom_range=0.1,           # Phóng to/thu nhỏ
    horizontal_flip=True,     # Lật ngang ảnh
    fill_mode='nearest'       # Bù pixel khi biến đổi
)

def extract_features(img_input, augment=False):
    """
    Trích xuất đặc trưng từ ảnh sử dụng MobileNetV2.
    Nếu `augment=True`, sẽ tạo ảnh tăng cường và trích xuất đặc trưng từ đó.
    """
    if isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input)
    else:
        img = image.load_img(img_input, target_size=(224, 224))

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Nếu bật augment, sinh thêm 1 biến thể từ ảnh
    if augment:
        aug_iter = datagen.flow(img_array, batch_size=1)
        img_array = next(aug_iter)

    features = feature_extractor.predict(img_array)
    return features.flatten()
