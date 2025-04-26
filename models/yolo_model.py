from ultralytics import YOLO

def load_yolo_model():
    model_path = "models/best.pt"  # Đường dẫn đến mô hình
    model = YOLO(model_path)  # Load trực tiếp bằng YOLO
    return model
