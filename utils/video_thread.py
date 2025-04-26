import os
import cv2
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from models.yolo_model import load_yolo_model

# Load YOLO model
yolo_model = load_yolo_model()


class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray)

    def __init__(self, source=0, threshold=0.5, iou_threshold=0.5):
        super().__init__()
        self.source = source
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.running = True
        self.prev_time = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        is_image = isinstance(self.source, str) and self.source.lower().endswith(('.jpg', '.jpeg', '.png'))

        while self.running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame, conf=self.threshold, iou=self.iou_threshold)[0]

            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if conf >= self.threshold:
                    label = f"{results.names[int(cls)]} ({conf:.2f})"
                    color = (0, 255, 0) if 'mask' in results.names[int(cls)] else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Hiển thị FPS nếu là video
            if not is_image:
                fps = 1.0 / (start_time - self.prev_time) if self.prev_time > 0 else 0
                self.prev_time = start_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            self.update_frame.emit(frame)

            if is_image:
                break

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
