import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QFileDialog, QWidget, QSlider, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from utils.video_thread import VideoThread

class MaskDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.thread = None

    def initUI(self):
        self.setWindowTitle('YOLO Mask Detection')
        self.setGeometry(100, 100, 900, 700)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setFixedSize(1080, 720)

        self.btn_open = QPushButton('📁 Chọn File', self)
        self.btn_open.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_open.clicked.connect(self.open_file)

        self.btn_cam = QPushButton('📷 Mở Camera', self)
        self.btn_cam.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_cam.clicked.connect(self.open_camera)

        self.btn_stop_cam = QPushButton('🛑 Tắt Camera', self)
        self.btn_stop_cam.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setEnabled(False)

        self.threshold_label = QLabel(f'Ngưỡng: 50%', self)
        self.threshold_label.setFont(QFont("Arial", 12))
        self.threshold_label.setAlignment(Qt.AlignCenter)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.iou_label = QLabel(f'IOU: 50%', self)
        self.iou_label.setFont(QFont("Arial", 12))
        self.iou_label.setAlignment(Qt.AlignCenter)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(10)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(50)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.valueChanged.connect(self.update_iou)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_open)
        button_layout.addWidget(self.btn_cam)
        button_layout.addWidget(self.btn_stop_cam)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.video_label)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.threshold_label)
        self.layout.addWidget(self.threshold_slider)
        self.layout.addWidget(self.iou_label)
        self.layout.addWidget(self.iou_slider)
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(self.layout)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn Video hoặc Ảnh", "",
                                                   "Video Files (*.mp4 *.avi);;Image Files (*.jpg *.png)")
        if file_path:
            self.start_detection(file_path)

    def open_camera(self):
        self.start_detection(0)

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
            self.video_label.clear()
            self.btn_stop_cam.setEnabled(False)

    def start_detection(self, source):
        if self.thread:
            self.thread.stop()
            self.thread.wait()

        self.thread = VideoThread(source, self.threshold_slider.value() / 100, self.iou_slider.value() / 100)
        self.thread.update_frame.connect(self.display_frame)
        self.thread.start()

        if source == 0:
            self.btn_stop_cam.setEnabled(True)
        else:
            self.btn_stop_cam.setEnabled(False)

    def update_threshold(self):
        value = self.threshold_slider.value()
        self.threshold_label.setText(f'Ngưỡng: {value}%')
        if self.thread:
            self.thread.threshold = value / 100

    def update_iou(self):
        value = self.iou_slider.value()
        self.iou_label.setText(f'IOU: {value}%')
        if self.thread:
            self.thread.iou_threshold = value / 100

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1080, 720), interpolation=cv2.INTER_AREA)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MaskDetectionApp()
    window.show()
    sys.exit(app.exec_())
