
#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QMessageBox
)

class ProcessingThread(QThread):
    processed_image = pyqtSignal(QImage)

    def __init__(self, cam_index=0, blur=15, threshold=0.5):
        super().__init__()
        # Initialize camera
        self.capture = cv2.VideoCapture(cam_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_index}")
        # Set parameters
        self.blur_radius = blur
        self.threshold = threshold
        self.bg_frame = None

        # Load DeepLabV3 model with pretrained weights
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=weights).eval()

        # Build preprocessing pipeline matching model's expected transforms
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        mean, std = weights.transforms().mean, weights.transforms().std
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_height, frame_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def run(self):
        # Warm up camera exposure
        for _ in range(30):
            self.capture.read()

        while not self.isInterruptionRequested():
            ret, frame_bgr = self.capture.read()
            if not ret:
                break

            # Convert frame to RGB numpy array
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Preprocess and infer
            input_tensor = self.preprocess(frame_rgb).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)["out"][0]
            mask = output.argmax(0).byte().cpu().numpy()

            # Create and smooth person mask
            person_mask = (mask == 15).astype(np.uint8)
            blurred = cv2.GaussianBlur(
                person_mask.astype(np.float32),
                (self.blur_radius, self.blur_radius),
                0
            )
            _, person_mask = cv2.threshold(
                blurred, self.threshold, 1, cv2.THRESH_BINARY
            )
            person_mask = person_mask[..., None]

            # Capture background on first iteration
            if self.bg_frame is None:
                self.bg_frame = frame_rgb.copy()

            # Composite foreground and background
            foreground = frame_rgb * (1 - person_mask)
            background = self.bg_frame * person_mask
            composite  = (foreground + background).astype(np.uint8)

            # Convert to QImage and emit signal
            qt_image = QImage(composite.data, w, h, w * 3, QImage.Format_RGB888)
            self.processed_image.emit(qt_image)

            # Throttle to reduce CPU usage
            self.msleep(30)

    def update_params(self, blur, threshold):
        self.blur_radius = blur
        self.threshold   = threshold

    def capture_background(self):
        ret, frame_bgr = self.capture.read()
        if ret:
            self.bg_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def set_background(self, bg_frame):
        self.bg_frame = bg_frame

    def stop(self):
        self.requestInterruption()
        self.wait()
        self.capture.release()

class InvisibilityCloakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invisibility Cloak GUI")

        # Start processing thread
        try:
            self.thread = ProcessingThread()
        except RuntimeError as e:
            QMessageBox.critical(self, "Camera Error", str(e))
            sys.exit(1)

        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)

        # Buttons
        self.btn_capture_bg = QPushButton("Capture Background")
        self.btn_capture_bg.clicked.connect(self.capture_background)
        self.btn_load_bg    = QPushButton("Load Background")
        self.btn_load_bg.clicked.connect(self.load_background)

        # Sliders
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 51)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.setValue(self.thread.blur_radius)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)

        self.th_slider = QSlider(Qt.Horizontal)
        self.th_slider.setRange(1, 99)
        self.th_slider.setValue(int(self.thread.threshold * 100))
        self.th_slider.valueChanged.connect(self.on_threshold_changed)

        # Layout setup
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(self.btn_capture_bg)
        ctrl_layout.addWidget(self.btn_load_bg)
        ctrl_layout.addWidget(QLabel("Blur Radius"))
        ctrl_layout.addWidget(self.blur_slider)
        ctrl_layout.addWidget(QLabel("Threshold"))
        ctrl_layout.addWidget(self.th_slider)
        controls.setLayout(ctrl_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(controls)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect signal and start thread
        self.thread.processed_image.connect(self.update_image)
        self.thread.start()

    def capture_background(self):
        self.thread.capture_background()

    def load_background(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Background Image")
        if path:
            img = cv2.imread(path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.thread.set_background(rgb)

    def on_blur_changed(self, val):
        br = val if val % 2 == 1 else val + 1
        self.thread.update_params(br, self.thread.threshold)

    def on_threshold_changed(self, val):
        th = val / 100.0
        self.thread.update_params(self.thread.blur_radius, th)

    def update_image(self, qt_img):
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InvisibilityCloakApp()
    window.show()
    sys.exit(app.exec_())

