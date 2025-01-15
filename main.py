import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from skimage import restoration
from PIL import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from rembg import remove

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.current_image = None
        self.load_object_detection_model()

    def initUI(self):
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Feature selection
        feature_layout = QHBoxLayout()
        self.feature_combo = QComboBox(self)
        self.feature_combo.addItems([
            "Image Enhancement",
            "Image Restoration",
            "Image Coloring",
            "Face Retouch",
            "Object Detection",
            "Image Segmentation",
            "Image Compression",
            "Remove Background"
        ])
        feature_layout.addWidget(self.feature_combo)

        # Buttons
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        feature_layout.addWidget(self.load_button)

        self.process_button = QPushButton('Process Image', self)
        self.process_button.clicked.connect(self.process_image)
        feature_layout.addWidget(self.process_button)

        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.save_image)
        feature_layout.addWidget(self.save_button)

        layout.addLayout(feature_layout)

        self.setLayout(layout)

    def load_object_detection_model(self):
        # Load the model
        model_path = 'ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model'
        self.detect_fn = tf.saved_model.load(model_path)

        # Load the label map
        label_map_path = 'mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_image(self):
        if self.current_image is None:
            return

        feature = self.feature_combo.currentText()

        try:
            if feature == "Image Enhancement":
                self.current_image = self.enhance_image(self.current_image)
            elif feature == "Image Restoration":
                self.current_image = self.restore_image(self.current_image)
            elif feature == "Image Coloring":
                self.current_image = self.colorize_image(self.current_image)
            elif feature == "Face Retouch":
                self.current_image = self.retouch_face(self.current_image)
            elif feature == "Object Detection":
                self.current_image = self.detect_objects(self.current_image)
            elif feature == "Image Segmentation":
                self.current_image = self.segment_image(self.current_image)
            elif feature == "Image Compression":
                self.current_image = self.compress_image(self.current_image)
            elif feature == "Remove Background":
                self.current_image = self.remove_background(self.current_image)

            self.display_image(self.current_image)
        except Exception as e:
            print(f"Error processing image: {str(e)}")

    def save_image(self):
        if self.current_image is None:
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            cv2.imwrite(file_name, self.current_image)

    def enhance_image(self, image):
        alpha = 1.2  # contrast control (1.0-3.0)
        beta = 50    # brightness control (0-100)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)

        image = cv2.medianBlur(image, 5)

        return image

    def restore_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = restoration.denoise_tv_chambolle(image, weight=0.1)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def colorize_image(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.convert("LA").convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def retouch_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_color = image[y:y+h, x:x+w]
            roi_color = cv2.bilateralFilter(roi_color, 9, 75, 75)
            image[y:y+h, x:x+w] = roi_color

        return image

    def detect_objects(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_expanded, dtype=tf.float32)

        # Perform detection
        detections = self.detect_fn(input_tensor)

        # Process detections
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Visualize the detections
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_rgb,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def segment_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.reshape((-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        _, label, center = cv2.kmeans(np.float32(image), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))

        return cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)

    def compress_image(self, image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

    def remove_background(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = remove(image_rgb)
        return cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
	