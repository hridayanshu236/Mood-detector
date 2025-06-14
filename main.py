import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import tensorflow as tf

# Print debug information
print("Current working directory:", os.getcwd())
model_path = 'BestModel.keras'
if os.path.exists(model_path):
    print(f"Model file exists: {model_path}")
else:
    print(f"Model file NOT found: {model_path}")
    print("Files in directory:", os.listdir())

# Verify TensorFlow version
print("TensorFlow version:", tf.__version__)

MOODS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
MOOD_EMOJIS = ["😐", "😊", "😮", "😢", "😠", "🤢", "😨"]

class StylizedLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            background-color: rgba(255, 255, 255, 220);
            border-radius: 10px;
            padding: 5px;
            margin: 5px;
            color: #333333;
        """)

class StylizedButton(QPushButton):
    def __init__(self, text=""):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4A86E8;
                color: white;
                border: none;
                border-radius: 15px;
                padding: 10px;
                margin: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3B78DE;
            }
            QPushButton:pressed {
                background-color: #2D5BB9;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #888888;
            }
        """)

class RealTimeMoodApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Mood Predictor")
        self.setGeometry(200, 200, 800, 600)
        self.setStyleSheet("background-color: #F0F0F0;")

        # UI
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet("""
            background-color: #E0E0E0;
            border-radius: 15px;
            margin: 10px;
            padding: 10px;
        """)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(550, 450)
        self.image_label.setStyleSheet("""
            border: 2px solid #CCCCCC;
            border-radius: 10px;
            background-color: #333333;
        """)
        
        image_layout = QVBoxLayout(self.image_frame)
        image_layout.addWidget(self.image_label, 0, Qt.AlignCenter)

        # Results and status
        self.result_label = StylizedLabel("Waiting for prediction...")
        self.result_label.setFont(QFont("Arial", 16))
        
        self.emoji_label = QLabel("")
        self.emoji_label.setAlignment(Qt.AlignCenter)
        self.emoji_label.setFont(QFont("Arial", 48))
        
        self.status_label = StylizedLabel("Camera not started")
        self.status_label.setFont(QFont("Arial", 10))
        
        # Camera control buttons
        self.start_btn = StylizedButton("Start Camera")
        self.start_btn.clicked.connect(self.start_webcam)

        self.stop_btn = StylizedButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_webcam)
        self.stop_btn.setEnabled(False)
        
        # Camera selector
        self.camera_prev_btn = StylizedButton("◀ Prev Camera")
        self.camera_prev_btn.clicked.connect(lambda: self.change_camera(-1))
        
        self.camera_next_btn = StylizedButton("Next Camera ▶")
        self.camera_next_btn.clicked.connect(lambda: self.change_camera(1))

        # Layouts
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Add title
        title_label = QLabel("Real-Time Emotion Detection")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333; margin: 10px 0;")
        main_layout.addWidget(title_label)
        
        # Add camera view
        main_layout.addWidget(self.image_frame)
        
        # Results section
        results_frame = QFrame()
        results_frame.setStyleSheet("""
            background-color: white;
            border-radius: 15px;
            padding: 10px;
            margin: 5px;
        """)
        results_layout = QHBoxLayout(results_frame)
        results_layout.addWidget(self.emoji_label, 1)
        results_layout.addWidget(self.result_label, 2)
        main_layout.addWidget(results_frame)
        
        # Status
        main_layout.addWidget(self.status_label)
        
        # Camera controls
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.camera_prev_btn)
        camera_layout.addWidget(self.start_btn)
        camera_layout.addWidget(self.stop_btn)
        camera_layout.addWidget(self.camera_next_btn)
        main_layout.addLayout(camera_layout)
        
        self.setLayout(main_layout)

        # Load TensorFlow model with error handling
        try:
            print("Attempting to load model...")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully")
            self.status_label.setText("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                print("Trying with model.h5 instead...")
                self.model = tf.keras.models.load_model('model.h5', compile=False)
                print("Model loaded successfully from model.h5")
                self.status_label.setText("Model loaded successfully from model.h5")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                self.model = None
                self.status_label.setText("Model loading error - check console")

        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Camera settings
        self.current_camera = 0
        
        # Webcam
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_webcam(self):
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.current_camera)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.current_camera}")
            
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {self.current_camera} opened: {width}x{height} at {fps} FPS")
            
            self.status_label.setText(f"Camera {self.current_camera} active • {width:.0f}×{height:.0f} • {fps:.0f} FPS")
            self.timer.start(30)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            print(f"Camera error: {e}")
            self.status_label.setText(f"Camera error: {str(e)}")

    def stop_webcam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image_label.clear()
        self.result_label.setText("Waiting for prediction...")
        self.emoji_label.setText("")
        self.status_label.setText("Camera stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def change_camera(self, direction):
        was_running = self.cap is not None and self.cap.isOpened()
        if was_running:
            self.stop_webcam()
            
        self.current_camera = max(0, self.current_camera + direction)
        self.status_label.setText(f"Selected camera {self.current_camera}")
        
        if was_running:
            self.start_webcam()

    def preprocess_image(self, img):
        try:
            face = cv2.resize(img, (48, 48))
            if face.ndim == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            return face
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.setText("Error reading from camera")
                return

            try:
                h, w = frame.shape[:2]
                
                # Apply subtle vignette effect for aesthetics
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                radius = int(min(w, h) * 0.9)
                cv2.circle(mask, center, radius, 255, -1, cv2.LINE_AA)
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                
                # Normalize mask to [0, 1]
                mask = mask.astype(float) / 255
                mask = np.stack([mask] * 3, axis=2)
                
                # Apply the mask to the frame for aesthetic effect
                display_frame = (frame * mask).astype(np.uint8)
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # For display (BGR to RGB)
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(pixmap)
                
                # Make prediction if face is detected and model is loaded
                if len(faces) > 0 and self.model is not None:
                    # Get the largest face for prediction
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w_face, h_face = face
                    
                    # Extract face region
                    face_img = frame[y:y+h_face, x:x+w_face]
                    
                    # Process and predict
                    pred_face = self.preprocess_image(face_img)
                    if pred_face is not None:
                        pred = self.model.predict(pred_face, verbose=0)
                        mood_idx = np.argmax(pred)
                        mood = MOODS[mood_idx]
                        confidence = float(pred[0][mood_idx]) * 100
                        
                        # Update result with mood and emoji
                        self.result_label.setText(f"{mood.capitalize()} ({confidence:.1f}%)")
                        self.emoji_label.setText(MOOD_EMOJIS[mood_idx])
                        
                        # Update status with additional info
                        self.status_label.setText(f"Camera {self.current_camera} active • Face detected")
                        
                        # Set color based on emotion
                        if mood == "happiness":
                            self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #2E8B57; font-weight: bold;")
                        elif mood in ["anger", "disgust", "fear"]:
                            self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #B22222; font-weight: bold;")
                        elif mood == "sadness":
                            self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #4682B4; font-weight: bold;")
                        elif mood == "surprise":
                            self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #FF8C00; font-weight: bold;")
                        else:
                            self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #333333; font-weight: bold;")
                        
                    else:
                        self.result_label.setText("Error processing face")
                        self.emoji_label.setText("❌")
                elif len(faces) == 0:
                    self.status_label.setText("No face detected - Position yourself in the frame")
                    self.result_label.setText("Waiting for face...")
                    self.emoji_label.setText("🔍")
                    self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #333333;")
                elif self.model is None:
                    self.status_label.setText("Model not loaded")
                    self.result_label.setText("Model error")
                    self.emoji_label.setText("❌")
                    self.result_label.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; padding: 5px; color: #FF0000;")

            except Exception as e:
                print(f"Error in frame update: {e}")
                self.status_label.setText(f"Error: {str(e)[:30]}")

    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = RealTimeMoodApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")