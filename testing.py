import cv2
import numpy as np
from keras.models import model_from_json
import time
import os
from datetime import datetime
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion_detection_secret_key_123'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class EmotionDetector:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
                      4: 'neutral', 5: 'sad', 6: 'surprise'}
        self.is_running = False
        self.webcam = None
        self.current_data = {
            'faces_count': 0,
            'fps': 0,
            'total_detections': 0,
            'dominant_emotion': '',
            'emotions': []
        }
        
        # Statistics
        self.emotion_count = {emotion: 0 for emotion in self.labels.values()}
        self.total_faces = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        
    def load_model(self):
        """Load the emotion detection model"""
        try:
            print("🔧 Loading emotion detection model...")
            if not os.path.exists("emotiondetector.json"):
                print("❌ Model file 'emotiondetector.json' not found!")
                return False
            if not os.path.exists("emotiondetector.h5"):
                print("❌ Weights file 'emotiondetector.h5' not found!")
                return False
                
            with open("emotiondetector.json", "r") as json_file:
                model_json = json_file.read()
            
            self.model = model_from_json(model_json)
            self.model.load_weights("emotiondetector.h5")
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_face_detector(self):
        """Load face detection classifier"""
        try:
            print("🔧 Loading face detection classifier...")
            # Try multiple possible paths for the cascade file
            possible_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            ]
            
            for haar_file in possible_paths:
                if os.path.exists(haar_file):
                    self.face_cascade = cv2.CascadeClassifier(haar_file)
                    if not self.face_cascade.empty():
                        print(f"✅ Face detector loaded from: {haar_file}")
                        return True
            
            # If no file found, try the default OpenCV path
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not self.face_cascade.empty():
                print("✅ Face detector loaded from OpenCV defaults")
                return True
                
            raise Exception("Could not load any face detection classifier")
            
        except Exception as e:
            print(f"❌ Error loading face detector: {e}")
            return False
    
    def extract_features(self, image):
        """Preprocess image for the model"""
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def get_emotion_color(self, emotion):
        """Get color for each emotion"""
        colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'neutral': (128, 128, 128), # Gray
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 165, 255)  # Orange
        }
        return colors.get(emotion, (255, 255, 255))
    
    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        try:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            frame_emotions = []
            
            for (x, y, w, h) in faces:
                try:
                    # Extract and preprocess face region
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (48, 48))
                    
                    # Extract features and predict
                    img_features = self.extract_features(face_roi)
                    predictions = self.model.predict(img_features, verbose=0)
                    
                    # Get dominant emotion
                    dominant_idx = np.argmax(predictions)
                    dominant_emotion = self.labels[dominant_idx]
                    confidence = np.max(predictions)
                    
                    # Update statistics
                    self.emotion_count[dominant_emotion] += 1
                    self.total_faces += 1
                    
                    frame_emotions.append({
                        'name': dominant_emotion,
                        'confidence': float(confidence)
                    })
                    
                    # Draw face rectangle
                    color = self.get_emotion_color(dominant_emotion)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw emotion label
                    label = f"{dominant_emotion} ({confidence:.1%})"
                    cv2.putText(frame, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Calculate FPS
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_time >= 1.0:
                self.current_data['fps'] = self.fps_counter
                self.fps_counter = 0
                self.fps_time = current_time
            
            # Update current data
            self.current_data['faces_count'] = len(faces)
            self.current_data['total_detections'] = self.total_faces
            self.current_data['emotions'] = frame_emotions
            
            if frame_emotions:
                self.current_data['dominant_emotion'] = max(frame_emotions, key=lambda x: x['confidence'])['name']
            else:
                self.current_data['dominant_emotion'] = ''
            
            # Add info text to frame
            info_text = f"Faces: {len(faces)} | FPS: {self.current_data['fps']}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame
    
    def start_detection(self):
        """Start emotion detection"""
        if not self.load_model() or not self.load_face_detector():
            return False
        
        print("🔧 Initializing webcam...")
        
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            self.webcam = cv2.VideoCapture(camera_index)
            if self.webcam.isOpened():
                print(f"✅ Webcam found at index {camera_index}")
                break
        else:
            print("❌ Error: Could not open any webcam")
            return False
        
        # Set camera properties
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.webcam.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam initialized successfully!")
        print("🚀 Starting emotion detection...")
        
        self.is_running = True
        return True
    
    def get_frame(self):
        """Get current frame from webcam"""
        if not self.is_running or self.webcam is None or not self.webcam.isOpened():
            return None
        
        ret, frame = self.webcam.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            return None
        
        processed_frame = self.process_frame(frame)
        return processed_frame
    
    def stop_detection(self):
        """Stop emotion detection"""
        self.is_running = False
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        self.print_session_summary()
        return True
    
    def print_session_summary(self):
        """Print session statistics"""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        print(f"Total faces analyzed: {self.total_faces}")
        
        if self.total_faces > 0:
            print("\nEmotion Distribution:")
            for emotion, count in sorted(self.emotion_count.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / self.total_faces) * 100
                    bar = "█" * int(percentage / 5)
                    print(f"  {emotion:8} : {bar} {count:3d} ({percentage:5.1f}%)")
            
            most_common = max(self.emotion_count.items(), key=lambda x: x[1])
            print(f"\n🎭 Most common emotion: {most_common[0]} ({most_common[1]} times)")
        
        print("👋 Session ended successfully!")

# Global detector instance
detector = EmotionDetector()
is_detection_active = False
detection_thread = None

@app.route('/')
def index():
    return render_template('index.html')

def detection_loop():
    """Main detection loop that runs in background"""
    global is_detection_active
    
    print("🎥 Starting detection loop...")
    while is_detection_active and detector.is_running:
        try:
            frame = detector.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    
                    # Convert to base64 for WebSocket transmission
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                    
                    # Send frame and data via WebSocket
                    socketio.emit('video_frame', {
                        'frame': f'data:image/jpeg;base64,{frame_b64}',
                        'faces_count': detector.current_data['faces_count'],
                        'fps': detector.current_data['fps'],
                        'total_detections': detector.current_data['total_detections'],
                        'dominant_emotion': detector.current_data['dominant_emotion'],
                        'emotions': detector.current_data['emotions']
                    })
            
            # Control frame rate
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in detection loop: {e}")
            break
    
    print("🛑 Detection loop stopped")

@socketio.on('connect')
def handle_connect():
    print('✅ Client connected')
    socketio.emit('connection_status', {'status': 'connected', 'message': 'Connected to emotion detection server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ Client disconnected')

@socketio.on('control')
def handle_control(data):
    global is_detection_active, detection_thread
    
    command = data.get('command')
    print(f'🎮 Received command: {command}')
    
    if command == 'start_detection':
        if not is_detection_active:
            if detector.start_detection():
                is_detection_active = True
                # Start detection in background thread
                detection_thread = threading.Thread(target=detection_loop, daemon=True)
                detection_thread.start()
                socketio.emit('control_response', {
                    'status': 'started',
                    'message': 'Emotion detection started successfully!'
                })
            else:
                socketio.emit('control_response', {
                    'status': 'error',
                    'message': 'Failed to start emotion detection. Check camera and model files.'
                })
        else:
            socketio.emit('control_response', {
                'status': 'error',
                'message': 'Detection is already running.'
            })
    
    elif command == 'stop_detection':
        if is_detection_active:
            is_detection_active = False
            detector.stop_detection()
            socketio.emit('control_response', {
                'status': 'stopped',
                'message': 'Emotion detection stopped.'
            })
        else:
            socketio.emit('control_response', {
                'status': 'error',
                'message': 'Detection is not running.'
            })
    
    elif command == 'save_frame':
        if is_detection_active:
            frame = detector.get_frame()
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                socketio.emit('control_response', {
                    'status': 'success',
                    'message': f'Frame saved as {filename}'
                })
        else:
            socketio.emit('control_response', {
                'status': 'error',
                'message': 'Start detection first to save frames.'
            })
    
    elif command == 'reset_stats':
        detector.emotion_count = {emotion: 0 for emotion in detector.labels.values()}
        detector.total_faces = 0
        socketio.emit('control_response', {
            'status': 'success',
            'message': 'Statistics reset successfully.'
        })

if __name__ == "__main__":
    print("🎭 Starting Real-Time Emotion Detection Server...")
    print("📁 Make sure you have the following files in the same directory:")
    print("   - emotiondetector.json")
    print("   - emotiondetector.h5")
    print("\n🌐 Access the web interface at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates directory")
    
    # Enable debug mode
    app.debug = True
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)