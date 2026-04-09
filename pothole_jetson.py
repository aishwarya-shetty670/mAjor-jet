import os
import time
import cv2
import numpy as np
import threading
from ultralytics import YOLO

# Import Jetson GPIO and Serial, with fallback for local testing
try:
    import Jetson.GPIO as GPIO
    import serial
    MOCK_HARDWARE = False
except ImportError:
    print("⚠️ Hardware libraries (Jetson.GPIO/serial) not found. Running in MOCK MODE.")
    MOCK_HARDWARE = True

# --- Configuration ---
# Look for the user's pre-trained Roboflow YOLOv8 model
MODEL_PATH = "best.pt" 
INPUT_PATH = "testpothole.mp4"
SERIAL_PORT = "/dev/ttyTHS1" # Change to /dev/ttyUSB0 if using USB adapter
BAUD_RATE = 9600

# GPIO Pins (BCM mode)
LED_GREEN = 18   # Normal
LED_YELLOW = 12  # Moderate
LED_RED = 13     # Severe
BUZZER_PIN = 23

# Processing parameters
FRAME_SKIP = 3  # Process every 3rd frame
LOG_FILE = "pothole_log.csv"
CONFIDENCE_THRESHOLD = 0.55  # Lowered slightly to catch missed potholes
MIN_AREA = 200               # Lowered to catch smaller/distant potholes

class HardwareController:
    def __init__(self):
        self.gps_data = "Unknown"
        self.stop_hw = False
        self.beep_frequency = 0 # 0 = off, >0 = pulse speed
        self.setup_gpio()
        self.setup_gps()
        
        # Buzzer thread
        self.buzzer_thread = threading.Thread(target=self._buzzer_loop, daemon=True)
        self.buzzer_thread.start()

    def setup_gpio(self):
        if MOCK_HARDWARE: return
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([LED_GREEN, LED_YELLOW, LED_RED, BUZZER_PIN], GPIO.OUT, initial=GPIO.LOW)

    def setup_gps(self):
        if MOCK_HARDWARE: return
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            # Run GPS reading in a separate thread to avoid blocking main loop
            self.gps_thread = threading.Thread(target=self._read_gps, daemon=True)
            self.gps_thread.start()
        except Exception as e:
            print(f"❌ GPS Setup Error: {e}")

    def _read_gps(self):
        while not self.stop_hw:
            try:
                if MOCK_HARDWARE: 
                    time.sleep(1)
                    continue
                line = self.ser.readline().decode('ascii', errors='replace')
                if line.startswith('$GPRMC'): # Basic NMEA sentence for location
                    # Simplified parsing - in real usage, use pynmeagps or similar
                    parts = line.split(',')
                    if parts[2] == 'A': # Status A = Active
                        lat = parts[3]
                        lon = parts[5]
                        self.gps_data = f"{lat}, {lon}"
            except:
                pass

    def update_indicators(self, severity):
        """Controls LEDs and Buzzer based on severity."""
        if MOCK_HARDWARE:
            print(f" [MOCK HW] Severity: {severity} | GPS: {self.gps_data}")
            return

        # Reset indicators
        GPIO.output([LED_GREEN, LED_YELLOW, LED_RED], GPIO.LOW)
        
        if severity == "Normal":
            GPIO.output(LED_GREEN, GPIO.HIGH)
            self.beep_frequency = 0 # No sound
        elif severity == "Moderate":
            GPIO.output(LED_YELLOW, GPIO.HIGH)
            self.beep_frequency = 0.5 # Slow beep
            self.log_detection("Moderate")
        elif severity == "Severe":
            GPIO.output(LED_RED, GPIO.HIGH)
            self.beep_frequency = 0.1 # Fast beep
            self.log_detection("Severe")

    def _buzzer_loop(self):
        """Non-blocking buzzer control in separate thread."""
        while not self.stop_hw:
            if self.beep_frequency > 0:
                if not MOCK_HARDWARE:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                time.sleep(self.beep_frequency)
                if not MOCK_HARDWARE:
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                time.sleep(self.beep_frequency)
            else:
                time.sleep(0.1) # Idle check

    def log_detection(self, severity):
        print(f"📍 Location [{self.gps_data}]: {severity} Pothole detected!")
        with open(LOG_FILE, "a") as f:
            f.write(f"{time.ctime()},{severity},{self.gps_data}\n")

    def cleanup(self):
        self.stop_hw = True
        self.beep_frequency = 0
        if not MOCK_HARDWARE:
            GPIO.cleanup()

class PotholeSystem:
    def __init__(self):
        self.load_model()
        self.hw = HardwareController()

    def load_model(self):
        print("Loading YOLOv8 model for Detection...")
        actual_model = MODEL_PATH
        if not os.path.exists(actual_model):
            alt_path = "D:/detection system/runs/detect/train2/weights/best.pt"
            if os.path.exists(alt_path):
                actual_model = alt_path
            else:
                print(f"⚠️ Model file {MODEL_PATH} not found. Falling back to default yolov8n.pt")
                actual_model = "yolov8n.pt"
            
        self.model = YOLO(actual_model)
        print(f"✅ YOLOv8 Model loaded successfully from {actual_model}")

        print("Loading CNN Classifier for Severity...")
        from tensorflow.keras.models import load_model as tf_load
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide tf logs
        cnn_path = "cnn_model.h5"
        if not os.path.exists(cnn_path):
            cnn_path = "severity_final.keras"
        self.cnn_model = tf_load(cnn_path)
        self.cnn_classes = ["Moderate", "Normal", "Severe"]
        print(f"✅ CNN Severity Model loaded from {cnn_path}")

    def preprocess_cnn(self, img_crop):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        # Convert BGR (OpenCV) to RGB (PIL/Keras)
        rgb_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        # Resize to CNN input size
        resized = cv2.resize(rgb_crop, (224, 224))
        # Convert to float and apply the exact MobileNetV2 preprocessing
        img_array = resized.astype("float32")
        preprocessed = preprocess_input(img_array)
        # Expand dims for batch size 1
        return np.expand_dims(preprocessed, axis=0)

    def run(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {source}")
            return

        # Calculate FPS and Delay
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 # Default if unknown
        wait_time = int(1000 / fps)
        print(f"🎬 Video Source detected at {fps} FPS. Using {wait_time}ms delay.")

        frame_count = 0
        detections = [] 

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or empty frame.")
                    break

                # Performance Optimization: Process every Nth frame
                if frame_count % FRAME_SKIP == 0:
                    detections = []
                    highest_severity = "Normal"
                    
                    # 1. Run YOLO inference to get bounding boxes
                    # Increased iou to 0.45 so adjacent/close potholes aren't merged or ignored
                    results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, iou=0.45)[0]
                    
                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Filter out tiny noisy detections that are likely false positives
                        area = (x2 - x1) * (y2 - y1)
                        if area < MIN_AREA:
                            continue
                        
                        # 2. Extract crop and classify severity using CNN
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        
                        input_data = self.preprocess_cnn(crop)
                        preds = self.cnn_model.predict(input_data, verbose=0)[0]
                        
                        # CNN Stabilization Logic
                        # classes: 0='Moderate', 1='Normal', 2='Severe'
                        idx = np.argmax(preds)
                        cnn_conf = preds[idx]
                        severity = self.cnn_classes[idx]
                        
                        # Only classify as "Severe" if the CNN is highly confident (> 65%).
                        # Otherwise, fallback to "Moderate" to prevent false alarms.
                        if severity == "Severe" and cnn_conf < 0.65:
                            severity = "Moderate"
                            
                        # If CNN is entirely unsure about it being a pothole, but YOLO found it,
                        # YOLO knows it's a pothole. We default to "Moderate" if CNN says "Normal".
                        if severity == "Normal":
                            severity = "Moderate"
                        
                        if severity != "Normal":
                            # We will show the CNN's confidence instead of YOLO's to debug severity
                            color = (0, 255, 255) if severity == "Moderate" else (0, 0, 255)
                            detections.append(((x1, y1, x2-x1, y2-y1), severity, color, cnn_conf))
                            
                            # Keep track of highest severity for hardware indicators
                            if highest_severity == "Normal": highest_severity = severity
                            elif highest_severity == "Moderate" and severity == "Severe": highest_severity = "Severe"
                    
                    # Update Hardware with the worst case found
                    self.hw.update_indicators(highest_severity)

                # Display Visual Output
                # Draw detections
                for (rect, text, color, conf) in detections:
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{text} ({conf:.2f})", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show summary at top left
                cv2.putText(frame, f"Potholes Detected: {len(detections)}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow("Jetson Pothole Detection (YOLOv8)", frame)
                
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
                
                frame_count += 1

        except Exception as e:
            print(f"❌ Runtime Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hw.cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_PATH, help="Path to video or '0' for camera")
    args = parser.parse_args()

    # Convert '0' to int for webcam
    src = int(args.input) if args.input.isdigit() else args.input

    try:
        system = PotholeSystem()
        system.run(src)
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
