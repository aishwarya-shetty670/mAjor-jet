import cv2
from ultralytics import YOLO
import time

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

# 1. Environment Setting
# Set this to False if you are testing on your local Windows PC
# Set this to True if you are deploying on the Jetson Nano
RUNNING_ON_JETSON = False 

# 2. Camera Setting
# Set to "USB" for standard webcams (Camera Index 0)
# Set to "CSI" for Jetson-specific Raspberry Pi Camera modules
CAMERA_TYPE = "USB" 

# ==========================================

def get_camera_source():
    """Returns the correct camera source/string based on the configuration."""
    if CAMERA_TYPE == "USB":
        return 0 # Default USB camera index
    elif CAMERA_TYPE == "CSI":
        # Standard GStreamer pipeline for Jetson Nano + Pi Camera
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    else:
        print(f"Unknown CAMERA_TYPE: {CAMERA_TYPE}. Defaulting to USB (0).")
        return 0

def main():
    # Load Model based on environment
    # Local PCs run raw PyTorch (.pt) just fine. Jetson Nano needs TensorRT (.engine) for FPS.
    if RUNNING_ON_JETSON:
        model_path = "best.engine" 
    else:
        model_path = "best.pt"
        
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"Failed to load model '{model_path}'. Ensure the file exists in the directory.")
        print(f"Error details: {e}")
        return
        
    # Setup Camera 
    cam_source = get_camera_source()
    print(f"Initializing camera source: {CAMERA_TYPE}")
    
    # We use cv2.CAP_GSTREAMER explicitly if CSI is selected
    if CAMERA_TYPE == "CSI":
        cap = cv2.VideoCapture(cam_source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(cam_source)
        
    if not cap.isOpened():
        print("Error: Could not open video source. Check camera connections.")
        return

    # OpenCV window sizing (Optimization for USB webcam)
    if CAMERA_TYPE == "USB":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variables for FPS Calculation
    prev_time = 0
    fps = 0

    print("Starting real-time pothole detection inference loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Stream ended or disconnected.")
            break

        # Inference Setup:
        # imgsz=320 drastically improves FPS on the Jetson Nano edge device.
        # conf=0.5 hides weak detections (reduces false positives)
        results = model.predict(source=frame, imgsz=320, conf=0.5, verbose=False)

        # Draw bounding boxes dynamically onto the frame
        annotated_frame = results[0].plot()

        # Calculate live Framerate
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Overlay FPS indicator
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # Overlay Environment Mode indicator
        mode_text = "JETSON EDGE - TensorRT" if RUNNING_ON_JETSON else "PC LOCAL - PyTorch"
        cv2.putText(annotated_frame, f"Env: {mode_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show standard window view
        cv2.imshow("Real-Time Pothole Detection", annotated_frame)

        # Break loop constraint if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up operations 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
