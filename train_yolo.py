from ultralytics import YOLO
import os
from multiprocessing import freeze_support

def main():
    # Provide the absolute path to data.yaml
    data_yaml_path = os.path.abspath("yolo_dataset/data.yaml")
    
    # Load a pretrained YOLOv8 Nano model (optimized for edge devices like Jetson Nano)
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    print(f"Starting training using {data_yaml_path}...")
    # Train the model
    # epochs=10 for a quick demo. Increase to 50-100 for better results.
    # imgsz=224 to match the previous CNN and keep training fast.
    results = model.train(
        data=data_yaml_path,
        epochs=10,
        imgsz=224,
        batch=16,
        name="pothole_yolo_model",
        device="cpu"  # Force CPU if no CUDA is available, ultralytics usually auto-detects though
    )
    
    print("✨ Training complete! Model saved in runs/detect/pothole_yolo_model/weights/")

if __name__ == "__main__":
    freeze_support() # Required for Windows multiprocessing
    main()
