from ultralytics import YOLO
import os

print("Loading model...")
model = YOLO("best.pt")

video_path = r"D:\mAjor jet\testpothole.mp4"

print("Checking file...")
print("Exists:", os.path.exists(video_path))

if not os.path.exists(video_path):
    print("❌ ERROR: Video not found!")
    exit()

print("Running detection on:", video_path)

results = model.predict(
    source=video_path,
    show=True,
    save=True,
    conf=0.5
)
print("Done!")