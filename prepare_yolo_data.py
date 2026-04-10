import os
import shutil
import random
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "d:/combinee"
CLASSES = ["moderate", "normal", "severe"]
DATASET_NAME = "yolo_dataset"
SPLIT_RATIO = {"train": 0.8, "val": 0.15, "test": 0.05}

# Target directory structure
BASE_DIR = os.path.join(SOURCE_DIR, DATASET_NAME)
IMAGE_SUBDIR = "images"
LABEL_SUBDIR = "labels"

def setup_directories():
    for split in SPLIT_RATIO.keys():
        os.makedirs(os.path.join(BASE_DIR, IMAGE_SUBDIR, split), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, LABEL_SUBDIR, split), exist_ok=True)

def generate_labels_and_split():
    img_list = []
    
    # Collect all images and their classes
    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(cls_path):
            print(f"⚠️ Warning: {cls_path} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            img_list.append((os.path.join(cls_path, f), idx))
    
    # Shuffle for random splitting
    random.shuffle(img_list)
    
    count = len(img_list)
    train_end = int(count * SPLIT_RATIO["train"])
    val_end = train_end + int(count * SPLIT_RATIO["val"])
    
    splits = {
        "train": img_list[:train_end],
        "val": img_list[train_end:val_end],
        "test": img_list[val_end:]
    }
    
    for split_name, data in splits.items():
        print(f"Processing {split_name} split...")
        for img_path, cls_idx in tqdm(data):
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            
            # 1. Copy Image
            target_img_path = os.path.join(BASE_DIR, IMAGE_SUBDIR, split_name, img_name)
            shutil.copy(img_path, target_img_path)
            
            # 2. Create YOLO Label (assuming the pothole is the central part of the image)
            # YOLO format: <class_idx> <x_center> <y_center> <width> <height> (all normalized 0-1)
            target_label_path = os.path.join(BASE_DIR, LABEL_SUBDIR, split_name, label_name)
            with open(target_label_path, "w") as f:
                # Default box: centered, 80% width, 80% height
                f.write(f"{cls_idx} 0.5 0.5 0.8 0.8\n")

def create_yaml():
    abs_path = os.path.abspath(BASE_DIR).replace('\\', '/')
    yaml_content = f"""
path: {abs_path}
train: images/train
val: images/val
test: images/test

names:
  0: moderate
  1: normal
  2: severe
"""
    with open(os.path.join(BASE_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"✅ data.yaml created at {os.path.join(BASE_DIR, 'data.yaml')}")

if __name__ == "__main__":
    setup_directories()
    generate_labels_and_split()
    create_yaml()
    print("✨ Dataset preparation complete!")
