"""
Tiger Detection Model Training Script (YOLOv8)

This script:
1. Downloads the tiger dataset from Kaggle
2. Extracts and prepares the dataset
3. Creates a YOLO-compatible YAML file
4. Trains a YOLOv8 model for tiger detection
5. Evaluates model performance
6. Runs inference on test images
"""

import os
import zipfile
import random
import shutil
import yaml
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_NAME = "gauravpendharkar/tiger-detection-dataset"
BASE_DIR = "/content"
DATASET_ZIP = os.path.join(BASE_DIR, "tiger-detection-dataset.zip")
DATASET_PATH = os.path.join(BASE_DIR, "final_data/dataset")
TEST_DIR = os.path.join(DATASET_PATH, "images/test")
TEST_1000_DIR = os.path.join(DATASET_PATH, "images/test_1000")
YAML_PATH = os.path.join(DATASET_PATH, "tiger.yaml")

EPOCHS = 50
IMG_SIZE = 512
BATCH_SIZE = 16

# -----------------------------
# STEP 1: DOWNLOAD DATASET
# -----------------------------
print("📥 Downloading dataset from Kaggle...")
os.system("mkdir -p ~/.kaggle")
os.system("cp kaggle.json ~/.kaggle/")
os.system("chmod 600 ~/.kaggle/kaggle.json")
os.system(f"kaggle datasets download -d {DATASET_NAME} -p {BASE_DIR}")

# -----------------------------
# STEP 2: EXTRACT DATASET
# -----------------------------
print("📦 Extracting dataset...")
with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
    zip_ref.extractall(BASE_DIR)

print(f"✅ Dataset extracted to: {BASE_DIR}")

# -----------------------------
# STEP 3: CREATE REDUCED TEST SET (1000 IMAGES)
# -----------------------------
print("🧪 Creating reduced test set...")

os.makedirs(TEST_1000_DIR, exist_ok=True)

all_images = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

random.seed(42)  # reproducibility
sampled_images = random.sample(all_images, min(1000, len(all_images)))

for img in sampled_images:
    shutil.copy(os.path.join(TEST_DIR, img),
                os.path.join(TEST_1000_DIR, img))

print(f"✅ Test set created with {len(sampled_images)} images")

# -----------------------------
# STEP 4: CREATE YAML FILE
# -----------------------------
print("📝 Creating dataset YAML...")

data_yaml = {
    'train': os.path.join(DATASET_PATH, 'images/train'),
    'val': os.path.join(DATASET_PATH, 'images/val'),
    'test': TEST_1000_DIR,
    'nc': 1,
    'names': ['tiger']
}

with open(YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ YAML file created")

# -----------------------------
# STEP 5: LOAD MODEL
# -----------------------------
print("🧠 Loading YOLOv8 model...")
model = YOLO("yolov8s.pt")  # change to yolov8m.pt for higher accuracy

# -----------------------------
# STEP 6: TRAIN MODEL
# -----------------------------
print("🚀 Training model...")

model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=2
)

# -----------------------------
# STEP 7: EVALUATE MODEL
# -----------------------------
print("📊 Evaluating model...")

metrics = model.val(data=YAML_PATH, split="test")

print("\n📊 Evaluation Results:")
print(f"mAP: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map75:.4f}")
print(f"Per-class mAP: {metrics.box.maps}")

# -----------------------------
# STEP 8: RUN INFERENCE
# -----------------------------
print("🔍 Running inference...")

results = model.predict(
    source=TEST_1000_DIR,
    imgsz=416,
    conf=0.25,
    max_det=50,
    save=False
)

print("✅ Inference completed")
