# 🐯 Tiger Detection and Re-Identification System

An end-to-end **AI-powered Tiger Detection and Re-Identification System** built using **YOLOv8, ResNet50, and Computer Vision techniques**.  
This system detects tigers in real-time video/images and assigns unique IDs to individual tigers using feature matching.

---

## 📌 Overview

Wildlife monitoring often requires identifying individual animals across multiple sightings.  
This project automates that process using:

- **YOLOv8** for real-time tiger detection  
- **ResNet50** for feature extraction (embeddings)  
- **Cosine Similarity + ORB Matching** for re-identification  
- **Centroid Tracking** for maintaining identity across frames  

---

## 🚀 Features

- 🎯 Real-time tiger detection using YOLOv8  
- 🧠 Feature-based tiger identification (Re-ID)  
- 🔁 Maintains consistent Tiger IDs across frames  
- 🧩 Handles ambiguous cases using ORB keypoint matching  
- 📁 Automatically builds a database of detected tigers  
- 🖥️ Live visualization with bounding boxes and IDs  

---

## 🧠 System Architecture


Video Input → YOLOv8 Detector → Crop Extraction → ResNet50 Embedding
→ Re-ID Matching (Cosine + ORB) → Centroid Tracker → Output Visualization


---

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)**
- **PyTorch & TorchVision**
- **OpenCV**
- **NumPy**
- **Pillow**

---

## 📂 Project Structure


Tiger-Detection-and-Re-Identification-System/
│
├── train_tiger_detector.py # Model training script

├── tiger_detector.py # Main detection + Re-ID pipeline

├── tiger_detector1.py # Alternate / experimental script

├── requirements.txt # Dependencies

├── README.md # Project documentation


---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/piyushsangore/Tiger-Detection-and-Re-Identification-System.git
cd Tiger-Detection-and-Re-Identification-System
```
2️⃣ Install Dependencies
```bash

pip install -r requirements.txt
```
3️⃣ Setup Kaggle API (for dataset download)

Download kaggle.json from your Kaggle account

Place it in the project directory
```bash

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
🧪 Model Training

Run the training script:

```bash

python train_tiger_detector.py
Training Details:
```

Model: YOLOv8 (yolov8s)

Epochs: 50

Image Size: 512

Dataset: Tiger Detection Dataset (Kaggle)

After training, the best model will be saved in:

runs/detect/train/weights/best.pt
🎥 Running Detection & Re-Identification
python tiger_detector.py
What happens:

Webcam starts

Tigers are detected using YOLOv8

Each tiger is assigned a unique ID (Tiger_1, Tiger_2, ...)

IDs remain consistent across frames

New tigers are automatically added to the database

🧩 How Re-Identification Works

Detection
YOLOv8 detects tiger bounding boxes

Feature Extraction
ResNet50 converts cropped tiger image into a feature vector

Matching

Cosine similarity compares embeddings

If ambiguous → ORB feature matching is used

Tracking
Centroid tracking ensures identity stability across frames

Database Update

New tiger → saved in database

Existing tiger → embedding updated

📊 Evaluation Metrics

Precision

mAP50

mAP50-95

These metrics evaluate how accurately the model detects tigers.

📸 Output Example

Bounding boxes drawn around detected tigers

Labels like Tiger_1, Tiger_2

Live count of sightings

Console logs showing matching results

⚠️ Notes

Do NOT upload kaggle.json to GitHub

Ensure GPU is enabled for faster training (recommended)

Adjust thresholds in script for better accuracy:

Similarity threshold

ORB match threshold

🔮 Future Improvements

Multi-camera tiger tracking

Cloud-based tiger database

Support for multiple animal species

Improved deep learning Re-ID models

Mobile or web dashboard for monitoring

👨‍💻 Author

Piyush Sangore
AI Student | Developer

⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
