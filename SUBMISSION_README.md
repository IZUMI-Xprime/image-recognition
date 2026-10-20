# 🦅 Wildlife Sentinel — OpenCV AI Competition 2026

> **Real-time wildlife detection and unique individual counting** powered by OpenCV 5, YOLOv8 segmentation, ByteTrack, and AWS cloud integration.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-green?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Rekognition%20%7C%20DynamoDB-orange?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🏆 Competition Requirements Checklist

| Requirement | Status | Details |
|---|---|---|
| **OpenCV 5 Integration** | ✅ | Frame processing pipeline, DNN, CLAHE, background subtraction, optical flow stabilization, ORB/AKAZE re-ID features |
| **AWS Component** | ✅ | S3 clip storage, Rekognition species verification, DynamoDB event logging, SNS alerts, Lambda deployment |
| **Pinned Dependencies** | ✅ | `requirements.txt` with version ranges |
| **Reproducible Setup** | ✅ | Step-by-step local deployment + AWS setup guide |
| **Technical Report** | ✅ | `docs/TECHNICAL_REPORT.md` — full architecture, algorithms, evaluation |
| **Architecture Document** | ✅ | `docs/ARCHITECTURE.md` — component diagrams, data flow |

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Quick Start — Local Deployment](#quick-start--local-deployment)
3. [OpenCV 5 Features Used](#opencv-5-features-used)
4. [AWS Cloud Integration](#aws-cloud-integration)
5. [Project Structure](#project-structure)
6. [Configuration Reference](#configuration-reference)
7. [Testing the Application](#testing-the-application)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Technical Documentation](#technical-documentation)

---

## What It Does

Wildlife Sentinel is a **full-stack computer vision system** for real-time wildlife monitoring that solves the **double-counting problem** — one deer standing in frame for 10 minutes is counted as **1 individual**, not 18,000.

### Core Capabilities

- **Live webcam stream** with real-time YOLOv8 segmentation — polygon outlines, not rectangles
- **Unique individual counting** via ByteTrack persistent IDs — same animal across frames = same ID = counted once
- **Video analysis** of uploaded files (tested on 2+ hour videos) — species breakdown with annotated output
- **Detection heatmaps** from segmentation polygon masks — shows where in the frame animals appear
- **OpenCV 5 preprocessing pipeline** — denoising, CLAHE contrast enhancement, background subtraction, video stabilization
- **AWS cloud backend** — S3 storage, Rekognition verification, DynamoDB logging, SNS alerts
- **Performance dashboard** — FPS, inference latency, GPU memory, CPU, RAM, uptime

---

## Quick Start — Local Deployment

### Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 |
| RAM | 4 GB (8 GB+ recommended) |
| GPU | Optional (NVIDIA GTX 1060+ for GPU mode) |

### Step 1 — Clone & Install

```bash
git clone https://github.com/IZUMI-Xprime/image-recognition.git
cd image-recognition

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2 — Install CUDA PyTorch (GPU users only)

```bash
# Check CUDA version
nvidia-smi

# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only (skip this step — already installed via requirements.txt)
```

### Step 3 — Configure Environment

```bash
# Copy the example config
cp .env.example .env

# Edit .env with your settings (at minimum):
#   SECRET_KEY=<random-string>
#   ADMIN_PASSWORD=<your-password>
```

### Step 4 — Build Cython Extension (Optional)

```bash
# Requires GCC (MinGW on Windows)
python build_tracker.py

# If this fails, the app still works with pure-Python IoU (slightly slower)
```

### Step 5 — Run

```bash
python webcam_full_v5.py
```

Open **http://localhost:5000** in your browser. Login with `admin` / your password.

### Expected Startup Log

```
17:30:01 [INFO] GPU detected: NVIDIA GeForce RTX 3050  (batch=8)
17:30:01 [INFO] Cython tracker loaded — fast IoU active.
17:30:01 [INFO] OpenCV version: 5.x.x
17:30:01 [INFO] OpenCV 5 pipeline enabled: CLAHE=on, BgSub=on, Denoise=off, Stabilize=off
17:30:01 [INFO] AWS integration: disabled (set USE_AWS=true to enable)
17:30:03 [INFO] Camera connected.
 * Running on http://0.0.0.0:5000
```

---

## OpenCV 5 Features Used

Wildlife Sentinel leverages OpenCV 5 extensively across the processing pipeline. All OpenCV 5 features are encapsulated in [`opencv5_features.py`](opencv5_features.py).

| Feature | OpenCV 5 API | Purpose |
|---|---|---|
| **Frame Capture** | `cv2.VideoCapture` | Camera input and video file reading |
| **Color Conversion** | `cv2.cvtColor` | BGR↔RGB for display and model input |
| **Polygon Drawing** | `cv2.fillPoly`, `cv2.polylines` | Segmentation mask visualization |
| **Image Blending** | `cv2.addWeighted` | Semi-transparent mask overlays |
| **Gaussian Blur** | `cv2.GaussianBlur` | Heatmap smoothing |
| **CLAHE** | `cv2.createCLAHE` | Adaptive contrast enhancement for varying lighting |
| **Denoising** | `cv2.fastNlMeansDenoisingColored` | Low-light frame cleanup |
| **Background Subtraction** | `cv2.createBackgroundSubtractorMOG2` | Motion pre-filter to skip static frames |
| **Optical Flow** | `cv2.calcOpticalFlowPyrLK` | Video stabilization for trail cameras |
| **Homography** | `cv2.findHomography` (RANSAC) | Shake compensation |
| **Feature Extraction** | `cv2.ORB_create`, `cv2.AKAZE_create` | Animal re-identification descriptors |
| **Object Tracking** | `cv2.TrackerCSRT_create` | Complementary OpenCV-native tracking |
| **DNN Module** | `cv2.dnn.readNetFromONNX` | Secondary ONNX model inference |
| **Video I/O** | `cv2.VideoWriter`, `cv2.imencode` | Output recording and MJPEG streaming |
| **Text Rendering** | `cv2.putText` | Overlay labels and timestamps |

### Why OpenCV 5?

OpenCV 5 brings critical improvements for this application:
- **Improved ONNX support** — better compatibility with modern model exports
- **Enhanced DNN module** — faster inference for secondary classification
- **Better CUDA acceleration** — improved `cv2.cuda` operations for GPU-accelerated preprocessing
- **Modernized API** — cleaner Python bindings, type hints support
- **Optimized video I/O** — faster codec handling for real-time streaming

---

## AWS Cloud Integration

The AWS integration is modular and optional — controlled by `USE_AWS=true` in `.env`. See [`aws_integration.py`](aws_integration.py).

### Components

| AWS Service | Purpose | Config Variable |
|---|---|---|
| **S3** | Store analyzed clips, snapshots, reports | `AWS_S3_BUCKET` |
| **Rekognition** | Secondary species verification | `AWS_REKOGNITION_VERIFY` |
| **DynamoDB** | Detection event metadata logging | `AWS_DYNAMODB_TABLE` |
| **SNS** | Cloud-native alert notifications | `AWS_SNS_TOPIC_ARN` |
| **Lambda** | Serverless video analysis trigger | Built-in handler |

### AWS Setup

```bash
# 1. Configure AWS CLI
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Output format (json)

# 2. Create S3 bucket
aws s3 mb s3://wildlife-sentinel-data

# 3. Create DynamoDB table
aws dynamodb create-table \
    --table-name wildlife-sentinel-events \
    --attribute-definitions AttributeName=event_id,AttributeType=S \
    --key-schema AttributeName=event_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# 4. Create SNS topic (optional)
aws sns create-topic --name wildlife-alerts

# 5. Update .env
USE_AWS=true
AWS_REGION=us-east-1
AWS_S3_BUCKET=wildlife-sentinel-data
AWS_DYNAMODB_TABLE=wildlife-sentinel-events
```

---

## Project Structure

```
image-recognition/
│
├── webcam_full_v5.py           # Main application — Flask, routes, all UI
├── opencv5_features.py         # OpenCV 5 enhanced processing pipeline
├── aws_integration.py          # AWS S3/Rekognition/DynamoDB/SNS/Lambda
│
├── tracker_cy.pyx              # Cython IoU matching (compiled to C)
├── build_tracker.py            # Cython build script (GCC, no MSVC)
│
├── requirements.txt            # Pinned Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git exclusions
├── README.md                   # ← This file (submission README)
│
├── docs/
│   ├── TECHNICAL_REPORT.md     # Competition technical report
│   └── ARCHITECTURE.md         # System architecture document
│
├── videos/                     # Live stream clips (auto-created)
└── uploads/                    # Uploaded + analyzed videos (auto-created)
```

---

## Configuration Reference

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | Random | Flask session signing key |
| `ADMIN_PASSWORD` | `password123` | Login password — **change this** |
| `CONF_THRESHOLD` | `0.40` | YOLO detection confidence threshold |
| `TRACK_BUFFER` | `90` | Frames before track retirement (~10s at 9fps) |

### OpenCV 5 Pipeline

| Variable | Default | Description |
|---|---|---|
| `OPENCV5_PIPELINE` | `true` | Enable OpenCV 5 preprocessing |
| `OPENCV5_CLAHE` | `true` | Adaptive contrast enhancement |
| `OPENCV5_DENOISE` | `false` | Frame denoising (slow, for low-light) |
| `OPENCV5_BG_SUBTRACT` | `true` | Background subtraction pre-filter |
| `OPENCV5_STABILIZE` | `false` | Video stabilization |

### AWS Integration

| Variable | Default | Description |
|---|---|---|
| `USE_AWS` | `false` | Master toggle for all AWS features |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_S3_BUCKET` | — | S3 bucket for clip/snapshot storage |
| `AWS_DYNAMODB_TABLE` | — | DynamoDB table for event logging |
| `AWS_SNS_TOPIC_ARN` | — | SNS topic for alerts |

---

## Testing the Application

### 1. Basic Functionality Test

```bash
# Start the app
python webcam_full_v5.py

# Verify startup log shows:
#   - GPU/CPU detection
#   - OpenCV version
#   - OpenCV 5 pipeline status
#   - AWS integration status

# Open http://localhost:5000
# Login → Dashboard should show live feed (or "Camera stopped" if no webcam)
```

### 2. OpenCV 5 Pipeline Test

```bash
# Run the standalone test
python -c "
from opencv5_features import OpenCV5Pipeline, get_opencv_info
print(get_opencv_info())
pipeline = OpenCV5Pipeline(enable_clahe=True, enable_bg_subtract=True)
import numpy as np
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = pipeline.process(test_frame)
print(f'Pipeline output shape: {result.shape}')
print('OpenCV 5 pipeline: OK')
"
```

### 3. AWS Integration Test

```bash
# Test AWS connection (requires configured credentials)
python -c "
from aws_integration import init_aws, get_aws_status
status = get_aws_status()
print(status)
"
```

### 4. Video Analysis Test

```bash
# Upload a test video via the /analyse page
# Or use curl:
curl -X POST http://localhost:5000/api/analyse \
  -F "video=@test_video.mp4" \
  -b "session=<your-session-cookie>"
```

### 5. Full Integration Test

```bash
# Run all component checks
python -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')

import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

from ultralytics import YOLO
print('Ultralytics: OK')

from opencv5_features import OpenCV5Pipeline
print('OpenCV5 pipeline: OK')

from aws_integration import get_aws_status
print(f'AWS: {get_aws_status()}')

print('\\n✓ All components loaded successfully')
"
```

---

## Performance Benchmarks

*Tested on: Intel i5-12th Gen + NVIDIA RTX 3050 (4GB VRAM), 1080p webcam, Windows 11*

| Metric | GPU Mode | CPU Mode |
|---|---|---|
| Live stream FPS | 8–12 fps | 3–5 fps |
| Inference latency | 80–120 ms/frame | 250–400 ms/frame |
| Video analysis throughput | ~9 fps effective | ~3 fps effective |
| RAM usage | ~380 MB | ~280 MB |
| GPU memory | ~450 MB | — |
| Tracker overhead (Cython) | < 0.5 ms/frame | < 0.5 ms/frame |
| OpenCV 5 pipeline overhead | ~2 ms/frame | ~5 ms/frame |
| AWS upload latency | ~200 ms/clip | ~200 ms/clip |

---

## Technical Documentation

- 📄 **[Technical Report](docs/TECHNICAL_REPORT.md)** — Problem statement, architecture, algorithms, evaluation
- 🏗️ **[Architecture Document](docs/ARCHITECTURE.md)** — Component diagrams, data flow, deployment topology

---

## License

MIT License — see `LICENSE` for details.

---

*Wildlife Sentinel — Built by Darnish, Mechatronics student at PSG Polytechnic College, Coimbatore.*
*Submitted to the OpenCV AI Competition 2026, powered by AWS.*
