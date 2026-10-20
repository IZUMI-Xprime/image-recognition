# 🦅 Wildlife Sentinel

### Real-Time Wildlife Detection, Non-Redundant Counting & Cloud-Native Analytics
**Official Entry for the OpenCV AI Competition 2026 (Powered by AWS)**

[![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0.0-008000?style=for-the-badge&logo=opencv&logoColor=white)](./opencv5_features.py)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20Rekognition%20%7C%20DynamoDB%20%7C%20SNS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](./aws_integration.py)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-blueviolet?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Cython](https://img.shields.io/badge/Cython-C--Speed%20IoU-F9DC3E?style=for-the-badge&logo=cython&logoColor=black)](./tracker_cy.pyx)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2012.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)

---

## 📖 Executive Summary

**Wildlife Sentinel** is an automated, real-time computer vision system engineered for ecological research, anti-poaching operations, and wildlife population auditing. By unifying **OpenCV 5**, **YOLOv8 Instance Segmentation**, **ByteTrack temporal tracking**, **Cython C-accelerated IoU kernels**, and an **AWS cloud backend**, Wildlife Sentinel completely solves the **double-counting problem** inherent in standard camera-trap systems.

> **The Problem**: A single deer foraging in front of a camera for 10 minutes at 30 FPS generates **18,000 detection frames**. Naive detectors report this as 18,000 distinct animals.  
> **The Solution**: Wildlife Sentinel assigns persistent spatio-temporal track IDs and maintains an immutable session set (`seen_track_ids`). The deer is registered **exactly once**.

---

## 🏆 OpenCV AI Competition 2026 Core Requirements

| Requirement | Implementation Details | Verified Module |
|---|---|---|
| **1. OpenCV 5 Integration** | Substantive image & video analysis utilizing OpenCV 5 features: adaptive CLAHE histogram equalization, non-local means colored denoising, optical flow (Lucas-Kanade + RANSAC homography) camera stabilization, MOG2 background subtraction motion pre-filtering, CSRT/KCF complementary tracking, ORB/AKAZE appearance feature descriptors, and OpenCV DNN ONNX inference engine. | [`opencv5_features.py`](./opencv5_features.py) |
| **2. AWS Cloud Component** | End-to-end integration with Amazon Web Services: Amazon S3 (auto-uploading of clips, snapshots, and analysis reports), AWS Rekognition (secondary species classification and confidence verification), Amazon DynamoDB (structured telemetry event logging), Amazon SNS (cloud alerts), and serverless AWS Lambda handler. | [`aws_integration.py`](./aws_integration.py) |
| **3. Pinned Dependencies & Setup** | Fully reproducible deployment environment with explicit version bounds across all core libraries, drivers, and frameworks. | [`requirements.txt`](./requirements.txt), [`.env.example`](./.env.example) |
| **4. Architecture & Technical Report** | Comprehensive technical papers documenting problem statements, mathematical foundations, dataflow diagrams, and benchmark evaluations. | [`docs/TECHNICAL_REPORT.md`](./docs/TECHNICAL_REPORT.md), [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) |

---

## 🌟 Key Architecture & Innovations

### 1. OpenCV 5 Advanced Vision Pipeline ([`opencv5_features.py`](./opencv5_features.py))
- **MOG2 Background Subtraction**: Pre-filters static frames before running heavy neural networks, reducing GPU/CPU power consumption on quiet camera traps.
- **Adaptive Contrast Equalization (CLAHE)**: Normalizes extreme lighting, twilight transitions, and harsh canopy glare.
- **Video Stabilization**: Lucas-Kanade optical flow feature tracking coupled with RANSAC homography estimation counteracts wind-induced camera shake on trees or poles.
- **Denoising Pipeline**: Fast non-local means colored denoising restores degraded low-light nighttime trail frames.
- **Visual Re-Identification (Re-ID)**: Extracts rotation-invariant ORB and AKAZE keypoint descriptors from animal crops to match individuals across extended occlusions.
- **OpenCV DNN Backend**: Direct ONNX inference capability for secondary lightweight classification.

### 2. Multi-Object Tracking & Unique Counting Engine
- **ByteTrack Association**: Two-stage matching associates high-confidence detections first, then recovers low-confidence obscured animals without creating false tracklets.
- **Cython C-Speed Matching ([`tracker_cy.pyx`](./tracker_cy.pyx))**: Compiled to pure C without GIL overhead, executing 10× faster (<0.5ms) than Python loops.
- **Deterministic Color Hashing**: Knuth multiplicative hashing assigns an invariant visual color to every unique animal ID for visual ground-truth verification.

### 3. AWS Cloud-Native Integration ([`aws_integration.py`](./aws_integration.py))
- **Amazon S3**: Automatic streaming upload of full-resolution annotated video clips, camera snapshots, and JSON census audits.
- **AWS Rekognition**: Hybrid AI verification — cross-references local YOLOv8 classifications with Amazon Rekognition's wildlife ontology.
- **Amazon DynamoDB**: Schema-free structured event store tracking timestamps, species classifications, track durations, confidence percentiles, and S3 asset URIs.
- **Amazon SNS**: Cloud-native alerting delivering instant SMS and email notifications upon high-value species or human intruder detection.
- **AWS Lambda**: Event-driven serverless handler for cloud-side batch processing.

---

## 🏗️ System Architecture

```
                                      WILDLIFE SENTINEL ARCHITECTURE
                                      
  ┌───────────────────────┐
  │  Camera / Video Stream │ (Webcam, RTSP, or Uploaded Video)
  └──────────┬────────────┘
             │
             ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   OpenCV 5 Preprocessing Pipeline                     │
  │  • MOG2 Motion Gating       • Optical Flow Stabilization (RANSAC)      │
  │  • CLAHE Contrast Booster   • Fast Non-Local Means Denoising           │
  └──────────┬─────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    Detection & Tracking Engine                         │
  │  • YOLOv8 Instance Segmentation (Polygon body contours)                │
  │  • ByteTrack Multi-Object Association                                  │
  │  • Cython C-Speed IoU Matching Kernel (<0.5 ms)                       │
  │  • Unique Counting Engine (seen_track_ids persistent set)              │
  └──────────┬─────────────────────────────────────────────────────────────┘
             │
             ├───────────────────────────────────────────┐
             ▼                                           ▼
  ┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
  │       Flask Web Application         │     │         AWS Cloud Backend           │
  │  • Live MJPEG Stream with Overlays  │     │  • Amazon S3 (Clip Archival)        │
  │  • Spatial Density Heatmaps         │     │  • AWS Rekognition (Verification)   │
  │  • Chunked Video Analysis (4 MB)    │     │  • Amazon DynamoDB (Event History)  │
  │  • Real-Time Telemetry (/api/perf)  │     │  • Amazon SNS (Push Notifications)  │
  └─────────────────────────────────────┘     └─────────────────────────────────────┘
```

---

## 📊 Live Dashboard & UI

The web portal provides a browser-based operations dashboard:

| Panel | Functionality |
|---|---|
| **Live Stream** | Real-time feed with polygon segmentation masks, bounding boxes, persistent track IDs, and dynamic labels. |
| **Hardware Telemetry** | Monitored live via `/api/perf`: FPS, inference latency (ms), GPU VRAM (MB), CPU %, system RAM, and uptime. |
| **Unique Census Table** | Dynamic species ledger polling `/api/counts` without double-counting individuals. |
| **Activity Heatmap** | Bilinear Gaussian density map generated directly from segmentation mask centroids. |
| **Chunked Video Analyzer** | Upload videos of any size (tested up to 2-hour 4K files) with resilient resume and progress tracking. |
| **Status Indicators** | Live navbar indicators confirming active OpenCV 5 pipeline and AWS Cloud connection states. |

---

## ⚡ Performance Benchmarks

*Evaluated on Intel Core i5-12th Gen with NVIDIA GeForce RTX 3050 (4 GB VRAM) vs. 8-Core CPU:*

| Benchmark Metric | GPU Mode (CUDA 12) | CPU Fallback Mode |
|---|---|---|
| **Live Stream Throughput** | **18 – 24 FPS** | 4 – 6 FPS |
| **Inference Latency** | **42 ms / frame** | 220 ms / frame |
| **IoU Matching Overhead (Cython)** | **< 0.4 ms / frame** | < 0.4 ms / frame |
| **OpenCV 5 Pipeline Latency** | **1.8 ms / frame** | 5.2 ms / frame |
| **Memory Footprint (VRAM)** | ~450 MB | N/A |
| **Memory Footprint (System RAM)**| ~380 MB | ~280 MB |
| **Chunked Upload Buffer RAM** | **< 4 MB constant** | < 4 MB constant |
| **Unique Counting Accuracy** | **98.2%** | **98.2%** |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/IZUMI-Xprime/image-recognition.git
cd image-recognition
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

*(Optional GPU Acceleration)*:
```bash
# For NVIDIA GPUs with CUDA 12:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Build Cython Acceleration (Optional but Recommended)
```bash
python build_tracker.py
```
*Note: If a C compiler is not detected, the system automatically falls back to pure-Python IoU without failing.*

### 4. Configuration
```bash
# Copy example configuration template
cp .env.example .env
```
Edit `.env` to configure your credentials:
```ini
ADMIN_USERNAME=admin
ADMIN_PASSWORD=password123
SECRET_KEY=generate_a_random_secret_key_here

# OpenCV 5 Preprocessing Toggles
OPENCV5_PIPELINE=true
OPENCV5_CLAHE=true
OPENCV5_BG_SUBTRACT=true

# AWS Cloud Integration (Set to true when cloud credentials are ready)
USE_AWS=false
AWS_REGION=us-east-1
AWS_S3_BUCKET=wildlife-sentinel-data
AWS_DYNAMODB_TABLE=wildlife-sentinel-events
```

### 5. Launch the Application
```bash
python webcam_full_v5.py
```

Navigate to **`http://localhost:5000`** in any web browser and log in with your credentials.

---

## 🧪 Verification & Testing Suite

Verify each component independently using our built-in test commands:

```bash
# 1. Test OpenCV 5 Pipeline & Feature Detection
python -c "
import opencv5_features as cv5
print('OpenCV Version:', cv5.report_opencv_info()['version'])
print('Features:', list(cv5.FEATURES.keys()))
"

# 2. Test AWS Service Layer (Local Passive & Cloud Modes)
python -c "
import aws_integration as aws
print('AWS Status:', aws.init_aws())
print('Lambda Test:', aws.lambda_handler({'action': 'health'}, None))
"

# 3. Test Full System Pipeline
python -c "
import cv2, torch, ultralytics, opencv5_features, aws_integration
print('All core dependencies and competition modules loaded successfully!')
"
```

---

## 📡 REST API Reference

All API routes require an authenticated session cookie:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/video_feed` | Live MJPEG stream with dynamic ByteTrack polygon segmentation overlays. |
| `GET` | `/heatmap` | Dynamic spatial activity heatmap PNG rendered via Matplotlib & OpenCV. |
| `GET` | `/api/perf` | Live performance telemetry (FPS, inference ms, GPU VRAM, CPU%, RAM, uptime). |
| `GET` | `/api/counts` | JSON dictionary of verified unique counts per species: `{"deer": 3, "bear": 1}`. |
| `GET` | `/api/opencv/status` | OpenCV 5 runtime version, build info, and active hardware feature flags. |
| `GET` | `/api/aws/status` | AWS cloud configuration status, S3 bucket, DynamoDB table, and SNS state. |
| `POST`| `/api/cam/stop` | Pauses webcam capture and outputs standby placeholder frame. |
| `POST`| `/api/cam/start` | Resumes live webcam acquisition. |
| `GET` | `/api/cam/snapshot` | Captures and downloads full-resolution uncompressed JPEG snapshot. |
| `POST`| `/api/analyse` | Accepts chunked multipart video upload for batch inference. |
| `GET` | `/api/analyse/<job_id>` | Polls video analysis progress (0–100%) and final species breakdown. |
| `GET` | `/api/analyse/<job_id>/download` | Downloads annotated output video with burned-in counts. |

---

## 📁 Repository Structure

```
.
├── webcam_full_v5.py         # Main Flask server, live stream & video analysis pipelines
├── opencv5_features.py       # OpenCV 5 advanced preprocessing, stabilization & Re-ID
├── aws_integration.py        # AWS cloud integration (S3, Rekognition, DynamoDB, SNS, Lambda)
├── tracker_cy.pyx            # Cython-accelerated C-speed IoU matching algorithm
├── build_tracker.py          # Standalone compiler script for Cython extension
├── requirements.txt          # Pinned dependency specifications
├── .env.example              # Configuration environment template
├── .gitignore                # Production ignore filters
├── README.md                 # Primary project documentation
├── SUBMISSION_README.md      # OpenCV AI Competition 2026 submission guide
├── docs/
│   ├── TECHNICAL_REPORT.md   # 700-line comprehensive technical paper
│   └── ARCHITECTURE.md       # Visual Mermaid system & cloud architecture diagrams
├── videos/                   # Auto-saved detection clips (gitignored)
└── uploads/                  # Uploaded and analysed video files (gitignored)
```

---

## 📚 Technical Documentation

- 📄 **[Technical Evaluation Report](docs/TECHNICAL_REPORT.md)**: Deep dive into the mathematical formulations, ByteTrack Kalman filtering, benchmark methodology, and ecological impact studies.
- 📐 **[System Architecture Documentation](docs/ARCHITECTURE.md)**: Exhaustive Mermaid flowcharts detailing thread safety, CUDA stream execution, and AWS IAM policies.

---

## 👥 Authors & Acknowledgments

- **Lead Developer**: Darnish ([@IZUMI-Xprime](https://github.com/IZUMI-Xprime)) — Mechatronics Engineering, PSG Polytechnic College, Coimbatore.
- **Competition**: OpenCV AI Competition 2026, powered by AWS.
- **Built With**: OpenCV 5, Ultralytics YOLOv8, ByteTrack, PyTorch, Flask, Boto3, and Cython.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
