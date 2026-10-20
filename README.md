# 🦅 Wildlife Sentinel

### Cloud-Native, Physics-Informed & Agentic Wildlife Surveillance
**Official Entry for the OpenCV AI Competition 2026 (Powered by AWS)**  
*Targeting the **"Agentic Vision"** and **"Best Use of COOL"** Special Prizes*

[![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0.0-008000?style=for-the-badge&logo=opencv&logoColor=white)](./opencv5_features.py)
[![AWS Graviton](https://img.shields.io/badge/AWS-Graviton3%20%7C%20Arm64-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](./benchmark_cool.py)
[![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-Claude%203.5%20Sonnet-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](./agentic_orchestrator.py)
[![AWS IoT Core](https://img.shields.io/badge/AWS%20IoT-Digital%20Twin-blue?style=for-the-badge&logo=internet-of-things&logoColor=white)](./aws_iot_twin.py)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-blueviolet?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Cython](https://img.shields.io/badge/Cython-C--Speed%20IoU-F9DC3E?style=for-the-badge&logo=cython&logoColor=black)](./tracker_cy.pyx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)

---

## 📖 Executive Summary

**Wildlife Sentinel** is a cloud-native, agentic computer vision platform engineered for long-term ecological monitoring, anti-poaching operations, and wildlife population audits on **AWS EC2 Graviton (Arm64)** instances.

By unifying **OpenCV 5 with Cloud-Optimized OpenCV Library (COOL)**, **YOLOv8 Instance Segmentation**, **ByteTrack**, and a custom **Continuous-Discrete Kalman Filter (`RobustKinematicTracker`)**, Wildlife Sentinel:
1. **Solves the Double-Counting Dilemma**: Tracks unique individuals persistently across multi-minute field encounters.
2. **Eliminates Centroid Kinematic Noise**: Replaces naive differencing with morphological covariance inflation, Mahalanobis outlier gating, and $SO(2)$ circular heading smoothing.
3. **Optimizes Cloud Energy via Active Perception**: Dynamically scales inference from 1 FPS throttled idle to 30 FPS burst, cutting cloud compute costs by **>70%**.
4. **Deploys an Autonomous AI Agent (Bedrock + MCP)**: Claude 3.5 Sonnet autonomously detects kinematic anomalies, crops and enhances forensic images with OpenCV CLAHE, steers PTZ cameras based on Kalman forward predictions, and broadcasts emergency alerts via Amazon SNS.

---

## 🏆 Special Prize Focus (OpenCV AI Competition 2026)

| Target Special Prize | Key Innovation & Architecture | Core Implementation |
|---|---|---|
| **🏆 Agentic Vision Prize** | Autonomous threat evaluation loop using **AWS Bedrock (Claude 3.5 Sonnet)** via the **Model Context Protocol (MCP)**. When pursuit velocities exceed $45\,\text{px/s}$, the agent orchestrates OpenCV CLAHE enhancement, Kalman forward-predicted PTZ camera steering, and AWS SNS warden alerting. | [`agentic_orchestrator.py`](./agentic_orchestrator.py) |
| **🏆 Best Use of COOL Prize** | Full hardware acceleration on **AWS EC2 Graviton (Arm64)** utilizing the **Cloud-Optimized OpenCV Library (COOL)** with Arm NEON vectorization and KleidiCV primitives for MOG2, spatial image moments, and covariance inversion, yielding a **1.34× throughput speedup and 28.4% cost reduction**. | [`benchmark_cool.py`](./benchmark_cool.py) |
| **🌐 Real-Time Spatial Digital Twin** | High-throughput asynchronous MQTT telemetry streaming to **AWS IoT Core**, ingesting continuous-time Kalman states into **Amazon Timestream** and rendering multi-agent trajectories in **Amazon Managed Grafana**. | [`aws_iot_twin.py`](./aws_iot_twin.py) |
| **⚡ Physics-Informed Active Perception** | Continuous monitoring of Kalman covariance $\mathbf{P}_k$ and Schmitt-trigger `is_moving` states. Automatically throttles video decoding to 1 FPS when animals are stationary, instantly bursting to 30 FPS upon motion. | [`main_pipeline.py`](./main_pipeline.py) |

---

## 🏗️ Cloud & Edge System Architecture

```
                                  [Edge Camera Trap / Wildlife Video]
                                                  │
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │   AWS EC2 Graviton (Arm64) Instance (c7g)       │
                         │   OpenCV 5 with COOL (NEON / KleidiCV) SIMD     │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │      Physics-Informed Active Perception          │
                        │    1 FPS Throttled Idle ◄──► 30 FPS Burst Mode   │
                        └─────────┬──────────────────────────────┬─────────┘
                                  │                              │
                                  ▼                              ▼
                    ┌───────────────────────────┐   ┌──────────────────────────────┐
                    │  YOLOv8 Instance Masks    │   │  RobustKinematicTracker      │
                    │      (yolov8n-seg)        │   │  Continuous-Discrete Kalman  │
                    └─────────────┬─────────────┘   └──────────────┬───────────────┘
                                  │                                │
                                  └────────────────┬───────────────┘
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         │                                                  │
                         ▼                                                  ▼
      ┌─────────────────────────────────────┐            ┌──────────────────────────────────────┐
      │   AWS IoT Core & Timestream Twin    │            │   Agentic Vision Threat Orchestrator │
      │   MQTT Telemetry Streaming:         │            │   AWS Bedrock (Claude 3.5 Sonnet)    │
      │   • Position (x, y) & Covariance    │            │   • Model Context Protocol (MCP)     │
      │   • Velocity, Speed & Heading       │            │   • OpenCV CLAHE Forensic Crop       │
      │   • Amazon Managed Grafana HUD      │            │   • Kalman Forward PTZ Steering      │
      └─────────────────────────────────────┘            │   • Amazon SNS Emergency Dispatch    │
                                                         └──────────────────────────────────────┘
```

---

## 🚀 Step-by-Step AWS Deployment Guide

### 1. Launch AWS EC2 Graviton3 Instance (Arm64)

```bash
# Launch c7g.xlarge instance running Amazon Linux 2023 or Ubuntu 24.04 (aarch64)
aws ec2 run-instances \
    --image-id ami-0c2b8ca1dad447f8a \
    --instance-type c7g.xlarge \
    --key-name wildlife-sentinel-key \
    --security-group-ids sg-0123456789abcdef0 \
    --subnet-id subnet-0123456789abcdef0 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WildlifeSentinel-Graviton}]'
```

### 2. Install Dependencies & Build OpenCV 5 with COOL (Arm NEON)

```bash
# Connect to your Graviton instance
ssh -i wildlife-sentinel-key.pem ec2-user@<GRAVITON_PUBLIC_IP>

# Install build essentials and Python 3.11
sudo dnf update -y
sudo dnf groupinstall "Development Tools" -y
sudo dnf install -y python3.11 python3.11-devel cmake git libpng-devel libjpeg-turbo-devel

# Clone Wildlife Sentinel repository
git clone https://github.com/IZUMI-Xprime/wildlife-sentinel-opencv-ai-2026.git
cd wildlife-sentinel-opencv-ai-2026

# Install Python requirements
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install awsiotsdk boto3 psutil
```

### 3. Setup AWS IAM Policy & Roles

Attach the following policy to your EC2 instance profile to enable Bedrock, IoT Core, Timestream, and SNS access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAgentInvocation",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:Converse"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-5-sonnet-*"
    },
    {
      "Sid": "IoTCorePublish",
      "Effect": "Allow",
      "Action": [
        "iot:Publish",
        "iot:Connect"
      ],
      "Resource": "*"
    },
    {
      "Sid": "TimestreamIngest",
      "Effect": "Allow",
      "Action": [
        "timestream:WriteRecords",
        "timestream:DescribeEndpoints"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SNSAlertPublish",
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:*:*:wildlife-alerts"
    }
  ]
}
```

### 4. Run the COOL Benchmarking Suite

Verify the Arm64 NEON acceleration and latency on your Graviton instance:

```bash
python benchmark_cool.py --iters 100 --out benchmark_cool_report.json
```

Output:
```
| OpenCV 5 Kernel / Operation          | Mean (ms) | p50 (ms) | Throughput (FPS) | Cost / 1M Frames |
| :---                                 | :---:     | :---:    | :---:            | :---:            |
| 1. MOG2 Motion Subtraction           | 7.43 ms   | 7.36 ms  | 134.4 FPS        | $0.3510          |
| 2. Spatial Moments (M00, M10, M01)   | 0.001 ms  | 0.001 ms | 324,254 FPS      | $0.0001          |
| 3. Covariance Inversion S_k^(-1)     | 0.001 ms  | 0.001 ms | 396,196 FPS      | $0.0001          |
| 4. CLAHE Contrast Equalization       | 0.22 ms   | 0.19 ms  | 4,471 FPS        | $0.0105          |
| 5. Dense Optical Flow (Farneback)    | 3.26 ms   | 3.22 ms  | 306.2 FPS        | $0.1541          |
| 6. End-to-End Preprocessing Pipeline | 3.88 ms   | 3.89 ms  | 257.0 FPS        | $0.1835          |
```

### 5. Run the Integrated Active Perception Pipeline

```bash
# Set your environment parameters
export AWS_REGION="us-east-1"
export AWS_IOT_ENDPOINT="xxxxxxxxxxxxxx-ats.iot.us-east-1.amazonaws.com"
export AWS_SNS_TOPIC_ARN="arn:aws:sns:us-east-1:123456789012:wildlife-alerts"

# Execute active perception with live IoT twin and Bedrock agent
python main_pipeline.py --source 0 --camera-id camera_01 --model yolov8n-seg.pt
```

---

## 📁 Repository Structure

```
.
├── main_pipeline.py          # Feature 1: Physics-informed active perception loop (1 vs 30 FPS)
├── benchmark_cool.py         # Feature 2: COOL vs x86 Graviton benchmark suite & cost estimator
├── aws_iot_twin.py           # Feature 3: AWS IoT Core streamer, Timestream DDL & Grafana JSON
├── agentic_orchestrator.py   # Feature 4: Bedrock Claude 3.5 Sonnet + MCP OpenCV/PTZ agent
├── webcam_full_v5.py         # Live web dashboard & interactive surveillance feed
├── opencv5_features.py       # CD-KF RobustKinematicTracker & OpenCV 5 preprocessing
├── aws_integration.py        # Base AWS S3, Rekognition, DynamoDB, and SNS integrations
├── tracker_cy.pyx            # Cython-accelerated C-speed IoU matching kernel
├── build_tracker.py          # In-place Cython compiler
├── evaluate.py               # Comprehensive MOTA/MOTP/IDF1 tracking evaluation harness
├── requirements.txt          # Pinned multi-platform dependencies
├── docs/
│   ├── TECHNICAL_REPORT.md   # Academic technical report with mathematical formulations
│   └── ARCHITECTURE.md       # Exhaustive Mermaid system & cloud architecture diagrams
└── videos/                   # Recorded detection clips and forensic anomaly crops
```

---

## 👥 Authors & Acknowledgments

- **Lead Developer & Architect**: Darnish ([@IZUMI-Xprime](https://github.com/IZUMI-Xprime)) — Mechatronics Engineering, PSG Polytechnic College, Coimbatore.
- **Competition**: OpenCV AI Competition 2026, powered by AWS.
- **Technologies**: OpenCV 5, AWS Graviton3, AWS Bedrock (Claude 3.5 Sonnet), AWS IoT Core, Amazon Timestream, Ultralytics YOLOv8, ByteTrack.

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
