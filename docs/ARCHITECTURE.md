# Wildlife Sentinel — Architecture Document

> Companion document to the [Technical Report](./TECHNICAL_REPORT.md).
> This document provides visual architectural diagrams using Mermaid,
> covering the full system from video ingestion to cloud backend.

---

## 1. System Components and Data Flow

The following diagram shows the complete Wildlife Sentinel system, from video
source ingestion through detection, tracking, counting, and output delivery to
both the web dashboard and AWS cloud backend.

```mermaid
flowchart TD
    subgraph Input["Video Input Layer"]
        WC["Webcam\n(cv2.VideoCapture)"]
        VF["Video File\n(.mp4 / .avi)"]
        RTSP["RTSP Stream\n(Network Camera)"]
    end

    subgraph Preprocessing["OpenCV 5 Advanced Preprocessing & Adaptation"]
        CC["Color Conversion\n(BGR → RGB / LAB)"]
        ENV_ADAPT["Environmental Adaptive Controller\n(Auto-tune CLAHE & MOG2 thresholds)"]
        CLAHE["Dynamic CLAHE Adaptive\nHistogram Equalization"]
        DENOISE["Fast Non-Local Means\nLow-Light Denoising"]
        BGS["Background Subtraction\n(MOG2 Motion Pre-Filter)"]
        OF["Optical Flow Stabilization\n(Lucas-Kanade + RANSAC)"]
    end

    subgraph Detection["YOLOv8 Detection Engine"]
        YOLO["YOLOv8x-seg\n(Instance Segmentation)"]
        NMS["Non-Maximum Suppression\n(IoU threshold = 0.5)"]
        FILTER["Class + Confidence\nFiltering"]
    end

    subgraph Tracking["ByteTrack Multi-Object Tracker"]
        ASSOC_HI["High-Confidence Association\n(conf ≥ 0.5)"]
        ASSOC_LO["Low-Confidence Recovery\n(0.1 ≤ conf < 0.5)"]
        IOU["Cython IoU Kernel\n(Optimized C Extension)"]
        HUNGARIAN["Hungarian Algorithm\n(Optimal Assignment)"]
        TRACKS["Active Track State\n(Kalman Filter Predictions)"]
    end

    subgraph Counting["Unique Individual Counter"]
        SEEN["seen_track_ids\n(Set per Species)"]
        COUNTS["Per-Species\nUnique Counts"]
        TOTAL["Total Unique\nIndividual Count"]
    end

    subgraph Output["Output Layer"]
        ANNOT["Annotated Frame\n(BBoxes + Masks + Labels)"]
        HEAT["Detection Heatmap\n(Spatial Activity Density)"]
        SPATIAL["Spatial Kinematics Engine\n(Velocity Vectors & Trajectory Trails)"]
        STATS["Statistics JSON\n(Counts + Confidence + FPS)"]
        TELEM["Enterprise Observability\n(/api/telemetry Health & Drops)"]
    end

    subgraph Dashboard["Flask Web Dashboard"]
        MJPEG["MJPEG Live Stream\n(/video_feed)"]
        REST["REST API\n(/api/stats, /api/heatmap)"]
        UI["Browser UI\n(Charts + Controls)"]
    end

    subgraph Cloud["AWS Cloud Backend"]
        S3["Amazon S3\n(Detection Images)"]
        REKOG["Amazon Rekognition\n(Species Verification)"]
        DYNAMO["Amazon DynamoDB\n(Event Logging)"]
        SNS["Amazon SNS\n(Real-Time Alerts)"]
    end

    WC --> CC
    VF --> CC
    RTSP --> CC

    CC --> CLAHE
    CLAHE --> GB
    GB --> BGS
    BGS --> OF
    OF --> YOLO

    YOLO --> NMS
    NMS --> FILTER
    FILTER --> ASSOC_HI

    ASSOC_HI --> IOU
    ASSOC_LO --> IOU
    IOU --> HUNGARIAN
    HUNGARIAN --> TRACKS
    TRACKS --> ASSOC_HI
    TRACKS --> ASSOC_LO
    FILTER --> ASSOC_LO

    TRACKS --> SEEN
    SEEN --> COUNTS
    COUNTS --> TOTAL

    TRACKS --> ANNOT
    TRACKS --> HEAT
    COUNTS --> STATS

    ANNOT --> MJPEG
    STATS --> REST
    HEAT --> REST
    MJPEG --> UI
    REST --> UI

    ANNOT --> S3
    COUNTS --> DYNAMO
    S3 --> REKOG
    REKOG --> DYNAMO
    DYNAMO --> SNS
```

---

## 2. OpenCV Processing Pipeline Stages

This diagram details the sequential OpenCV operations applied to each incoming
frame before it reaches the YOLO detection engine. Each stage is annotated with
the specific OpenCV API calls used.

```mermaid
flowchart LR
    subgraph Capture["Stage 1: Capture"]
        CAP["cv2.VideoCapture.read()\nRaw BGR frame\n1280×720×3"]
    end

    subgraph ColorConv["Stage 2: Color"]
        BGR2LAB["cv2.cvtColor()\nBGR → LAB\nfor CLAHE"]
        SPLIT["cv2.split()\nExtract L channel"]
    end

    subgraph Enhance["Stage 3: Enhance"]
        CLAHE_OP["cv2.createCLAHE()\nclipLimit=2.0\ntileGrid=8×8"]
        MERGE["cv2.merge()\nRecombine LAB"]
        LAB2BGR["cv2.cvtColor()\nLAB → BGR"]
    end

    subgraph Denoise["Stage 4: Denoise"]
        GAUSS["cv2.GaussianBlur()\nkernel=5×5, σ=0"]
    end

    subgraph MotionDet["Stage 5: Motion (Optional)"]
        MOG2["cv2.createBackground\nSubtractorMOG2()"]
        FG["Foreground mask\n(skip frame if empty)"]
    end

    subgraph Stabilize["Stage 6: Stabilize (Optional)"]
        GFTT["cv2.goodFeaturesToTrack()"]
        LK["cv2.calcOpticalFlowPyrLK()"]
        WARP["cv2.warpAffine()\n(Compensate camera shake)"]
    end

    subgraph Convert["Stage 7: Final Convert"]
        BGR2RGB["cv2.cvtColor()\nBGR → RGB\n(for YOLO input)"]
    end

    CAP --> BGR2LAB
    BGR2LAB --> SPLIT
    SPLIT --> CLAHE_OP
    CLAHE_OP --> MERGE
    MERGE --> LAB2BGR
    LAB2BGR --> GAUSS
    GAUSS --> MOG2
    MOG2 --> FG
    FG --> GFTT
    GFTT --> LK
    LK --> WARP
    WARP --> BGR2RGB
```

### Pipeline Timing Breakdown

```mermaid
xychart-beta
    title "OpenCV Preprocessing Latency per Stage (ms)"
    x-axis ["Capture", "Color Conv", "CLAHE", "Gauss Blur", "BG Sub", "Opt Flow", "Final Conv"]
    y-axis "Latency (ms)" 0 --> 2.5
    bar [0.5, 0.3, 0.8, 0.2, 0.6, 1.5, 0.2]
```

---

## 3. AWS Cloud Integration Flow

This diagram illustrates the data flow between Wildlife Sentinel's local
processing engine and the AWS cloud backend services. It shows decision points,
async operations, and the alert pipeline.

```mermaid
flowchart TD
    DET["New Detection\n(species, confidence,\nbbox, crop image)"]

    DET --> CONF_CHECK{"Confidence\n≥ 0.6?"}

    CONF_CHECK -->|"Yes (high confidence)"| S3_UP["Upload crop to S3\ns3://bucket/detections/\nYYYY/MM/DD/uuid.jpg"]
    CONF_CHECK -->|"No (uncertain)"| REKOG_CHECK{"Confidence\n≥ 0.4?"}

    REKOG_CHECK -->|"Yes (uncertain zone)"| REKOG["Send crop to\nAmazon Rekognition\nDetectLabels API"]
    REKOG_CHECK -->|"No (below threshold)"| DISCARD["Discard detection\n(too low confidence)"]

    REKOG --> REKOG_RESULT{"Rekognition\nagrees with\nYOLO label?"}
    REKOG_RESULT -->|"Yes"| S3_UP
    REKOG_RESULT -->|"No"| FLAG["Flag for\nhuman review"]
    FLAG --> S3_UP

    S3_UP --> DDB_LOG["Log to DynamoDB\nWildlifeSentinelEvents\n(event_id, timestamp,\nspecies, confidence,\ntrack_id, s3_key)"]

    DDB_LOG --> ALERT_CHECK{"Matches\nalert rule?"}

    ALERT_CHECK -->|"Rare species\ndetected"| SNS_PUB["Publish to\nAmazon SNS topic"]
    ALERT_CHECK -->|"Activity spike\n> N in T min"| SNS_PUB
    ALERT_CHECK -->|"No rule match"| DONE["Processing\ncomplete"]

    SNS_PUB --> EMAIL["Email\nNotification"]
    SNS_PUB --> SMS["SMS\nAlert"]
    SNS_PUB --> SLACK["Slack\nWebhook"]
    SNS_PUB --> LAMBDA_HOOK["Lambda\n(Custom Action)"]

    style DISCARD fill:#f96,stroke:#333
    style FLAG fill:#ff9,stroke:#333
    style SNS_PUB fill:#9cf,stroke:#333
    style DONE fill:#9f9,stroke:#333
```

### Cloud Cost Estimation

```mermaid
xychart-beta
    title "Estimated Monthly AWS Cost by Detection Volume"
    x-axis ["1K dets", "5K dets", "10K dets", "50K dets", "100K dets"]
    y-axis "Cost (USD)" 0 --> 50
    bar [0.5, 2.1, 4.3, 18.5, 35.2]
```

> **Cost drivers**: S3 storage ($0.023/GB), Rekognition DetectLabels ($0.001/image,
> only uncertain detections), DynamoDB on-demand ($1.25/M writes), SNS ($0.50/100K
> notifications). Costs assume ~10% of detections are in the uncertain zone.

---

## 4. Deployment Topology Options

Wildlife Sentinel supports three deployment topologies, from fully local to
fully cloud-managed. The following diagrams illustrate each option.

### 4.1 Topology A — Local-Only Deployment

All processing runs on a single machine. No cloud dependencies. Suitable for
field stations with limited or no internet connectivity.

```mermaid
flowchart LR
    subgraph LocalMachine["Local Machine (GPU Workstation)"]
        CAM["Camera\n(USB/File)"]
        CV["OpenCV 5\nPreprocessing"]
        YOLO["YOLOv8x-seg\n(GPU Inference)"]
        BT["ByteTrack\nTracker"]
        UC["Unique\nCounter"]
        FLASK["Flask\nDashboard"]
        DB["SQLite\n(Local Log)"]
    end

    BROWSER["User Browser\n(LAN Access)"]

    CAM --> CV --> YOLO --> BT --> UC --> FLASK
    UC --> DB
    FLASK --> BROWSER
```

**Characteristics:**
| Property | Value |
|---|---|
| Internet required | No |
| Data persistence | Local SQLite + filesystem |
| Alerting | None (or local email via SMTP) |
| Scalability | Single camera, single machine |
| Setup complexity | Low |

### 4.2 Topology B — Hybrid Cloud Deployment (Recommended)

Local GPU handles real-time inference; AWS provides persistent storage, secondary
verification, and alerting. The recommended deployment for connected field
stations.

```mermaid
flowchart LR
    subgraph Edge["Edge / Local (GPU Workstation)"]
        CAM["Camera\n(USB/File/RTSP)"]
        CV["OpenCV 5"]
        YOLO["YOLOv8x-seg"]
        BT["ByteTrack"]
        UC["Counter"]
        FLASK["Flask\nDashboard"]
    end

    subgraph AWS["AWS Cloud"]
        S3["S3\n(Image Store)"]
        REKOG["Rekognition\n(Verification)"]
        DDB["DynamoDB\n(Event Log)"]
        SNS_T["SNS\n(Alerts)"]
    end

    BROWSER["User Browser\n(LAN / WAN)"]

    CAM --> CV --> YOLO --> BT --> UC --> FLASK
    FLASK --> BROWSER

    UC -->|"Detection crops"| S3
    S3 -->|"Uncertain dets"| REKOG
    UC -->|"Event records"| DDB
    DDB -->|"Alert rules"| SNS_T
    REKOG -->|"Verification"| DDB
```

**Characteristics:**
| Property | Value |
|---|---|
| Internet required | Yes (for cloud sync; degrades gracefully if offline) |
| Data persistence | S3 + DynamoDB (durable, queryable) |
| Alerting | SNS (email, SMS, Slack) |
| Scalability | Multiple cameras on one GPU, cloud scales independently |
| Setup complexity | Medium |

### 4.3 Topology C — Full Cloud / Serverless Deployment

All processing runs on AWS. Video frames are uploaded from lightweight edge
devices (Raspberry Pi, phone) and processed by Lambda functions. Suitable for
batch processing or mobile-uploaded images.

```mermaid
flowchart LR
    subgraph EdgeDevice["Edge Device (Lightweight)"]
        CAM_LITE["Camera\n(Pi / Phone)"]
        UPLOAD["Frame Upload\n(HTTPS)"]
    end

    subgraph AWS_Full["AWS Cloud"]
        APIGW["API Gateway\n(HTTPS Endpoint)"]
        LAMBDA["Lambda\n(OpenCV + YOLO\nin Docker)"]
        S3_F["S3\n(Frames + Results)"]
        REKOG_F["Rekognition"]
        DDB_F["DynamoDB"]
        SNS_F["SNS"]
        CLOUDFRONT["CloudFront\n(Dashboard CDN)"]
        STATIC["S3 Static Site\n(Dashboard UI)"]
    end

    USER["User Browser\n(Anywhere)"]

    CAM_LITE --> UPLOAD --> APIGW --> LAMBDA
    LAMBDA --> S3_F
    LAMBDA --> DDB_F
    S3_F --> REKOG_F --> DDB_F
    DDB_F --> SNS_F
    STATIC --> CLOUDFRONT --> USER
    LAMBDA -->|"Results JSON"| STATIC
```

**Characteristics:**
| Property | Value |
|---|---|
| Internet required | Yes (always) |
| Data persistence | S3 + DynamoDB |
| Alerting | SNS (email, SMS, Slack) |
| Scalability | Auto-scaling via Lambda concurrency |
| Setup complexity | High (Docker images, IAM, API Gateway) |
| Real-time processing | No (batch / near-real-time, ~2–5 sec latency per image) |
| GPU | Not available in Lambda; uses CPU YOLO or Rekognition only |

---

## 5. Deployment Topology Comparison

```mermaid
flowchart TD
    START{"Choose Deployment\nTopology"}

    START -->|"No internet\navailable"| A["Topology A\nLocal-Only"]
    START -->|"Internet available\nReal-time needed"| B["Topology B\nHybrid Cloud\n(Recommended)"]
    START -->|"No local GPU\nBatch processing OK"| C["Topology C\nFull Cloud"]

    A --> A_DESC["✅ Zero latency\n✅ Fully offline\n❌ No alerts\n❌ No backup"]
    B --> B_DESC["✅ Real-time + cloud\n✅ Alerts & logging\n✅ Offline fallback\n❌ Needs GPU locally"]
    C --> C_DESC["✅ No local hardware\n✅ Auto-scaling\n❌ Higher latency\n❌ Higher cost"]
```

---

## 6. Data Model

```mermaid
erDiagram
    CAMERA {
        string camera_id PK
        string location
        string stream_url
        string status
    }

    DETECTION_EVENT {
        string event_id PK
        string timestamp SK
        string species
        float confidence
        int track_id
        string bbox_json
        string s3_key
        string camera_id FK
        boolean is_verified
    }

    TRACK {
        int track_id PK
        string species
        string first_seen
        string last_seen
        int frame_count
        string appearance_descriptor
    }

    ALERT_RULE {
        string rule_id PK
        string species_filter
        float confidence_threshold
        int spike_count
        int spike_window_sec
        string sns_topic_arn
    }

    CAMERA ||--o{ DETECTION_EVENT : "produces"
    DETECTION_EVENT }o--|| TRACK : "belongs to"
    ALERT_RULE ||--o{ DETECTION_EVENT : "triggers on"
```

---

## 7. Thread and Process Architecture

```mermaid
flowchart TD
    subgraph MainProcess["Main Process (Python)"]
        subgraph MainThread["Main Thread"]
            CAPTURE["Frame Capture Loop\n(cv2.VideoCapture)"]
            PREPROCESS["OpenCV Preprocessing"]
            INFERENCE["YOLO Inference\n(GPU — blocks on result)"]
            TRACK_UPDATE["ByteTrack Update\n(Cython IoU)"]
            COUNT_UPDATE["Counter Update\n(seen_track_ids)"]
            ANNOTATE["Frame Annotation\n(OpenCV drawing)"]
        end

        subgraph FlaskThread["Flask Thread (daemon)"]
            HTTP["HTTP Server\n(port 5000)"]
            MJPEG_EP["/video_feed\n(MJPEG generator)"]
            API_EP["/api/stats\n(JSON endpoint)"]
            HEAT_EP["/api/heatmap\n(PNG endpoint)"]
        end

        subgraph CloudThread["Cloud Sync Thread (daemon)"]
            QUEUE["Detection Queue\n(thread-safe)"]
            S3_WORKER["S3 Upload Worker"]
            DDB_WORKER["DynamoDB Write Worker"]
            REKOG_WORKER["Rekognition Worker"]
        end

        CAPTURE --> PREPROCESS --> INFERENCE --> TRACK_UPDATE --> COUNT_UPDATE --> ANNOTATE
        ANNOTATE -->|"Latest frame\n(shared buffer)"| MJPEG_EP
        COUNT_UPDATE -->|"Stats dict\n(thread-safe)"| API_EP
        COUNT_UPDATE -->|"Detection event"| QUEUE
        QUEUE --> S3_WORKER
        QUEUE --> DDB_WORKER
        S3_WORKER --> REKOG_WORKER
    end

    GPU["NVIDIA GPU\n(CUDA Context)"]
    INFERENCE -.->|"cudaMemcpy +\nkernel launch"| GPU
```

---

## 8. Environmental Adaptation, Spatial Kinematics & Observability Architecture

This section details the international-grade OpenCV 5 enhancements implemented for edge-case resilience, behavioral tracking, and enterprise-grade telemetry.

### 8.1 Closed-Loop Environmental Adaptive Controller

```mermaid
flowchart TD
    RAW["Raw BGR Frame"] --> RESIZE["Fast Sub-sample Preview
(160×120, INTER_NEAREST)"]
    RESIZE --> LUM["Mean Luminance Y
& Contrast Std Dev"]
    RESIZE --> CHROMA["Cross-Channel Dispersion
(|R-G| + |G-B|)"]
    RESIZE --> LAP["Laplacian Variance
Var(Laplacian) Blur Index"]

    LUM & CHROMA & LAP --> PROFILE_ENGINE{"Environmental Scene Classifier"}

    PROFILE_ENGINE -->|"Chroma < 4.0 & Y < 90"| IR_MODE["INFRARED NIGHT
- CLAHE clip: 4.2
- MOG2 varThreshold: 36.0
- NL-Means Denoise: ON"]
    PROFILE_ENGINE -->|"Y < 45"| LOW_MODE["LOW LIGHT
- CLAHE clip: 3.8
- MOG2 varThreshold: 28.0
- NL-Means Denoise: ON"]
    PROFILE_ENGINE -->|"Y > 215 or Glare > 0.12"| GLARE_MODE["HARSH GLARE
- CLAHE clip: 1.2
- MOG2 varThreshold: 20.0
- Desaturate / Highlight suppress"]
    PROFILE_ENGINE -->|"Var(Lap) < 35 & Std < 30"| RAIN_MODE["FOG / PRECIPITATION
- CLAHE clip: 3.0
- MOG2 varThreshold: 24.0
- Edge-preserve smooth"]
    PROFILE_ENGINE -->|"Y < 5 or (Std < 3.5 & Var < 2)"| OBS_MODE["LENS OBSTRUCTED / BLACKOUT
- CLAHE clip: 1.0
- MOG2 varThreshold: 50.0
- Flag Fault Alert"]
    PROFILE_ENGINE -->|"Nominal range"| DAY_MODE["DAYLIGHT SAVANNAH
- CLAHE clip: 2.0
- MOG2 varThreshold: 16.0
- Fast bypass"]

    IR_MODE & LOW_MODE & GLARE_MODE & RAIN_MODE & OBS_MODE & DAY_MODE --> CALIB["Dynamic Parameter Application
(Live MOG2 & CLAHE auto-tune without history reset)"]
```

### 8.2 Spatial Kinematics & Ego-Motion Optical Flow Engine

```mermaid
flowchart LR
    subgraph InputData["Tracker & Video Stream"]
        BBOX["Target Bounding Box
(x1, y1, x2, y2)"]
        HOMO["Camera Homography Matrix H
(from Video Stabilizer)"]
        CROPS["Current & Previous
Target Image Crops"]
    end

    subgraph MotionAnalysis["SpatialMotionAnalyzer"]
        CENTROID["Centroid Moving Average
(k-frame window, dt)"]
        OPTFLOW["cv2.calcOpticalFlowFarneback
(Dense crop surface flow)"]
        COMPENSATE["Camera Motion Subtraction
v_true = v_measured - v_camera"]
        CLASSIFY["Locomotion Classifier
- Stationary (<25 px/s)
- Browsing (25-85 px/s)
- Walking (85-260 px/s)
- Running (>260 px/s)"]
    end

    subgraph OutputViz["Rendering & Telemetry"]
        TRAILS["Fading Polyline Motion Trails"]
        ARROWS["Heading Azimuth Direction Arrows"]
        LABELS["Compass Pill (e.g., 'NW 142px/s [running]')"]
    end

    BBOX --> CENTROID
    CROPS --> OPTFLOW
    CENTROID & HOMO --> COMPENSATE
    COMPENSATE --> CLASSIFY
    CLASSIFY --> TRAILS & ARROWS & LABELS
```

### 8.3 Enterprise Health & Observability Telemetry Pipeline

```mermaid
flowchart TD
    FRAME["Incoming Video Frame"] --> T0["Start Timer t0"]
    
    subgraph StageTimers["Latency Decomposition"]
        T_PREP["OpenCV 5 Preprocessing (ms)"]
        T_INFER["YOLOv8 Inference (ms)"]
        T_TRACK["ByteTrack Cython IoU (ms)"]
        T_SPATIAL["Spatial Kinematics & Flow (ms)"]
        T_ENC["MJPEG Encode (ms)"]
    end

    subgraph TelemetryCollector["TelemetryTracker (Rolling 60-frame Window)"]
        STATS["Compute Mean, p50, p95, p99 Latencies & Jitter"]
        DROPS["Frame Drop Estimator against 25 FPS Budget"]
        SYS_RES["psutil Resource Profiler (CPU %, RAM RSS/VMS)"]
        GPU_RES["CUDA Allocator Profiler (VRAM Alloc/Reserved)"]
        SLA["SLA Health Evaluator
- HEALTHY: Drop < 5%
- DEGRADED: Drop < 25%
- CRITICAL: Drop >= 25%"]
    end

    subgraph API_Endpoints["Enterprise Cloud Telemetry Endpoints"]
        EP1["GET /api/telemetry
(Full JSON snapshot)"]
        EP2["GET /api/perf
(High-frequency dashboard polling)"]
    end

    T0 --> T_PREP --> T_INFER --> T_TRACK --> T_SPATIAL --> T_ENC
    StageTimers --> STATS & DROPS & SYS_RES & GPU_RES
    STATS & DROPS & SYS_RES & GPU_RES --> SLA
    SLA --> EP1 & EP2
```


---

## 9. Cloud-Native Agentic Vision & AWS Graviton Architecture

### 9.1 Physics-Informed Active Perception Pipeline

```mermaid
flowchart TD
    subgraph StreamInput["Video Stream Acquisition"]
        CAM["Camera Trap / RTSP Stream"] --> FRAME["Raw BGR Frame"]
    end

    subgraph StateEstimation["Continuous-Discrete Kalman Filter (CD-KF)"]
        FRAME --> YOLOV8["YOLOv8-seg Detection"]
        YOLOV8 --> CD_KF["RobustKinematicTracker (DWNA Model)"]
        CD_KF --> K_STATE["KinematicState:
- Position & Covariance P_k
- Schmitt-Trigger is_moving Flag
- Speed & Azimuth Heading"]
    end

    subgraph ActivePerception["Physics-Informed Active Perception Manager"]
        K_STATE --> EVAL{"Are All Entities
Stationary > 5.0s?"}
        EVAL -- "Yes" --> THROTTLE["ComputeMode.THROTTLED_1FPS
- Throttles Decoding & Inference to 1 FPS
- Activates OpenCV MOG2 Motion Pre-Filter
- Saves >70% EC2 Compute Budget"]
        EVAL -- "No (Speed > 12 px/s)" --> BURST["ComputeMode.BURST_30FPS
- Scales Instantly to 30 FPS Burst Mode
- High-Speed Pursuit Tracking"]
        THROTTLE --> MOG2_CHECK{"MOG2 Foreground
Change > 0.8%?"}
        MOG2_CHECK -- "Yes" --> BURST
        MOG2_CHECK -- "No" --> THROTTLE
    end

    BURST --> OUT["Active Stream Processing & Telemetry"]
```

### 9.2 Real-Time Spatial Digital Twin (AWS IoT Core & Timestream)

```mermaid
flowchart LR
    subgraph Edge["AWS EC2 Graviton3 Edge"]
        K_DATA["Continuous Kinematic State
(Track ID, Pos, Vel, Heading, Covariance)"]
        STREAMER["AWSIoTStreamer
(Async Queue Buffer)"]
        K_DATA --> STREAMER
    end

    subgraph CloudIoT["AWS IoT Core Broker"]
        TOPIC["MQTT Topic:
'wildlife/kinematics/camera_1'
(TLS 1.3 / SigV4)"]
        STREAMER -->|"Publish Telemetry"| TOPIC
        RULE["IoT Topic SQL Rule:
SELECT * FROM 'wildlife/kinematics/+'"]
        TOPIC --> RULE
    end

    subgraph CloudStorage["Amazon Timestream Time-Series DB"]
        HOT["Memory Store (24 Hours)
Sub-second Real-Time Ingest"]
        COLD["Magnetic Store (365 Days)
Historical Migration Analytics"]
        RULE --> HOT
        HOT -->|"Auto-Tiering"| COLD
    end

    subgraph Dashboard["Amazon Managed Grafana"]
        GRAFANA["Multi-Agent Trajectory HUD:
- 2D Ground Plane XY Plot
- Real-time Velocity Spikes
- Anomaly Count Alarms"]
        HOT -->|"Direct Query"| GRAFANA
    end
```

### 9.3 AWS Bedrock (Claude 3.5 Sonnet) MCP Threat Orchestrator

```mermaid
sequenceDiagram
    autonumber
    participant KIN as RobustKinematicTracker
    participant AGENT as AgenticThreatOrchestrator
    participant BEDROCK as AWS Bedrock (Claude 3.5 Sonnet)
    participant OPENCV as OpenCV 5 CLAHE Tool
    participant PTZ as Camera PTZ Gimbal
    participant SNS as Amazon SNS Topic

    KIN->>AGENT: Anomaly Trigger (Speed > 45 px/s, Panic Flight)
    AGENT->>BEDROCK: Converse API with MCP Tool Specifications
    Note over BEDROCK: Claude analyzes kinematic context<br/>Decides tool execution order
    BEDROCK->>AGENT: Tool Use: crop_and_enhance_event(bbox, frame)
    AGENT->>OPENCV: Apply CLAHE contrast & edge-preserving filter
    OPENCV-->>AGENT: High-res forensic image saved to disk
    AGENT-->>BEDROCK: Tool Result: Evidence metadata & sharpness
    
    BEDROCK->>AGENT: Tool Use: steer_ptz_camera(pred_x, pred_y)
    AGENT->>PTZ: Kalman forward prediction actuation (Pan/Tilt slew)
    PTZ-->>AGENT: PTZ Gimbal aligned with target trajectory
    AGENT-->>BEDROCK: Tool Result: Actuation status
    
    BEDROCK->>AGENT: Tool Use: send_threat_alert_sns(subject, message)
    AGENT->>SNS: Publish high-priority alert to wardens
    SNS-->>AGENT: Alert broadcasted (SMS/Email)
    AGENT-->>BEDROCK: Tool Result: MessageId confirmed
    
    BEDROCK-->>AGENT: Final Incident Evaluation Summary
```

---
---

*Wildlife Sentinel — Architecture Document*
*Last updated: October 2026*
