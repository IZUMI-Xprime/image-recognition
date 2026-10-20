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

    subgraph Preprocessing["OpenCV 5 Preprocessing Pipeline"]
        CC["Color Conversion\n(BGR → RGB / LAB)"]
        CLAHE["CLAHE Adaptive\nHistogram Equalization"]
        GB["Gaussian Blur\n(Noise Reduction)"]
        BGS["Background Subtraction\n(MOG2 — optional)"]
        OF["Optical Flow\nStabilization (optional)"]
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
        STATS["Statistics JSON\n(Counts + Confidence + FPS)"]
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

*Wildlife Sentinel — Architecture Document*
*Last updated: October 2026*
