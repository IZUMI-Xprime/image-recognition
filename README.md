# 🦅 Wildlife Sentinel

> Real-time wildlife detection, unique individual counting, and video analysis — powered by YOLOv8 segmentation, ByteTrack, Cython, and GPU acceleration. Runs entirely locally with a custom browser dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-green?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Key Features](#key-features)
3. [How It Works — Architecture](#how-it-works--architecture)
4. [Tech Stack](#tech-stack)
5. [System Requirements](#system-requirements)
6. [Installation](#installation)
   - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
   - [Step 2 — Install Python Dependencies](#step-2--install-python-dependencies)
   - [Step 3 — Install CUDA PyTorch (GPU)](#step-3--install-cuda-pytorch-gpu)
   - [Step 4 — Build the Cython Extension](#step-4--build-the-cython-extension)
   - [Step 5 — Configure Environment Variables](#step-5--configure-environment-variables)
7. [Running the App](#running-the-app)
8. [Dashboard Pages](#dashboard-pages)
9. [API Reference](#api-reference)
10. [Environment Variables Reference](#environment-variables-reference)
11. [Deep Dive — Unique Individual Counting](#deep-dive--unique-individual-counting)
12. [Deep Dive — Cython Acceleration](#deep-dive--cython-acceleration)
13. [Deep Dive — GPU Batch Inference](#deep-dive--gpu-batch-inference)
14. [Deep Dive — Segmentation Masks](#deep-dive--segmentation-masks)
15. [Deep Dive — Chunked Video Upload](#deep-dive--chunked-video-upload)
16. [Performance Benchmarks](#performance-benchmarks)
17. [Project Structure](#project-structure)
18. [Troubleshooting](#troubleshooting)
19. [Contributing](#contributing)
20. [License](#license)

---

## What It Does

Wildlife Sentinel is a full-stack computer vision system that:

- **Streams your webcam live** with real-time animal/object detection drawn as segmentation polygon outlines (not rectangles)
- **Counts unique individuals** — a deer standing in frame for 10 minutes counts as **1**, not 18,000
- **Analyses uploaded videos** of any length (tested on 2-hour files) and produces a species breakdown report with an annotated output video
- **Generates heatmaps** showing where in the frame detections occurred, using exact mask shapes not bounding boxes
- **Sends Telegram + Email alerts** when animals are detected, with recorded clips attached
- **Shows live performance metrics** — FPS, inference latency, GPU memory, CPU %, RAM, uptime
- **Runs 100% locally** — no cloud API, no subscription, no data leaves your machine

---

## Key Features

### 🎯 Detection
- **YOLOv8n-seg** — nano segmentation model, fast and accurate for wildlife/animals/people
- **Polygon outlines** instead of bounding boxes — traces the exact body shape of each subject
- **Semi-transparent fill** per detection, unique colour per tracked individual
- **Confidence threshold** tunable via environment variable

### 🔢 Unique Counting (No Double-Counting)
- **ByteTrack** assigns a persistent integer ID to each individual across frames
- A **`seen_track_ids` set** ensures each ID is counted exactly once — forever
- Same animal re-entering frame keeps its ID — not re-counted
- Track buffer of 90 frames (~10 seconds) before a track is retired
- Each colour on screen = one individual — visually verifiable

### ⚡ Performance
- **CUDA GPU acceleration** — auto-detected, inference runs on GPU if available
- **Batch inference** — 8 frames per GPU call for ~3× throughput on video analysis
- **Cython C extension** — IoU matching loop compiled to pure C, sub-0.5ms per frame
- **Separate YOLO model instances** — live stream and video analysis never share a lock, run truly in parallel
- **JPEG quality 70** — optimal bandwidth/quality for maximum FPS

### 📹 Video Analysis
- Upload any `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` file — no size limit
- **Streaming upload** — writes to disk in 4 MB chunks, uses <4 MB RAM regardless of file size
- **Progress bar** survives page reload — job ID saved in `localStorage`, auto-resumes
- Species results table with unique counts and distribution bar chart
- Download annotated output video with masks and unique counts burned in

### 🗺️ Heatmap
- Built from segmentation polygon masks — not rectangles
- Gaussian-blurred for smooth heat gradient visualization
- Overlaid directly on the live camera frame for spatial context

### 🔔 Alerts
- **Telegram bot** — sends clip video to your chat when animal detected
- **Email (Gmail SMTP)** — sends species list with one-minute cooldown to prevent spam
- Both run in a background thread pool — never block the live stream

### 🛡️ Security
- Passwords stored and compared as **SHA-256 hashes**
- **Constant-time comparison** (`hmac.compare_digest`) prevents timing attacks
- `secure_filename()` on all uploads — prevents path traversal
- All error responses return **JSON**, never HTML — prevents frontend parse errors
- Session cookies: `HttpOnly`, `SameSite=Lax`
- All secrets via environment variables — nothing hardcoded

---

## How It Works — Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask App (threaded)                      │
│                                                                   │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────┐  │
│  │  Live Stream     │   │  Video Analysis  │   │  I/O Pool   │  │
│  │  Thread          │   │  ThreadPool (1)  │   │  ThreadPool │  │
│  │                  │   │                  │   │  (2)        │  │
│  │  model_stream    │   │  model_analysis  │   │  Telegram   │  │
│  │  (yolov8n-seg)   │   │  (yolov8n-seg)   │   │  Email      │  │
│  │  + ByteTrack     │   │  + batch GPU     │   │             │  │
│  │  + seen_ids set  │   │  + IoU tracker   │   └─────────────┘  │
│  │  + Cython IoU    │   │  + Cython IoU    │                    │
│  └──────────────────┘   └──────────────────┘                    │
│         │                        │                               │
│    _stream_lock             _analysis_lock                       │
│    (independent)            (independent)                        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         │
    MJPEG stream → Browser (Chrome/Firefox)
    JSON APIs    → Dashboard JS (polled every 1–2s)
```

**Four concurrent concerns run simultaneously:**
1. **Live MJPEG stream** — continuous generator loop, yields JPEG frames
2. **Video analysis worker** — GPU batch inference in dedicated ThreadPoolExecutor
3. **I/O pool** — Telegram uploads + email in separate pool (never blocks stream)
4. **Flask request handler** — threaded mode for concurrent API calls

**Two completely separate YOLO model instances** are loaded at startup — one for the live stream, one for video analysis. This is critical: a single shared model with a shared lock caused the analysis thread to deadlock waiting for the stream thread to release it. Two models = two independent CUDA contexts = truly parallel.

---

## Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Detection | YOLOv8n-seg (Ultralytics) | Instance segmentation |
| Tracking | ByteTrack (built into Ultralytics) | Persistent individual IDs |
| GPU | PyTorch CUDA 12.4 | GPU inference acceleration |
| Web framework | Flask 3.x | HTTP server + MJPEG streaming |
| Computer vision | OpenCV (cv2) | Frame processing, polygon drawing |
| Hot loop | Cython + GCC | C-speed IoU matching |
| Alerts | python-telegram-bot | Telegram clip delivery |
| Heatmap | Matplotlib + NumPy | Detection density visualization |
| Metrics | psutil | CPU/RAM monitoring |
| Security | hmac, hashlib, werkzeug | Auth + path safety |

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, Ubuntu 20.04, macOS 12 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.12 |
| RAM | 4 GB | 8 GB+ |
| GPU | None (CPU fallback works) | NVIDIA GTX 1060+ (CUDA 11.8+) |
| Webcam | Any USB/built-in webcam | 1080p USB webcam |
| Disk | 500 MB (models + deps) | 2 GB+ (for video storage) |
| C compiler | GCC (MinGW on Windows) | GCC 12+ |

---

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/wildlife-sentinel.git
cd wildlife-sentinel
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, Ultralytics (YOLOv8), OpenCV, NumPy, Seaborn, Matplotlib, python-telegram-bot, werkzeug, psutil, and Cython.

> **Note:** The first time you run the app, `yolov8n-seg.pt` (~6 MB) will be auto-downloaded by Ultralytics.

### Step 3 — Install CUDA PyTorch (GPU)

**Check your CUDA version first:**
```bash
nvidia-smi
```
Look at the top-right corner for the CUDA version number.

**Then install the matching PyTorch build:**

```bash
# CUDA 12.4 / 12.x (most modern laptops)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (no GPU)
pip install torch torchvision torchaudio
```

**Verify GPU is detected:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected output:
```
True
NVIDIA GeForce RTX XXXX
```

### Step 4 — Build the Cython Extension

The Cython extension compiles the IoU matching loop to pure C for maximum speed. This step is optional — the app falls back to pure Python if the extension is not built.

**Windows — Install MinGW first (skip if already have GCC):**

Option A — Chocolatey (run PowerShell as Administrator):
```powershell
choco install mingw
```

Option B — Manual download:
1. Go to [winlibs.com](https://winlibs.com)
2. Download **UCRT, x86_64, latest release** `.zip`
3. Extract to `C:\mingw64`
4. Add `C:\mingw64\bin` to your PATH environment variable

Verify GCC is available:
```bash
gcc --version
```

**Build the extension:**
```bash
python build_tracker.py
```

Expected output:
```
Found gcc at: C:\mingw64\bin\gcc.exe
[1/3] Cythonizing tracker_cy.pyx → tracker_cy.c
[2/3] Compiling tracker_cy.c
[3/3] Linking → tracker_cy.cp312-win_amd64.pyd
✓ Built successfully: tracker_cy.cp312-win_amd64.pyd  (XX KB)
  tracker_cy loaded OK — Cython IoU active
```

**Linux / macOS:**
```bash
# Install GCC if not present
sudo apt install gcc   # Ubuntu/Debian
brew install gcc       # macOS

python build_tracker.py
```

> If the build fails, the app still works — it just uses pure-Python IoU which is slightly slower. You will see this in the startup log: `tracker_cy not found — using pure-Python IoU`.

### Step 5 — Configure Environment Variables

Copy the example file:
```bash
cp .env.example .env
```

Edit `.env` with your values. At minimum set:
```
SECRET_KEY=any_long_random_string_here
ADMIN_PASSWORD=your_chosen_password
```

See the full [Environment Variables Reference](#environment-variables-reference) below.

**Windows — Set variables permanently:**

Search for "Edit the system environment variables" → Environment Variables → New (under User variables), add each key/value pair. Close and reopen your terminal after.

**Or create a launcher batch file** (`run.bat`):
```bat
@echo off
set SECRET_KEY=your_secret_key_here
set ADMIN_PASSWORD=your_password_here
set TELEGRAM_TOKEN=your_bot_token
set TELEGRAM_CHAT_ID=your_chat_id
set EMAIL_SENDER=your@gmail.com
set EMAIL_RECEIVER=alerts@gmail.com
set EMAIL_PASSWORD=your_app_password
python webcam_full_v5.py
pause
```

**Linux / macOS:**
```bash
export SECRET_KEY="your_secret_key_here"
export ADMIN_PASSWORD="your_password_here"
python webcam_full_v5.py
```

---

## Running the App

```bash
python webcam_full_v5.py
```

**Expected startup log:**
```
13:42:01 [INFO] GPU detected: NVIDIA GeForce RTX 3050  (batch=8)
13:42:01 [INFO] Cython tracker loaded — fast IoU active.
13:42:03 [INFO] Camera connected.
 * Running on http://0.0.0.0:5000
```

Open your browser and go to:
```
http://localhost:5000
```

Login with:
- **Username:** `admin` (or whatever `ADMIN_USERNAME` is set to)
- **Password:** whatever you set for `ADMIN_PASSWORD` (default: `password123`)

> **To access from another device on your network:** use your computer's local IP instead of `localhost`, e.g. `http://192.168.1.x:5000`

---

## Dashboard Pages

### `/dashboard` — Live Feed

The main dashboard with:

| Section | Description |
|---|---|
| Stats row | Total detections, unique classes, clips saved, camera status |
| Performance row | FPS, inference ms, GPU memory, CPU%, RAM, uptime, total frames, tracker mode |
| Live feed | MJPEG stream with ByteTrack polygon overlays, unique colour per individual |
| Detections panel | Real-time count per species, updated every 2 seconds |
| Heatmap panel | Detection density map overlaid on camera frame |
| Controls | ⏹ Stop / ▶ Start camera, 📷 Snapshot, Heatmap refresh, Clear Counts |

### `/analyse` — Video Analysis

Upload any video file for species analysis:

1. Drag and drop or click to browse (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
2. Click **Run Analysis**
3. Watch the progress bar — shows upload speed, MB/s, ETA, then inference progress
4. Results show: total unique individuals, species count, frames analysed, device used
5. Species breakdown table with unique counts and distribution bars
6. Download the annotated output video

> **Page reload safe:** If you reload the page mid-analysis, it automatically resumes showing the correct progress. The job ID is saved in `localStorage`.

### `/recordings` — Saved Clips

Lists all `.mp4` clips recorded from the live stream when animals/objects were detected. Shows filename, file size, and download link.

---

## API Reference

All API endpoints require authentication (active session cookie). Unauthenticated requests return `401 JSON`.

### Pages

| Route | Method | Description |
|---|---|---|
| `GET /` | GET | Redirects to `/login` |
| `GET /login` | GET, POST | Login page |
| `GET /logout` | GET | Clears session, redirects to login |
| `GET /dashboard` | GET | Main dashboard |
| `GET /analyse` | GET | Video analysis page |
| `GET /recordings` | GET | Recordings list |
| `GET /download/<filename>` | GET | Download a saved clip |

### Stream

| Route | Method | Description |
|---|---|---|
| `GET /video_feed` | GET | MJPEG live stream (multipart/x-mixed-replace) |
| `GET /heatmap` | GET | Current heatmap PNG (captures one frame, runs inference) |

### Camera Control

| Route | Method | Description |
|---|---|---|
| `POST /api/cam/stop` | POST | Pause the live stream (shows static placeholder) |
| `POST /api/cam/start` | POST | Resume the live stream |
| `GET /api/cam/status` | GET | `{"active": true/false}` |
| `GET /api/cam/snapshot` | GET | Single JPEG snapshot download (full quality) |

### Detection Data

| Route | Method | Description |
|---|---|---|
| `GET /api/counts` | GET | `{"person": 3, "dog": 1, ...}` — unique individual counts |
| `GET /api/clip_count` | GET | `{"count": 5}` — number of saved clips |
| `POST /api/clear_counts` | POST | Reset all counts and seen track IDs |

### Performance

| Route | Method | Description |
|---|---|---|
| `GET /api/perf` | GET | `{"fps": 9.2, "infer_ms": 108.3, "gpu_mem_mb": 412.0, "cpu_pct": 22.1, "ram_mb": 380, "uptime_s": 3600, "uptime_fmt": "01:00:00", "frame_count": 33120, "device": "cuda:0", "gpu_name": "RTX 3050", "cython": true}` |

### Video Analysis

| Route | Method | Description |
|---|---|---|
| `POST /api/analyse` | POST | Upload video file (`multipart/form-data`, field name `video`). Returns `{"job_id": "abc123"}` with status 202 |
| `GET /api/analyse/<job_id>` | GET | Poll job status. Returns `{"status": "queued/processing/done/error", "progress": 0-100, "species": [...], "total_unique": N, "frames_analysed": N, "device": "cuda:0"}` |
| `GET /api/analyse/<job_id>/download` | GET | Download annotated output video |

**Job status values:**

| Status | Meaning |
|---|---|
| `queued` | Uploaded, waiting for analysis worker |
| `processing` | Inference running, `progress` updates 0→100 |
| `done` | Complete, `species` array populated |
| `error` | Failed, `error` field contains message |

---

## Environment Variables Reference

Copy `.env.example` to `.env` and fill in your values.

### Required

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | Random (changes every restart) | Flask session signing key. Set a fixed value to keep sessions alive across restarts |
| `ADMIN_USERNAME` | `admin` | Login username |
| `ADMIN_PASSWORD` | `password123` | Login password. **Change this.** |

### Optional — Alerts

| Variable | Default | Description |
|---|---|---|
| `EMAIL_SENDER` | — | Gmail address to send alerts from |
| `EMAIL_RECEIVER` | — | Email address to receive alerts |
| `EMAIL_PASSWORD` | — | Gmail **App Password** (not your login password). Generate at myaccount.google.com → Security → App passwords |
| `TELEGRAM_TOKEN` | — | Bot token from @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | — | Your Telegram chat ID. Get it from @userinfobot |
| `EMAIL_COOLDOWN` | `60` | Minimum seconds between email alerts |

### Optional — Storage

| Variable | Default | Description |
|---|---|---|
| `VIDEO_DIR` | `videos` | Where live stream clips are saved |
| `UPLOAD_DIR` | `uploads` | Where uploaded and analysed videos are saved |
| `CLIP_DURATION` | `60` | Length of each recorded clip in seconds |

### Optional — Detection Tuning

| Variable | Default | Description |
|---|---|---|
| `CONF_THRESHOLD` | `0.40` | Minimum YOLO confidence to count a detection (0.0–1.0). Lower = more detections, more false positives |
| `IOU_THRESHOLD` | `0.35` | Minimum IoU overlap to match a detection to an existing track |
| `TRACK_BUFFER` | `90` | Frames a ByteTrack track survives without a match before retiring. At 9fps, 90 = ~10 seconds |
| `MAX_ABSENT_FRAMES` | `90` | Same as TRACK_BUFFER, controls the Python-side IoU tracker |

### Optional — Performance

| Variable | Default | Description |
|---|---|---|
| `BATCH_SIZE` | `8` (GPU) / `1` (CPU) | Frames per GPU inference batch for video analysis. Increase on high-VRAM GPUs |
| `FRAME_SKIP` | `3` | Video analysis: analyse every Nth frame. Higher = faster but less accurate |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode (never use in production) |

---

## Deep Dive — Unique Individual Counting

### The Problem

Naive detection counts every detection on every frame:
```
Frame 1: 1 person detected → count = 1
Frame 2: same person → count = 2
Frame 3: same person → count = 3
...
Frame 900: same person → count = 900
```

One person standing still for 30 seconds at 30fps = count of 900. Completely useless.

### The Solution: ByteTrack + seen_track_ids

**Step 1 — ByteTrack assigns persistent IDs**

YOLOv8's built-in ByteTrack tracker uses a Kalman filter + IoU matching pipeline:
1. First detection → new track ID assigned (e.g., `#3`)
2. Next frame — prediction of where `#3` will be
3. New detection overlaps prediction → matched to `#3` (same individual)
4. No new ID created, no new count

**Step 2 — seen_track_ids ensures each ID counted exactly once**

```python
seen_track_ids = set()   # global, lives for the session

# Per detection, per frame:
track_id = int(boxes.id[i])
if track_id not in seen_track_ids:
    seen_track_ids.add(track_id)
    animal_counts[label] += 1   # counted exactly ONCE, ever
# else: same individual seen again — do nothing
```

**Result by scenario:**

| Scenario | Behaviour |
|---|---|
| Person A enters frame | ID `#3` assigned → count = 1 |
| Person A stands still for 10 min | ID `#3` matched every frame → count stays 1 |
| Person A leaves frame | Track `#3` goes absent, retired after 10s |
| Person B enters frame | New ID `#7` → count = 2 |
| Person A comes back quickly (< 10s) | ByteTrack re-matches `#3` → count stays 2 |

**`seen_track_ids` is cleared in two situations:**
- User clicks **Clear Counts** — fresh session
- A clip recording finalizes — each clip has its own independent count

---

## Deep Dive — Cython Acceleration

The IoU matching loop — comparing every detection box against every active track — runs on every frame. At 9fps with 10 active tracks and 5 detections, that's 9 × 10 × 5 = 450 IoU calculations per second in Python's interpreter. In Cython compiled C, that's essentially free.

### What is Compiled

`tracker_cy.pyx` compiles the following to pure C:

```cython
# cython: boundscheck=False    — no Python array bounds checking
# cython: wraparound=False     — no negative-index handling
# cython: cdivision=True       — C division, no Python zero-division guard
# cython: nonecheck=False      — no None checks on typed variables

cdef struct BBox:              # stack-allocated — zero heap/GC pressure
    int x1, y1, x2, y2

cdef inline float _iou_c(BBox a, BBox b) nogil:   # inlined at call site, GIL released
    # Pure C arithmetic, no Python objects
```

### Compiler Flags

```
-O3           maximum optimisation level
-march=native use all CPU instruction sets available on this machine (AVX2, etc.)
-ffast-math   allow FP reassociation (safe for IoU)
```

### Graceful Fallback

```python
try:
    import tracker_cy as _cy
    _match_fn = _cy.match_detections_to_tracks
    log.info("Cython tracker loaded — fast IoU active.")
except ImportError:
    _match_fn = _match_py   # pure Python, identical results, slightly slower
```

No code change needed — just build the extension and restart.

---

## Deep Dive — GPU Batch Inference

### Live Stream

Each frame is sent to YOLO individually via `model.track()` with `persist=True` — ByteTrack needs this to maintain tracklet state between calls.

### Video Analysis

Frames are accumulated into batches before GPU inference:

```python
BATCH_SIZE = 8   # on GPU; 1 on CPU

# Accumulate frames
batch_frames.append(frame)
if len(batch_frames) >= BATCH_SIZE:
    _flush_batch()   # send all 8 at once

# In _flush_batch():
results = model_analysis(batch_frames, conf=CONF_THRESHOLD, device=DEVICE)
```

Sending 8 frames at once vs 1 frame 8 times gives ~3× throughput because:
- GPU kernel launch overhead amortized over 8 frames
- Better utilization of CUDA parallelism
- Single Python→C→CUDA transition per batch

### Two Independent Models

```python
model_stream   = YOLO("yolov8n-seg.pt")   # live stream only
model_analysis = YOLO("yolov8n-seg.pt")   # video analysis only

_stream_lock   = Lock()    # only serializes model_stream
_analysis_lock = Lock()    # only serializes model_analysis
```

The live stream holds `_stream_lock` continuously. With a single shared model, the analysis thread would wait forever. Two models = zero contention.

---

## Deep Dive — Segmentation Masks

YOLOv8n-seg outputs per-detection polygon masks as `result.masks.xy[i]` — a NumPy array of (N, 2) float pixel coordinates tracing the object outline.

```python
def _draw_seg(frame, mask_xy, colour, label, alpha=0.35):
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))

    # 1. Semi-transparent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 2. Crisp outline
    cv2.polylines(frame, [pts], isClosed=True, color=colour,
                  thickness=2, lineType=cv2.LINE_AA)

    # 3. Label at topmost polygon point (floats above the subject's head)
    top_pt = tuple(pts[pts[:, 0, 1].argmin()][0])
    cv2.putText(frame, label, (top_pt[0], top_pt[1] - 8), ...)
```

Each individual gets a **deterministic unique colour** from their track ID:
```python
def _track_colour(track_id):
    rng = (track_id * 2654435761) & 0xFFFFFF   # Knuth multiplicative hash
    r = (rng >> 16) & 0xFF
    g = (rng >> 8)  & 0xFF
    b =  rng        & 0xFF
    return (b, g, r)   # OpenCV BGR order
```

Same ID → same colour, always. If two shapes on screen share a colour, they're the same individual.

---

## Deep Dive — Chunked Video Upload

Standard Flask file upload buffers the entire file in RAM before processing. A 2-hour video (4–8 GB) would exhaust memory and crash.

Instead, the upload route streams to disk in 4 MB chunks:

```python
CHUNK_SIZE = 4 * 1024 * 1024   # 4 MB

with open(save_path, "wb") as out:
    while True:
        chunk = f.stream.read(CHUNK_SIZE)
        if not chunk:
            break
        out.write(chunk)
```

`MAX_CONTENT_LENGTH` is intentionally **not set** in Flask config — no size limit.

The browser uses `XMLHttpRequest` with `upload.progress` to show:
- Bytes transferred / total
- Upload speed (MB/s)
- ETA in seconds

After upload, the file is verified with `cv2.VideoCapture` before queuing — rejects non-video files disguised with a video extension.

---

## Performance Benchmarks

*Tested on: Intel i5-12th Gen + NVIDIA RTX 3050 (4GB VRAM), 1080p webcam, Windows 11*

| Metric | GPU Mode | CPU Mode |
|---|---|---|
| Live stream FPS | 8–12 fps | 3–5 fps |
| Inference latency | 80–120 ms/frame | 250–400 ms/frame |
| Video analysis throughput | ~9 fps effective | ~3 fps effective |
| RAM usage (live stream) | ~380 MB | ~280 MB |
| GPU memory | ~450 MB | — |
| Upload RAM (any file size) | < 4 MB | < 4 MB |
| Tracker overhead (Cython) | < 0.5 ms/frame | < 0.5 ms/frame |
| Tracker overhead (Python) | 2–5 ms/frame | 2–5 ms/frame |

> FPS depends heavily on webcam resolution and GPU model. YOLOv8n-seg is significantly slower than YOLOv8n (detection only) because segmentation masks add compute. If FPS is too low, set `CONF_THRESHOLD=0.6` to reduce false positives, or switch the model to `yolov8n.pt` (detection only, no polygon masks).

---

## Project Structure

```
wildlife-sentinel/
│
├── webcam_full_v5.py        # Main application — Flask app, all routes, all logic
│                            # All HTML templates embedded as strings (no templates/ folder)
│
├── tracker_cy.pyx           # Cython source — IoU matching loop in typed C
│                            # Compiled to .pyd (Windows) or .so (Linux/macOS)
│
├── build_tracker.py         # Cython build script — calls Cython + GCC directly
│                            # Bypasses distutils/setuptools/MSVC entirely
│
├── requirements.txt         # Python dependencies
│
├── .env.example             # Environment variable template (safe to commit)
├── .env                     # Your actual secrets — NEVER commit this
│
├── .gitignore               # Excludes .env, .pyd, videos/, uploads/, etc.
│
├── README.md                # This file
│
├── videos/                  # Live stream clips (auto-created, gitignored)
│   └── clip_20250101_120000.mp4
│
├── uploads/                 # Uploaded + analysed videos (auto-created, gitignored)
│   ├── upload_abc123.mkv    # Raw uploaded file (deleted after analysis)
│   └── analysed_abc123.mp4  # Annotated output
│
└── bytetrack_sentinel.yaml  # Auto-generated ByteTrack config (gitignored)
```

**Key design decision:** The entire app is a single Python file with no external template files. HTML pages are embedded as `render_template_string()` strings. This makes the project trivially easy to deploy — copy one file and run it.

---

## Troubleshooting

### App starts but camera shows black screen
- Check if another app (Zoom, Teams, OBS) is using the webcam
- Try setting `VIDEO_DIR` to a different camera index: edit `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `webcam_full_v5.py`
- On Linux: add your user to the `video` group: `sudo usermod -aG video $USER`

### GPU not detected (shows "No GPU — running on CPU")
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# If False, reinstall PyTorch with correct CUDA version — see Step 3
```

### Cython build fails on Windows: "gcc not found"
```powershell
# Install MinGW via Chocolatey (run as Administrator)
choco install mingw
# Then close and reopen terminal, verify:
gcc --version
```

### Analysis stuck at 0% / "Queued"
- This was fixed in v5 by using separate model instances. If you see this, ensure you're running `webcam_full_v5.py`, not an older version.

### Re-entering frame counted as new individual
- Increase `TRACK_BUFFER` (default 90 = ~10 seconds at 9fps):
  ```
  set TRACK_BUFFER=150   # ~16 seconds
  ```
- Restart the app after changing this value

### Telegram alerts not sending
1. Verify `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are set correctly
2. Start a conversation with your bot first — bots can't initiate messages
3. Check terminal for error messages from `_telegram_async`

### Email alerts not sending
1. Use a **Gmail App Password**, not your Gmail login password
2. Enable 2-Step Verification on your Google account first
3. Generate App Password: myaccount.google.com → Security → App passwords

### Heatmap shows "Camera unavailable"
- The heatmap captures a fresh frame from the live stream
- If the camera is stopped (you clicked ⏹ Stop), start it first with ▶ Start, then refresh the heatmap

### Analysis page doesn't resume after reload
- The `localStorage` key `sentinel_job_id` stores the active job ID
- If the server was restarted while a job was running, the job is lost (in-memory registry cleared)
- Re-upload the video to start a new analysis

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first.

**Areas that would benefit from contributions:**
- Re-ID model integration (OSNet/FastReID) for appearance-based matching across long absences
- RTSP/IP camera support as alternative to USB webcam
- PostgreSQL/SQLite persistence for job registry (survives server restarts)
- Docker compose file for easy deployment
- Support for multiple simultaneous camera streams

**Code style:** The project uses no formatter — just follow the existing style. All functions are documented with docstrings. Security-sensitive code should be accompanied by a comment explaining why.

---

## License

MIT License — see `LICENSE` for details.

---

*Built by Darnish — Mechatronics student at PSG Polytechnic College, Coimbatore.*  
*Part of an ongoing project in wildlife monitoring and sustainable agriculture technology.*
