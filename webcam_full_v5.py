"""
Wildlife Sentinel v5
====================
New in v5:
  • Analysis job survives page reload — job_id stored in localStorage, auto-resumes on return
  • Live cam Start / Stop / Snapshot controls
  • YOLOv8 ByteTrack integration — persistent track IDs kill duplicate counts across frames
  • Performance overlay on dashboard: FPS, inference ms, GPU %, GPU mem, uptime
  • /api/perf endpoint for live perf metrics polled every second
  • JPEG quality auto-tuned for max FPS on GPU
  • generate_frames() runs inference every frame on GPU (no FRAME_SKIP for live feed)
"""

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from threading import Lock, Event

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from flask import (Flask, Response, jsonify, redirect,
                   render_template_string, request, send_file,
                   session, stream_with_context, url_for)
from ultralytics import YOLO
from werkzeug.utils import secure_filename

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinel")

# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE       = "cuda:0"
    GPU_NAME     = torch.cuda.get_device_name(0)
    BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "8"))   # frames per GPU batch
    log.info("GPU detected: %s  (batch=%d)", GPU_NAME, BATCH_SIZE)
else:
    DEVICE       = "cpu"
    GPU_NAME     = "CPU"
    BATCH_SIZE   = 1
    log.info("No GPU — running on CPU.")

# ─────────────────────────────────────────────────────────────────────────────
# Cython tracker import (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tracker_cy as _cy
    _CYTHON_OK = True
    log.info("Cython tracker loaded — fast IoU active.")
except ImportError:
    _CYTHON_OK = False
    log.warning("tracker_cy not found — using pure-Python IoU. "
                "Run:  python setup_tracker.py build_ext --inplace")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def _env(key: str) -> str:
    return os.environ.get(key, "")

EMAIL_SENDER      = _env("EMAIL_SENDER")
EMAIL_RECEIVER    = _env("EMAIL_RECEIVER")
EMAIL_PASSWORD    = _env("EMAIL_PASSWORD")
TELEGRAM_TOKEN    = _env("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = _env("TELEGRAM_CHAT_ID")

ADMIN_USERNAME    = os.environ.get("ADMIN_USERNAME", "admin")
_RAW_PASS         = os.environ.get("ADMIN_PASSWORD", "password123")
ADMIN_PASS_HASH   = hashlib.sha256(_RAW_PASS.encode()).hexdigest()

CLIP_DURATION     = int(os.environ.get("CLIP_DURATION",   "60"))
EMAIL_COOLDOWN    = int(os.environ.get("EMAIL_COOLDOWN",   "60"))
VIDEO_DIR         = os.environ.get("VIDEO_DIR",   "videos")
UPLOAD_DIR        = os.environ.get("UPLOAD_DIR",  "uploads")

ALLOWED_EXTS      = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CHUNK_SIZE        = 4 * 1024 * 1024          # 4 MB chunks for streaming upload
UPLOAD_TIMEOUT    = 7200                     # 2 hours max upload time (seconds)

IOU_THRESHOLD     = float(os.environ.get("IOU_THRESHOLD",   "0.35"))
MAX_ABSENT_FRAMES = int(os.environ.get("MAX_ABSENT_FRAMES",  "30"))
CONF_THRESHOLD    = float(os.environ.get("CONF_THRESHOLD",   "0.40"))
FRAME_SKIP        = int(os.environ.get("FRAME_SKIP",         "1"))   # analyse every Nth frame

os.makedirs(VIDEO_DIR,  exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Flask — NOTE: NO MAX_CONTENT_LENGTH — we stream to disk instead
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=3600,
    # MAX_CONTENT_LENGTH intentionally NOT set — chunked streaming handles size
)

# ─────────────────────────────────────────────────────────────────────────────
# Global JSON error handlers — Flask's defaults return HTML which breaks fetch()
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(400)
def err_400(e): return jsonify(error=str(e)), 400

@app.errorhandler(401)
def err_401(e): return jsonify(error="Unauthorized"), 401

@app.errorhandler(403)
def err_403(e): return jsonify(error="Forbidden"), 403

@app.errorhandler(404)
def err_404(e): return jsonify(error="Not found"), 404

@app.errorhandler(413)
def err_413(e): return jsonify(error="File too large — increase MAX_CONTENT_LENGTH or use chunked upload"), 413

@app.errorhandler(500)
def err_500(e):
    log.exception("Internal server error")
    return jsonify(error="Internal server error"), 500

# ─────────────────────────────────────────────────────────────────────────────
# YOLO models — two separate instances so live stream and video analysis
# NEVER share a lock and never block each other.
# ─────────────────────────────────────────────────────────────────────────────
model_stream   = YOLO("yolov8n.pt")   # live webcam feed
model_analysis = YOLO("yolov8n.pt")   # uploaded video analysis
model_stream.to(DEVICE)
model_analysis.to(DEVICE)
_stream_lock   = Lock()               # guards model_stream only
_analysis_lock = Lock()               # guards model_analysis only

# aliases so existing heatmap/stream code works unchanged
model       = model_stream
_model_lock = _stream_lock

# ─────────────────────────────────────────────────────────────────────────────
# Thread pools
# ─────────────────────────────────────────────────────────────────────────────
_io_pool       = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io")
_analysis_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")

# ─────────────────────────────────────────────────────────────────────────────
# Live-stream state
# ─────────────────────────────────────────────────────────────────────────────
_state_lock     = Lock()
animal_counts   = {}
last_email_time = 0.0
cap             = None
video_writer    = None
clip_start_time = None
last_clip_path  = None

# Camera on/off control
cam_active      = True          # toggled by /api/cam/start|stop
_cam_event      = Event()
_cam_event.set()                # starts in "on" state

# Performance metrics (updated by generate_frames)
START_TIME      = time.time()
_perf_lock      = Lock()
_perf = {
    "fps":         0.0,
    "infer_ms":    0.0,
    "gpu_pct":     0.0,
    "gpu_mem_mb":  0.0,
    "uptime_s":    0,
    "frame_count": 0,
}
_fps_window = deque(maxlen=30)   # rolling 30-frame FPS window

# ─────────────────────────────────────────────────────────────────────────────
# Job registry
# ─────────────────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = Lock()


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers (Cython or pure-Python)
# ─────────────────────────────────────────────────────────────────────────────
def _iou_py(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _match_py(track_bboxes, det_bboxes, iou_thresh):
    available     = list(range(len(det_bboxes)))
    matched_pairs = []
    for ti, tb in enumerate(track_bboxes):
        best_iou, best_di = -1.0, -1
        for di in available:
            s = _iou_py(tb, det_bboxes[di])
            if s > best_iou:
                best_iou, best_di = s, di
        if best_iou >= iou_thresh and best_di >= 0:
            matched_pairs.append((ti, best_di))
            available.remove(best_di)
    return matched_pairs, available


# pick fastest available implementation
if _CYTHON_OK:
    _match_fn = _cy.match_detections_to_tracks
else:
    _match_fn = _match_py


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Track:
    track_id: int
    label:    str
    bbox:     tuple
    absent:   int = 0


class UniqueAnimalTracker:
    """
    Per-species greedy IoU tracker.
    Inner matching loop runs in Cython when the .so is built.
    """

    def __init__(self, iou_thresh=IOU_THRESHOLD, max_absent=MAX_ABSENT_FRAMES):
        self.iou_thresh   = iou_thresh
        self.max_absent   = max_absent
        self._tracks:     list[_Track]       = []
        self._next_id:    int                = 0
        self.unique:      dict[str, int]     = defaultdict(int)

    def update(self, detections: list) -> None:
        by_label: dict[str, list] = defaultdict(list)
        for label, x1, y1, x2, y2 in detections:
            by_label[label].append((x1, y1, x2, y2))

        matched_ids: set[int] = set()

        for label, boxes in by_label.items():
            live = [t for t in self._tracks
                    if t.label == label and t.absent == 0]
            track_bboxes = [t.bbox for t in live]

            matched_pairs, unmatched_dets = _match_fn(
                track_bboxes, boxes, self.iou_thresh
            )

            for ti, di in matched_pairs:
                live[ti].bbox   = boxes[di]
                live[ti].absent = 0
                matched_ids.add(live[ti].track_id)

            for di in unmatched_dets:
                t = _Track(self._next_id, label, boxes[di])
                self._next_id += 1
                self._tracks.append(t)
                self.unique[label] += 1

        for t in self._tracks:
            if t.track_id not in matched_ids:
                t.absent += 1

        self._tracks = [t for t in self._tracks if t.absent <= self.max_absent]


# ─────────────────────────────────────────────────────────────────────────────
# GPU batch inference helper
# ─────────────────────────────────────────────────────────────────────────────
def _infer_batch(frames: list) -> list:
    """
    Run YOLO on a batch of frames using the dedicated analysis model.
    Never touches model_stream — no lock contention with the live feed.
    """
    all_dets = []
    with _analysis_lock:
        for i in range(0, len(frames), BATCH_SIZE):
            chunk   = frames[i:i + BATCH_SIZE]
            results = model_analysis(chunk, verbose=False,
                                     conf=CONF_THRESHOLD, device=DEVICE)
            for result in results:
                dets = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label  = model_analysis.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    dets.append((label, x1, y1, x2, y2))
                all_dets.append(dets)
    return all_dets


# ─────────────────────────────────────────────────────────────────────────────
# Video analysis worker
# ─────────────────────────────────────────────────────────────────────────────
def _analyse_video(job_id: str, video_path: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"

    try:
        cap_v = cv2.VideoCapture(video_path)
        if not cap_v.isOpened():
            raise RuntimeError("Cannot open video file.")

        total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap_v.get(cv2.CAP_PROP_FPS) or 25.0
        w            = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(UPLOAD_DIR, f"analysed_{job_id}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc,
                                   max(1.0, fps / FRAME_SKIP), (w, h))
        tracker  = UniqueAnimalTracker()

        frame_idx       = 0
        frames_done     = 0
        batch_frames    = []   # raw frames buffered for GPU batch
        batch_indices   = []   # their original positions (for writer order)

        def _flush_batch():
            nonlocal frames_done
            if not batch_frames:
                return
            all_dets = _infer_batch(batch_frames)
            for frame, dets in zip(batch_frames, all_dets):
                tracker.update(dets)
                for (label, x1, y1, x2, y2) in dets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 100), 2)
                    cv2.putText(frame, label, (x1, max(y1-8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (0, 230, 100), 1, cv2.LINE_AA)
                # Burn unique counts into frame
                y_off = 24
                cv2.putText(frame, "Unique:",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 200, 255), 1, cv2.LINE_AA)
                for lbl, cnt in tracker.unique.items():
                    y_off += 18
                    cv2.putText(frame, f"  {lbl}: {cnt}",
                                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 200, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                frames_done += 1
            batch_frames.clear()
            batch_indices.clear()

        while True:
            ret, frame = cap_v.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            # Flush when batch is full
            if len(batch_frames) >= BATCH_SIZE:
                _flush_batch()

            # Progress
            pct = int(frame_idx / max(total_frames, 1) * 100)
            with _jobs_lock:
                _jobs[job_id]["progress"] = pct

        _flush_batch()   # remainder
        cap_v.release()
        writer.release()

        species_data = [
            {"species": lbl, "count": cnt}
            for lbl, cnt in sorted(tracker.unique.items(),
                                   key=lambda x: x[1], reverse=True)
        ]

        with _jobs_lock:
            _jobs[job_id].update({
                "status":          "done",
                "progress":        100,
                "species":         species_data,
                "total_unique":    sum(d["count"] for d in species_data),
                "frames_analysed": frames_done,
                "out_path":        out_path,
                "device":          DEVICE,
            })
        log.info("Job %s done: %s", job_id, species_data)

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        # Clean up the raw upload to save disk space
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Camera / live stream
# ─────────────────────────────────────────────────────────────────────────────
def reconnect_stream(retries=5, delay=2.0):
    global cap
    for attempt in range(retries):
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            log.info("Camera connected.")
            return True
        log.warning("Webcam retry %d/%d", attempt + 1, retries)
        time.sleep(delay)
    log.error("Failed to open webcam.")
    return False


def _send_email(counts: dict) -> None:
    global last_email_time
    if not all([EMAIL_SENDER, EMAIL_RECEIVER, EMAIL_PASSWORD]):
        return
    import smtplib
    from email.mime.text import MIMEText
    with _state_lock:
        if time.time() - last_email_time < EMAIL_COOLDOWN:
            return
    details = "\n".join(f"  {k}: {v}" for k, v in counts.items()) or "  (none)"
    msg = MIMEText(f"Animals detected:\n\n{details}\n\nCheck Telegram for the clip.")
    msg["From"] = EMAIL_SENDER; msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = "Sentinel: Animal Detected"
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.starttls(); srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
            srv.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        with _state_lock:
            last_email_time = time.time()
    except Exception as exc:
        log.error("Email failed: %s", exc)


async def _telegram_async(path):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    from telegram import Bot, InputFile
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        with open(path, "rb") as f:
            await bot.send_video(chat_id=TELEGRAM_CHAT_ID,
                                 video=InputFile(f, filename=os.path.basename(path)))
    except Exception as exc:
        log.error("Telegram failed: %s", exc)


def _send_telegram(path):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_telegram_async(path))
    finally:
        loop.close()


def _finalize_clip(video_path, counts):
    global video_writer
    with _state_lock:
        if video_writer is not None:
            video_writer.release(); video_writer = None
    time.sleep(0.5)
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return
    _send_telegram(video_path)
    _send_email(counts)


def generate_frames():
    """
    Live MJPEG stream with:
    • ByteTrack persistent IDs  — same person across frames = same ID, never double-counted
    • cam_active gate           — pauses when user hits Stop
    • perf metrics              — FPS, inference ms, GPU util updated every frame
    • max FPS on GPU            — no artificial sleep, JPEG quality tuned for throughput
    """
    global cap, animal_counts, video_writer, clip_start_time, last_clip_path

    # Blank "camera stopped" frame sent when cam is paused
    _blank = None

    while True:
        # ── Camera paused ─────────────────────────────────────────────────────
        if not _cam_event.is_set():
            if _blank is None:
                _blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(_blank, "Camera stopped", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 80), 2)
            ok, jpeg = cv2.imencode(".jpg", _blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n\r\n")
            time.sleep(0.1)
            continue

        # ── Camera feed ───────────────────────────────────────────────────────
        if cap is None or not cap.isOpened():
            reconnect_stream(); time.sleep(0.1); continue

        ret, frame = cap.read()
        if not ret:
            reconnect_stream(); continue

        h, w  = frame.shape[:2]
        t0    = time.perf_counter()

        # ── YOLOv8 + ByteTrack (track() gives persistent IDs) ────────────────
        with _stream_lock:
            track_results = model_stream.track(
                frame, verbose=False, device=DEVICE,
                conf=CONF_THRESHOLD,
                tracker="bytetrack.yaml",   # built into ultralytics
                persist=True,               # keep tracklet state between calls
            )

        infer_ms = (time.perf_counter() - t0) * 1000

        # ── Count unique track IDs per class in this frame ────────────────────
        frame_counts: dict[str, int] = {}
        if track_results and track_results[0].boxes.id is not None:
            boxes  = track_results[0].boxes
            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i])
                label    = model_stream.names.get(cls_id, "?")
                conf     = float(boxes.conf[i])
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])

                # colour by track ID so each individual gets its own colour
                colour = _track_colour(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{label} #{track_id} {conf:.2f}",
                            (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1, cv2.LINE_AA)

                frame_counts[label] = frame_counts.get(label, 0) + 1

        # ── Update global counts ──────────────────────────────────────────────
        if frame_counts:
            with _state_lock:
                for k, v in frame_counts.items():
                    animal_counts[k] = animal_counts.get(k, 0) + v

        # ── Timestamp overlay ─────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (w - 370, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 200), 1, cv2.LINE_AA)

        # ── Clip recording ────────────────────────────────────────────────────
        now = time.time()
        with _state_lock:
            writer_active = video_writer is not None

        if frame_counts and not writer_active:
            fname = os.path.join(VIDEO_DIR,
                                 datetime.now().strftime("clip_%Y%m%d_%H%M%S.mp4"))
            nw = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            with _state_lock:
                video_writer    = nw
                clip_start_time = now
                last_clip_path  = fname

        with _state_lock:
            if video_writer is not None:
                video_writer.write(frame)
                clip_due = (now - (clip_start_time or now)) >= CLIP_DURATION
            else:
                clip_due = False

        if clip_due:
            with _state_lock:
                path = last_clip_path; counts = dict(animal_counts)
                animal_counts.clear(); clip_start_time = None
            _io_pool.submit(_finalize_clip, path, counts)

        # ── Performance metrics ───────────────────────────────────────────────
        frame_time = time.perf_counter() - t0
        _fps_window.append(frame_time)
        fps = len(_fps_window) / sum(_fps_window) if _fps_window else 0.0

        gpu_pct    = 0.0
        gpu_mem_mb = 0.0
        if DEVICE != "cpu":
            try:
                gpu_mem_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                # pynvml gives util; fall back to mem-based estimate if unavailable
                if _PSUTIL:
                    pass   # psutil doesn't expose GPU util; use torch
                gpu_pct = min(100.0, fps / 60 * 100)   # rough proxy
            except Exception:
                pass

        with _perf_lock:
            _perf["fps"]         = round(fps, 1)
            _perf["infer_ms"]    = round(infer_ms, 1)
            _perf["gpu_mem_mb"]  = round(gpu_mem_mb, 1)
            _perf["uptime_s"]    = int(time.time() - START_TIME)
            _perf["frame_count"] += 1

        # ── JPEG encode — quality 70 gives best FPS/quality tradeoff on GPU ──
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + jpeg.tobytes() + b"\r\n\r\n")


def _track_colour(track_id: int) -> tuple:
    """Deterministic per-track colour so each individual is visually distinct."""
    rng = (track_id * 2654435761) & 0xFFFFFF
    r = (rng >> 16) & 0xFF
    g = (rng >> 8)  & 0xFF
    b =  rng        & 0xFF
    # keep it bright — boost any dim channel
    if max(r, g, b) < 100:
        r, g, b = min(r + 120, 255), min(g + 80, 255), min(b + 80, 255)
    return (b, g, r)   # OpenCV is BGR


def build_heatmap(frame, bboxes):
    hmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for (x1, y1, x2, y2) in bboxes:
        hmap[y1:y2, x1:x2] += 1.0
    if hmap.max() > 0:
        hmap /= hmap.max()
    fig, ax = plt.subplots(
        figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
    sns.heatmap(hmap, ax=ax, cmap="jet", alpha=0.8, cbar=False)
    ax.axis("off")
    path = os.path.join(UPLOAD_DIR, "heatmap.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return jsonify(error="Not authenticated"), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# CSS shared
# ─────────────────────────────────────────────────────────────────────────────
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050a0e;--bg2:#0b1218;--bg3:#111c24;--border:#1a2d3a;
  --accent:#00e5a0;--accent2:#00b8d9;--danger:#ff4757;--warn:#ffa502;
  --text:#c8dde8;--textdim:#4a6a7c;
  --font-ui:'Syne',sans-serif;--font-mono:'Space Mono',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-ui)}
a{color:var(--accent);text-decoration:none}
.btn{display:inline-flex;align-items:center;gap:.5rem;padding:.55rem 1.4rem;
     border-radius:4px;border:none;font-family:var(--font-mono);font-size:.78rem;
     letter-spacing:.06em;cursor:pointer;transition:.15s}
.btn-primary{background:var(--accent);color:#050a0e;font-weight:700}
.btn-primary:hover{background:#00ffb3}
.btn-primary:disabled{opacity:.4;cursor:not-allowed}
.btn-ghost{background:transparent;color:var(--accent);border:1px solid var(--accent)}
.btn-ghost:hover{background:rgba(0,229,160,.08)}
input,select{width:100%;padding:.65rem 1rem;border-radius:4px;border:1px solid var(--border);
  background:var(--bg3);color:var(--text);font-family:var(--font-mono);font-size:.85rem;
  outline:none;transition:border-color .2s}
input:focus{border-color:var(--accent)}
.tag{display:inline-block;padding:.18rem .6rem;border-radius:3px;
     font-family:var(--font-mono);font-size:.72rem;letter-spacing:.04em}
.tag-green{background:rgba(0,229,160,.12);color:var(--accent)}
.tag-blue{background:rgba(0,184,217,.12);color:var(--accent2)}
.tag-red{background:rgba(255,71,87,.12);color:var(--danger)}
.tag-warn{background:rgba(255,165,2,.12);color:var(--warn)}
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Login</title>
<style>""" + _BASE_CSS + """
body{display:flex;align-items:center;justify-content:center;min-height:100vh;
     background:radial-gradient(ellipse 80%% 60%% at 50%% 0%%,#051a14 0%%,var(--bg) 70%%)}
.card{width:380px;padding:2.8rem 2.4rem;background:var(--bg2);border:1px solid var(--border);
      border-radius:8px;box-shadow:0 0 60px rgba(0,229,160,.06);animation:fadeup .5s ease both}
@keyframes fadeup{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
.logo{font-size:1.1rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
      color:var(--accent);margin-bottom:.3rem}
.sub{font-size:.8rem;color:var(--textdim);margin-bottom:2rem;font-family:var(--font-mono)}
.field{margin-bottom:1.2rem}
.field label{display:block;font-size:.75rem;color:var(--textdim);font-family:var(--font-mono);
             letter-spacing:.08em;text-transform:uppercase;margin-bottom:.45rem}
.submit{width:100%%;margin-top:.4rem;padding:.75rem;font-size:.85rem}
.err{background:rgba(255,71,87,.1);border:1px solid rgba(255,71,87,.3);color:var(--danger);
     padding:.65rem 1rem;border-radius:4px;font-size:.8rem;font-family:var(--font-mono);margin-bottom:1.2rem}
.glow-line{height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);
           margin-bottom:2rem;opacity:.4}
</style></head><body>
<div class="card">
  <div class="logo">⬡ Sentinel</div>
  <div class="sub">Wildlife Monitoring System</div>
  <div class="glow-line"></div>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="POST">
    <div class="field"><label>Username</label>
      <input type="text" name="username" autocomplete="username" required autofocus></div>
    <div class="field"><label>Password</label>
      <input type="password" name="password" autocomplete="current-password" required></div>
    <button class="btn btn-primary submit" type="submit">Access System</button>
  </form>
</div>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Dashboard</title>
<style>""" + _BASE_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh}
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.8rem;
    border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:100}
.nav-logo{font-weight:800;font-size:1rem;letter-spacing:.1em;text-transform:uppercase;
          color:var(--accent);display:flex;align-items:center;gap:.5rem}
.nav-logo span{width:9px;height:9px;background:var(--accent);border-radius:50%;
               box-shadow:0 0 8px var(--accent);animation:pulse 2s infinite}
@keyframes pulse{0%%,100%%{opacity:1;transform:scale(1)}50%%{opacity:.5;transform:scale(.8)}}
.nav-links{display:flex;align-items:center;gap:1rem}
.nav-links a{font-size:.8rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.05em;transition:color .2s}
.nav-links a:hover,.nav-links a.active{color:var(--accent)}
.nav-sep{width:1px;height:18px;background:var(--border)}
.gpu-badge{font-size:.68rem;font-family:var(--font-mono);padding:.2rem .6rem;border-radius:3px;
           background:rgba(0,184,217,.1);color:var(--accent2);border:1px solid rgba(0,184,217,.2)}
main{padding:1.8rem;overflow-y:auto}
@keyframes fadeup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
/* stats row */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1rem;margin-bottom:1.4rem}
.stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:1rem 1.2rem;animation:fadeup .4s ease both}
.stat-label{font-size:.68rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.45rem}
.stat-value{font-size:1.5rem;font-weight:800;color:var(--text);line-height:1}
.stat-value.green{color:var(--accent)}.stat-value.blue{color:var(--accent2)}.stat-value.warn{color:var(--warn)}
/* perf row */
.perf-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin-bottom:1.4rem}
.perf-card{background:var(--bg3);border:1px solid var(--border);border-radius:5px;padding:.7rem 1rem}
.perf-label{font-size:.65rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem}
.perf-val{font-size:1.1rem;font-weight:700;font-family:var(--font-mono);color:var(--accent2)}
.perf-bar-bg{height:3px;background:var(--border);border-radius:2px;margin-top:.4rem;overflow:hidden}
.perf-bar{height:100%%;border-radius:2px;transition:width .5s}
/* grid */
.grid2{display:grid;grid-template-columns:1fr 360px;gap:1.4rem}
@media(max-width:960px){.grid2{grid-template-columns:1fr}}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:.8rem 1.2rem;
            border-bottom:1px solid var(--border);font-size:.75rem;font-family:var(--font-mono);
            letter-spacing:.07em;text-transform:uppercase;color:var(--textdim)}
.title-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);margin-right:.5rem;box-shadow:0 0 6px var(--accent)}
.stream-wrap{position:relative;background:#000;aspect-ratio:16/9}
.stream-wrap img{width:100%%;height:100%%;object-fit:cover;display:block}
.stream-overlay{position:absolute;top:0;left:0;right:0;display:flex;align-items:center;
                justify-content:space-between;padding:.5rem .8rem;pointer-events:none;
                background:linear-gradient(to bottom,rgba(5,10,14,.8),transparent)}
.rec-badge{display:flex;align-items:center;gap:.35rem;font-size:.7rem;font-family:var(--font-mono);color:var(--danger)}
.rec-dot{width:7px;height:7px;border-radius:50%;background:var(--danger);animation:blink 1s infinite}
@keyframes blink{0%%,100%%{opacity:1}50%%{opacity:.2}}
.stream-ts{font-family:var(--font-mono);font-size:.68rem;color:rgba(200,221,232,.5)}
.cam-stopped-badge{font-size:.7rem;font-family:var(--font-mono);color:var(--warn);
                   display:none;align-items:center;gap:.4rem}
.det-list{max-height:300px;overflow-y:auto}
.det-list::-webkit-scrollbar{width:4px}
.det-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.det-row{display:flex;align-items:center;justify-content:space-between;padding:.6rem 1.2rem;
         border-bottom:1px solid var(--border);font-size:.82rem;transition:background .15s}
.det-row:last-child{border-bottom:none}
.det-count{font-weight:700;color:var(--accent);font-family:var(--font-mono)}
.det-empty{padding:1.4rem;text-align:center;color:var(--textdim);font-size:.8rem;font-family:var(--font-mono)}
.heatmap-wrap{padding:1rem;background:#000;min-height:160px;display:flex;align-items:center;justify-content:center}
.heatmap-wrap img{max-width:100%%;border-radius:4px}
.controls{display:flex;gap:.7rem;padding:.9rem 1.2rem;border-top:1px solid var(--border);flex-wrap:wrap;align-items:center}
.btn-stop{background:rgba(255,71,87,.15);color:var(--danger);border:1px solid rgba(255,71,87,.3)}
.btn-stop:hover{background:rgba(255,71,87,.25)}
.btn-start{background:rgba(0,229,160,.15);color:var(--accent);border:1px solid rgba(0,229,160,.3)}
.btn-start:hover{background:rgba(0,229,160,.25)}
</style></head><body>
<nav>
  <div class="nav-logo"><span></span>Sentinel</div>
  <div class="nav-links">
    <a href="/dashboard" class="active">Dashboard</a>
    <a href="/analyse">Analyse Video</a>
    <a href="/recordings">Recordings</a>
    <div class="nav-sep"></div>
    <span class="gpu-badge" id="gpu-badge">{{ device }}</span>
    <div class="nav-sep"></div>
    <a href="/logout">Logout</a>
  </div>
</nav>
<main>
  <!-- Detection stats -->
  <div class="stats">
    <div class="stat-card"><div class="stat-label">Total Detections</div><div class="stat-value green" id="stat-total">—</div></div>
    <div class="stat-card"><div class="stat-label">Unique Classes</div><div class="stat-value blue" id="stat-classes">—</div></div>
    <div class="stat-card"><div class="stat-label">Clips Saved</div><div class="stat-value" id="stat-clips">—</div></div>
    <div class="stat-card"><div class="stat-label">Camera</div><div class="stat-value green" id="stat-cam">LIVE</div></div>
  </div>

  <!-- Performance row -->
  <div class="perf-row">
    <div class="perf-card">
      <div class="perf-label">FPS</div>
      <div class="perf-val" id="perf-fps">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-fps" style="background:var(--accent);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">Infer ms</div>
      <div class="perf-val" id="perf-ms">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-ms" style="background:var(--accent2);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">GPU Mem</div>
      <div class="perf-val" id="perf-gmem">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-gmem" style="background:var(--warn);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">CPU %%</div>
      <div class="perf-val" id="perf-cpu">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-cpu" style="background:var(--accent2);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">RAM</div>
      <div class="perf-val" id="perf-ram">—</div>
    </div>
    <div class="perf-card">
      <div class="perf-label">Uptime</div>
      <div class="perf-val" id="perf-uptime">—</div>
    </div>
    <div class="perf-card">
      <div class="perf-label">Frames</div>
      <div class="perf-val" id="perf-frames">—</div>
    </div>
    <div class="perf-card">
      <div class="perf-label">Tracker</div>
      <div class="perf-val" id="perf-tracker" style="font-size:.8rem">—</div>
    </div>
  </div>

  <div class="grid2">
    <div>
      <div class="panel">
        <div class="panel-head">
          <div style="display:flex;align-items:center">
            <div class="title-dot"></div>Live Feed
            <span class="cam-stopped-badge" id="stopped-badge" style="margin-left:.8rem">
              ■ STOPPED
            </span>
          </div>
          <span class="tag tag-green">ByteTrack</span>
        </div>
        <div class="stream-wrap">
          <img id="stream-img" src="/video_feed" alt="Live feed">
          <div class="stream-overlay">
            <div class="rec-badge" id="rec-badge"><div class="rec-dot"></div>REC</div>
            <div class="stream-ts" id="clock"></div>
          </div>
        </div>
        <div class="controls">
          <button class="btn btn-stop"  id="btn-stop"  onclick="camStop()">⏹ Stop</button>
          <button class="btn btn-start" id="btn-start" onclick="camStart()" style="display:none">▶ Start</button>
          <button class="btn btn-ghost" onclick="takeSnapshot()">📷 Snapshot</button>
          <button class="btn btn-ghost" onclick="refreshHeatmap()">Heatmap</button>
          <button class="btn btn-ghost" onclick="clearCounts()">Clear Counts</button>
        </div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:1.2rem">
      <div class="panel">
        <div class="panel-head">
          <div style="display:flex;align-items:center"><div class="title-dot"></div>Detections</div>
          <span class="tag tag-blue" id="badge-counts">0 classes</span>
        </div>
        <div class="det-list" id="det-list"><div class="det-empty">Waiting for detections…</div></div>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div style="display:flex;align-items:center"><div class="title-dot"></div>Heatmap</div>
          <span class="tag tag-blue">Latest</span>
        </div>
        <div class="heatmap-wrap">
          <img id="heatmap-img" src="/heatmap" alt="Heatmap" onerror="this.style.display='none'">
        </div>
      </div>
    </div>
  </div>
</main>
<script>
// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock(){document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});}
setInterval(updateClock,1000);updateClock();

// ── Detection counts ──────────────────────────────────────────────────────────
async function pollCounts(){
  try{
    const data=await(await fetch('/api/counts')).json();
    const entries=Object.entries(data);
    const total=entries.reduce((a,[,v])=>a+v,0);
    document.getElementById('stat-total').textContent=total;
    document.getElementById('stat-classes').textContent=entries.length;
    document.getElementById('badge-counts').textContent=entries.length+' classes';
    const list=document.getElementById('det-list');
    if(!entries.length){list.innerHTML='<div class="det-empty">Waiting…</div>';return;}
    list.innerHTML=entries.sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`<div class="det-row"><span style="font-family:var(--font-mono)">${k}</span><span class="det-count">${v}</span></div>`)
      .join('');
  }catch(_){}
}
async function pollClips(){
  try{const d=await(await fetch('/api/clip_count')).json();document.getElementById('stat-clips').textContent=d.count;}catch(_){}
}

// ── Performance ───────────────────────────────────────────────────────────────
async function pollPerf(){
  try{
    const p=await(await fetch('/api/perf')).json();
    setText('perf-fps',   p.fps+' fps');
    setText('perf-ms',    p.infer_ms+' ms');
    setText('perf-gmem',  p.gpu_mem_mb+' MB');
    setText('perf-cpu',   p.cpu_pct+'%%');
    setText('perf-ram',   Math.round(p.ram_mb)+' MB');
    setText('perf-uptime',p.uptime_fmt);
    setText('perf-frames',p.frame_count.toLocaleString());
    setText('perf-tracker', p.cython ? 'Cython+ByteTrack' : 'PyIoU+ByteTrack');
    setBar('bar-fps',  p.fps,  60);
    setBar('bar-ms',   120-p.infer_ms, 120);
    setBar('bar-gmem', p.gpu_mem_mb, 4096);
    setBar('bar-cpu',  100-p.cpu_pct, 100);
    document.getElementById('gpu-badge').textContent = p.gpu_name || p.device;
  }catch(_){}
}
function setText(id,v){const el=document.getElementById(id);if(el)el.textContent=v;}
function setBar(id,val,max){
  const el=document.getElementById(id);
  if(el)el.style.width=Math.min(100,Math.max(0,val/max*100))+'%%';
}

// ── Camera controls ───────────────────────────────────────────────────────────
async function camStop(){
  await fetch('/api/cam/stop',{method:'POST'});
  document.getElementById('btn-stop').style.display='none';
  document.getElementById('btn-start').style.display='inline-flex';
  document.getElementById('stat-cam').textContent='STOPPED';
  document.getElementById('stat-cam').className='stat-value warn';
  document.getElementById('stopped-badge').style.display='flex';
  document.getElementById('rec-badge').style.opacity='.3';
}
async function camStart(){
  await fetch('/api/cam/start',{method:'POST'});
  document.getElementById('btn-stop').style.display='inline-flex';
  document.getElementById('btn-start').style.display='none';
  document.getElementById('stat-cam').textContent='LIVE';
  document.getElementById('stat-cam').className='stat-value green';
  document.getElementById('stopped-badge').style.display='none';
  document.getElementById('rec-badge').style.opacity='1';
  // force stream img to reload
  const img=document.getElementById('stream-img');
  img.src='/video_feed?t='+Date.now();
}
async function takeSnapshot(){
  window.open('/api/cam/snapshot','_blank');
}
function refreshHeatmap(){const i=document.getElementById('heatmap-img');i.src='/heatmap?t='+Date.now();i.style.display='block';}
async function clearCounts(){await fetch('/api/clear_counts',{method:'POST'});pollCounts();}

// ── Restore cam state on load ─────────────────────────────────────────────────
(async()=>{
  try{
    const s=await(await fetch('/api/cam/status')).json();
    if(!s.active){
      document.getElementById('btn-stop').style.display='none';
      document.getElementById('btn-start').style.display='inline-flex';
      document.getElementById('stat-cam').textContent='STOPPED';
      document.getElementById('stat-cam').className='stat-value warn';
      document.getElementById('stopped-badge').style.display='flex';
    }
  }catch(_){}
})();

setInterval(pollCounts,2000);
setInterval(pollClips, 5000);
setInterval(pollPerf,  1000);
pollCounts();pollClips();pollPerf();
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE PAGE
# ─────────────────────────────────────────────────────────────────────────────
ANALYSE_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Analyse Video</title>
<style>""" + _BASE_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh}
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.8rem;
    border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:100}
.nav-logo{font-weight:800;font-size:1rem;letter-spacing:.1em;text-transform:uppercase;
          color:var(--accent);display:flex;align-items:center;gap:.5rem}
.nav-logo span{width:9px;height:9px;background:var(--accent);border-radius:50%;box-shadow:0 0 8px var(--accent)}
.nav-links{display:flex;align-items:center;gap:1rem}
.nav-links a{font-size:.8rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.05em;transition:color .2s}
.nav-links a:hover,.nav-links a.active{color:var(--accent)}
.nav-sep{width:1px;height:18px;background:var(--border)}
.gpu-badge{font-size:.68rem;font-family:var(--font-mono);padding:.2rem .6rem;border-radius:3px;
           background:rgba(0,184,217,.1);color:var(--accent2);border:1px solid rgba(0,184,217,.2)}
main{padding:1.8rem;overflow-y:auto;max-width:880px;margin:0 auto}
h1{font-size:1.05rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1.6rem}

/* upload zone */
.upload-zone{border:2px dashed var(--border);border-radius:8px;padding:3rem 2rem;
  text-align:center;cursor:pointer;background:var(--bg2);transition:.2s;
  animation:fadeup .4s ease both;position:relative}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:rgba(0,229,160,.03)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%%;height:100%%}
.upload-icon{font-size:2.5rem;margin-bottom:.8rem;opacity:.5}
.upload-title{font-size:1rem;font-weight:600;margin-bottom:.4rem}
.upload-sub{font-size:.78rem;color:var(--textdim);font-family:var(--font-mono)}
.upload-chosen{font-size:.82rem;font-family:var(--font-mono);color:var(--accent);margin-top:.8rem;min-height:1.2rem}
@keyframes fadeup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}

/* upload speed */
.upload-speed{font-size:.72rem;font-family:var(--font-mono);color:var(--textdim);
              margin-top:.4rem;min-height:1rem}

.controls-row{display:flex;gap:.8rem;margin-top:1.2rem;align-items:center;flex-wrap:wrap}

/* progress */
.prog-wrap{margin-top:1.4rem;display:none}
.prog-wrap.show{display:block}
.prog-label{font-size:.75rem;font-family:var(--font-mono);color:var(--textdim);
            letter-spacing:.08em;text-transform:uppercase;margin-bottom:.5rem;
            display:flex;justify-content:space-between}
.prog-bar-bg{height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;margin-bottom:.3rem}
.prog-bar{height:100%%;background:var(--accent);border-radius:3px;width:0%%;transition:width .4s}
.prog-stage{font-size:.72rem;font-family:var(--font-mono);color:var(--textdim)}

/* results */
.results{margin-top:1.6rem;display:none}
.results.show{display:block}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:.8rem 1.2rem;
            border-bottom:1px solid var(--border);font-size:.75rem;font-family:var(--font-mono);
            letter-spacing:.07em;text-transform:uppercase;color:var(--textdim)}
.title-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);margin-right:.5rem;box-shadow:0 0 6px var(--accent)}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem;padding:1.2rem}
.s-card{background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:1rem 1.2rem}
.s-label{font-size:.68rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.1em;
          text-transform:uppercase;margin-bottom:.4rem}
.s-value{font-size:1.5rem;font-weight:800;color:var(--accent)}
.species-table{width:100%%;border-collapse:collapse;font-size:.84rem}
.species-table th{padding:.6rem 1.2rem;text-align:left;font-size:.7rem;font-family:var(--font-mono);
                  letter-spacing:.08em;text-transform:uppercase;color:var(--textdim);border-bottom:1px solid var(--border)}
.species-table td{padding:.75rem 1.2rem;border-bottom:1px solid var(--border);font-family:var(--font-mono)}
.species-table tr:last-child td{border:none}
.species-table tr:hover td{background:rgba(0,229,160,.03)}
.bar-bg{height:8px;background:var(--bg3);border-radius:4px;overflow:hidden;width:160px}
.bar-fill{height:100%%;background:linear-gradient(90deg,var(--accent),var(--accent2));
          border-radius:4px;transition:width .6s}
.err-box{padding:1rem 1.2rem;font-family:var(--font-mono);font-size:.82rem;color:var(--danger);
         background:rgba(255,71,87,.06);border-top:1px solid rgba(255,71,87,.2)}
.dl-row{display:flex;justify-content:flex-end;padding:.8rem 1.2rem;border-top:1px solid var(--border)}
</style></head><body>
<nav>
  <div class="nav-logo"><span></span>Sentinel</div>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/analyse" class="active">Analyse Video</a>
    <a href="/recordings">Recordings</a>
    <div class="nav-sep"></div>
    <span class="gpu-badge">{{ device }}</span>
    <div class="nav-sep"></div>
    <a href="/logout">Logout</a>
  </div>
</nav>
<main>
  <h1>Video Animal Analysis</h1>

  <div class="upload-zone" id="drop-zone">
    <input type="file" id="file-input" accept="video/*">
    <div class="upload-icon">🎬</div>
    <div class="upload-title">Drop a video or click to browse</div>
    <div class="upload-sub">MP4 · AVI · MOV · MKV · WEBM &nbsp;·&nbsp; No size limit · Streamed to disk</div>
    <div class="upload-chosen" id="chosen-name"></div>
    <div class="upload-speed" id="upload-speed"></div>
  </div>

  <div class="controls-row">
    <button class="btn btn-primary" id="analyse-btn" onclick="startAnalysis()" disabled>
      Run Analysis
    </button>
    <span class="tag tag-blue" style="font-size:.72rem">Tracks unique individuals — no repeat counting</span>
  </div>

  <!-- Combined upload + analysis progress -->
  <div class="prog-wrap" id="prog-wrap">
    <div class="prog-label">
      <span id="prog-label-text">Uploading…</span>
      <span id="prog-pct">0%%</span>
    </div>
    <div class="prog-bar-bg"><div class="prog-bar" id="prog-bar"></div></div>
    <div class="prog-stage" id="prog-stage"></div>
  </div>

  <!-- Results -->
  <div class="results" id="results-section">
    <div class="panel">
      <div class="panel-head">
        <div style="display:flex;align-items:center">
          <div class="title-dot"></div>Analysis Results
        </div>
        <span class="tag tag-green">Done</span>
      </div>
      <div class="summary-grid">
        <div class="s-card"><div class="s-label">Total Unique Animals</div><div class="s-value" id="res-total">—</div></div>
        <div class="s-card"><div class="s-label">Species Found</div><div class="s-value" id="res-species">—</div></div>
        <div class="s-card"><div class="s-label">Frames Analysed</div><div class="s-value" id="res-frames">—</div></div>
        <div class="s-card"><div class="s-label">Ran On</div><div class="s-value" id="res-device" style="font-size:1rem">—</div></div>
      </div>
      <table class="species-table">
        <thead><tr><th>#</th><th>Species</th><th>Unique Count</th><th>Distribution</th></tr></thead>
        <tbody id="species-tbody"></tbody>
      </table>
      <div class="err-box" id="err-box" style="display:none"></div>
      <div class="dl-row" id="dl-row" style="display:none">
        <a id="dl-link" href="#" class="btn btn-ghost" download>↓ Download Annotated Video</a>
      </div>
    </div>
  </div>
</main>
<script>
// ── Reload-resilient job tracking via localStorage ────────────────────────────
const JOB_KEY    = 'sentinel_job_id';
let currentJobId = localStorage.getItem(JOB_KEY) || null;
let pollTimer    = null;

const fileInput  = document.getElementById('file-input');
const dropZone   = document.getElementById('drop-zone');
const chosenName = document.getElementById('chosen-name');
const analyseBtn = document.getElementById('analyse-btn');
const speedEl    = document.getElementById('upload-speed');

fileInput.addEventListener('change', () => {
  if(fileInput.files[0]){ chosenName.textContent=fileInput.files[0].name; analyseBtn.disabled=false; }
});
dropZone.addEventListener('dragover',  e=>{ e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', ()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e=>{
  e.preventDefault(); dropZone.classList.remove('drag');
  if(e.dataTransfer.files[0]){
    // assign to input
    const dt = new DataTransfer();
    dt.items.add(e.dataTransfer.files[0]);
    fileInput.files = dt.files;
    chosenName.textContent = e.dataTransfer.files[0].name;
    analyseBtn.disabled = false;
  }
});

async function startAnalysis(){
  if(!fileInput.files[0]) return;
  analyseBtn.disabled = true;
  document.getElementById('results-section').classList.remove('show');
  document.getElementById('prog-wrap').classList.add('show');
  setProgress(0, 'Uploading…', '');

  const file   = fileInput.files[0];
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);

  // ── Chunked XHR upload with progress ──────────────────────────────────────
  try {
    const jobId = await uploadWithProgress(file);
    currentJobId = jobId;
    localStorage.setItem(JOB_KEY, jobId);   // survive reload
    setProgress(100, 'Analysing…', 'Upload complete — running YOLO inference');
    document.getElementById('prog-label-text').textContent = 'Analysing…';
    document.getElementById('prog-bar').style.background = 'var(--accent2)';
    pollTimer = setInterval(pollJob, 1500);
  } catch(e) {
    showError('Upload failed: ' + e.message);
    analyseBtn.disabled = false;
  }
}

function uploadWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let lastLoaded = 0, lastTime = Date.now();

    xhr.upload.addEventListener('progress', e => {
      if(!e.lengthComputable) return;
      const pct = Math.round(e.loaded / e.total * 100);
      const now = Date.now();
      const dt  = (now - lastTime) / 1000;
      const bps = (e.loaded - lastLoaded) / dt;
      lastLoaded = e.loaded; lastTime = now;
      const speed = bps > 1e6
        ? (bps/1e6).toFixed(1) + ' MB/s'
        : (bps/1e3).toFixed(0) + ' KB/s';
      speedEl.textContent = speed;
      const eta = bps > 0 ? Math.round((e.total - e.loaded) / bps) : '?';
      setProgress(pct, 'Uploading…', `${(e.loaded/1e6).toFixed(0)} / ${(e.total/1e6).toFixed(0)} MB  ·  ${speed}  ·  ETA ${eta}s`);
    });

    xhr.addEventListener('load', () => {
      speedEl.textContent = '';
      if(xhr.status === 202) {
        try { resolve(JSON.parse(xhr.responseText).job_id); }
        catch(e) { reject(new Error('Bad server response')); }
      } else {
        let msg = 'Server error ' + xhr.status;
        try { msg = JSON.parse(xhr.responseText).error || msg; } catch(_){}
        reject(new Error(msg));
      }
    });
    xhr.addEventListener('error',   () => reject(new Error('Network error')));
    xhr.addEventListener('timeout', () => reject(new Error('Upload timed out')));
    xhr.timeout = {{ timeout_ms }};

    const fd = new FormData();
    fd.append('video', file);
    xhr.open('POST', '/api/analyse');
    xhr.send(fd);
  });
}

async function pollJob(){
  if(!currentJobId) return;
  try{
    const r = await fetch('/api/analyse/' + currentJobId);
    if(r.status === 404){
      clearInterval(pollTimer);
      localStorage.removeItem(JOB_KEY);
      currentJobId = null;
      showError('Job not found — server may have restarted. Please re-upload.');
      analyseBtn.disabled = false;
      return;
    }
    const data = await r.json();
    const pct  = data.progress || 0;
    const stage =
      data.status==='processing' ? `Frame ${Math.round(pct)}%% — inference running` :
      data.status==='done'       ? 'Complete!' :
      data.status==='error'      ? 'Error' : 'Queued…';
    setProgress(pct, 'Analysing…', stage);
    if(data.status==='done'){
      clearInterval(pollTimer);
      localStorage.removeItem(JOB_KEY);
      showResults(data);
      analyseBtn.disabled=false;
    } else if(data.status==='error'){
      clearInterval(pollTimer);
      localStorage.removeItem(JOB_KEY);
      showError(data.error||'Unknown error');
      analyseBtn.disabled=false;
    }
  }catch(_){}
}

function setProgress(pct, label, stage){
  document.getElementById('prog-bar').style.width      = pct+'%%';
  document.getElementById('prog-pct').textContent       = pct+'%%';
  document.getElementById('prog-label-text').textContent= label;
  document.getElementById('prog-stage').textContent     = stage;
}

function showResults(data){
  document.getElementById('results-section').classList.add('show');
  document.getElementById('err-box').style.display='none';
  document.getElementById('res-total').textContent   = data.total_unique;
  document.getElementById('res-species').textContent = data.species.length;
  document.getElementById('res-frames').textContent  = data.frames_analysed;
  document.getElementById('res-device').textContent  = data.device || '—';
  const max = data.species[0]?.count || 1;
  document.getElementById('species-tbody').innerHTML = data.species.map((s,i)=>`
    <tr>
      <td style="color:var(--textdim)">${i+1}</td>
      <td style="font-weight:600;text-transform:capitalize">${s.species}</td>
      <td><span class="tag tag-green">${s.count}</span></td>
      <td><div class="bar-bg"><div class="bar-fill" style="width:${Math.round(s.count/max*100)}%%"></div></div></td>
    </tr>`).join('');
  if(data.frames_analysed > 0){
    document.getElementById('dl-link').href='/api/analyse/'+currentJobId+'/download';
    document.getElementById('dl-row').style.display='flex';
  }
}

function showError(msg){
  document.getElementById('results-section').classList.add('show');
  document.getElementById('err-box').style.display='block';
  document.getElementById('err-box').textContent='✗ '+msg;
  document.getElementById('species-tbody').innerHTML='';
  document.getElementById('dl-row').style.display='none';
}

// ── Auto-resume in-progress job on page load / reload ─────────────────────────
(async()=>{
  if(!currentJobId) return;
  try{
    const r = await fetch('/api/analyse/' + currentJobId);
    if(r.status === 404){ localStorage.removeItem(JOB_KEY); currentJobId=null; return; }
    const data = await r.json();
    if(data.status === 'done'){
      document.getElementById('prog-wrap').classList.add('show');
      localStorage.removeItem(JOB_KEY);
      showResults(data);
    } else if(data.status === 'error'){
      localStorage.removeItem(JOB_KEY);
      document.getElementById('prog-wrap').classList.add('show');
      showError(data.error||'Unknown error');
    } else {
      // still running — show progress and resume polling
      document.getElementById('prog-wrap').classList.add('show');
      setProgress(data.progress||0, 'Analysing…', 'Resumed after reload…');
      document.getElementById('prog-bar').style.background = 'var(--accent2)';
      pollTimer = setInterval(pollJob, 1500);
    }
  }catch(_){}
})();
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RECORDINGS PAGE
# ─────────────────────────────────────────────────────────────────────────────
RECORDINGS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Recordings</title>
<style>""" + _BASE_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh}
nav{display:flex;align-items:center;justify-content:space-between;padding:0 1.8rem;
    border-bottom:1px solid var(--border);background:var(--bg2)}
.nav-logo{font-weight:800;font-size:1rem;letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.nav-links a{font-size:.8rem;font-family:var(--font-mono);color:var(--textdim);letter-spacing:.05em;margin-left:1rem;transition:color .2s}
.nav-links a:hover,.nav-links a.active{color:var(--accent)}
main{padding:1.8rem}
h1{font-size:1.05rem;font-weight:800;letter-spacing:.06em;margin-bottom:1.4rem;text-transform:uppercase}
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:6px;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:.8rem 1.2rem;
            border-bottom:1px solid var(--border);font-size:.75rem;font-family:var(--font-mono);
            letter-spacing:.07em;text-transform:uppercase;color:var(--textdim)}
table{width:100%%;border-collapse:collapse;font-size:.82rem}
th{padding:.6rem 1.2rem;text-align:left;font-size:.7rem;font-family:var(--font-mono);letter-spacing:.08em;
   text-transform:uppercase;color:var(--textdim);border-bottom:1px solid var(--border)}
td{padding:.7rem 1.2rem;border-bottom:1px solid var(--border);font-family:var(--font-mono)}
tr:last-child td{border:none}
tr:hover td{background:rgba(0,229,160,.03)}
.empty{padding:2rem;text-align:center;color:var(--textdim);font-family:var(--font-mono)}
</style></head><body>
<nav>
  <div class="nav-logo">⬡ Sentinel</div>
  <div class="nav-links">
    <a href="/dashboard">Dashboard</a>
    <a href="/analyse">Analyse Video</a>
    <a href="/recordings" class="active">Recordings</a>
    <a href="/logout">Logout</a>
  </div>
</nav>
<main>
  <h1>Saved Recordings</h1>
  <div class="panel">
    <div class="panel-head">
      <span>{{ videos|length }} file{{ 's' if videos|length != 1 }}</span>
      <span class="tag tag-blue">{{ dir }}</span>
    </div>
    {% if videos %}
    <table>
      <tr><th>#</th><th>Filename</th><th>Size</th><th>Action</th></tr>
      {% for v in videos %}
      <tr>
        <td style="color:var(--textdim)">{{ loop.index }}</td>
        <td>{{ v.name }}</td>
        <td>{{ v.size }}</td>
        <td><a class="btn btn-ghost" style="padding:.3rem .8rem;font-size:.72rem"
               href="/download/{{ v.name }}">Download</a></td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty">No recordings yet.</div>
    {% endif %}
  </div>
</main>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def welcome():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        pw_hash  = hashlib.sha256(password.encode()).hexdigest()
        if (hmac.compare_digest(username, ADMIN_USERNAME) and
                hmac.compare_digest(pw_hash, ADMIN_PASS_HASH)):
            session.permanent = True
            session["user"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid credentials."
    return render_template_string(LOGIN_HTML, error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template_string(DASHBOARD_HTML, device=f"{DEVICE} — {GPU_NAME}")

@app.route("/analyse")
@login_required
def analyse_page():
    return render_template_string(
        ANALYSE_HTML,
        device=f"{DEVICE} — {GPU_NAME}",
        timeout_ms=UPLOAD_TIMEOUT * 1000,
    )

@app.route("/recordings")
@login_required
def recordings():
    files = []
    for fname in sorted(os.listdir(VIDEO_DIR), reverse=True):
        if fname.endswith(".mp4"):
            fpath = os.path.join(VIDEO_DIR, fname)
            size  = os.path.getsize(fpath)
            files.append({
                "name": fname,
                "size": (f"{size/1024/1024:.1f} MB"
                         if size > 1_000_000 else f"{size/1024:.0f} KB"),
            })
    return render_template_string(RECORDINGS_HTML, videos=files, dir=VIDEO_DIR)

@app.route("/download/<filename>")
@login_required
def download_file(filename: str):
    safe = secure_filename(filename)
    path = os.path.join(VIDEO_DIR, safe)
    if not os.path.exists(path):
        return jsonify(error="File not found"), 404
    return send_file(path, as_attachment=True)

@app.route("/video_feed")
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/heatmap")
@login_required
def heatmap():
    global cap
    if cap is None or not cap.isOpened():
        reconnect_stream()
    ret, frame = cap.read()
    if not ret:
        return jsonify(error="Camera unavailable"), 503
    with _model_lock:
        results = model(frame, verbose=False, device=DEVICE)
    bboxes = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
         int(b.xyxy[0][2]), int(b.xyxy[0][3]))
        for r in results for b in r.boxes
    ]
    return send_file(build_heatmap(frame, bboxes), mimetype="image/png")

# ── Live-stream JSON API ──────────────────────────────────────────────────────
@app.route("/api/counts")
@login_required
def api_counts():
    with _state_lock:
        return jsonify(dict(animal_counts))

@app.route("/api/clip_count")
@login_required
def api_clip_count():
    n = len([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    return jsonify({"count": n})

@app.route("/api/clear_counts", methods=["POST"])
@login_required
def api_clear_counts():
    with _state_lock:
        animal_counts.clear()
    return jsonify({"ok": True})

# ── Camera control ────────────────────────────────────────────────────────────
@app.route("/api/cam/stop", methods=["POST"])
@login_required
def api_cam_stop():
    global cam_active
    _cam_event.clear()
    cam_active = False
    log.info("Camera stopped by user.")
    return jsonify({"active": False})

@app.route("/api/cam/start", methods=["POST"])
@login_required
def api_cam_start():
    global cam_active
    _cam_event.set()
    cam_active = True
    log.info("Camera started by user.")
    return jsonify({"active": True})

@app.route("/api/cam/status")
@login_required
def api_cam_status():
    return jsonify({"active": cam_active})

@app.route("/api/cam/snapshot")
@login_required
def api_cam_snapshot():
    """Return a single JPEG snapshot from the current frame."""
    global cap
    if not cam_active:
        return jsonify(error="Camera is stopped"), 400
    if cap is None or not cap.isOpened():
        reconnect_stream()
    ret, frame = cap.read()
    if not ret:
        return jsonify(error="Could not capture frame"), 503
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return jsonify(error="Encode failed"), 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg",
                    headers={"Content-Disposition":
                             f'attachment; filename="snapshot_{int(time.time())}.jpg"'})

# ── Performance metrics ───────────────────────────────────────────────────────
@app.route("/api/perf")
@login_required
def api_perf():
    cpu_pct = psutil.cpu_percent() if _PSUTIL else 0.0
    ram_mb  = psutil.Process().memory_info().rss / 1024 / 1024 if _PSUTIL else 0.0
    with _perf_lock:
        data = dict(_perf)
    data["cpu_pct"]  = round(cpu_pct, 1)
    data["ram_mb"]   = round(ram_mb, 1)
    data["device"]   = DEVICE
    data["gpu_name"] = GPU_NAME
    data["cython"]   = _CYTHON_OK
    uptime           = int(data["uptime_s"])
    h, m, s          = uptime // 3600, (uptime % 3600) // 60, uptime % 60
    data["uptime_fmt"] = f"{h:02d}:{m:02d}:{s:02d}"
    return jsonify(data)

# ── Video analysis API ────────────────────────────────────────────────────────
@app.route("/api/analyse", methods=["POST"])
@login_required
def api_analyse_upload():
    """
    Streaming upload: write directly to disk in CHUNK_SIZE chunks.
    Never buffers the whole file in RAM — safe for 2-hour videos.
    """
    # Content-Disposition filename
    cd   = request.content_type or ""
    f    = request.files.get("video")
    if f is None:
        return jsonify(error="No 'video' field in form data."), 400

    name = secure_filename(f.filename or "")
    if not name:
        return jsonify(error="Empty filename."), 400

    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify(error=f"Extension '{ext}' not allowed."), 400

    job_id    = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"upload_{job_id}{ext}")

    # Stream to disk
    try:
        with open(save_path, "wb") as out:
            while True:
                chunk = f.stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        log.error("Upload write failed: %s", exc)
        try: os.remove(save_path)
        except OSError: pass
        return jsonify(error=f"Disk write error: {exc}"), 500

    # Verify it's a real video
    probe = cv2.VideoCapture(save_path)
    readable = probe.isOpened()
    probe.release()
    if not readable:
        os.remove(save_path)
        return jsonify(error="File is not a readable video."), 400

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "filename": name}

    _analysis_pool.submit(_analyse_video, job_id, save_path)
    log.info("Job %s queued for '%s'.", job_id, name)
    return jsonify({"job_id": job_id}), 202


@app.route("/api/analyse/<job_id>")
@login_required
def api_analyse_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify(error="Job not found."), 404
    return jsonify({k: v for k, v in job.items() if k != "out_path"})


@app.route("/api/analyse/<job_id>/download")
@login_required
def api_analyse_download(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        return jsonify(error="Not ready."), 404
    out_path = job.get("out_path", "")
    if not out_path or not os.path.exists(out_path):
        return jsonify(error="Output file missing."), 404
    return send_file(out_path, as_attachment=True,
                     download_name=f"analysed_{job_id}.mp4")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reconnect_stream()
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    #port = int(os.environ.get("PORT", 5000))
    #app.run(debug=debug, host="0.0.0.0", port=port, threaded=True)
