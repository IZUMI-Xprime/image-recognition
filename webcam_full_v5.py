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
MAX_ABSENT_FRAMES = int(os.environ.get("MAX_ABSENT_FRAMES",  "90"))   # 90 ≈ 10s at 9fps
CONF_THRESHOLD    = float(os.environ.get("CONF_THRESHOLD",   "0.40"))
FRAME_SKIP        = int(os.environ.get("FRAME_SKIP",         "3"))

# ByteTrack buffer — how many frames a track survives without a match
# At 8-9 fps, 90 frames = ~10 seconds before an individual is "forgotten"
# Increase this if the same person re-entering is being counted twice
TRACK_BUFFER      = int(os.environ.get("TRACK_BUFFER", "90"))

os.makedirs(VIDEO_DIR,  exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Write a custom ByteTrack config with extended track_buffer so tracks
# survive longer between frames — prevents re-entry being counted as new person
_BYTETRACK_CFG = "bytetrack_sentinel.yaml"
with open(_BYTETRACK_CFG, "w") as _f:
    _f.write(f"""# Sentinel ByteTrack config — extended buffer for slow cameras
tracker_type: bytetrack
track_high_thresh: 0.5      # min confidence to start a new track
track_low_thresh: 0.1       # low-confidence detections still used for matching
new_track_thresh: 0.6       # min confidence to create a brand-new track
track_buffer: {TRACK_BUFFER} # frames a track survives without matching (90 = ~10s at 9fps)
match_thresh: 0.8           # IoU threshold for first-round matching
fuse_score: true
""")

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
# YOLO segmentation models
# yolov8s-seg (small) gives significantly better mask accuracy than nano.
# Two separate instances: live stream and video analysis never share a lock.
# ─────────────────────────────────────────────────────────────────────────────
_SEG_MODEL     = os.environ.get("SEG_MODEL", "yolov8s-seg.pt")   # override via env
model_stream   = YOLO(_SEG_MODEL)   # live webcam feed
model_analysis = YOLO(_SEG_MODEL)   # uploaded video analysis
model_stream.to(DEVICE)
model_analysis.to(DEVICE)
_stream_lock   = Lock()               # guards model_stream only
_analysis_lock = Lock()               # guards model_analysis only

# aliases so heatmap route works unchanged
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
_state_lock      = Lock()
animal_counts    = {}          # {label: unique_individual_count}
seen_track_ids   = set()       # track IDs already counted — never double-count
last_email_time  = 0.0
cap              = None
video_writer     = None
clip_start_time  = None
last_clip_path   = None

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
    Run ByteTrack segmentation on a list of frames using model_analysis.
    Uses model_analysis.track() with persist=True so track IDs are consistent
    across the entire video — same individual = same ID = counted once.

    Returns list of (track_dets, masks_xy) per frame:
      track_dets : [(label, track_id, x1,y1,x2,y2, conf), …]
      masks_xy   : [np.ndarray(N,2) or None, …]
    """
    all_results = []
    with _analysis_lock:
        for frame in frames:
            # track() with persist=True maintains ByteTrack state across calls
            results = model_analysis.track(
                frame, verbose=False, conf=CONF_THRESHOLD, device=DEVICE,
                tracker=_BYTETRACK_CFG, persist=True,
            )
            track_dets = []
            masks_xy   = []

            if results and results[0].boxes.id is not None:
                result    = results[0]
                boxes     = result.boxes
                has_masks = result.masks is not None

                for j in range(len(boxes)):
                    cls_id   = int(boxes.cls[j])
                    label    = model_analysis.names.get(cls_id, str(cls_id))
                    track_id = int(boxes.id[j])
                    conf     = float(boxes.conf[j])
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[j])
                    track_dets.append((label, track_id, x1, y1, x2, y2, conf))
                    masks_xy.append(
                        result.masks.xy[j]
                        if has_masks and j < len(result.masks.xy)
                        else None
                    )

            all_results.append((track_dets, masks_xy))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Video analysis worker
# ─────────────────────────────────────────────────────────────────────────────
def _analyse_video(job_id: str, video_path: str) -> None:
    """
    Process uploaded video with ByteTrack + seen_track_ids.
    Same counting logic as the live stream:
      - Each track_id counted exactly once per species
      - Unique colour per track_id so viewer can verify visually
      - Label shows species + track ID + confidence
    """
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

        # ── Counting state (mirrors live stream logic exactly) ─────────────────
        job_seen_ids: set[int]      = set()        # track IDs already counted
        job_counts:   dict[str,int] = {}           # {label: unique_count}

        frame_idx   = 0
        frames_done = 0

        # ByteTrack needs persist=True and processes one frame at a time
        # (batch tracking breaks track continuity — each batch would restart IDs)
        while True:
            ret, frame = cap_v.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Run ByteTrack on this single frame (persist keeps state across calls)
            track_dets, masks_xy = _infer_batch([frame])[0]

            for idx, (label, track_id, x1, y1, x2, y2, conf) in enumerate(track_dets):
                colour = _track_colour(track_id)
                mask   = masks_xy[idx] if idx < len(masks_xy) else None

                # Draw segmentation polygon or fallback box
                disp_label = f"{label} #{track_id} {conf:.2f}"
                if mask is not None and len(mask) >= 3:
                    _draw_seg(frame, mask, colour, disp_label)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, disp_label,
                                (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                colour, 1, cv2.LINE_AA)

                # Count only on first sighting of this track ID
                if track_id not in job_seen_ids:
                    job_seen_ids.add(track_id)
                    job_counts[label] = job_counts.get(label, 0) + 1

            # Burn running unique counts into top-left corner
            y_off = 24
            cv2.putText(frame, "Unique individuals:",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (200, 200, 200), 1, cv2.LINE_AA)
            for lbl, cnt in sorted(job_counts.items(), key=lambda x: x[1], reverse=True):
                y_off += 19
                colour_lbl = (200, 200, 200)
                cv2.putText(frame, f"  {lbl}: {cnt}",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                            0.44, colour_lbl, 1, cv2.LINE_AA)

            writer.write(frame)
            frames_done += 1

            # Progress
            pct = int(frame_idx / max(total_frames, 1) * 100)
            with _jobs_lock:
                _jobs[job_id]["progress"] = pct

        cap_v.release()
        writer.release()

        species_data = [
            {"species": lbl, "count": cnt}
            for lbl, cnt in sorted(job_counts.items(),
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
        log.info("Job %s done — %d unique individuals: %s",
                 job_id, sum(job_counts.values()), species_data)

        # ── Send annotated video to Telegram ──────────────────────────────────
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            caption_lines = ["🦅 *Video Analysis Complete*\n"]
            caption_lines.append(f"Total unique individuals: *{sum(job_counts.values())}*")
            caption_lines.append(f"Species found: *{len(species_data)}*\n")
            for sp in species_data:
                caption_lines.append(f"• {sp['species'].capitalize()}: {sp['count']}")
            caption = "\n".join(caption_lines)
            _io_pool.submit(_send_telegram_video, out_path, caption)

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
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
    """Send a video clip to Telegram (no caption — used for live clips)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_telegram_async(path))
    finally:
        loop.close()


async def _telegram_video_async(path: str, caption: str) -> None:
    """Async Telegram video send with a caption string."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    from telegram import Bot, InputFile
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        with open(path, "rb") as f:
            await bot.send_video(
                chat_id=TELEGRAM_CHAT_ID,
                video=InputFile(f, filename=os.path.basename(path)),
                caption=caption,
                parse_mode="Markdown",
            )
        log.info("Telegram: analysis video sent.")
    except Exception as exc:
        log.error("Telegram analysis send failed: %s", exc)


def _send_telegram_video(path: str, caption: str) -> None:
    """Thread-safe Telegram sender with caption — used for analysis results."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_telegram_video_async(path, caption))
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


def _draw_seg(frame: np.ndarray,
              mask_xy: np.ndarray,
              colour: tuple,
              label: str,
              alpha: float = 0.35) -> None:
    """
    Draw segmentation polygon + semi-transparent fill + clearly visible label.
    Label has a solid background pill so it's readable on any background.
    """
    if mask_xy is None or len(mask_xy) < 3:
        return

    h, w = frame.shape[:2]
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))

    # ── Semi-transparent fill ─────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Solid outline ─────────────────────────────────────────────────────────
    cv2.polylines(frame, [pts], isClosed=True, color=colour,
                  thickness=2, lineType=cv2.LINE_AA)

    # ── Label with solid background pill ─────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness  = 1
    padding    = 4

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Place label above the topmost mask point, clamped inside frame
    top_pt = pts[pts[:, 0, 1].argmin()][0]
    lx = int(np.clip(top_pt[0], 0, w - tw - padding * 2 - 2))
    ly = int(np.clip(top_pt[1] - 10, th + padding + 4, h - padding - 4))

    # Solid background rectangle
    cv2.rectangle(frame,
                  (lx - padding, ly - th - padding),
                  (lx + tw + padding, ly + baseline + padding),
                  colour, cv2.FILLED)

    # White text on coloured background — always readable
    cv2.putText(frame, label,
                (lx, ly),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def generate_frames():
    """
    Live MJPEG stream with:
    • ByteTrack persistent IDs  — same person across frames = same ID, never double-counted
    • cam_active gate           — pauses when user hits Stop
    • perf metrics              — FPS, inference ms, GPU util updated every frame
    • max FPS on GPU            — no artificial sleep, JPEG quality tuned for throughput
    """
    global cap, animal_counts, seen_track_ids, video_writer, clip_start_time, last_clip_path

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
                tracker=_BYTETRACK_CFG,   # custom config with extended track_buffer
                persist=True,
            )

        infer_ms = (time.perf_counter() - t0) * 1000

        # ── Count only NEW track IDs — same ID seen again = same individual ──
        new_detections: dict[str, int] = {}
        frame_has_detections = False

        if track_results and track_results[0].boxes.id is not None:
            result   = track_results[0]
            boxes    = result.boxes
            masks    = result.masks   # None if model didn't return masks

            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i])
                label    = model_stream.names.get(cls_id, "?")
                conf     = float(boxes.conf[i])
                track_id = int(boxes.id[i])
                colour   = _track_colour(track_id)

                frame_has_detections = True

                if masks is not None and i < len(masks.xy):
                    # Segmentation path — polygon outline + fill
                    _draw_seg(frame, masks.xy[i], colour,
                              f"{label} #{track_id} {conf:.2f}")
                else:
                    # Fallback to box if mask unavailable for this detection
                    x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{label} #{track_id} {conf:.2f}",
                                (x1, max(y1 - 8, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1,
                                cv2.LINE_AA)

                # Only count this individual if we've NEVER seen this track ID
                with _state_lock:
                    if track_id not in seen_track_ids:
                        seen_track_ids.add(track_id)
                        animal_counts[label] = animal_counts.get(label, 0) + 1
                        new_detections[label] = new_detections.get(label, 0) + 1

        # ── Timestamp overlay ─────────────────────────────────────────────────
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (w - 370, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 200), 1, cv2.LINE_AA)

        # ── Clip recording ────────────────────────────────────────────────────
        now = time.time()
        with _state_lock:
            writer_active = video_writer is not None

        if frame_has_detections and not writer_active:
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
                path = last_clip_path
                counts = dict(animal_counts)
                animal_counts.clear()
                seen_track_ids.clear()   # new clip = new session, reset seen IDs
                clip_start_time = None
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


def build_heatmap(frame: np.ndarray, masks_xy: list) -> str:
    """
    Build a detection heatmap overlaid on the camera frame.
    Saved at native camera resolution — CSS scales it to 50% of stream width.
    """
    h, w = frame.shape[:2]
    hmap = np.zeros((h, w), dtype=np.float32)

    for mask in masks_xy:
        if mask is None or len(mask) < 3:
            continue
        pts = mask.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(hmap, [pts], 1.0)

    sigma = max(w // 25, 8)
    if hmap.max() > 0:
        hmap = cv2.GaussianBlur(hmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
        hmap /= hmap.max()

    # BGR → RGB — critical, prevents colour inversion
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Figure at exact camera pixel size — CSS does the display scaling
    dpi = 72
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0b1218")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(frame_rgb, aspect="auto")
    ax.imshow(hmap, cmap="jet", alpha=0.5, vmin=0, vmax=1,
              interpolation="bilinear", aspect="auto")
    ax.axis("off")

    path = os.path.join(UPLOAD_DIR, "heatmap.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0,
                facecolor=fig.get_facecolor(), dpi=dpi)
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

# ─────────────────────────────────────────────────────────────────────────────
# CSS + HTML TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --white:      #ffffff;
  --bg:         #f5f4f1;
  --bg2:        #ffffff;
  --bg3:        #f0efe9;
  --border:     #e2e0d8;
  --border2:    #d0cdc3;
  --ink:        #1a1916;
  --ink2:       #4a4840;
  --ink3:       #8c897f;
  --sage:       #3d6b52;
  --sage-l:     #eef4f0;
  --sage-m:     #c5dccb;
  --amber:      #b5620a;
  --amber-l:    #fef3e2;
  --rose:       #b03a2e;
  --rose-l:     #fdf0ee;
  --blue:       #2c5f8a;
  --blue-l:     #edf3f9;
  --shadow:     0 1px 3px rgba(26,25,22,.06), 0 4px 16px rgba(26,25,22,.04);
  --shadow-md:  0 2px 8px rgba(26,25,22,.08), 0 8px 32px rgba(26,25,22,.06);
  --r:          10px;
  --r-sm:       6px;
  --font:       'DM Sans', sans-serif;
  --mono:       'DM Mono', monospace;
}

html,body{height:100%;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--sage);text-decoration:none}
a:hover{color:var(--ink)}

.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.5rem 1.2rem;border-radius:var(--r-sm);
     border:1px solid transparent;font-family:var(--font);font-size:.82rem;font-weight:500;
     cursor:pointer;transition:all .15s;white-space:nowrap;letter-spacing:.01em}
.btn-primary{background:var(--sage);color:#fff;border-color:var(--sage)}
.btn-primary:hover{background:#2f5340;border-color:#2f5340}
.btn-primary:disabled{opacity:.45;cursor:not-allowed}
.btn-outline{background:var(--white);color:var(--ink2);border-color:var(--border2)}
.btn-outline:hover{background:var(--bg3);border-color:var(--border2);color:var(--ink)}
.btn-danger{background:var(--white);color:var(--rose);border-color:var(--border2)}
.btn-danger:hover{background:var(--rose-l);border-color:var(--rose)}

input,select{width:100%;padding:.6rem .9rem;border-radius:var(--r-sm);
  border:1px solid var(--border2);background:var(--white);color:var(--ink);
  font-family:var(--font);font-size:.9rem;outline:none;transition:border-color .15s,box-shadow .15s}
input:focus,select:focus{border-color:var(--sage);box-shadow:0 0 0 3px rgba(61,107,82,.12)}

.badge{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .65rem;border-radius:20px;
       font-size:.74rem;font-weight:500;letter-spacing:.02em}
.badge-sage{background:var(--sage-l);color:var(--sage);border:1px solid var(--sage-m)}
.badge-amber{background:var(--amber-l);color:var(--amber)}
.badge-rose{background:var(--rose-l);color:var(--rose)}
.badge-blue{background:var(--blue-l);color:var(--blue)}
.badge-neutral{background:var(--bg3);color:var(--ink2);border:1px solid var(--border)}

.card{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow)}
.card-head{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1.25rem;
           border-bottom:1px solid var(--border)}
.card-title{font-size:.78rem;font-weight:600;color:var(--ink2);letter-spacing:.04em;text-transform:uppercase}
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Sign In</title>
<style>""" + _BASE_CSS + """
body{display:flex;min-height:100vh;background:var(--bg)}

/* Left panel — illustration side */
.left-panel{
  flex:0 0 42%%;background:var(--sage);display:flex;flex-direction:column;
  justify-content:space-between;padding:3rem;position:relative;overflow:hidden
}
.left-panel::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 120%% 80%% at 110%% 110%%,rgba(255,255,255,.06) 0%%,transparent 60%%);
  pointer-events:none
}
.brand{display:flex;align-items:center;gap:.7rem}
.brand-icon{width:34px;height:34px;background:rgba(255,255,255,.15);border-radius:8px;
            display:flex;align-items:center;justify-content:center;font-size:1.1rem}
.brand-name{font-size:1rem;font-weight:600;color:#fff;letter-spacing:.02em}
.left-body{flex:1;display:flex;flex-direction:column;justify-content:center;padding:2rem 0}
.left-headline{font-size:2rem;font-weight:300;color:#fff;line-height:1.25;margin-bottom:1rem;letter-spacing:-.02em}
.left-headline strong{font-weight:600;display:block}
.left-sub{font-size:.88rem;color:rgba(255,255,255,.65);line-height:1.65;max-width:300px}
.left-dots{display:flex;gap:.5rem}
.left-dot{width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,.3)}
.left-dot.active{background:#fff}

/* Right panel — form */
.right-panel{flex:1;display:flex;align-items:center;justify-content:center;padding:2rem}
.form-card{width:100%%;max-width:380px}
.form-top{margin-bottom:2.5rem}
.form-title{font-size:1.5rem;font-weight:600;color:var(--ink);letter-spacing:-.02em;margin-bottom:.35rem}
.form-sub{font-size:.87rem;color:var(--ink3)}
.field{margin-bottom:1.2rem}
.field label{display:block;font-size:.8rem;font-weight:500;color:var(--ink2);margin-bottom:.4rem}
.submit{width:100%%;padding:.7rem;font-size:.9rem;font-weight:500;margin-top:.4rem}
.err{display:flex;align-items:center;gap:.5rem;padding:.6rem .9rem;background:var(--rose-l);
     border:1px solid rgba(176,58,46,.2);border-radius:var(--r-sm);color:var(--rose);
     font-size:.82rem;margin-bottom:1.2rem}

@media(max-width:700px){.left-panel{display:none}.right-panel{padding:1.5rem}}
@keyframes fadein{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
.form-card{animation:fadein .4s ease both}
</style></head><body>

<div class="left-panel">
  <div class="brand">
    <div class="brand-icon">🦅</div>
    <span class="brand-name">Sentinel</span>
  </div>
  <div class="left-body">
    <div class="left-headline">Wildlife<strong>Monitoring</strong>System</div>
    <p class="left-sub">Real-time detection, unique individual tracking, and deep video analysis — powered by YOLOv8 and ByteTrack.</p>
  </div>
  <div class="left-dots">
    <div class="left-dot active"></div>
    <div class="left-dot"></div>
    <div class="left-dot"></div>
  </div>
</div>

<div class="right-panel">
  <div class="form-card">
    <div class="form-top">
      <div class="form-title">Sign in</div>
      <div class="form-sub">Access your monitoring dashboard</div>
    </div>
    {% if error %}
    <div class="err">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/><path d="M8 5v3.5M8 11v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      {{ error }}
    </div>
    {% endif %}
    <form method="POST">
      <div class="field">
        <label>Username</label>
        <input type="text" name="username" autocomplete="username" required autofocus placeholder="admin">
      </div>
      <div class="field">
        <label>Password</label>
        <input type="password" name="password" autocomplete="current-password" required placeholder="••••••••">
      </div>
      <button class="btn btn-primary submit" type="submit">Continue</button>
    </form>
  </div>
</div>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# SHARED NAV (injected into dashboard + analyse + recordings)
# ─────────────────────────────────────────────────────────────────────────────
_NAV_CSS = """
nav{
  height:56px;background:var(--white);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 2rem;position:sticky;top:0;z-index:100;
  box-shadow:0 1px 0 var(--border)
}
.nav-brand{display:flex;align-items:center;gap:.6rem;font-weight:600;font-size:.95rem;color:var(--ink)}
.nav-icon{width:28px;height:28px;background:var(--sage);border-radius:7px;
          display:flex;align-items:center;justify-content:center;font-size:.85rem}
.nav-links{display:flex;align-items:center;gap:.2rem}
.nav-link{padding:.38rem .85rem;border-radius:var(--r-sm);font-size:.84rem;font-weight:500;
          color:var(--ink2);transition:.15s;cursor:pointer;text-decoration:none}
.nav-link:hover{background:var(--bg3);color:var(--ink)}
.nav-link.active{background:var(--sage-l);color:var(--sage)}
.nav-right{display:flex;align-items:center;gap:.8rem}
.nav-sep{width:1px;height:18px;background:var(--border)}
.device-pill{font-size:.74rem;font-weight:500;padding:.22rem .7rem;border-radius:20px;
             background:var(--blue-l);color:var(--blue);border:1px solid rgba(44,95,138,.15)}
.nav-logout{font-size:.82rem;color:var(--ink3);font-weight:500;text-decoration:none;
            padding:.38rem .7rem;border-radius:var(--r-sm);transition:.15s}
.nav-logout:hover{background:var(--bg3);color:var(--ink)}
"""

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Dashboard</title>
<style>""" + _BASE_CSS + _NAV_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh;background:var(--bg)}
main{padding:1.75rem 2rem;overflow-y:auto}

/* ── Stat cards ── */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}
.stat-card{background:var(--white);border:1px solid var(--border);border-radius:var(--r);
           padding:1.1rem 1.3rem;box-shadow:var(--shadow)}
.stat-label{font-size:.73rem;font-weight:600;color:var(--ink3);letter-spacing:.05em;
            text-transform:uppercase;margin-bottom:.5rem}
.stat-value{font-size:1.65rem;font-weight:600;color:var(--ink);line-height:1;letter-spacing:-.02em}
.stat-value.sage{color:var(--sage)}
.stat-value.amber{color:var(--amber)}
.stat-value.rose{color:var(--rose)}

/* ── Perf row ── */
.perf-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:.75rem;margin-bottom:1.5rem}
.perf-card{background:var(--white);border:1px solid var(--border);border-radius:var(--r);
           padding:.8rem 1rem;box-shadow:var(--shadow)}
.perf-label{font-size:.7rem;font-weight:600;color:var(--ink3);letter-spacing:.05em;
            text-transform:uppercase;margin-bottom:.3rem}
.perf-val{font-size:1rem;font-weight:600;font-family:var(--mono);color:var(--ink);letter-spacing:-.01em}
.perf-bar-bg{height:2px;background:var(--border);border-radius:1px;margin-top:.5rem}
.perf-bar{height:100%%;border-radius:1px;transition:width .6s}

/* ── Main grid ── */
.grid-main{display:grid;grid-template-columns:2fr 1fr;gap:1.25rem;align-items:start}
@media(max-width:960px){.grid-main{grid-template-columns:1fr}}

/* ── Left column ── */
.left-col{display:flex;flex-direction:column;gap:1.25rem}

/* ── Stream ── */
.stream-panel{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.stream-wrap{position:relative;background:#0f0f0d;overflow:hidden;line-height:0}
.stream-wrap img{display:block;width:100%%;height:auto}
.stream-overlay{position:absolute;top:0;left:0;right:0;display:flex;align-items:center;
                justify-content:space-between;padding:.6rem .9rem;pointer-events:none;
                background:linear-gradient(to bottom,rgba(0,0,0,.55),transparent)}
.rec-badge{display:flex;align-items:center;gap:.35rem;font-size:.72rem;font-weight:600;
           color:#fff;letter-spacing:.04em;font-family:var(--mono)}
.rec-dot{width:6px;height:6px;border-radius:50%;background:#ef4444;animation:blink 1.2s infinite}
@keyframes blink{0%%,100%%{opacity:1}50%%{opacity:.25}}
.stream-ts{font-family:var(--mono);font-size:.68rem;color:rgba(255,255,255,.5)}
.cam-stopped{display:none;align-items:center;gap:.4rem;font-size:.72rem;font-weight:600;
             color:var(--amber);letter-spacing:.04em}
.cam-stopped-dot{width:6px;height:6px;border-radius:50%;background:var(--amber)}
.stream-controls{display:flex;gap:.6rem;padding:.9rem 1.1rem;border-top:1px solid var(--border);flex-wrap:wrap}

/* ── Heatmap ── */
.heatmap-panel{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.heatmap-body{background:#0f0f0d;padding:.5rem;display:flex;justify-content:center}
.heatmap-body img{display:block;width:50%%;height:auto;border-radius:4px}

/* ── Right column ── */
.right-col{display:flex;flex-direction:column;gap:1.25rem}
.det-panel{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.det-list{max-height:480px;overflow-y:auto}
.det-list::-webkit-scrollbar{width:3px}
.det-list::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.det-row{display:flex;align-items:center;justify-content:space-between;padding:.65rem 1.25rem;
         border-bottom:1px solid var(--border);transition:background .12s}
.det-row:last-child{border-bottom:none}
.det-row:hover{background:var(--bg)}
.det-name{font-size:.85rem;font-weight:500;color:var(--ink);text-transform:capitalize}
.det-count{font-family:var(--mono);font-size:.82rem;font-weight:500;color:var(--sage);
           background:var(--sage-l);padding:.15rem .55rem;border-radius:20px;
           border:1px solid var(--sage-m)}
.det-empty{padding:2rem;text-align:center;color:var(--ink3);font-size:.85rem}
</style></head><body>

<nav>
  <div class="nav-brand">
    <div class="nav-icon">🦅</div>
    Sentinel
  </div>
  <div class="nav-links">
    <a href="/dashboard" class="nav-link active">Dashboard</a>
    <a href="/analyse"   class="nav-link">Analyse</a>
    <a href="/recordings" class="nav-link">Recordings</a>
  </div>
  <div class="nav-right">
    <span class="device-pill" id="gpu-badge">{{ device }}</span>
    <div class="nav-sep"></div>
    <a href="/logout" class="nav-logout">Sign out</a>
  </div>
</nav>

<main>
  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Unique detections</div>
      <div class="stat-value sage" id="stat-total">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Species</div>
      <div class="stat-value" id="stat-classes">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Clips saved</div>
      <div class="stat-value" id="stat-clips">—</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Camera</div>
      <div class="stat-value sage" id="stat-cam">Live</div>
    </div>
  </div>

  <!-- Performance -->
  <div class="perf-row">
    <div class="perf-card">
      <div class="perf-label">FPS</div>
      <div class="perf-val" id="perf-fps">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-fps" style="background:var(--sage);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">Infer ms</div>
      <div class="perf-val" id="perf-ms">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-ms" style="background:var(--blue);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">GPU mem</div>
      <div class="perf-val" id="perf-gmem">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-gmem" style="background:var(--amber);width:0%%"></div></div>
    </div>
    <div class="perf-card">
      <div class="perf-label">CPU</div>
      <div class="perf-val" id="perf-cpu">—</div>
      <div class="perf-bar-bg"><div class="perf-bar" id="bar-cpu" style="background:var(--blue);width:0%%"></div></div>
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
      <div class="perf-val" id="perf-tracker" style="font-size:.75rem">—</div>
    </div>
  </div>

  <!-- Main grid -->
  <div class="grid-main">
    <div class="left-col">

      <!-- Stream -->
      <div class="stream-panel">
        <div class="card-head">
          <div style="display:flex;align-items:center;gap:.75rem">
            <span class="card-title">Live feed</span>
            <span class="cam-stopped" id="stopped-badge"><span class="cam-stopped-dot"></span>Stopped</span>
          </div>
          <span class="badge badge-sage">ByteTrack</span>
        </div>
        <div class="stream-wrap">
          <img id="stream-img" src="/video_feed" alt="Live feed">
          <div class="stream-overlay">
            <div class="rec-badge" id="rec-badge"><span class="rec-dot"></span>REC</div>
            <div class="stream-ts" id="clock"></div>
          </div>
        </div>
        <div class="stream-controls">
          <button class="btn btn-danger" id="btn-stop"  onclick="camStop()">⏹ Stop</button>
          <button class="btn btn-primary" id="btn-start" onclick="camStart()" style="display:none">▶ Resume</button>
          <button class="btn btn-outline" onclick="takeSnapshot()">📷 Snapshot</button>
          <button class="btn btn-outline" onclick="refreshHeatmap()">Refresh heatmap</button>
          <button class="btn btn-outline" onclick="clearCounts()">Clear counts</button>
        </div>
      </div>

      <!-- Heatmap -->
      <div class="heatmap-panel">
        <div class="card-head">
          <span class="card-title">Detection heatmap</span>
          <span class="badge badge-neutral">Segmentation masks</span>
        </div>
        <div class="heatmap-body">
          <img id="heatmap-img" src="/heatmap" alt="Heatmap" onerror="this.style.display='none'">
        </div>
      </div>

    </div>

    <!-- Right: detections -->
    <div class="right-col">
      <div class="det-panel">
        <div class="card-head">
          <span class="card-title">Detections</span>
          <span class="badge badge-blue" id="badge-counts">0 species</span>
        </div>
        <div class="det-list" id="det-list">
          <div class="det-empty">Waiting for detections…</div>
        </div>
      </div>
    </div>
  </div>
</main>

<script>
function updateClock(){document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});}
setInterval(updateClock,1000);updateClock();

async function pollCounts(){
  try{
    const d=await(await fetch('/api/counts')).json();
    const e=Object.entries(d);
    const total=e.reduce((a,[,v])=>a+v,0);
    document.getElementById('stat-total').textContent=total||'0';
    document.getElementById('stat-classes').textContent=e.length||'0';
    document.getElementById('badge-counts').textContent=e.length+' species';
    const list=document.getElementById('det-list');
    if(!e.length){list.innerHTML='<div class="det-empty">Waiting for detections…</div>';return;}
    list.innerHTML=e.sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`<div class="det-row"><span class="det-name">${k}</span><span class="det-count">${v}</span></div>`)
      .join('');
  }catch(_){}
}

async function pollClips(){
  try{const d=await(await fetch('/api/clip_count')).json();document.getElementById('stat-clips').textContent=d.count||'0';}catch(_){}
}

async function pollPerf(){
  try{
    const p=await(await fetch('/api/perf')).json();
    const set=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};
    const bar=(id,v,mx)=>{const el=document.getElementById(id);if(el)el.style.width=Math.min(100,Math.max(0,v/mx*100))+'%%';};
    set('perf-fps',   p.fps+' fps');
    set('perf-ms',    p.infer_ms+' ms');
    set('perf-gmem',  p.gpu_mem_mb+' MB');
    set('perf-cpu',   p.cpu_pct+'%%');
    set('perf-ram',   Math.round(p.ram_mb)+' MB');
    set('perf-uptime',p.uptime_fmt);
    set('perf-frames',p.frame_count.toLocaleString());
    set('perf-tracker',p.cython?'Cython':'Pure Python');
    bar('bar-fps', p.fps, 30);
    bar('bar-ms',  Math.max(0,200-p.infer_ms), 200);
    bar('bar-gmem',p.gpu_mem_mb, 4096);
    bar('bar-cpu', 100-p.cpu_pct, 100);
    const badge=document.getElementById('gpu-badge');
    if(badge)badge.textContent=p.gpu_name||p.device;
  }catch(_){}
}

async function camStop(){
  await fetch('/api/cam/stop',{method:'POST'});
  document.getElementById('btn-stop').style.display='none';
  document.getElementById('btn-start').style.display='inline-flex';
  document.getElementById('stat-cam').textContent='Stopped';
  document.getElementById('stat-cam').className='stat-value amber';
  document.getElementById('stopped-badge').style.display='flex';
  document.getElementById('rec-badge').style.opacity='.4';
}
async function camStart(){
  await fetch('/api/cam/start',{method:'POST'});
  document.getElementById('btn-stop').style.display='inline-flex';
  document.getElementById('btn-start').style.display='none';
  document.getElementById('stat-cam').textContent='Live';
  document.getElementById('stat-cam').className='stat-value sage';
  document.getElementById('stopped-badge').style.display='none';
  document.getElementById('rec-badge').style.opacity='1';
  const img=document.getElementById('stream-img');
  img.src='/video_feed?t='+Date.now();
}
function takeSnapshot(){window.open('/api/cam/snapshot','_blank');}
function refreshHeatmap(){const i=document.getElementById('heatmap-img');i.src='/heatmap?t='+Date.now();i.style.display='block';}
async function clearCounts(){await fetch('/api/clear_counts',{method:'POST'});pollCounts();}

(async()=>{
  try{const s=await(await fetch('/api/cam/status')).json();
    if(!s.active){
      document.getElementById('btn-stop').style.display='none';
      document.getElementById('btn-start').style.display='inline-flex';
      document.getElementById('stat-cam').textContent='Stopped';
      document.getElementById('stat-cam').className='stat-value amber';
      document.getElementById('stopped-badge').style.display='flex';
    }}catch(_){}
})();

setInterval(pollCounts,2000);
setInterval(pollClips, 5000);
setInterval(pollPerf,  1000);
pollCounts();pollClips();pollPerf();
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
ANALYSE_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Analyse</title>
<style>""" + _BASE_CSS + _NAV_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh;background:var(--bg)}
main{padding:1.75rem 2rem;overflow-y:auto;max-width:820px;margin:0 auto}
h1{font-size:1.3rem;font-weight:600;color:var(--ink);letter-spacing:-.02em;margin-bottom:.3rem}
.page-sub{font-size:.87rem;color:var(--ink3);margin-bottom:2rem}

/* Upload zone */
.drop-zone{
  border:1.5px dashed var(--border2);border-radius:var(--r);padding:2.5rem 2rem;
  text-align:center;cursor:pointer;background:var(--white);
  transition:border-color .15s,background .15s;position:relative;box-shadow:var(--shadow)
}
.drop-zone:hover,.drop-zone.drag{border-color:var(--sage);background:var(--sage-l)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%%;height:100%%}
.drop-icon{font-size:2rem;margin-bottom:.8rem;opacity:.45}
.drop-title{font-size:.95rem;font-weight:600;color:var(--ink);margin-bottom:.3rem}
.drop-sub{font-size:.82rem;color:var(--ink3)}
.drop-chosen{font-size:.82rem;font-weight:500;color:var(--sage);margin-top:.65rem;min-height:1rem}
.drop-speed{font-size:.78rem;color:var(--ink3);margin-top:.2rem;font-family:var(--mono);min-height:1rem}

.controls-row{display:flex;gap:.75rem;margin-top:1.1rem;align-items:center;flex-wrap:wrap}

/* Progress */
.prog-wrap{margin-top:1.5rem;display:none}
.prog-wrap.show{display:block}
.prog-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem}
.prog-label{font-size:.8rem;font-weight:600;color:var(--ink2)}
.prog-pct{font-size:.8rem;font-family:var(--mono);color:var(--ink3)}
.prog-bg{height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:.4rem}
.prog-bar{height:100%%;background:var(--sage);border-radius:2px;width:0%%;transition:width .4s}
.prog-stage{font-size:.78rem;color:var(--ink3)}

/* Results */
.results{margin-top:1.75rem;display:none}
.results.show{display:block}
.results-card{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
.sum-cell{background:var(--white);padding:1.1rem 1.3rem}
.sum-label{font-size:.72rem;font-weight:600;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.4rem}
.sum-value{font-size:1.5rem;font-weight:600;color:var(--sage);letter-spacing:-.02em}
.sum-value.plain{color:var(--ink)}

table{width:100%%;border-collapse:collapse}
th{padding:.6rem 1.25rem;text-align:left;font-size:.72rem;font-weight:600;color:var(--ink3);
   letter-spacing:.05em;text-transform:uppercase;border-bottom:1px solid var(--border)}
td{padding:.7rem 1.25rem;border-bottom:1px solid var(--border);font-size:.87rem;color:var(--ink)}
tr:last-child td{border:none}
tr:hover td{background:var(--bg)}
.bar-cell{width:35%%}
.bar-bg{height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;border:1px solid var(--border)}
.bar-fill{height:100%%;background:var(--sage);border-radius:3px;transition:width .7s}
.species-name{font-weight:500;text-transform:capitalize}
.count-badge{display:inline-flex;padding:.15rem .6rem;background:var(--sage-l);color:var(--sage);
             border:1px solid var(--sage-m);border-radius:20px;font-family:var(--mono);font-size:.8rem;font-weight:500}

.err-banner{margin:.75rem 1.25rem;padding:.7rem 1rem;background:var(--rose-l);border:1px solid rgba(176,58,46,.2);
            border-radius:var(--r-sm);color:var(--rose);font-size:.83rem;display:none}
.dl-row{display:flex;justify-content:flex-end;padding:.9rem 1.25rem;border-top:1px solid var(--border)}
</style></head><body>

<nav>
  <div class="nav-brand"><div class="nav-icon">🦅</div>Sentinel</div>
  <div class="nav-links">
    <a href="/dashboard"  class="nav-link">Dashboard</a>
    <a href="/analyse"    class="nav-link active">Analyse</a>
    <a href="/recordings" class="nav-link">Recordings</a>
  </div>
  <div class="nav-right">
    <span class="device-pill">{{ device }}</span>
    <div class="nav-sep"></div>
    <a href="/logout" class="nav-logout">Sign out</a>
  </div>
</nav>

<main>
  <h1>Video Analysis</h1>
  <p class="page-sub">Upload a video to count unique individuals per species — no double-counting across frames.</p>

  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="video/*">
    <div class="drop-icon">🎬</div>
    <div class="drop-title">Drop a video file or click to browse</div>
    <div class="drop-sub">MP4 · AVI · MOV · MKV · WEBM &nbsp;·&nbsp; No file size limit</div>
    <div class="drop-chosen" id="chosen-name"></div>
    <div class="drop-speed" id="upload-speed"></div>
  </div>

  <div class="controls-row">
    <button class="btn btn-primary" id="analyse-btn" onclick="startAnalysis()" disabled>
      Run analysis
    </button>
    <span class="badge badge-neutral">Tracks unique individuals — no repeat counting</span>
  </div>

  <div class="prog-wrap" id="prog-wrap">
    <div class="prog-header">
      <span class="prog-label" id="prog-label-text">Uploading…</span>
      <span class="prog-pct" id="prog-pct">0%%</span>
    </div>
    <div class="prog-bg"><div class="prog-bar" id="prog-bar"></div></div>
    <div class="prog-stage" id="prog-stage"></div>
  </div>

  <div class="results" id="results-section">
    <div class="results-card">
      <div class="card-head">
        <span class="card-title">Results</span>
        <span class="badge badge-sage">Complete</span>
      </div>
      <div class="summary-grid">
        <div class="sum-cell"><div class="sum-label">Unique animals</div><div class="sum-value" id="res-total">—</div></div>
        <div class="sum-cell"><div class="sum-label">Species found</div><div class="sum-value plain" id="res-species">—</div></div>
        <div class="sum-cell"><div class="sum-label">Frames analysed</div><div class="sum-value plain" id="res-frames">—</div></div>
        <div class="sum-cell"><div class="sum-label">Ran on</div><div class="sum-value plain" id="res-device" style="font-size:1rem">—</div></div>
      </div>
      <table>
        <thead><tr><th>#</th><th>Species</th><th>Unique individuals</th><th class="bar-cell">Distribution</th></tr></thead>
        <tbody id="species-tbody"></tbody>
      </table>
      <div class="err-banner" id="err-box"></div>
      <div class="dl-row" id="dl-row" style="display:none">
        <a id="dl-link" href="#" class="btn btn-outline" download>↓ Download annotated video</a>
      </div>
    </div>
  </div>
</main>

<script>
const JOB_KEY='sentinel_job_id';
let currentJobId=localStorage.getItem(JOB_KEY)||null;
let pollTimer=null;
const fileInput=document.getElementById('file-input');
const dropZone=document.getElementById('drop-zone');
const chosenName=document.getElementById('chosen-name');
const analyseBtn=document.getElementById('analyse-btn');
const speedEl=document.getElementById('upload-speed');

fileInput.addEventListener('change',()=>{
  if(fileInput.files[0]){chosenName.textContent=fileInput.files[0].name;analyseBtn.disabled=false;}
});
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag');});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop',e=>{
  e.preventDefault();dropZone.classList.remove('drag');
  if(e.dataTransfer.files[0]){
    const dt=new DataTransfer();dt.items.add(e.dataTransfer.files[0]);
    fileInput.files=dt.files;chosenName.textContent=e.dataTransfer.files[0].name;analyseBtn.disabled=false;
  }
});

async function startAnalysis(){
  if(!fileInput.files[0])return;
  analyseBtn.disabled=true;
  document.getElementById('results-section').classList.remove('show');
  document.getElementById('prog-wrap').classList.add('show');
  setProgress(0,'Uploading…','');
  try{
    const jobId=await uploadWithProgress(fileInput.files[0]);
    currentJobId=jobId;localStorage.setItem(JOB_KEY,jobId);
    document.getElementById('prog-bar').style.background='var(--blue)';
    setProgress(100,'Analysing…','Upload complete — running inference');
    pollTimer=setInterval(pollJob,1500);
  }catch(e){showError('Upload failed: '+e.message);analyseBtn.disabled=false;}
}

function uploadWithProgress(file){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();
    let lastLoaded=0,lastTime=Date.now();
    xhr.upload.addEventListener('progress',e=>{
      if(!e.lengthComputable)return;
      const pct=Math.round(e.loaded/e.total*100);
      const now=Date.now(),dt=(now-lastTime)/1000;
      const bps=(e.loaded-lastLoaded)/dt;
      lastLoaded=e.loaded;lastTime=now;
      const speed=bps>1e6?(bps/1e6).toFixed(1)+' MB/s':(bps/1e3).toFixed(0)+' KB/s';
      speedEl.textContent=speed;
      const eta=bps>0?Math.round((e.total-e.loaded)/bps)+'s':'…';
      setProgress(pct,'Uploading…',`${(e.loaded/1e6).toFixed(0)} / ${(e.total/1e6).toFixed(0)} MB  ·  ${speed}  ·  ETA ${eta}`);
    });
    xhr.addEventListener('load',()=>{
      speedEl.textContent='';
      if(xhr.status===202){try{resolve(JSON.parse(xhr.responseText).job_id);}catch(e){reject(new Error('Bad response'));}}
      else{let msg='Server error '+xhr.status;try{msg=JSON.parse(xhr.responseText).error||msg;}catch(_){}reject(new Error(msg));}
    });
    xhr.addEventListener('error',()=>reject(new Error('Network error')));
    xhr.addEventListener('timeout',()=>reject(new Error('Timed out')));
    xhr.timeout={{ timeout_ms }};
    const fd=new FormData();fd.append('video',file);
    xhr.open('POST','/api/analyse');xhr.send(fd);
  });
}

async function pollJob(){
  if(!currentJobId)return;
  try{
    const r=await fetch('/api/analyse/'+currentJobId);
    if(r.status===404){clearInterval(pollTimer);localStorage.removeItem(JOB_KEY);currentJobId=null;
      showError('Job not found — server may have restarted. Please re-upload.');analyseBtn.disabled=false;return;}
    const data=await r.json();
    const pct=data.progress||0;
    const stage=data.status==='processing'?`Frame ${Math.round(pct)}%% — inference running`:
                data.status==='done'?'Complete':data.status==='error'?'Error':'Queued…';
    setProgress(pct,'Analysing…',stage);
    if(data.status==='done'){clearInterval(pollTimer);localStorage.removeItem(JOB_KEY);showResults(data);analyseBtn.disabled=false;}
    else if(data.status==='error'){clearInterval(pollTimer);localStorage.removeItem(JOB_KEY);showError(data.error||'Unknown error');analyseBtn.disabled=false;}
  }catch(_){}
}

function setProgress(pct,label,stage){
  document.getElementById('prog-bar').style.width=pct+'%%';
  document.getElementById('prog-pct').textContent=pct+'%%';
  document.getElementById('prog-label-text').textContent=label;
  document.getElementById('prog-stage').textContent=stage;
}

function showResults(data){
  document.getElementById('results-section').classList.add('show');
  document.getElementById('err-box').style.display='none';
  document.getElementById('res-total').textContent=data.total_unique;
  document.getElementById('res-species').textContent=data.species.length;
  document.getElementById('res-frames').textContent=data.frames_analysed.toLocaleString();
  document.getElementById('res-device').textContent=data.device||'—';
  const max=data.species[0]?.count||1;
  document.getElementById('species-tbody').innerHTML=data.species.map((s,i)=>`
    <tr>
      <td style="color:var(--ink3);font-family:var(--mono)">${i+1}</td>
      <td><span class="species-name">${s.species}</span></td>
      <td><span class="count-badge">${s.count}</span></td>
      <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${Math.round(s.count/max*100)}%%"></div></div></td>
    </tr>`).join('');
  if(data.frames_analysed>0){
    document.getElementById('dl-link').href='/api/analyse/'+currentJobId+'/download';
    document.getElementById('dl-row').style.display='flex';
  }
}

function showError(msg){
  document.getElementById('results-section').classList.add('show');
  const box=document.getElementById('err-box');box.style.display='block';box.textContent='⚠ '+msg;
  document.getElementById('species-tbody').innerHTML='';
  document.getElementById('dl-row').style.display='none';
}

(async()=>{
  if(!currentJobId)return;
  try{
    const r=await fetch('/api/analyse/'+currentJobId);
    if(r.status===404){localStorage.removeItem(JOB_KEY);currentJobId=null;return;}
    const data=await r.json();
    if(data.status==='done'){document.getElementById('prog-wrap').classList.add('show');localStorage.removeItem(JOB_KEY);showResults(data);}
    else if(data.status==='error'){localStorage.removeItem(JOB_KEY);document.getElementById('prog-wrap').classList.add('show');showError(data.error||'Unknown error');}
    else{document.getElementById('prog-wrap').classList.add('show');setProgress(data.progress||0,'Analysing…','Resumed…');document.getElementById('prog-bar').style.background='var(--blue)';pollTimer=setInterval(pollJob,1500);}
  }catch(_){}
})();
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# RECORDINGS
# ─────────────────────────────────────────────────────────────────────────────
RECORDINGS_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentinel — Recordings</title>
<style>""" + _BASE_CSS + _NAV_CSS + """
body{display:grid;grid-template-rows:56px 1fr;min-height:100vh;background:var(--bg)}
main{padding:1.75rem 2rem;overflow-y:auto}
h1{font-size:1.3rem;font-weight:600;color:var(--ink);letter-spacing:-.02em;margin-bottom:.3rem}
.page-sub{font-size:.87rem;color:var(--ink3);margin-bottom:2rem}
.rec-card{background:var(--white);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow);overflow:hidden}
table{width:100%%;border-collapse:collapse}
th{padding:.65rem 1.25rem;text-align:left;font-size:.72rem;font-weight:600;color:var(--ink3);
   letter-spacing:.05em;text-transform:uppercase;border-bottom:1px solid var(--border);background:var(--bg)}
td{padding:.75rem 1.25rem;border-bottom:1px solid var(--border);font-size:.87rem;color:var(--ink)}
tr:last-child td{border:none}
tr:hover td{background:var(--bg)}
.fname{font-family:var(--mono);font-size:.81rem;color:var(--ink2)}
.fsize{font-family:var(--mono);font-size:.81rem;color:var(--ink3)}
.empty{padding:3rem;text-align:center;color:var(--ink3);font-size:.87rem}
</style></head><body>

<nav>
  <div class="nav-brand"><div class="nav-icon">🦅</div>Sentinel</div>
  <div class="nav-links">
    <a href="/dashboard"  class="nav-link">Dashboard</a>
    <a href="/analyse"    class="nav-link">Analyse</a>
    <a href="/recordings" class="nav-link active">Recordings</a>
  </div>
  <div class="nav-right">
    <a href="/logout" class="nav-logout">Sign out</a>
  </div>
</nav>

<main>
  <h1>Recordings</h1>
  <p class="page-sub">Auto-saved clips from the live stream — recorded when animals are detected.</p>
  <div class="rec-card">
    {% if videos %}
    <table>
      <thead><tr><th>#</th><th>Filename</th><th>Size</th><th></th></tr></thead>
      <tbody>
        {% for v in videos %}
        <tr>
          <td style="color:var(--ink3);font-family:var(--mono);width:3rem">{{ loop.index }}</td>
          <td><span class="fname">{{ v.name }}</span></td>
          <td><span class="fsize">{{ v.size }}</span></td>
          <td style="text-align:right">
            <a href="/download/{{ v.name }}" class="btn btn-outline" style="padding:.3rem .8rem;font-size:.78rem">
              ↓ Download
            </a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty">No recordings yet. Clips are saved automatically when animals are detected on the live feed.</div>
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

    # Extract polygon masks from segmentation results
    masks_xy = []
    for r in results:
        if r.masks is not None:
            for poly in r.masks.xy:
                masks_xy.append(poly)
        else:
            # model returned no masks — fall back to box-centre dot
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # synthesise a small polygon around centroid
                r2 = max((x2 - x1) // 4, 8)
                masks_xy.append(np.array([
                    [cx - r2, cy - r2], [cx + r2, cy - r2],
                    [cx + r2, cy + r2], [cx - r2, cy + r2],
                ], dtype=np.float32))

    return send_file(build_heatmap(frame, masks_xy), mimetype="image/png")

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
        seen_track_ids.clear()   # reset so new individuals are counted fresh
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
    app.run(debug=debug, host="0.0.0.0", port=5000, threaded=True)
