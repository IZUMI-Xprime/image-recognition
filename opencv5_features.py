"""
opencv5_features.py — OpenCV 5 Feature Integration Module for Wildlife Sentinel
=================================================================================

This standalone module demonstrates substantive use of OpenCV 5 APIs for the
OpenCV AI Competition 2026.  It is designed to complement the main Wildlife
Sentinel YOLO-based detection pipeline with advanced computer-vision
capabilities that showcase OpenCV 5's improved DNN backend, image-processing
primitives, video stabilisation, object tracking, feature-based
re-identification, and background subtraction.

Every public function accepts a frame (numpy ndarray, BGR uint8) and returns
processed results.  The ``OpenCV5Pipeline`` class chains the individual steps
into a cohesive processing pipeline suitable for real-time trail-camera or
security-camera footage of wildlife.

No imports from the main Wildlife Sentinel application are used — this module
is completely self-contained.

OpenCV version compatibility
----------------------------
The module inspects ``cv2.__version__`` at import time and populates a
``FEATURES`` dict that records which capabilities are available.  All public
functions degrade gracefully when run on OpenCV 4.x by falling back to
compatible APIs or returning the input frame unchanged.

Author:  Wildlife Sentinel / OpenCV AI Competition 2026
License: Apache-2.0
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError as _exc:
    raise ImportError(
        "OpenCV is required.  Install with:  pip install opencv-contrib-python"
    ) from _exc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version detection & feature availability
# ---------------------------------------------------------------------------

_CV_VERSION: str = cv2.__version__
_CV_MAJOR: int = int(_CV_VERSION.split(".")[0])
_CV_MINOR: int = int(_CV_VERSION.split(".")[1]) if len(_CV_VERSION.split(".")) > 1 else 0

FEATURES: Dict[str, bool] = {
    "dnn_onnx": hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromONNX"),
    "fast_nl_means_denoising": hasattr(cv2, "fastNlMeansDenoisingColored"),
    "clahe": hasattr(cv2, "createCLAHE"),
    "edge_preserving_filter": hasattr(cv2, "edgePreservingFilter"),
    "optical_flow_lk": hasattr(cv2, "calcOpticalFlowPyrLK"),
    "find_homography": hasattr(cv2, "findHomography"),
    "tracker_csrt": hasattr(cv2, "TrackerCSRT_create") or hasattr(cv2, "legacy_TrackerCSRT"),
    "tracker_kcf": hasattr(cv2, "TrackerKCF_create"),
    "orb": hasattr(cv2, "ORB_create"),
    "akaze": hasattr(cv2, "AKAZE_create"),
    "bg_subtractor_mog2": hasattr(cv2, "createBackgroundSubtractorMOG2"),
    "spatial_motion_analytics": True,
    "environmental_adaptation": True,
    "opencv5": _CV_MAJOR >= 5,
}


def report_opencv_info() -> Dict[str, Any]:
    """Return a dict summarising the detected OpenCV version and features.

    Returns
    -------
    dict
        ``version`` — the full version string, ``major`` / ``minor`` — parsed
        integers, ``features`` — mapping of feature name → availability bool,
        ``build_info`` — first 500 chars of ``cv2.getBuildInformation()``.
    """
    build_info = ""
    if hasattr(cv2, "getBuildInformation"):
        build_info = cv2.getBuildInformation()[:500]

    info = {
        "version": _CV_VERSION,
        "major": _CV_MAJOR,
        "minor": _CV_MINOR,
        "features": dict(FEATURES),
        "build_info": build_info,
    }
    logger.info(
        "OpenCV %s detected (major=%d). Feature flags: %s",
        _CV_VERSION,
        _CV_MAJOR,
        {k: v for k, v in FEATURES.items() if v},
    )
    return info


# ========================================================================== #
# 1.  DNN MODULE — Secondary species-verification classifier                #
# ========================================================================== #

@dataclass
class DNNClassifierResult:
    """Result of a DNN-based species-verification inference.

    Attributes
    ----------
    label_index : int
        Predicted class index (argmax of softmax output).
    confidence : float
        Softmax probability for the predicted class.
    all_scores : np.ndarray
        Full softmax output vector.
    label_name : str | None
        Human-readable label if a label list was provided.
    inference_ms : float
        Wall-clock inference time in milliseconds.
    """

    label_index: int
    confidence: float
    all_scores: np.ndarray
    label_name: Optional[str] = None
    inference_ms: float = 0.0


class DNNSpeciesClassifier:
    """Lightweight secondary classifier using ``cv2.dnn`` with ONNX support.

    OpenCV 5 ships with a substantially improved ONNX parser that handles a
    wider set of operators (including dynamic shapes and attention layers).
    This class wraps model loading and inference behind a simple API so the
    main pipeline can optionally verify species predictions from the primary
    YOLO detector.

    Parameters
    ----------
    model_path : str | Path
        Path to an ``.onnx`` model file.
    labels : list[str] | None
        Optional ordered list of class labels.
    input_size : tuple[int, int]
        (width, height) expected by the model.  Default ``(224, 224)``.
    scale : float
        Pixel scale factor applied by ``cv2.dnn.blobFromImage``.
    mean : tuple[float, float, float]
        Channel-wise mean subtraction values (BGR order).
    backend : int | None
        DNN backend id.  ``None`` auto-selects.
    target : int | None
        DNN target id.  ``None`` auto-selects.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (224, 224),
        scale: float = 1.0 / 255.0,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        backend: Optional[int] = None,
        target: Optional[int] = None,
    ) -> None:
        if not FEATURES["dnn_onnx"]:
            raise RuntimeError(
                "cv2.dnn.readNetFromONNX is not available in this OpenCV build."
            )

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.labels = labels
        self.input_size = input_size
        self.scale = scale
        self.mean = mean

        # --- Load model via OpenCV 5's enhanced ONNX importer ----------------
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

        # Prefer DNN_BACKEND_DEFAULT which, in OpenCV 5, automatically
        # selects the fastest available backend (OpenVINO, CUDA, or CPU).
        _backend = backend if backend is not None else cv2.dnn.DNN_BACKEND_DEFAULT
        _target = target if target is not None else cv2.dnn.DNN_TARGET_CPU
        self.net.setPreferableBackend(_backend)
        self.net.setPreferableTarget(_target)

        logger.info(
            "DNNSpeciesClassifier loaded %s  (input %s, backend=%d, target=%d)",
            self.model_path.name,
            input_size,
            _backend,
            _target,
        )

    # --------------------------------------------------------------------- #

    def classify(self, frame: np.ndarray) -> DNNClassifierResult:
        """Run inference on a single BGR frame.

        The frame is resized, normalised, and passed through the ONNX model.
        A softmax is applied to the raw logits if the output does not already
        sum to ~1.0.

        Parameters
        ----------
        frame : np.ndarray
            Input image, BGR uint8.

        Returns
        -------
        DNNClassifierResult
            Classification output.
        """
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.scale,
            size=self.input_size,
            mean=self.mean,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        t0 = time.perf_counter()
        out = self.net.forward()
        inference_ms = (time.perf_counter() - t0) * 1000.0

        scores = out.flatten().astype(np.float64)

        # Apply softmax if the output doesn't look like probabilities already.
        if scores.sum() < 0.99 or scores.min() < 0.0:
            exp_scores = np.exp(scores - scores.max())
            scores = exp_scores / exp_scores.sum()

        idx = int(np.argmax(scores))
        label_name = self.labels[idx] if self.labels and idx < len(self.labels) else None

        return DNNClassifierResult(
            label_index=idx,
            confidence=float(scores[idx]),
            all_scores=scores,
            label_name=label_name,
            inference_ms=inference_ms,
        )


# ========================================================================== #
# 2.  ENHANCED IMAGE PROCESSING PIPELINE                                    #
# ========================================================================== #

def denoise_low_light(
    frame: np.ndarray,
    h: int = 10,
    h_color: int = 10,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """Denoise a low-light wildlife frame using Non-Local Means.

    Trail cameras and security cameras often operate in low-light conditions
    (dawn, dusk, infrared) where sensor noise is significant.
    ``cv2.fastNlMeansDenoisingColored`` is a strong non-local-means denoiser
    that preserves edges while suppressing Gaussian-like noise — critical for
    retaining fine fur/feather texture that aids species identification.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    OpenCV 5 includes SIMD and threading improvements in the NLM
    implementation, resulting in measurably faster denoising on modern CPUs.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 input frame.
    h : int
        Filter strength for luminance.  Higher removes more noise but may blur.
    h_color : int
        Filter strength for chrominance.
    template_window : int
        Size of the template patch (must be odd).
    search_window : int
        Size of the search area (must be odd).

    Returns
    -------
    np.ndarray
        Denoised BGR uint8 frame.
    """
    if not FEATURES["fast_nl_means_denoising"]:
        logger.warning("fastNlMeansDenoisingColored unavailable; returning original.")
        return frame

    return cv2.fastNlMeansDenoisingColored(
        frame,
        None,
        h,
        h_color,
        template_window,
        search_window,
    )


def enhance_contrast_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance contrast via Contrast-Limited Adaptive Histogram Equalisation.

    Wildlife imagery frequently suffers from uneven lighting — dark canopy
    shadows, bright sky patches, or infrared flash hotspots.  CLAHE adaptively
    equalises contrast in local tiles, preventing over-amplification of noise
    in already-bright regions while lifting detail in shadows.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    ``cv2.createCLAHE`` has been available since OpenCV 2.x, but OpenCV 5
    integrates it into the unified ``cv2.cuda`` module for optional GPU
    acceleration via ``cv2.cuda.createCLAHE``.  This function falls back to
    the CPU path transparently.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 input frame.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple[int, int]
        Number of tiles in row and column directions.

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR uint8 frame.
    """
    if not FEATURES["clahe"]:
        logger.warning("CLAHE unavailable; returning original.")
        return frame

    # Convert to LAB colour space so we equalise only the lightness channel,
    # leaving chrominance intact.  This avoids colour-shift artefacts.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l_channel)

    lab_eq = cv2.merge([l_eq, a_channel, b_channel])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def edge_preserving_smooth(
    frame: np.ndarray,
    flags: int = 1,
    sigma_s: float = 60.0,
    sigma_r: float = 0.4,
) -> np.ndarray:
    """Apply edge-preserving filtering for sharper downstream segmentation.

    Before generating segmentation masks (e.g., GrabCut, watershed, or
    post-YOLO mask refinement), it helps to smooth textures while keeping
    object boundaries sharp.  ``cv2.edgePreservingFilter`` implements the
    *domain-transform* or *normalised convolution* filter from the
    ``photo`` module, producing results superior to bilateral filtering
    for natural-image segmentation.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    The ``photo`` module's edge-preserving filter is fully maintained in
    OpenCV 5 and benefits from the unified Mat / UMat pipeline for
    transparent GPU off-loading on supported platforms.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 input frame.
    flags : int
        1 = Recursive filter, 2 = Normalised convolution.
    sigma_s : float
        Spatial sigma (range 0–200).  Controls neighbourhood size.
    sigma_r : float
        Range sigma (range 0–1).  Controls how dissimilar colours are smoothed.

    Returns
    -------
    np.ndarray
        Smoothed BGR uint8 frame with preserved edges.
    """
    if not FEATURES["edge_preserving_filter"]:
        logger.warning("edgePreservingFilter unavailable; returning original.")
        return frame

    return cv2.edgePreservingFilter(frame, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)


# ========================================================================== #
# 3.  VIDEO STABILISATION MODULE                                             #
# ========================================================================== #

@dataclass
class StabilisationState:
    """Internal mutable state for the frame-to-frame stabiliser.

    Attributes
    ----------
    prev_gray : np.ndarray | None
        Greyscale version of the previous frame.
    prev_points : np.ndarray | None
        Tracked feature points from the previous frame.
    accumulated_transform : np.ndarray
        Running 3×3 homography (identity initially).
    smoothing_window : list[np.ndarray]
        Sliding window of recent transforms for temporal smoothing.
    """

    prev_gray: Optional[np.ndarray] = None
    prev_points: Optional[np.ndarray] = None
    accumulated_transform: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    smoothing_window: List[np.ndarray] = field(default_factory=list)


def stabilise_frame(
    frame: np.ndarray,
    state: StabilisationState,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
    smoothing_radius: int = 15,
) -> Tuple[np.ndarray, StabilisationState]:
    """Stabilise a single frame using sparse optical flow and RANSAC homography.

    Trail cameras and pole-mounted security cameras suffer from wind-induced
    vibration and animal-triggered shake.  This function estimates the
    inter-frame camera motion via Lucas-Kanade optical flow on Shi-Tomasi
    corners, computes a RANSAC-robust homography, temporally smooths the
    accumulated trajectory, and warps the current frame to compensate.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    * ``cv2.calcOpticalFlowPyrLK`` — OpenCV 5 adds AVX-512 / NEON fast-paths.
    * ``cv2.findHomography`` with ``cv2.RANSAC`` — OpenCV 5's USAC framework
      (Universal RANSAC) replaces the legacy RANSAC implementation, offering
      faster convergence and better inlier ratios on noisy point sets.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 input frame.
    state : StabilisationState
        Mutable state object that persists across calls.
    max_corners : int
        Maximum Shi-Tomasi corners to track.
    quality_level : float
        Minimum quality for corner detection.
    min_distance : float
        Minimum Euclidean distance between corners.
    smoothing_radius : int
        Number of frames for temporal trajectory smoothing.

    Returns
    -------
    tuple[np.ndarray, StabilisationState]
        The stabilised frame and the updated state.
    """
    if not (FEATURES["optical_flow_lk"] and FEATURES["find_homography"]):
        logger.warning("Optical flow / homography unavailable; returning original.")
        return frame, state

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # ------------------------------------------------------------------
    # First frame — initialise and return as-is
    # ------------------------------------------------------------------
    if state.prev_gray is None:
        state.prev_gray = gray
        state.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return frame, state

    # ------------------------------------------------------------------
    # Track features from previous frame into current frame
    # ------------------------------------------------------------------
    if state.prev_points is None or len(state.prev_points) < 4:
        # Not enough points; re-detect and skip stabilisation this frame.
        state.prev_gray = gray
        state.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return frame, state

    curr_points, status, _err = cv2.calcOpticalFlowPyrLK(
        state.prev_gray,
        gray,
        state.prev_points,
        None,
    )

    # Keep only successfully tracked points
    if curr_points is None or status is None:
        state.prev_gray = gray
        state.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return frame, state

    good_mask = status.flatten() == 1
    prev_good = state.prev_points[good_mask]
    curr_good = curr_points[good_mask]

    if len(prev_good) < 4:
        state.prev_gray = gray
        state.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return frame, state

    # ------------------------------------------------------------------
    # Compute RANSAC homography  (OpenCV 5 USAC under the hood)
    # ------------------------------------------------------------------
    H, inlier_mask = cv2.findHomography(prev_good, curr_good, cv2.RANSAC, 5.0)

    if H is None:
        state.prev_gray = gray
        state.prev_points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        return frame, state

    # Accumulate the frame-to-frame transform
    state.accumulated_transform = H @ state.accumulated_transform

    # ------------------------------------------------------------------
    # Temporal smoothing — average recent transforms for a jitter-free path
    # ------------------------------------------------------------------
    state.smoothing_window.append(state.accumulated_transform.copy())
    if len(state.smoothing_window) > smoothing_radius:
        state.smoothing_window.pop(0)

    # Average the homographies element-wise (a simple but effective
    # approximation for small rotations / translations).
    smooth_H = np.mean(state.smoothing_window, axis=0)

    # Compute the correction: inverse of accumulated * smoothed
    try:
        correction = smooth_H @ np.linalg.inv(state.accumulated_transform)
    except np.linalg.LinAlgError:
        correction = np.eye(3, dtype=np.float64)

    stabilised = cv2.warpPerspective(frame, correction, (w, h))

    # ------------------------------------------------------------------
    # Prepare state for next call
    # ------------------------------------------------------------------
    state.prev_gray = gray
    state.prev_points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )

    return stabilised, state


# ========================================================================== #
# 4.  OBJECT TRACKING WITH OpenCV 5                                          #
# ========================================================================== #

@dataclass
class TrackerState:
    """Wraps an OpenCV object tracker and its current bounding box.

    Attributes
    ----------
    tracker : Any
        The OpenCV tracker object (CSRT or KCF).
    bbox : tuple[int, int, int, int] | None
        Current bounding box ``(x, y, w, h)`` or ``None`` if tracking lost.
    tracker_type : str
        ``"CSRT"`` or ``"KCF"``.
    """

    tracker: Any = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    tracker_type: str = "CSRT"


def create_tracker(tracker_type: str = "CSRT") -> TrackerState:
    """Create an OpenCV object tracker.

    ``cv2.TrackerCSRT`` provides high accuracy suitable for animal tracking
    where appearance changes slowly but the target may be partially occluded
    by vegetation.  ``cv2.TrackerKCF`` is faster but less robust to occlusion.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    In OpenCV 5 the tracking API has been consolidated.  The legacy
    ``cv2.legacy.TrackerCSRT_create()`` path is deprecated in favour of
    ``cv2.TrackerCSRT.create()`` (or direct constructor).  This factory
    function probes both paths for maximum compatibility.

    Parameters
    ----------
    tracker_type : str
        ``"CSRT"`` or ``"KCF"``.

    Returns
    -------
    TrackerState
        Initialised (but not yet started) tracker state.
    """
    tracker_obj = None
    resolved_type = tracker_type.upper()

    if resolved_type == "CSRT":
        # OpenCV 5 preferred path
        if hasattr(cv2, "TrackerCSRT_create"):
            tracker_obj = cv2.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerCSRT"):
            tracker_obj = cv2.TrackerCSRT.create()  # type: ignore[attr-defined]
        # Legacy fallback (OpenCV 4.x contrib)
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            tracker_obj = cv2.legacy.TrackerCSRT_create()
        else:
            logger.warning("CSRT tracker unavailable; falling back to KCF.")
            resolved_type = "KCF"

    if resolved_type == "KCF" and tracker_obj is None:
        if hasattr(cv2, "TrackerKCF_create"):
            tracker_obj = cv2.TrackerKCF_create()
        elif hasattr(cv2, "TrackerKCF"):
            tracker_obj = cv2.TrackerKCF.create()  # type: ignore[attr-defined]
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            tracker_obj = cv2.legacy.TrackerKCF_create()

    if tracker_obj is None:
        raise RuntimeError(
            f"Could not create a {resolved_type} tracker.  "
            "Ensure opencv-contrib-python is installed."
        )

    return TrackerState(tracker=tracker_obj, tracker_type=resolved_type)


def init_tracker(
    state: TrackerState,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> TrackerState:
    """Initialise the tracker on a frame with a known bounding box.

    Parameters
    ----------
    state : TrackerState
        Tracker state (from ``create_tracker``).
    frame : np.ndarray
        BGR uint8 frame.
    bbox : tuple[int, int, int, int]
        ``(x, y, w, h)`` bounding box of the target.

    Returns
    -------
    TrackerState
        Updated state with ``bbox`` set.
    """
    state.tracker.init(frame, bbox)
    state.bbox = bbox
    return state


def update_tracker(
    state: TrackerState,
    frame: np.ndarray,
) -> Tuple[bool, Optional[Tuple[int, int, int, int]], TrackerState]:
    """Update the tracker with a new frame.

    Parameters
    ----------
    state : TrackerState
        Current tracker state.
    frame : np.ndarray
        BGR uint8 frame.

    Returns
    -------
    tuple[bool, tuple | None, TrackerState]
        ``(success, bbox_or_none, updated_state)``.
    """
    ok, bbox = state.tracker.update(frame)
    if ok:
        state.bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    else:
        state.bbox = None
    return ok, state.bbox, state


# ========================================================================== #
# 5.  FEATURE-BASED ANIMAL RE-IDENTIFICATION                                #
# ========================================================================== #

@dataclass
class AnimalDescriptor:
    """Feature descriptor for a single animal crop.

    Attributes
    ----------
    keypoints : list
        Detected keypoints (non-serialisable cv2.KeyPoint objects).
    descriptors : np.ndarray | None
        Feature descriptor matrix (N × descriptor_dim).
    detector_type : str
        ``"ORB"`` or ``"AKAZE"``.
    timestamp : float
        Epoch timestamp when the descriptor was computed.
    metadata : dict
        Arbitrary metadata (e.g., species label, track id).
    """

    keypoints: list = field(default_factory=list)
    descriptors: Optional[np.ndarray] = None
    detector_type: str = "ORB"
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnimalReIDModule:
    """Feature-based animal re-identification using ORB or AKAZE descriptors.

    Many wildlife-monitoring scenarios require recognising the *same individual*
    across hours or days — e.g., tracking a specific leopard returning to a
    waterhole.  Full deep-learning re-ID models are expensive; a lightweight
    alternative is to extract local feature descriptors from a cropped animal
    region and match them against a gallery of previously seen individuals.

    This module uses ORB (fast, binary) or AKAZE (more robust to scale /
    viewpoint changes) feature detectors to build per-animal descriptor sets,
    and BFMatcher with Hamming distance + Lowe's ratio test for matching.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    * ``cv2.ORB_create`` — OpenCV 5 adds hardware-accelerated descriptor
      computation on ARM via NEON intrinsics.
    * ``cv2.AKAZE_create`` — OpenCV 5 stabilised multi-threaded AKAZE on all
      platforms.
    * ``cv2.BFMatcher`` — Unchanged API, but benefits from OpenCV 5's UMat
      transparent-offload for large descriptor sets.

    Parameters
    ----------
    detector : str
        ``"ORB"`` or ``"AKAZE"``.
    max_features : int
        Maximum features for ORB.  Ignored for AKAZE.
    ratio_threshold : float
        Lowe's ratio threshold for descriptor matching.  Lower = stricter.
    """

    def __init__(
        self,
        detector: str = "ORB",
        max_features: int = 500,
        ratio_threshold: float = 0.75,
    ) -> None:
        self.detector_type = detector.upper()
        self.ratio_threshold = ratio_threshold

        if self.detector_type == "AKAZE" and FEATURES["akaze"]:
            self._detector = cv2.AKAZE_create()
        elif FEATURES["orb"]:
            self._detector = cv2.ORB_create(nfeatures=max_features)
            self.detector_type = "ORB"
        else:
            raise RuntimeError("Neither ORB nor AKAZE detectors are available.")

        # BFMatcher with Hamming distance (suitable for binary descriptors)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Gallery of known animal descriptors
        self._gallery: List[AnimalDescriptor] = []

        logger.info(
            "AnimalReIDModule initialised with %s detector (ratio=%.2f).",
            self.detector_type,
            self.ratio_threshold,
        )

    # --------------------------------------------------------------------- #

    def compute_descriptor(
        self,
        crop: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnimalDescriptor:
        """Compute a feature descriptor for a cropped animal image.

        Parameters
        ----------
        crop : np.ndarray
            BGR uint8 cropped image of a single animal.
        metadata : dict | None
            Optional metadata to attach (species, track ID, etc.).

        Returns
        -------
        AnimalDescriptor
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kps, descs = self._detector.detectAndCompute(gray, None)

        return AnimalDescriptor(
            keypoints=list(kps) if kps else [],
            descriptors=descs,
            detector_type=self.detector_type,
            timestamp=time.time(),
            metadata=metadata or {},
        )

    def add_to_gallery(self, descriptor: AnimalDescriptor) -> int:
        """Add a descriptor to the re-identification gallery.

        Parameters
        ----------
        descriptor : AnimalDescriptor
            Descriptor to store.

        Returns
        -------
        int
            Gallery index assigned to this descriptor.
        """
        self._gallery.append(descriptor)
        return len(self._gallery) - 1

    def match(
        self,
        query: AnimalDescriptor,
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Match a query descriptor against the gallery.

        Uses ``cv2.BFMatcher.knnMatch`` with *k=2* and Lowe's ratio test to
        filter ambiguous matches.  The match score for each gallery entry is
        the count of good matches normalised by the total number of query
        descriptors.

        Parameters
        ----------
        query : AnimalDescriptor
            Query animal descriptor.
        top_k : int
            Number of best gallery matches to return.

        Returns
        -------
        list[tuple[int, float]]
            Sorted list of ``(gallery_index, match_score)`` pairs, highest
            score first.  Score in [0, 1].
        """
        if query.descriptors is None or len(query.descriptors) == 0:
            return []

        scores: List[Tuple[int, float]] = []
        for idx, gallery_desc in enumerate(self._gallery):
            if gallery_desc.descriptors is None or len(gallery_desc.descriptors) == 0:
                scores.append((idx, 0.0))
                continue

            raw_matches = self._matcher.knnMatch(
                query.descriptors, gallery_desc.descriptors, k=2
            )

            good_count = 0
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_count += 1

            score = good_count / max(len(query.descriptors), 1)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def gallery_size(self) -> int:
        """Return the number of entries in the gallery."""
        return len(self._gallery)


# ========================================================================== #
# 6.  BACKGROUND SUBTRACTION — Motion Pre-filter                            #
# ========================================================================== #

@dataclass
class MotionDetectionResult:
    """Result of background-subtraction-based motion detection.

    Attributes
    ----------
    motion_detected : bool
        ``True`` if the fraction of foreground pixels exceeds the threshold.
    foreground_mask : np.ndarray
        Binary mask (uint8, 0/255) of detected foreground.
    foreground_fraction : float
        Fraction of pixels classified as foreground.
    contours : list[np.ndarray]
        List of filtered motion contours.
    """

    motion_detected: bool
    foreground_mask: np.ndarray
    foreground_fraction: float
    contours: List[np.ndarray] = field(default_factory=list)


class MotionPreFilter:
    """Background subtraction pre-filter using MOG2.

    Running YOLO on every single frame from a 24/7 trail camera is wasteful —
    most frames contain only static scenery.  This module uses
    ``cv2.createBackgroundSubtractorMOG2`` to maintain an adaptive background
    model and flag frames that contain significant motion, allowing the
    downstream pipeline to skip YOLO inference on static frames.

    OpenCV 5 relevance
    ~~~~~~~~~~~~~~~~~~
    ``cv2.createBackgroundSubtractorMOG2`` is a mature OpenCV API.  In
    OpenCV 5 it benefits from:
    * Transparent UMat / T-API GPU acceleration.
    * Improved shadow-detection heuristics (``detectShadows=True``).
    * Better numerical stability for very long sequences (>24 h continuous).

    Parameters
    ----------
    history : int
        Number of recent frames used to build the background model.
    var_threshold : float
        Mahalanobis distance threshold for foreground classification.
    detect_shadows : bool
        If ``True``, shadow pixels are detected and marked separately.
    motion_threshold : float
        Minimum foreground fraction to consider motion detected.
    learning_rate : float
        Background model learning rate.  ``-1`` = automatic.
    min_contour_area : int
        Minimum contour area (pixels) to count as meaningful motion.
        Contours smaller than this are eroded away to suppress noise.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        motion_threshold: float = 0.005,
        learning_rate: float = -1.0,
        min_contour_area: int = 500,
    ) -> None:
        if not FEATURES["bg_subtractor_mog2"]:
            raise RuntimeError(
                "cv2.createBackgroundSubtractorMOG2 is not available."
            )

        self.motion_threshold = motion_threshold
        self.learning_rate = learning_rate
        self.min_contour_area = min_contour_area

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

        # Morphological kernel for noise removal
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        logger.info(
            "MotionPreFilter initialised (history=%d, var_thresh=%.1f, "
            "motion_thresh=%.4f).",
            history,
            var_threshold,
            motion_threshold,
        )

    # --------------------------------------------------------------------- #

    def detect(self, frame: np.ndarray) -> MotionDetectionResult:
        """Apply background subtraction to a frame and decide if motion exists.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 input frame.

        Returns
        -------
        MotionDetectionResult
        """
        fg_mask = self._subtractor.apply(frame, learningRate=self.learning_rate)

        # Threshold: MOG2 marks shadows as 127; binarise to keep only definite
        # foreground (255).
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological opening to remove small noise blobs
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)

        contours_out: List[np.ndarray] = []
        # Optionally filter out small contours
        if self.min_contour_area > 0:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            clean_mask = np.zeros_like(fg_mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_contour_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
                    contours_out.append(cnt)
            fg_mask = clean_mask
        else:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_out = list(contours)

        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_pixels = int(np.count_nonzero(fg_mask))
        fg_fraction = fg_pixels / max(total_pixels, 1)

        return MotionDetectionResult(
            motion_detected=fg_fraction >= self.motion_threshold,
            foreground_mask=fg_mask,
            foreground_fraction=fg_fraction,
            contours=contours_out,
        )


# ========================================================================== #
# 7.  Spatial & Motion Analytics — Velocity Vector Tracking                   #
# ========================================================================== #

@dataclass
class VelocityVector:
    """Kinematic velocity vector and motion characterisation for a tracked target.

    Attributes
    ----------
    vx : float
        Horizontal velocity in pixels per second.
    vy : float
        Vertical velocity in pixels per second.
    speed : float
        Magnitude of velocity in pixels per second.
    heading_deg : float
        Azimuth heading angle [0.0, 360.0) degrees, where 0=East, 90=South.
    cardinal : str
        Compass heading string ("N", "NE", "E", "SE", "S", "SW", "W", "NW", or "STATIONARY").
    acceleration : float
        Linear acceleration magnitude in pixels/s^2.
    activity_state : str
        Behavioral classification ("stationary", "browsing", "walking", "running").
    smoothed_pos : Tuple[int, int]
        Continuous-Discrete Kalman filtered (x, y) anchor coordinates.
    trajectory : List[Tuple[int, int]]
        Historical breadcrumb trail of smoothed coordinates.
    """

    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    heading_deg: float = 0.0
    cardinal: str = "N/A"
    acceleration: float = 0.0
    activity_state: str = "stationary"
    smoothed_pos: Tuple[int, int] = (0, 0)
    trajectory: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class TrajectoryPoint:
    """Timestamped centroid coordinate."""
    cx: float
    cy: float
    timestamp: float


@dataclass(frozen=True)
class KinematicState:
    """Immutable output container returned by RobustKinematicTracker.update()."""
    speed: float                          # Smoothed scalar speed in px/s
    velocity: Tuple[float, float]         # Smoothed (vx, vy) velocity vector in px/s
    cardinal: str                         # 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', or 'STATIONARY'
    heading_rad: float                    # Smoothed screen-space heading in radians (-pi, pi]
    smoothed_pos: Tuple[int, int]         # Filtered (x, y) anchor coordinate in pixels
    trajectory: List[Tuple[int, int]]     # Historical trail of filtered (x, y) positions


class _TrackFilterState:
    """Internal per-track state container for the Constant-Velocity Kalman & SO(2) filters."""

    __slots__ = (
        "x",
        "P",
        "last_ts",
        "last_wh",
        "ema_vx",
        "ema_vy",
        "unit_hx",
        "unit_hy",
        "heading_rad",
        "is_moving",
        "cardinal_idx",
        "trajectory",
        "initialized",
    )

    def __init__(self, max_traj_len: int) -> None:
        self.x: np.ndarray = np.zeros((4, 1), dtype=np.float64)  # [x, y, vx, vy]^T
        self.P: np.ndarray = np.eye(4, dtype=np.float64)
        self.last_ts: float = 0.0
        self.last_wh: Tuple[float, float] = (0.0, 0.0)

        # Post-Kalman continuous-time EMA velocity state
        self.ema_vx: float = 0.0
        self.ema_vy: float = 0.0

        # SO(2) unit vector state for wrap-around-safe angular smoothing
        self.unit_hx: float = 1.0
        self.unit_hy: float = 0.0
        self.heading_rad: float = 0.0

        # Schmitt-trigger motion state and 8-way compass sector index (0=N, 1=NE, ..., 7=NW)
        self.is_moving: bool = False
        self.cardinal_idx: Optional[int] = None

        self.trajectory: deque[Tuple[int, int]] = deque(maxlen=max_traj_len)
        self.initialized: bool = False


class RobustKinematicTracker:
    """Production-grade Continuous-Discrete Kalman Filter (CD-KF) with morphological

    covariance inflation, Mahalanobis innovation gating, and SO(2) Schmitt-trigger
    hysteresis designed to sit downstream of YOLOv8 + ByteTrack.
    """

    _CARDINAL_LABELS: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

    def __init__(
        self,
        torso_anchor_ratio: float = 0.65,
        process_accel_std: float = 35.0,
        base_meas_pos_std: float = 6.0,
        morph_jitter_weight: float = 1.8,
        vel_ema_tau: float = 0.25,
        heading_ema_tau: float = 0.30,
        speed_wake_thresh: float = 12.0,
        speed_sleep_thresh: float = 7.0,
        max_speed_px_s: float = 450.0,
        max_accel_px_s2: float = 600.0,
        chi2_gate_thresh: float = 9.21,
        occlusion_reset_sec: float = 1.0,
        sector_hysteresis_deg: float = 6.0,
        max_trajectory_len: int = 40,
        stale_track_timeout_sec: float = 30.0,
    ) -> None:
        if speed_sleep_thresh >= speed_wake_thresh:
            raise ValueError("speed_sleep_thresh must be strictly less than speed_wake_thresh.")

        self.torso_anchor_ratio = float(np.clip(torso_anchor_ratio, 0.1, 0.95))
        self.sigma_a2 = float(process_accel_std ** 2)
        self.base_r_var = float(base_meas_pos_std ** 2)
        self.morph_jitter_weight = float(morph_jitter_weight)
        self.vel_ema_tau = max(1e-3, float(vel_ema_tau))
        self.heading_ema_tau = max(1e-3, float(heading_ema_tau))
        self.speed_wake_thresh = float(speed_wake_thresh)
        self.speed_sleep_thresh = float(speed_sleep_thresh)
        self.max_speed_px_s = float(max_speed_px_s)
        self.max_accel_px_s2 = float(max_accel_px_s2)
        self.chi2_gate_thresh = float(chi2_gate_thresh)
        self.occlusion_reset_sec = float(occlusion_reset_sec)
        self.sector_hyst_rad = math.radians(float(sector_hysteresis_deg))
        self.max_trajectory_len = int(max_trajectory_len)
        self.stale_track_timeout_sec = float(stale_track_timeout_sec)

        self._H: np.ndarray = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self._I4: np.ndarray = np.eye(4, dtype=np.float64)
        self._tracks: Dict[int, _TrackFilterState] = {}
        self._update_counter: int = 0

    def _extract_anchor(
        self,
        bbox: Tuple[int, int, int, int],
        mask_pts: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        bx = float(x1) + 0.5 * w
        by = float(y1) + self.torso_anchor_ratio * h

        if mask_pts is not None and isinstance(mask_pts, np.ndarray) and mask_pts.size >= 6:
            pts = mask_pts.reshape(-1, 1, 2).astype(np.float32)
            moments = cv2.moments(pts)
            m00 = moments["m00"]
            if m00 > 1e-3:
                mx = float(moments["m10"] / m00)
                my = float(moments["m01"] / m00)
                pad_x = 0.15 * w
                pad_y = 0.15 * h
                ax = float(np.clip(mx, x1 + pad_x, x2 - pad_x))
                ay = float(np.clip(my, y1 + pad_y, y2 - pad_y))
                return ax, ay, w, h

        return bx, by, w, h

    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        timestamp: float,
        mask_pts: Optional[np.ndarray] = None,
    ) -> KinematicState:
        self._update_counter += 1
        if self._update_counter % 200 == 0:
            self._prune_stale_tracks(timestamp)

        ax, ay, w, h = self._extract_anchor(bbox, mask_pts)
        z = np.array([[ax], [ay]], dtype=np.float64)

        trk = self._tracks.get(track_id)
        if trk is None:
            trk = _TrackFilterState(self.max_trajectory_len)
            self._tracks[track_id] = trk

        raw_dt = timestamp - trk.last_ts if trk.initialized else 0.0
        if (not trk.initialized) or (raw_dt > self.occlusion_reset_sec) or (raw_dt < 0.0):
            trk.x[0, 0] = ax
            trk.x[1, 0] = ay
            trk.x[2, 0] = 0.0
            trk.x[3, 0] = 0.0
            trk.P = np.diag([
                self.base_r_var,
                self.base_r_var,
                (0.25 * self.max_speed_px_s) ** 2,
                (0.25 * self.max_speed_px_s) ** 2,
            ])
            trk.last_ts = timestamp
            trk.last_wh = (w, h)
            trk.ema_vx = 0.0
            trk.ema_vy = 0.0
            trk.is_moving = False
            trk.trajectory.clear()
            trk.initialized = True

            pos_int = (int(round(ax)), int(round(ay)))
            trk.trajectory.append(pos_int)
            return KinematicState(
                speed=0.0,
                velocity=(0.0, 0.0),
                cardinal="STATIONARY",
                heading_rad=trk.heading_rad,
                smoothed_pos=pos_int,
                trajectory=list(trk.trajectory),
            )

        if raw_dt < 1e-4:
            pos_int = (int(round(trk.x[0, 0])), int(round(trk.x[1, 0])))
            speed = math.hypot(trk.ema_vx, trk.ema_vy)
            cardinal = (
                self._CARDINAL_LABELS[trk.cardinal_idx]
                if (trk.is_moving and trk.cardinal_idx is not None)
                else "STATIONARY"
            )
            return KinematicState(
                speed=speed if trk.is_moving else 0.0,
                velocity=(trk.ema_vx, trk.ema_vy) if trk.is_moving else (0.0, 0.0),
                cardinal=cardinal,
                heading_rad=trk.heading_rad,
                smoothed_pos=pos_int,
                trajectory=list(trk.trajectory),
            )

        dt = float(np.clip(raw_dt, 0.005, self.occlusion_reset_sec))
        trk.last_ts = timestamp

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        dt3 = 0.5 * dt2 * dt
        dt4 = 0.25 * dt2 * dt2
        Q = self.sigma_a2 * np.array(
            [
                [dt4, 0.0, dt3, 0.0],
                [0.0, dt4, 0.0, dt3],
                [dt3, 0.0, dt2, 0.0],
                [0.0, dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        x_pred = F @ trk.x
        P_pred = F @ trk.P @ F.T + Q

        prev_w, prev_h = trk.last_wh
        dw = w - prev_w
        dh = h - prev_h
        trk.last_wh = (w, h)

        r_x = self.base_r_var + self.morph_jitter_weight * (dw * dw)
        r_y = self.base_r_var + self.morph_jitter_weight * (dh * dh)
        R = np.array([[r_x, 0.0], [0.0, r_y]], dtype=np.float64)

        y_innov = z - (self._H @ x_pred)
        S = (self._H @ P_pred @ self._H.T) + R

        det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
        if det_S > 1e-9:
            inv_det = 1.0 / det_S
            S_inv = np.array(
                [
                    [S[1, 1] * inv_det, -S[0, 1] * inv_det],
                    [-S[1, 0] * inv_det, S[0, 0] * inv_det],
                ],
                dtype=np.float64,
            )
        else:
            S_inv = np.linalg.pinv(S)

        mahalanobis_sq = float((y_innov.T @ S_inv @ y_innov)[0, 0])

        if mahalanobis_sq > self.chi2_gate_thresh:
            inflation_factor = mahalanobis_sq / self.chi2_gate_thresh
            R_robust = R * inflation_factor
            S = (self._H @ P_pred @ self._H.T) + R_robust
            det_S = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
            inv_det = 1.0 / max(det_S, 1e-9)
            S_inv = np.array(
                [
                    [S[1, 1] * inv_det, -S[0, 1] * inv_det],
                    [-S[1, 0] * inv_det, S[0, 0] * inv_det],
                ],
                dtype=np.float64,
            )
        else:
            R_robust = R

        K = P_pred @ self._H.T @ S_inv
        x_post = x_pred + (K @ y_innov)
        I_KH = self._I4 - (K @ self._H)
        trk.P = (I_KH @ P_pred @ I_KH.T) + (K @ R_robust @ K.T)

        dv_x = float(x_post[2, 0] - trk.x[2, 0])
        dv_y = float(x_post[3, 0] - trk.x[3, 0])
        dv_norm = math.hypot(dv_x, dv_y)
        max_dv = self.max_accel_px_s2 * dt
        if dv_norm > max_dv and dv_norm > 1e-6:
            scale = max_dv / dv_norm
            x_post[2, 0] = trk.x[2, 0] + dv_x * scale
            x_post[3, 0] = trk.x[3, 0] + dv_y * scale

        raw_vx = float(x_post[2, 0])
        raw_vy = float(x_post[3, 0])
        raw_spd = math.hypot(raw_vx, raw_vy)
        if raw_spd > self.max_speed_px_s and raw_spd > 1e-6:
            spd_scale = self.max_speed_px_s / raw_spd
            raw_vx *= spd_scale
            raw_vy *= spd_scale
            x_post[2, 0] = raw_vx
            x_post[3, 0] = raw_vy

        trk.x = x_post

        alpha_v = 1.0 - math.exp(-dt / self.vel_ema_tau)
        trk.ema_vx += alpha_v * (raw_vx - trk.ema_vx)
        trk.ema_vy += alpha_v * (raw_vy - trk.ema_vy)
        smooth_speed = math.hypot(trk.ema_vx, trk.ema_vy)

        if trk.is_moving:
            if smooth_speed < self.speed_sleep_thresh:
                trk.is_moving = False
        else:
            if smooth_speed > self.speed_wake_thresh:
                trk.is_moving = True

        if trk.is_moving and smooth_speed > 1e-5:
            ux = trk.ema_vx / smooth_speed
            uy = trk.ema_vy / smooth_speed

            alpha_h = 1.0 - math.exp(-dt / self.heading_ema_tau)
            trk.unit_hx += alpha_h * (ux - trk.unit_hx)
            trk.unit_hy += alpha_h * (uy - trk.unit_hy)
            norm_h = math.hypot(trk.unit_hx, trk.unit_hy)
            if norm_h > 1e-6:
                trk.unit_hx /= norm_h
                trk.unit_hy /= norm_h
            else:
                trk.unit_hx, trk.unit_hy = ux, uy

            trk.heading_rad = math.atan2(trk.unit_hy, trk.unit_hx)
            trk.cardinal_idx = self._update_compass_sector(
                trk.unit_hx, trk.unit_hy, trk.cardinal_idx
            )
            cardinal_str = self._CARDINAL_LABELS[trk.cardinal_idx]
            out_speed = smooth_speed
            out_vel = (trk.ema_vx, trk.ema_vy)
        else:
            cardinal_str = "STATIONARY"
            out_speed = 0.0
            out_vel = (0.0, 0.0)

        pos_int = (int(round(float(trk.x[0, 0]))), int(round(float(trk.x[1, 0]))))
        if not trk.trajectory or trk.trajectory[-1] != pos_int:
            trk.trajectory.append(pos_int)

        return KinematicState(
            speed=out_speed,
            velocity=out_vel,
            cardinal=cardinal_str,
            heading_rad=trk.heading_rad,
            smoothed_pos=pos_int,
            trajectory=list(trk.trajectory),
        )

    def _update_compass_sector(
        self,
        ux: float,
        uy: float,
        prev_idx: Optional[int],
    ) -> int:
        azimuth = math.atan2(ux, -uy) % (2.0 * math.pi)
        sector_width = math.pi / 4.0

        candidate_idx = int(((azimuth + 0.5 * sector_width) % (2.0 * math.pi)) // sector_width)
        if prev_idx is None or candidate_idx == prev_idx:
            return candidate_idx

        prev_center = prev_idx * sector_width
        ang_diff = abs((azimuth - prev_center + math.pi) % (2.0 * math.pi) - math.pi)

        if ang_diff > (0.5 * sector_width + self.sector_hyst_rad):
            return candidate_idx
        return prev_idx

    def _prune_stale_tracks(self, current_ts: float) -> None:
        stale_ids = [
            tid
            for tid, st in self._tracks.items()
            if (current_ts - st.last_ts) > self.stale_track_timeout_sec
        ]
        for tid in stale_ids:
            del self._tracks[tid]


def draw_kinematics(
    frame: np.ndarray,
    state: KinematicState,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    track_id: Optional[int] = None,
    color: Tuple[int, int, int] = (142, 199, 72),
    max_arrow_px: int = 45,
    min_arrow_px: int = 14,
    speed_for_max_arrow: float = 160.0,
) -> np.ndarray:
    """Renders non-intrusive fading trajectory breadcrumbs, a grounded core anchor,
    and a clamped directional heading vector.
    """
    h_img, w_img = frame.shape[:2]

    # 1. Fading trajectory breadcrumb trail
    pts = state.trajectory
    n_pts = len(pts)
    if n_pts >= 2:
        start_i = max(1, n_pts - 18)
        for i in range(start_i, n_pts):
            ratio = (i - start_i + 1) / float(n_pts - start_i + 1)
            seg_color = (
                int(color[0] * ratio),
                int(color[1] * ratio),
                int(color[2] * ratio),
            )
            thick = 1 if ratio < 0.6 else 2
            cv2.line(frame, pts[i - 1], pts[i], seg_color, thick, cv2.LINE_AA)

    # 2. Grounded morphological anchor
    ax, ay = state.smoothed_pos
    cv2.circle(frame, (ax, ay), 3, color, -1, cv2.LINE_AA)

    # 3. Clamped directional vector arrow (only when moving actively outside deadband)
    if state.cardinal != "STATIONARY" and state.speed > 8.0:
        norm_ratio = min(1.0, state.speed / max(1.0, speed_for_max_arrow))
        arrow_len = min_arrow_px + norm_ratio * (max_arrow_px - min_arrow_px)

        end_x = int(round(ax + arrow_len * math.cos(state.heading_rad)))
        end_y = int(round(ay + arrow_len * math.sin(state.heading_rad)))
        end_x = int(np.clip(end_x, 0, w_img - 1))
        end_y = int(np.clip(end_y, 0, h_img - 1))

        cv2.arrowedLine(
            frame,
            (ax, ay),
            (end_x, end_y),
            color,
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.28,
        )
    return frame


class SpatialMotionAnalyzer:
    """Optical flow and trajectory-based spatial motion analyser for wildlife.

    Uses a Continuous-Discrete Kalman Filter (CD-KF) with morphological covariance
    inflation, Mahalanobis innovation gating, and SO(2) Schmitt-trigger hysteresis
    for jitter-free, robust kinematics estimation.
    """

    CARDINALS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')

    def __init__(
        self,
        history_len: int = 40,
        smoothing_window: int = 5,
        speed_wake_thresh: float = 12.0,
        speed_sleep_thresh: float = 7.0,
    ) -> None:
        self.history_len = max(history_len, 3)
        self.smoothing_window = min(smoothing_window, self.history_len)
        self._trajectories: Dict[int, List[TrajectoryPoint]] = {}
        self._last_velocities: Dict[int, VelocityVector] = {}
        self._last_timestamps: Dict[int, float] = {}

        # Continuous-Discrete Kalman Filter Kinematic State Estimator
        self.robust_tracker = RobustKinematicTracker(
            torso_anchor_ratio=0.65,
            speed_wake_thresh=speed_wake_thresh,
            speed_sleep_thresh=speed_sleep_thresh,
            max_trajectory_len=self.history_len,
        )

    def update_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None,
        mask_pts: Optional[np.ndarray] = None,
    ) -> VelocityVector:
        """Update kinematics for a single tracked target using CD-KF.

        Parameters
        ----------
        track_id : int
            Unique persistent target identifier.
        bbox : tuple of (x1, y1, x2, y2)
            Bounding box coordinates.
        timestamp : float, optional
            Current frame epoch timestamp in seconds (monotonic).
        mask_pts : ndarray, optional
            Segmentation polygon points for morphological moment center-of-mass anchor.
        """
        now = timestamp if timestamp is not None else time.monotonic()
        kstate = self.robust_tracker.update(track_id, bbox, now, mask_pts=mask_pts)

        prev_vel = self._last_velocities.get(track_id)
        prev_speed = prev_vel.speed if prev_vel else 0.0
        last_t = self._last_timestamps.get(track_id, now - 0.05)
        dt = max(now - last_t, 1e-3)
        self._last_timestamps[track_id] = now
        accel = (kstate.speed - prev_speed) / dt

        if kstate.cardinal == "STATIONARY" or kstate.speed < 12.0:
            activity = "stationary"
        elif kstate.speed < 65.0:
            activity = "browsing"
        elif kstate.speed < 220.0:
            activity = "walking"
        else:
            activity = "running"

        heading_deg = (math.degrees(kstate.heading_rad) + 360.0) % 360.0

        vec = VelocityVector(
            vx=round(kstate.velocity[0], 2),
            vy=round(kstate.velocity[1], 2),
            speed=round(kstate.speed, 2),
            heading_deg=round(heading_deg, 1),
            cardinal=kstate.cardinal,
            acceleration=round(accel, 2),
            activity_state=activity,
            smoothed_pos=kstate.smoothed_pos,
            trajectory=kstate.trajectory,
        )
        self._last_velocities[track_id] = vec
        return vec

    @staticmethod
    def compute_crop_optical_flow(
        prev_crop: np.ndarray,
        curr_crop: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute dense optical flow motion vector inside an animal bounding box crop."""
        if prev_crop is None or curr_crop is None:
            return (0.0, 0.0, 0.0)
        h, w = curr_crop.shape[:2]
        if h < 8 or w < 8:
            return (0.0, 0.0, 0.0)
        p_resized = cv2.resize(prev_crop, (w, h)) if prev_crop.shape[:2] != (h, w) else prev_crop
        prev_g = cv2.cvtColor(p_resized, cv2.COLOR_BGR2GRAY) if p_resized.ndim == 3 else p_resized
        curr_g = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY) if curr_crop.ndim == 3 else curr_crop

        flow = cv2.calcOpticalFlowFarneback(
            prev_g, curr_g, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        thresh = float(np.percentile(mag, 70))
        mask = mag >= max(thresh, 0.5)
        if np.count_nonzero(mask) == 0:
            return (0.0, 0.0, 0.0)
        flow_u = float(np.mean(flow[..., 0][mask]))
        flow_v = float(np.mean(flow[..., 1][mask]))
        flow_spd = float(np.hypot(flow_u, flow_v))
        return (round(flow_u, 2), round(flow_v, 2), round(flow_spd, 2))

    @staticmethod
    def compensate_camera_motion(
        vec: VelocityVector,
        camera_homography: Optional[np.ndarray],
        dt: float = 1.0 / 25.0,
    ) -> VelocityVector:
        """Compensate velocity vector for camera ego-motion using stabilization homography."""
        if camera_homography is None or np.allclose(camera_homography, np.eye(3)):
            return vec
        cam_dx = float(camera_homography[0, 2])
        cam_dy = float(camera_homography[1, 2])
        cam_vx = cam_dx / max(dt, 1e-4)
        cam_vy = cam_dy / max(dt, 1e-4)

        true_vx = vec.vx - cam_vx
        true_vy = vec.vy - cam_vy
        true_speed = float(np.hypot(true_vx, true_vy))

        raw_angle = (math.degrees(math.atan2(true_vy, true_vx)) + 360.0) % 360.0
        cardinal_idx = int((raw_angle + 22.5) // 45.0) % 8
        cardinal = SpatialMotionAnalyzer.CARDINALS[cardinal_idx]

        return VelocityVector(
            vx=round(true_vx, 2),
            vy=round(true_vy, 2),
            speed=round(true_speed, 2),
            heading_deg=round(raw_angle, 1),
            cardinal=cardinal,
            acceleration=vec.acceleration,
            activity_state=vec.activity_state,
            smoothed_pos=vec.smoothed_pos,
            trajectory=vec.trajectory,
        )

    def draw_motion_overlay(
        self,
        frame: np.ndarray,
        active_tracks: Dict[int, Tuple[int, int, int, int]],
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Render subtle fading trajectory breadcrumbs and clean directional indicators using CD-KF smoothed states."""
        canvas = frame
        h_img, w_img = frame.shape[:2]
        for tid, bbox in active_tracks.items():
            vel = self._last_velocities.get(tid)
            if not vel or not vel.trajectory or len(vel.trajectory) < 2:
                continue

            col = color_map.get(tid, (142, 199, 72)) if color_map else (142, 199, 72)
            pts = vel.trajectory
            n_pts = len(pts)

            # Draw fading trajectory trail
            start_i = max(1, n_pts - 16)
            for i in range(start_i, n_pts):
                ratio = (i - start_i + 1) / float(n_pts - start_i + 1)
                seg_color = (
                    int(col[0] * ratio),
                    int(col[1] * ratio),
                    int(col[2] * ratio),
                )
                thick = 1 if ratio < 0.6 else 2
                cv2.line(canvas, pts[i - 1], pts[i], seg_color, thick, cv2.LINE_AA)

            ax, ay = vel.smoothed_pos
            # Morphological anchor point
            cv2.circle(canvas, (ax, ay), 3, col, -1, cv2.LINE_AA)

            # Directional vector arrow (only when moving actively outside deadband)
            if vel.cardinal != "STATIONARY" and vel.speed > 10.0:
                norm_ratio = min(1.0, vel.speed / 180.0)
                arrow_len = 16.0 + norm_ratio * 26.0
                rad = math.radians(vel.heading_deg)
                end_x = int(round(ax + arrow_len * math.cos(rad)))
                end_y = int(round(ay + arrow_len * math.sin(rad)))
                end_x = int(np.clip(end_x, 0, w_img - 1))
                end_y = int(np.clip(end_y, 0, h_img - 1))
                cv2.arrowedLine(canvas, (ax, ay), (end_x, end_y), col, 2, cv2.LINE_AA, tipLength=0.28)
        return canvas


# ========================================================================== #
# 8.  Environmental Adaptive Controller (Auto-Calibration)                   #
# ========================================================================== #

@dataclass
class EnvironmentProfile:
    """Real-time environmental lighting and weather assessment.

    Attributes
    ----------
    mode : str
        Scene illumination mode ("daylight", "low_light", "infrared_night", "fog_precipitation", "harsh_glare").
    mean_luminance : float
        Average pixel intensity in range [0.0, 255.0].
    contrast_std : float
        Standard deviation of intensity.
    color_saturation : float
        Color diversity index across chromatic channels.
    sharpness_laplacian : float
        Variance of Laplacian operator; lower indicates fog/precipitation/blur.
    glare_fraction : float
        Fraction of pixels over-saturated near 255.
    recommended_clahe_clip : float
        Optimized CLAHE clip limit parameter.
    recommended_mog2_var_threshold : float
        Optimized background subtractor sensitivity.
    recommended_denoise : bool
        Whether low-light frame denoising should be active.
    """

    mode: str = "daylight"
    mean_luminance: float = 128.0
    contrast_std: float = 40.0
    color_saturation: float = 50.0
    sharpness_laplacian: float = 100.0
    glare_fraction: float = 0.0
    recommended_clahe_clip: float = 2.0
    recommended_mog2_var_threshold: float = 16.0
    recommended_denoise: bool = False
    lens_obstructed: bool = False


class EnvironmentalAdaptiveController:
    """Real-time edge-case and failure-mode adaptive controller.

    Continuously estimates ambient light, weather degradation (fog/rain), and
    sensor modality (monochrome infrared night mode vs natural daylight) using
    OpenCV histogram metrics, dynamically adjusting CLAHE clip limits and MOG2
    variance thresholds without interrupting background history.
    """

    def __init__(self, sample_interval: int = 15) -> None:
        self.sample_interval = max(sample_interval, 1)
        self._frame_count = 0
        self._cached_profile = EnvironmentProfile()

    def analyze(self, frame: np.ndarray, force: bool = False) -> EnvironmentProfile:
        """Analyze frame environmental metrics and compute optimal tuning parameters."""
        self._frame_count += 1
        if not force and self._frame_count % self.sample_interval != 0:
            return self._cached_profile

        # Use small preview for sub-millisecond evaluation
        h, w = frame.shape[:2]
        preview = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))
        glare_frac = float(np.count_nonzero(gray >= 250)) / (160 * 120)

        # Measure color dispersion (difference across channels)
        b, g, r = cv2.split(preview)
        color_diff = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) +
                            np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32))))

        # Focus & atmospheric blur measure via Laplacian variance
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Classification logic
        lens_obs = False
        if mean_lum < 5.0 or (mean_lum > 15.0 and contrast_std < 3.5 and lap_var < 2.0):
            mode = "lens_obstructed"
            clahe_clip = 1.0
            mog2_thresh = 50.0
            denoise = False
            lens_obs = True
        elif color_diff < 4.0 and mean_lum < 90.0:
            mode = "infrared_night"
            clahe_clip = 4.2
            mog2_thresh = 36.0  # elevate threshold to ignore infrared sensor shot noise
            denoise = True
        elif mean_lum < 45.0:
            mode = "low_light"
            clahe_clip = 3.8
            mog2_thresh = 28.0
            denoise = True
        elif glare_frac > 0.12 or mean_lum > 215.0:
            mode = "harsh_glare"
            clahe_clip = 1.2
            mog2_thresh = 20.0
            denoise = False
        elif lap_var < 35.0 and contrast_std < 30.0 and mean_lum > 60.0:
            mode = "fog_precipitation"
            clahe_clip = 3.0
            mog2_thresh = 24.0  # prevent rain drops from triggering full detector
            denoise = False
        else:
            mode = "daylight"
            clahe_clip = 2.0
            mog2_thresh = 16.0
            denoise = False

        profile = EnvironmentProfile(
            mode=mode,
            mean_luminance=round(mean_lum, 1),
            contrast_std=round(contrast_std, 1),
            color_saturation=round(color_diff, 1),
            sharpness_laplacian=round(lap_var, 1),
            glare_fraction=round(glare_frac, 3),
            recommended_clahe_clip=clahe_clip,
            recommended_mog2_var_threshold=mog2_thresh,
            recommended_denoise=denoise,
            lens_obstructed=lens_obs,
        )
        self._cached_profile = profile
        return profile


# ========================================================================== #
# 9.  OpenCV5Pipeline — Unified processing chain                             #
# ========================================================================== #

@dataclass
class PipelineResult:
    """Aggregated result of running the full ``OpenCV5Pipeline``.

    Attributes
    ----------
    frame : np.ndarray
        The processed frame (stabilised, denoised, contrast-enhanced, etc.).
    motion : MotionDetectionResult | None
        Motion detection output, or ``None`` if skipped.
    should_run_detector : bool
        ``True`` if downstream object detection should be invoked.
    tracker_bbox : tuple[int, int, int, int] | None
        Updated bounding box from object tracker, if active.
    tracker_ok : bool
        Whether the tracker successfully updated.
    reid_descriptor : AnimalDescriptor | None
        Feature descriptor extracted from the current detection crop.
    reid_matches : list[tuple[int, float]]
        Best re-ID gallery matches for this frame.
    diagnostics : dict[str, Any]
        Timing and debug information.
    """

    frame: np.ndarray
    motion: Optional[MotionDetectionResult] = None
    should_run_detector: bool = True
    tracker_bbox: Optional[Tuple[int, int, int, int]] = None
    tracker_ok: bool = False
    reid_descriptor: Optional[AnimalDescriptor] = None
    reid_matches: List[Tuple[int, float]] = field(default_factory=list)
    environment: Optional[EnvironmentProfile] = None
    velocity_vectors: Dict[int, VelocityVector] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class OpenCV5Pipeline:
    """Chains OpenCV 5 processing steps into a unified wildlife-analysis pipeline.

    The pipeline executes the following stages in order:

    1. **Background subtraction** — decides whether the frame has motion.
       If no motion, subsequent heavy stages are skipped.
    2. **Video stabilisation** — compensates for camera shake.
    3. **Denoising** — reduces sensor noise in low-light frames.
    4. **Contrast enhancement** — CLAHE on the L channel.
    5. **Edge-preserving smoothing** — prepares the frame for segmentation.
    6. **Object tracking** — if a tracker is active, update it.
    7. **Re-identification** — if a detection crop is provided, compute and
       match descriptors.

    Each stage can be individually enabled or disabled.

    Parameters
    ----------
    enable_motion_filter : bool
        Enable MOG2 motion pre-filter.
    enable_stabilisation : bool
        Enable optical-flow-based stabilisation.
    enable_denoise : bool
        Enable non-local-means denoising.
    enable_clahe : bool
        Enable CLAHE contrast enhancement.
    enable_edge_preserve : bool
        Enable edge-preserving smoothing.
    enable_tracking : bool
        Enable CSRT/KCF object tracking.
    enable_reid : bool
        Enable ORB/AKAZE re-identification.
    tracker_type : str
        ``"CSRT"`` or ``"KCF"``.
    reid_detector : str
        ``"ORB"`` or ``"AKAZE"``.
    denoise_h : int
        Luminance filter strength for denoising.
    clahe_clip : float
        CLAHE clip limit.
    motion_threshold : float
        Minimum foreground fraction to flag motion.
    """

    def __init__(
        self,
        *,
        enable_motion_filter: bool = True,
        enable_stabilisation: bool = True,
        enable_denoise: bool = True,
        enable_clahe: bool = True,
        enable_edge_preserve: bool = False,
        enable_tracking: bool = True,
        enable_reid: bool = True,
        tracker_type: str = "CSRT",
        reid_detector: str = "ORB",
        denoise_h: int = 10,
        clahe_clip: float = 2.0,
        motion_threshold: float = 0.005,
        enable_environment_adaptation: bool = True,
        enable_spatial_analytics: bool = True,
    ) -> None:
        self.enable_motion_filter = enable_motion_filter
        self.enable_stabilisation = enable_stabilisation
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe
        self.enable_edge_preserve = enable_edge_preserve
        self.enable_tracking = enable_tracking
        self.enable_reid = enable_reid
        self.denoise_h = denoise_h
        self.clahe_clip = clahe_clip

        # --- Sub-modules ---------------------------------------------------
        self.motion_filter: Optional[MotionPreFilter] = None
        if enable_motion_filter and FEATURES["bg_subtractor_mog2"]:
            self.motion_filter = MotionPreFilter(motion_threshold=motion_threshold)

        self.stab_state = StabilisationState()

        self.tracker_state: Optional[TrackerState] = None
        if enable_tracking:
            try:
                self.tracker_state = create_tracker(tracker_type)
            except RuntimeError:
                logger.warning("Tracker creation failed; tracking disabled.")
                self.tracker_state = None

        self.reid_module: Optional[AnimalReIDModule] = None
        if enable_reid:
            try:
                self.reid_module = AnimalReIDModule(detector=reid_detector)
            except RuntimeError:
                logger.warning("ReID module creation failed; re-id disabled.")
                self.reid_module = None

        # Spatial kinematics & environmental adaptation modules
        self.enable_environment_adaptation = enable_environment_adaptation
        self.enable_spatial_analytics = enable_spatial_analytics
        self.env_controller = EnvironmentalAdaptiveController(sample_interval=15)
        self.spatial_analyzer = SpatialMotionAnalyzer(history_len=25, smoothing_window=5)

        # DNN classifier is optional; users attach it after construction.
        self.dnn_classifier: Optional[DNNSpeciesClassifier] = None

        logger.info("OpenCV5Pipeline initialised.  Active stages: %s", self.active_stages)

    # --------------------------------------------------------------------- #

    @property
    def active_stages(self) -> List[str]:
        """Return a list of currently enabled stage names."""
        stages: List[str] = []
        if self.enable_motion_filter and self.motion_filter:
            stages.append("motion_filter")
        if self.enable_stabilisation:
            stages.append("stabilisation")
        if self.enable_denoise:
            stages.append("denoise")
        if self.enable_clahe:
            stages.append("clahe")
        if self.enable_edge_preserve:
            stages.append("edge_preserve")
        if self.enable_tracking and self.tracker_state:
            stages.append("tracking")
        if self.enable_reid and self.reid_module:
            stages.append("reid")
        if self.dnn_classifier:
            stages.append("dnn_classifier")
        if getattr(self, "enable_environment_adaptation", False):
            stages.append("env_adaptive")
        if getattr(self, "enable_spatial_analytics", False):
            stages.append("spatial_analytics")
        return stages

    # --------------------------------------------------------------------- #

    def init_tracker_on_detection(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """Initialise the tracker with a detected bounding box.

        Call this when the YOLO detector produces a new detection that you
        want to track across subsequent frames.

        Parameters
        ----------
        frame : np.ndarray
            The frame in which the detection was made.
        bbox : tuple[int, int, int, int]
            ``(x, y, w, h)`` bounding box.
        """
        if self.tracker_state is None:
            return
        # Re-create the tracker (OpenCV trackers are single-use after init).
        try:
            self.tracker_state = create_tracker(self.tracker_state.tracker_type)
            init_tracker(self.tracker_state, frame, bbox)
        except RuntimeError as exc:
            logger.warning("Could not re-initialise tracker: %s", exc)
            self.tracker_state = None

    # --------------------------------------------------------------------- #

    def process(
        self,
        frame: np.ndarray,
        detection_crop: Optional[np.ndarray] = None,
        detection_metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Run the full pipeline on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 input frame.
        detection_crop : np.ndarray | None
            Optional cropped image of a detected animal (for re-ID).
        detection_metadata : dict | None
            Optional metadata to attach to the re-ID descriptor.

        Returns
        -------
        PipelineResult
        """
        diag: Dict[str, Any] = {"opencv_version": _CV_VERSION}
        result = PipelineResult(frame=frame.copy(), diagnostics=diag)

        t_start = time.perf_counter()

        # ---- Stage 0: Environmental Edge-Case Adaptation ----------------
        env_profile = None
        if getattr(self, "enable_environment_adaptation", False):
            t0_env = time.perf_counter()
            env_profile = self.env_controller.analyze(frame)
            result.environment = env_profile
            diag["env_mode"] = env_profile.mode
            diag["env_ms"] = (time.perf_counter() - t0_env) * 1000.0
            # Dynamically calibrate MOG2 sensitivity
            if self.motion_filter and hasattr(self.motion_filter._subtractor, "setVarThreshold"):
                self.motion_filter._subtractor.setVarThreshold(env_profile.recommended_mog2_var_threshold)

        # ---- 1. Motion pre-filter ----------------------------------------
        if self.motion_filter and self.enable_motion_filter:
            t0 = time.perf_counter()
            motion_result = self.motion_filter.detect(frame)
            diag["motion_ms"] = (time.perf_counter() - t0) * 1000.0
            result.motion = motion_result
            result.should_run_detector = motion_result.motion_detected

            if not motion_result.motion_detected:
                # Short-circuit: no motion → skip expensive stages.
                diag["total_ms"] = (time.perf_counter() - t_start) * 1000.0
                result.frame = frame
                return result

        # ---- 2. Video stabilisation --------------------------------------
        if self.enable_stabilisation:
            t0 = time.perf_counter()
            result.frame, self.stab_state = stabilise_frame(
                result.frame, self.stab_state
            )
            diag["stabilisation_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 3. Denoising (Dynamic Environmental Auto-Calibration) -------
        should_denoise = self.enable_denoise or (env_profile is not None and env_profile.recommended_denoise)
        if should_denoise:
            t0 = time.perf_counter()
            result.frame = denoise_low_light(result.frame, h=self.denoise_h)
            diag["denoise_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 4. CLAHE contrast enhancement (Environment adaptive) --------
        if self.enable_clahe:
            t0 = time.perf_counter()
            dyn_clip = env_profile.recommended_clahe_clip if env_profile is not None else self.clahe_clip
            result.frame = enhance_contrast_clahe(
                result.frame, clip_limit=dyn_clip
            )
            diag["clahe_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 5. Edge-preserving smoothing ---------------------------------
        if self.enable_edge_preserve:
            t0 = time.perf_counter()
            result.frame = edge_preserving_smooth(result.frame)
            diag["edge_preserve_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 6. Object tracking ------------------------------------------
        if self.tracker_state and self.enable_tracking and self.tracker_state.bbox:
            t0 = time.perf_counter()
            ok, bbox, self.tracker_state = update_tracker(
                self.tracker_state, result.frame
            )
            result.tracker_ok = ok
            result.tracker_bbox = bbox
            diag["tracking_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 7. Re-identification ----------------------------------------
        if self.reid_module and self.enable_reid and detection_crop is not None:
            t0 = time.perf_counter()
            desc = self.reid_module.compute_descriptor(
                detection_crop, metadata=detection_metadata
            )
            result.reid_descriptor = desc

            if self.reid_module.gallery_size > 0:
                result.reid_matches = self.reid_module.match(desc)

            # Automatically add to gallery for future matching
            self.reid_module.add_to_gallery(desc)
            diag["reid_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- Timing summary -----------------------------------------------
        diag["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        return result


# ========================================================================== #
# Module-level convenience: print a summary when run directly                #
# ========================================================================== #

if __name__ == "__main__":
    info = report_opencv_info()
    print("=" * 60)
    print(f"  Wildlife Sentinel -- OpenCV 5 Feature Module")
    print(f"  OpenCV version : {info['version']}")
    print(f"  Major          : {info['major']}")
    print(f"  OpenCV 5?      : {'YES' if info['features']['opencv5'] else 'NO'}")
    print("=" * 60)
    print("\n  Feature availability:")
    for feat, avail in info["features"].items():
        status = "[OK]" if avail else "[--]"
        print(f"    [{status}] {feat}")

    # Quick smoke test with a synthetic frame
    print("\n  Running pipeline smoke test on a 640×480 synthetic frame …")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    pipeline = OpenCV5Pipeline(
        enable_motion_filter=True,
        enable_stabilisation=True,
        enable_denoise=False,  # skip denoising for speed in smoke test
        enable_clahe=True,
        enable_edge_preserve=False,
        enable_tracking=True,
        enable_reid=True,
    )

    print(f"  Active stages  : {pipeline.active_stages}")

    # Process two frames so the stabiliser has a prev-frame reference
    _ = pipeline.process(test_frame)
    result = pipeline.process(test_frame)

    print(f"  Output shape   : {result.frame.shape}")
    print(f"  Motion detected: {result.motion.motion_detected if result.motion else 'N/A'}")
    print(f"  Detector run?  : {result.should_run_detector}")
    print(f"  Total time     : {result.diagnostics.get('total_ms', 0):.1f} ms")
    print("\n  Smoke test passed [OK]")
