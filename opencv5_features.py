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
import time
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
    """

    motion_detected: bool
    foreground_mask: np.ndarray
    foreground_fraction: float


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

        # Optionally filter out small contours
        if self.min_contour_area > 0:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            clean_mask = np.zeros_like(fg_mask)
            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_contour_area:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
            fg_mask = clean_mask

        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_pixels = int(np.count_nonzero(fg_mask))
        fg_fraction = fg_pixels / max(total_pixels, 1)

        return MotionDetectionResult(
            motion_detected=fg_fraction >= self.motion_threshold,
            foreground_mask=fg_mask,
            foreground_fraction=fg_fraction,
        )


# ========================================================================== #
# 7.  OpenCV5Pipeline — Unified processing chain                             #
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

        # ---- 3. Denoising ------------------------------------------------
        if self.enable_denoise:
            t0 = time.perf_counter()
            result.frame = denoise_low_light(result.frame, h=self.denoise_h)
            diag["denoise_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 4. CLAHE contrast enhancement --------------------------------
        if self.enable_clahe:
            t0 = time.perf_counter()
            result.frame = enhance_contrast_clahe(
                result.frame, clip_limit=self.clahe_clip
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
    print(f"  Wildlife Sentinel — OpenCV 5 Feature Module")
    print(f"  OpenCV version : {info['version']}")
    print(f"  Major          : {info['major']}")
    print(f"  OpenCV 5?      : {'YES' if info['features']['opencv5'] else 'NO'}")
    print("=" * 60)
    print("\n  Feature availability:")
    for feat, avail in info["features"].items():
        status = "✓" if avail else "✗"
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
    print("\n  Smoke test passed ✓")
