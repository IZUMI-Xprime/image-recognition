# tracker_cy.pyx  –  Cython-accelerated IoU tracker core
# Compile with:  python setup_tracker.py build_ext --inplace
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Pure-C inner loops for the IoU tracker.
Everything that was Python-level iteration in UniqueAnimalTracker.update()
is now typed C code running at near-native speed.
"""

from libc.math cimport fmaxf, fminf
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Typed structs (stack-allocated — zero GC pressure)
# ─────────────────────────────────────────────────────────────────────────────
cdef struct BBox:
    int x1, y1, x2, y2


# ─────────────────────────────────────────────────────────────────────────────
# IoU — inlined, no Python objects
# ─────────────────────────────────────────────────────────────────────────────
cdef inline float _iou_c(BBox a, BBox b) nogil:
    cdef int ix1, iy1, ix2, iy2
    cdef float inter, area_a, area_b
    ix1 = a.x1 if a.x1 > b.x1 else b.x1
    iy1 = a.y1 if a.y1 > b.y1 else b.y1
    ix2 = a.x2 if a.x2 < b.x2 else b.x2
    iy2 = a.y2 if a.y2 < b.y2 else b.y2
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = <float>((ix2 - ix1) * (iy2 - iy1))
    area_a = <float>((a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = <float>((b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / (area_a + area_b - inter)


# ─────────────────────────────────────────────────────────────────────────────
# Public Python-callable IoU (used for single-pair checks from Python)
# ─────────────────────────────────────────────────────────────────────────────
def iou(tuple a, tuple b):
    """IoU between two (x1,y1,x2,y2) tuples. Returns float."""
    cdef BBox ba, bb
    ba.x1, ba.y1, ba.x2, ba.y2 = a[0], a[1], a[2], a[3]
    bb.x1, bb.y1, bb.x2, bb.y2 = b[0], b[1], b[2], b[3]
    return _iou_c(ba, bb)


# ─────────────────────────────────────────────────────────────────────────────
# match_detections_to_tracks
#
# Greedy O(T*D) matching for one label bucket.
# Returns (matched_track_indices, unmatched_box_indices)
# Both are Python lists of ints — small so list alloc is fine.
# ─────────────────────────────────────────────────────────────────────────────
def match_detections_to_tracks(
        list track_bboxes,    # [(x1,y1,x2,y2), …]  active tracks
        list det_bboxes,      # [(x1,y1,x2,y2), …]  detections this frame
        float iou_thresh):
    """
    Greedy nearest-IoU matching.
    Each track grabs its best unmatched detection if IoU >= threshold.

    Returns:
        matched_pairs   : list of (track_idx, det_idx)
        unmatched_dets  : list of det_idx with no track match
    """
    cdef int nt = len(track_bboxes)
    cdef int nd = len(det_bboxes)
    cdef int ti, di, best_di
    cdef float score, best_iou
    cdef BBox bt, bd

    if nt == 0 or nd == 0:
        return [], list(range(nd))

    # Track which detections are still available
    cdef list available = list(range(nd))
    cdef list matched_pairs = []

    for ti in range(nt):
        t = track_bboxes[ti]
        bt.x1, bt.y1, bt.x2, bt.y2 = t[0], t[1], t[2], t[3]

        best_iou = -1.0
        best_di  = -1

        for di in available:
            d = det_bboxes[di]
            bd.x1, bd.y1, bd.x2, bd.y2 = d[0], d[1], d[2], d[3]
            score = _iou_c(bt, bd)
            if score > best_iou:
                best_iou = score
                best_di  = di

        if best_iou >= iou_thresh and best_di >= 0:
            matched_pairs.append((ti, best_di))
            available.remove(best_di)

    return matched_pairs, available   # available == unmatched dets


# ─────────────────────────────────────────────────────────────────────────────
# batch_iou
#
# Compute IoU between one query box and a list of boxes.
# Returns a Python list of floats.  Useful for debugging / testing.
# ─────────────────────────────────────────────────────────────────────────────
def batch_iou(tuple query, list boxes):
    """
    Returns list of IoU scores between `query` and each box in `boxes`.
    Each box is (x1, y1, x2, y2).
    """
    cdef BBox bq, bb
    bq.x1, bq.y1, bq.x2, bq.y2 = query[0], query[1], query[2], query[3]
    cdef list out = []
    for box in boxes:
        bb.x1, bb.y1, bb.x2, bb.y2 = box[0], box[1], box[2], box[3]
        out.append(_iou_c(bq, bb))
    return out
