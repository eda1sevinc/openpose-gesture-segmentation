# src/openpose_gesture_segmentation/io.py
import json
import re
from pathlib import Path

import numpy as np

FNAME_RE = re.compile(r"(\d+)_keypoints\.json$|_(\d+)_keypoints\.json$")


def frame_index(path: Path):
    """
        Extract the numeric frame index from an OpenPose filename.

        Supported patterns
        ------------------
        .../XYZ_<frame>_keypoints.json
        .../<frame>_keypoints.json

        Parameters
        ----------
        path : Path
            Path to a single OpenPose JSON file.

        Returns
        -------
        int or None
            Extracted frame index if the name matches, else None.
    """
    m = FNAME_RE.search(path.name)
    return int(m.group(1) or m.group(2)) if m else None


def _triplets_to_xyc(arr):
    """
        Convert a flat [x0, y0, c0, x1, y1, c1, ...] array into (x, y, c).

        Parameters
        ----------
        arr : sequence of float
            Flat list/array of OpenPose keypoint triplets.

        Returns
        -------
        (x, y, c) : tuple of ndarrays or (None, None, None)
            x, y, c each with shape (N_points,). If input size is invalid, returns (None, None, None).
    """
    a = np.asarray(arr, dtype=float)
    if a.size % 3 != 0 or a.size == 0:
        return None, None, None
    pts = a.reshape((-1, 3))
    return pts[:, 0], pts[:, 1], pts[:, 2]


def read_pose_and_hands(path: Path):
    """
        Read a single OpenPose JSON and extract wrists and 21-point hands (no shoulders).

        Picks the first detected person in the file. For each requested keypoint, if the
        confidence is too low or missing, returns NaNs (for pose points) or (None, None, None) for hands.

        Parameters
        ----------
        path : Path
            Path to one `*_keypoints.json`.

        Returns
        -------
        dict or None
            {
              "R_wrist": (x, y, c) or (nan, nan, nan),
              "L_wrist": (x, y, c) or (nan, nan, nan),
              "handR": (x21, y21, c21) or (None, None, None),
              "handL": (x21, y21, c21) or (None, None, None)
            }
            Returns None if no people are present.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people:
        return None

    p = people[0]  # pick first (you can change selection logic later)
    pose = p.get("pose_keypoints_2d", [])
    x, y, c = _triplets_to_xyc(pose)

    # BODY_25 indices
    IDX_RWR = 4
    IDX_LWR = 7

    def wrist(idx):
        if x is None or len(x) <= idx or c[idx] < 0.1:
            return (float("nan"), float("nan"), float("nan"))
        return (float(x[idx]), float(y[idx]), float(c[idx]))

    handR = _triplets_to_xyc(p.get("hand_right_keypoints_2d", []))
    handL = _triplets_to_xyc(p.get("hand_left_keypoints_2d", []))

    return {
        "R_wrist": wrist(IDX_RWR),
        "L_wrist": wrist(IDX_LWR),
        "handR": handR if handR[0] is not None else (None, None, None),
        "handL": handL if handL[0] is not None else (None, None, None),
    }


def read_pose_and_hands_for_pose_segmentation(path: Path):
    """
        Read a single OpenPose JSON and extract wrists, shoulders, shoulder width, and hands.

        This variant additionally provides:
          - R/L shoulder keypoints (BODY_25 indices 2 and 5)
          - Per-frame shoulder width (Euclidean distance between shoulders)

        Parameters
        ----------
        path : Path
            Path to one `*_keypoints.json`.

        Returns
        -------
        dict or None
            {
              "R_wrist": (x, y, c) or (nan, nan, nan),
              "L_wrist": (x, y, c) or (nan, nan, nan),
              "R_shoulder": (x, y, c) or (nan, nan, nan),
              "L_shoulder": (x, y, c) or (nan, nan, nan),
              "shoulder": float (shoulder width in pixels, NaN if unavailable),
              "handR": (x21, y21, c21) or (None, None, None),
              "handL": (x21, y21, c21) or (None, None, None)
            }
            Returns None if no people are present.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    people = data.get("people", [])
    if not people:
        return None

    p = people[0]
    pose = p.get("pose_keypoints_2d", [])
    x, y, c = _triplets_to_xyc(pose)

    # BODY_25 indices
    IDX_RWR = 4
    IDX_LWR = 7
    IDX_RSHO = 2
    IDX_LSHO = 5

    def kp(idx):
        if x is None or len(x) <= idx or c[idx] < 0.1:
            return (float("nan"), float("nan"), float("nan"))
        return (float(x[idx]), float(y[idx]), float(c[idx]))

    R_wrist = kp(IDX_RWR)
    L_wrist = kp(IDX_LWR)
    R_shoulder = kp(IDX_RSHO)
    L_shoulder = kp(IDX_LSHO)

    # shoulder width
    if all(np.isfinite([R_shoulder[0], R_shoulder[1], L_shoulder[0], L_shoulder[1]])):
        sw = float(np.hypot(R_shoulder[0] - L_shoulder[0], R_shoulder[1] - L_shoulder[1]))
    else:
        sw = float("nan")

    handR = _triplets_to_xyc(p.get("hand_right_keypoints_2d", []))
    handL = _triplets_to_xyc(p.get("hand_left_keypoints_2d", []))

    return {
        "R_wrist": R_wrist,
        "L_wrist": L_wrist,
        "R_shoulder": R_shoulder,
        "L_shoulder": L_shoulder,
        "shoulder": sw,
        "handR": handR if handR[0] is not None else (None, None, None),
        "handL": handL if handL[0] is not None else (None, None, None),
    }


def load_frames(folder: Path, fps: float):
    """
        Load and parse all OpenPose frame JSONs in a folder into a time-aligned list.

        The files are sorted by the numeric frame index extracted from their filenames.
        Each element in the returned list contains the selected person's key data and
        a `t_ms` field computed from frame index and FPS.

        Parameters
        ----------
        folder : Path
            Directory containing `*_keypoints.json` files.
        fps : float
            Video frame rate. Used to compute timestamp: t_ms = frame_index / fps * 1000.

        Returns
        -------
        list of dict
            Each dict includes:
              - "t_ms": timestamp in milliseconds
              - keypoints/arrays as returned by `read_pose_and_hands_for_pose_segmentation`
            Files without a valid frame index or without people are skipped.
    """
    files = sorted(Path(folder).glob("*_keypoints.json"),
                   key=lambda p: (frame_index(p) or 10 ** 18, p.name))
    out = []
    for f in files:
        idx = frame_index(f)
        if idx is None:
            continue
        # d = read_pose_and_hands(f)
        d = read_pose_and_hands_for_pose_segmentation(f)
        if d is None:
            continue
        d["t_ms"] = (idx / fps) * 1000.0
        out.append(d)
    return out
