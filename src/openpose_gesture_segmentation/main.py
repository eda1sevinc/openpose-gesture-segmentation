# src/openpose_gesture_segmentation/main.py
import argparse
from pathlib import Path

import numpy as np

from .io import load_frames
from .segmentation import (
    union_segments, wrist_velocity_normalized, segment_with_pose_extension,
    velocity_hand_normalized
)


def _safe_mid_shoulder(frame):
    """
        Return the midpoint between the two shoulders for a frame.

        Falls back to the midpoint between wrists if shoulder coordinates are missing.
        If both are unavailable, returns (nan, nan).

        Parameters
        ----------
        frame : dict
            A per-frame dict produced by `io.load_frames`.

        Returns
        -------
        (cx, cy) : tuple[float, float]
            Midpoint in pixel coordinates.
    """
    # mid-point between shoulders; fall back to wrist mid if shoulders missing
    rx, ry, _ = frame.get("R_shoulder", (np.nan, np.nan, np.nan))
    lx, ly, _ = frame.get("L_shoulder", (np.nan, np.nan, np.nan))
    if np.isfinite(rx) and np.isfinite(ry) and np.isfinite(lx) and np.isfinite(ly):
        return (0.5 * (rx + lx), 0.5 * (ry + ly))
    # fallback: mid wrist (worse, but avoids crash)
    rwx, rwy, _ = frame["R_wrist"]
    lwx, lwy, _ = frame["L_wrist"]
    if np.isfinite(rwx) and np.isfinite(rwy) and np.isfinite(lwx) and np.isfinite(lwy):
        return (0.5 * (rwx + lwx), 0.5 * (rwy + lwy))
    return (np.nan, np.nan)


def _shoulder_width_safe(frame, global_sw):
    """
        Get a per-frame shoulder width with a safe fallback to global median.

        Parameters
        ----------
        frame : dict
            A per-frame dict with a "shoulder" key (float, pixels) if available.
        global_sw : float
            Median shoulder width over the sequence (pixels), or NaN if unknown.

        Returns
        -------
        float
            Usable shoulder width (>0) for normalization.
    """
    sw = frame.get("shoulder", float("nan"))
    if np.isfinite(sw) and sw > 1e-6:
        return sw
    return global_sw if np.isfinite(global_sw) and global_sw > 1e-6 else 1.0


def _wrist_coords_norm(frames, side):
    """
       Build shoulder-normalized wrist coordinates relative to shoulder midpoint.

       For each frame, compute:
           ( (wx - cx)/shoulder_width , (wy - cy)/shoulder_width )

       Parameters
       ----------
       frames : list[dict]
           Sequence returned by `load_frames`.
       side : {"R","L"}
           Which wrist to use.

       Returns
       -------
       list[(x_norm, y_norm)]
           One tuple per frame; if inputs are missing, falls back to (0.0, 0.0).
    """
    # global fallback shoulder width (median of available)
    sws = np.array([f.get("shoulder", np.nan) for f in frames], float)
    global_sw = np.nanmedian(sws[np.isfinite(sws)]) if np.any(np.isfinite(sws)) else np.nan

    coords = []
    for f in frames:
        wx, wy, _ = f[f"{side}_wrist"]
        cx, cy = _safe_mid_shoulder(f)
        sw = _shoulder_width_safe(f, global_sw)
        if np.isfinite(wx) and np.isfinite(wy) and np.isfinite(cx) and np.isfinite(cy) and sw > 1e-6:
            coords.append(((wx - cx) / sw, (wy - cy) / sw))
        else:
            coords.append((0.0, 0.0))  # neutral when unknown
    return coords


def main():
    """
       CLI entry point: compute gesture segments from OpenPose JSON frames and export TSV.

       Pipeline
       --------
       1) Load frames (wrists, shoulders, hands, timestamps).
       2) Build normalized velocity per side: max(wrist_norm_velocity, hand_norm_velocity).
       3) Segment with hysteresis + pose-hold extension.
       4) Union left/right segments and export to TSV (ELAN-importable).

       Arguments
       ---------
       -i / --input : folder with `*_keypoints.json`
       --fps        : video FPS (use exact, e.g., 29.97)
       --out        : output TSV path
       --high-k     : MAD multiplier for start threshold
       --low-k      : MAD multiplier for end threshold
       --dist-thr   : pose-hold distance (in shoulder-width units)
       --min-len-ms : minimum segment duration (ms)
       --merge-gap-ms : merge gaps shorter than this (ms)
       --still-frames : consecutive low frames required to end a segment
    """
    ap = argparse.ArgumentParser(description="OpenPose → Gesture Segmentation (wrists + hands)")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Folder with *_keypoints.json")
    ap.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    ap.add_argument("--out", type=Path, default=Path("gestures.tsv"), help="Output TSV")

    ap.add_argument("--high-k", type=float, default=3.0, help="MAD multiplier for start")
    ap.add_argument("--low-k", type=float, default=1.5, help="MAD multiplier for end")
    ap.add_argument("--dist-thr", type=float, default=0.05, help="Pose-change threshold (shoulder units)")
    ap.add_argument("--min-len-ms", type=int, default=200, help="Minimum segment duration (ms)")
    ap.add_argument("--merge-gap-ms", type=int, default=150, help="Merge gap (ms)")
    ap.add_argument("--still-frames", type=int, default=3, help="Consecutive low frames to end")
    args = ap.parse_args()

    frames = load_frames(args.input, args.fps)
    if not frames:
        print("No frames found.")
        return

    # Right side
    # tR_w, vR_w = simple_velocity_wrist(frames, side="R")
    tR_w, vR_w = wrist_velocity_normalized(frames, side="R", fps=args.fps)
    tR_h, vR_h = velocity_hand_normalized(frames, side="R", conf_min=0.2)
    vR = np.nanmax(np.vstack([np.nan_to_num(vR_w), np.nan_to_num(vR_h)]), axis=0)
    # segR, thrR = segment_by_threshold(tR_w, vR)
    # segR, thrR = segment_by_hysteresis(tR_w, vR)
    coordsR = _wrist_coords_norm(frames, "R")
    segR, thrR = segment_with_pose_extension(
        tR_w, vR, coordsR,
        high_k=args.high_k, low_k=args.low_k,
        dist_thr=args.dist_thr,
        min_len_ms=args.min_len_ms,
        merge_gap_ms=args.merge_gap_ms,
        still_frames=args.still_frames
    )

    # Left side
    # tL_w, vL_w = simple_velocity_wrist(frames, side="L")
    tL_w, vL_w = wrist_velocity_normalized(frames, side="L", fps=args.fps)
    tL_h, vL_h = velocity_hand_normalized(frames, side="L", conf_min=0.2)
    vL = np.nanmax(np.vstack([np.nan_to_num(vL_w), np.nan_to_num(vL_h)]), axis=0)
    # segL, thrL = segment_by_threshold(tL_w, vL)
    # segL, thrL = segment_by_hysteresis(tL_w, vL)
    coordsL = _wrist_coords_norm(frames, "L")
    segL, thrL = segment_with_pose_extension(
        tL_w, vL, coordsL,
        high_k=args.high_k, low_k=args.low_k,
        dist_thr=args.dist_thr,
        min_len_ms=args.min_len_ms,
        merge_gap_ms=args.merge_gap_ms,
        still_frames=args.still_frames
    )

    # Union: gesture starts/ends if **either** side moves
    segU = union_segments(segL, segR)

    # Write one TSV tier (union) — ELAN import: CSV / Tab-delimited Text
    from .export import write_tsv
    write_tsv(segU, args.out, tier="gestures", label="gesture")

    # print(f"R thr={thrR:.3f}, L thr={thrL:.3f}, segments: L={len(segL)} R={len(segR)} U={len(segU)}")

    print(
        f"R thr_high={thrR['thr_high']:.3f}, R thr_low={thrR['thr_low']:.3f}, "
        f"L thr_high={thrL['thr_high']:.3f}, L thr_low={thrL['thr_low']:.3f}, "
        f"segments: L={len(segL)} R={len(segR)} U={len(segU)}"
    )
