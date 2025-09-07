# src/openpose_gesture_segmentation/segmentation.py
import numpy as np


def simple_velocity_wrist(frames, side="R"):
    """
       Compute per-frame wrist displacement (pixels per frame) for the given side.

       Parameters
       ----------
       frames : Sequence[dict]
           Per-frame records from `io.load_frames`, each containing e.g. "R_wrist", "L_wrist", "t_ms".
           Wrist entries are (x, y, c).
       side : {"R", "L"}, default "R"
           Which wrist to use.

       Returns
       -------
       t_ms : ndarray, shape (N,)
           Timestamps in milliseconds for each frame.
       v : ndarray, shape (N,)
           Euclidean displacement per frame in pixels/frame (first value is 0).
           Note: multiply by FPS to convert to pixels/second.
    """
    # side: "R" or "L"
    xs = np.array([f[f"{side}_wrist"][0] for f in frames], float)
    ys = np.array([f[f"{side}_wrist"][1] for f in frames], float)
    v = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0]))
    t = np.array([f["t_ms"] for f in frames], float)
    return t, v


# todo this is old velocity normalization (calculating only wrist-wise)
# def wrist_velocity_normalized(frames, side="R", fps=30.0):
#     xs = np.array([f[f"{side}_wrist"][0] for f in frames], float)
#     ys = np.array([f[f"{side}_wrist"][1] for f in frames], float)
#     sw = np.array([
#         np.hypot(f["R_wrist"][0] - f["L_wrist"][0],
#                  f["R_wrist"][1] - f["L_wrist"][1])
#         for f in frames
#     ], float)
#
#     dx = np.diff(xs, prepend=xs[0])
#     dy = np.diff(ys, prepend=ys[0])
#     v = np.hypot(dx, dy) * fps  # pixels/sec
#     v_norm = np.divide(v, sw, out=np.zeros_like(v), where=sw > 1e-6)  # normalize
#     t = np.array([f["t_ms"] for f in frames], float)
#     return t, v_norm

def wrist_velocity_normalized(frames, side="R", fps=30.0):
    """
       Compute wrist velocity normalized by shoulder width (shoulder-widths per second).

       Parameters
       ----------
       frames : Sequence[dict]
           Per-frame records containing `{side}_wrist` and "shoulder".
           "shoulder" is shoulder width in pixels (computed in io.py).
       side : {"R", "L"}, default "R"
           Which wrist to use.
       fps : float, default 30.0
           Video frame rate.

       Returns
       -------
       t_ms : ndarray, shape (N,)
           Timestamps in ms.
       v_norm : ndarray, shape (N,)
           Wrist velocity in shoulder-widths/second.
           (pixel displacement per frame * fps) / shoulder_width.
       """
    xs = np.array([f[f"{side}_wrist"][0] for f in frames], float)
    ys = np.array([f[f"{side}_wrist"][1] for f in frames], float)

    # use shoulder width computed in io.py
    sw = np.array([f.get("shoulder", np.nan) for f in frames], float)
    # safe fallback: median shoulder width if missing
    sw_med = np.nanmedian(sw[np.isfinite(sw)]) if np.any(np.isfinite(sw)) else np.nan
    sw_safe = np.where(np.isfinite(sw) & (sw > 1e-6),
                       sw,
                       sw_med if np.isfinite(sw_med) and sw_med > 1e-6 else 1.0)

    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    v = np.hypot(dx, dy) * fps  # pixels/sec
    v_norm = v / sw_safe  # shoulder-widths per second
    t = np.array([f["t_ms"] for f in frames], float)
    return t, v_norm


def velocity_hand_normalized(frames, side="R", conf_min=0.2, smooth_w=5):
    """
       Compute hand motion normalized by shoulder width (shoulder-widths per frame or per second if you scale).

       Parameters
       ----------
       frames : Sequence[dict]
           Per-frame records with "handR"/"handL" and "shoulder".
       side : {"R", "L"}, default "R"
           Which hand to use.
       conf_min : float, default 0.2
           Minimum keypoint confidence to consider a point valid.
       smooth_w : int, default 5
           Moving average window for smoothing.

       Returns
       -------
       t_ms : ndarray
           Timestamps in ms.
       V_norm : ndarray
           Hand motion normalized by shoulder width.
           (If you want per second, multiply the raw hand velocity by fps before normalizing.)
    """
    # reuse your existing velocity_hand to get raw speed
    t, V = velocity_hand(frames, side=side, conf_min=conf_min, smooth_w=smooth_w)

    sw = np.array([f.get("shoulder", np.nan) for f in frames], float)
    sw_med = np.nanmedian(sw[np.isfinite(sw)]) if np.any(np.isfinite(sw)) else np.nan
    sw_safe = np.where(np.isfinite(sw) & (sw > 1e-6),
                       sw,
                       sw_med if np.isfinite(sw_med) and sw_med > 1e-6 else 1.0)

    V_norm = V / sw_safe
    return t, V_norm


def velocity_hand(frames, side="R", conf_min=0.2, smooth_w=5):
    """
       Compute hand motion as the median per-frame speed across 21 hand keypoints.

       Each frame's hand speed is the median Euclidean displacement between consecutive
       frames for all keypoints whose confidence >= conf_min in both frames.
       A moving-average smoothing can be applied.

       Parameters
       ----------
       frames : Sequence[dict]
           Per-frame records with "handR"/"handL" = (x21, y21, c21) or (None, None, None).
       side : {"R", "L"}, default "R"
           Which hand to use.
       conf_min : float, default 0.2
           Minimum keypoint confidence to consider a point valid.
       smooth_w : int, default 5
           Moving average window size (frames). Use 1 to disable smoothing.

       Returns
       -------
       t_ms : ndarray, shape (N,)
           Timestamps in ms.
       V : ndarray, shape (N,)
           Hand speed (pixels/frame), smoothed (first valid frame is NaN by definition).
           Multiply by fps to get pixels/sec if needed.
    """
    """
    Median speed across the 21 hand points, using only points with c>=conf_min
    in BOTH consecutive frames. Returns per-frame velocity (length = n_frames).
    """
    V = []
    prev = None
    for f in frames:
        x, y, c = f[f"hand{side}"]
        if x is None:
            V.append(np.nan)
            prev = None
            continue
        x = np.asarray(x);
        y = np.asarray(y);
        c = np.asarray(c)
        good = c >= conf_min
        if prev is None:
            V.append(np.nan)
            prev = (x, y, good)
            continue
        px, py, pgood = prev
        vx = x - px;
        vy = y - py
        speed = np.hypot(vx, vy)
        mask = good & pgood
        V.append(np.nanmedian(speed[mask]) if np.any(mask) else np.nan)
        prev = (x, y, good)

    V = np.array(V, float)
    if smooth_w > 1:
        kernel = np.ones(smooth_w) / smooth_w
        V = np.convolve(V, kernel, mode="same")
    t = np.array([f["t_ms"] for f in frames], float)
    return t, V


def segment_by_threshold(times_ms, vel, k=3.0, min_len_ms=200):
    """
        Segment gestures using a single adaptive threshold.

        Start when velocity > median + k*MAD; end when it falls back below that threshold.
        Discard segments shorter than `min_len_ms`.

        Parameters
        ----------
        times_ms : Iterable[float]
            Timestamps in ms (one per frame).
        vel : Iterable[float]
            Velocity/activation signal (same length as times_ms).
        k : float, default 3.0
            MAD multiplier above the median to set the threshold.
        min_len_ms : int, default 200
            Minimum duration in ms for a segment to be kept.

        Returns
        -------
        segments : list[(start_ms, end_ms)]
            Start/end times in ms for each detected segment.
        thr : float
            The threshold used (median + k*MAD), helpful for debugging.
    """
    med = np.nanmedian(vel)
    mad = np.nanmedian(np.abs(vel - med)) or 1.0
    thr = med + k * mad
    segments = []
    active = None
    for t, val in zip(times_ms, vel):
        if active is None and np.isfinite(val) and val > thr:
            active = t
        elif active is not None and (not np.isfinite(val) or val <= thr):
            if t - active >= min_len_ms:
                segments.append((int(active), int(t)))
            active = None
    if active is not None:
        segments.append((int(active), int(times_ms[-1])))
    return segments, float(thr)


def segment_by_hysteresis(times_ms, vel,
                          high_k=3.0, low_k=1.5,
                          min_len_ms=200, merge_gap_ms=150,
                          still_frames=3):
    """
    Segment gestures with hysteresis:
      - Start when velocity > thr_high
      - End when velocity < thr_low for `still_frames` in a row
    Args:
        times_ms: array of timestamps in ms
        vel: velocity values
        high_k: how many MAD above median to start a gesture
        low_k: how many MAD above median to end a gesture
        min_len_ms: drop gestures shorter than this
        merge_gap_ms: merge gestures closer than this gap
        still_frames: require this many consecutive low-frames before ending
    Returns:
        segments: list[[start_ms, end_ms], ...]
        thresholds: dict with thr_high, thr_low
    """

    v = np.asarray(vel, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) or 1.0
    thr_high = med + high_k * mad
    thr_low = med + low_k * mad

    segments = []
    active = None
    low_count = 0

    for t, val in zip(times_ms, v):
        if active is None:
            if np.isfinite(val) and val > thr_high:
                active = t
                low_count = 0
        else:
            if not np.isfinite(val) or val < thr_low:
                low_count += 1
                if low_count >= still_frames:
                    if t - active >= min_len_ms:
                        segments.append([int(active), int(t)])
                    active = None
                    low_count = 0
            else:
                low_count = 0

    if active is not None:
        segments.append([int(active), int(times_ms[-1])])

    # Merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= merge_gap_ms:
                prev[1] = max(prev[1], seg[1])
            else:
                merged.append(seg)

    return merged, {"thr_high": thr_high, "thr_low": thr_low}


def segment_with_pose_extension(times_ms, vel, coords,
                                high_k=3.0, low_k=1.5,
                                dist_thr=0.05,  # normalized shoulder units
                                min_len_ms=200, merge_gap_ms=150,
                                still_frames=3):
    """
       Segment gestures with hysteresis AND pose-hold extension.

       Starts like hysteresis: start when vel > median + high_k*MAD.
       Ending is delayed if the pose (wrist position) remains near the start pose even
       after velocity falls below the low threshold—useful for static gestures (e.g., thumbs-up).

       Parameters
       ----------
       times_ms : Iterable[float]
           Timestamps in ms.
       vel : Iterable[float]
           Velocity/activation signal (e.g., max of wrist and hand, normalized).
       coords : Sequence[(x_norm, y_norm)]
           Wrist coordinates per frame, normalized by shoulder width and centered
           at shoulder midpoint (units: shoulder widths).
       high_k : float, default 3.0
           Start threshold factor (see hysteresis).
       low_k : float, default 1.5
           End threshold factor (see hysteresis).
       dist_thr : float, default 0.05
           If |coord - ref_coord| < dist_thr while vel is low, keep extending segment.
           Increase to end sooner; decrease to hold longer.
       min_len_ms : int, default 200
           Minimum segment duration (ms).
       merge_gap_ms : int, default 150
           Merge adjacent segments if gap <= this (ms).
       still_frames : int, default 3
           Require this many consecutive low frames (with pose change) to end.

       Returns
       -------
       segments : list[[start_ms, end_ms]]
           Cleaned segments with pose-hold extension.
       thresholds : dict
           {"thr_high": float, "thr_low": float, "dist_thr": float}.
    """
    v = np.asarray(vel, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) or 1.0
    thr_high = med + high_k * mad
    thr_low = med + low_k * mad

    segments = []
    active = None
    ref_coord = None
    low_count = 0

    for t, val, coord in zip(times_ms, v, coords):
        if active is None:
            if np.isfinite(val) and val > thr_high:
                active = t
                ref_coord = coord  # remember pose at gesture start
                low_count = 0
        else:
            dx = coord[0] - ref_coord[0] if coord[0] is not None else 0
            dy = coord[1] - ref_coord[1] if coord[1] is not None else 0
            dist = np.hypot(dx, dy)

            if (not np.isfinite(val) or val < thr_low) and dist < dist_thr:
                # velocity low but pose unchanged → extend gesture
                pass
            elif (not np.isfinite(val) or val < thr_low) and dist >= dist_thr:
                low_count += 1
                if low_count >= still_frames:
                    if t - active >= min_len_ms:
                        segments.append([int(active), int(t)])
                    active = None
                    ref_coord = None
                    low_count = 0
            else:
                low_count = 0

    if active is not None:
        segments.append([int(active), int(times_ms[-1])])

    # merge close segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            if seg[0] - prev[1] <= merge_gap_ms:
                prev[1] = max(prev[1], seg[1])
            else:
                merged.append(seg)

    return merged, {"thr_high": thr_high, "thr_low": thr_low, "dist_thr": dist_thr}


def union_segments(A, B, merge_gap_ms=150):
    """
       Merge two segment lists (e.g., left and right) into their timewise union.

       The result is sorted by start time and merges overlapping/nearby intervals
       within `merge_gap_ms`.

       Parameters
       ----------
       A, B : sequences of [start_ms, end_ms]
           Segment lists to union (e.g., left-hand segments and right-hand segments).
       merge_gap_ms : int, default 150
           Merge if the gap between two neighboring intervals <= this (ms).

       Returns
       -------
       union : list[[start_ms, end_ms]]
           Merged, non-overlapping segments covering any activity from A or B.
       """
    all_seg = sorted(A + B, key=lambda s: (s[0], s[1]))
    out = []
    for s in all_seg:
        seg = list(s)  # make mutable
        if not out:
            out.append(seg)
        else:
            prev = out[-1]
            if seg[0] <= prev[1] + merge_gap_ms:
                prev[1] = max(prev[1], seg[1])
            else:
                out.append(seg)
    return out
