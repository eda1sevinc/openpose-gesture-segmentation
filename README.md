
# OpenPose Gesture Segmentation

Python toolkit for extracting **gesture segments** from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) keypoint JSON files.  
The output is an **ELAN-compatible TSV** file that can be imported into ELAN for annotation and labeling.

---

## Features

- Reads OpenPose `*_keypoints.json` (BODY_25 + hand keypoints).
- Computes **wrist and hand velocities**, normalized by shoulder width (camera-distance invariant).
- Detects gesture segments using:
  - Adaptive thresholds (Median + MAD).
  - Hysteresis (separate start/end thresholds).
  - Pose extension: keeps static poses (e.g., thumbs-up) alive even after motion stops.
- Exports gesture segments to **Tab-delimited text** (ELAN importable).

---

## Installation

Clone the repository and install dependencies with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/eda1sevinc/openpose-gesture-segmentation.git
cd openpose-gesture-segmentation
poetry install
````

---

## Usage

Run the CLI with:

```bash
poetry run opgs \
  -i /path/to/openpose/keypoints \
  --fps 29.97 \
  --out gestures.tsv
```

### Main arguments

* `-i / --input` : folder containing `*_keypoints.json`
* `--fps` : video frame rate (e.g., `29.97`)
* `--out` : output TSV file (default: `gestures.tsv`)

### Advanced options

* `--high-k` : MAD multiplier for start threshold (default: 3.0)
* `--low-k` : MAD multiplier for end threshold (default: 1.5)
* `--dist-thr` : pose-hold threshold in shoulder units (default: 0.05)
* `--min-len-ms` : minimum segment duration in ms (default: 200)
* `--merge-gap-ms` : merge segments if gap smaller than this (default: 150)
* `--still-frames` : consecutive low frames to confirm gesture end (default: 3)

---

## Output

The output file is a **tab-separated text file** with gesture segments:

```
tier	annotator	start	end	value
gestures	system	1234	2345	gesture
gestures	system	3000	4100	gesture
```

You can import it into **ELAN** via:

```
File → Import → Tab-delimited text...
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.


