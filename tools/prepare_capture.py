"""
Prepare a saved raw-count NPZ as an unannotated white-hot display test image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """
    Scale a local capture's counts to display intensities; never modify the capture.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    args = parser.parse_args()
    with np.load(args.capture, allow_pickle=False) as capture:
        counts = capture["raw_counts"].astype(np.float32)
    if counts.ndim != 2 or not counts.size or not np.isfinite(counts).all():
        raise ValueError("Expected a finite two-dimensional raw_counts plane")
    low, high = float(counts.min()), float(counts.max())
    gray = (counts - low) * (255 / max(high - low, 1))
    gray = np.rot90(gray, args.rotate // 90).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), gray):
        raise ValueError("Could not write prepared display image")


if __name__ == "__main__":
    main()
