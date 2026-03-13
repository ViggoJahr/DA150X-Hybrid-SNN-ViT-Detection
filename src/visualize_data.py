#!/usr/bin/env python3
"""
DA150X Heatmap Overlay Visualizer
Generates videos showing the Gaussian target heatmaps (what the model learns
to predict) overlaid on event camera frames.

Usage:
  # One recording:
  python3 visualize_data.py "data/2026-03-10 da150x-trafficdata/week_32/box_3"

  # All recordings:
  python3 visualize_data.py "data/2026-03-10 da150x-trafficdata" --all

  # Custom fps or training data path:
  python3 visualize_data.py "data/2026-03-10 da150x-trafficdata" --all --fps 30 \
      --training_data data/training_output_scaled/
"""

import argparse
import glob
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

# ─── Class definitions ───────────────────────────────────────────────
CLASS_MAP = {
    0: ("person",  (0,  120, 255)),
    2: ("car",     (255, 178, 50)),
    5: ("bus",     (50,  205, 50)),
    7: ("truck",   (0,   215, 255)),
}

HEATMAP_COLORS_RGB = {
    "person": np.array([255, 80,  80]),
    "car":    np.array([80,  160, 255]),
    "bus":    np.array([80,  220, 80]),
    "truck":  np.array([255, 200, 50]),
}

DEFAULT_COLOR = (200, 200, 200)

RECORDING_TO_TRAINING = {
    "week_32-box_3": "week_32-box_3",
    "week_33-box_2": "week_33-box_2",
    "week_34-box_1": "week_34-box_1",
    "week_35-box_2": "week_35-box_2",
    "week_36-box_3": "week_36-box_3",
}


def parse_labels_for_frame(label_entry, img_w, img_h):
    detections = []
    if label_entry is None or len(label_entry) == 0:
        return detections
    arr = np.array(label_entry, dtype=np.float32).flatten()
    if len(arr) % 5 != 0:
        return detections
    for i in range(len(arr) // 5):
        cls_id = int(arr[i * 5])
        cx, cy, w, h = arr[i*5+1], arr[i*5+2], arr[i*5+3], arr[i*5+4]
        cx_px, cy_px = cx * img_w, cy * img_h
        w_px, h_px = w * img_w, h * img_h
        x1 = int(cx_px - w_px / 2)
        y1 = int(cy_px - h_px / 2)
        x2 = int(cx_px + w_px / 2)
        y2 = int(cy_px + h_px / 2)
        name, color = CLASS_MAP.get(cls_id, (f"cls_{cls_id}", DEFAULT_COLOR))
        detections.append({
            "class_name": name, "color": color,
            "x1": max(0, x1), "y1": max(0, y1),
            "x2": min(img_w - 1, x2), "y2": min(img_h - 1, y2),
        })
    return detections


def draw_detections(frame, detections, thickness=1, font_scale=0.4):
    for det in detections:
        color = det["color"]
        cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]),
                      color, thickness)
        label = det["class_name"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, 1)
        cv2.rectangle(frame,
                      (det["x1"], det["y1"] - th - baseline - 4),
                      (det["x1"] + tw + 4, det["y1"]),
                      color, -1)
        cv2.putText(frame, label, (det["x1"] + 2, det["y1"] - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)


def event_frame_to_bgr(event_frame, colormap=cv2.COLORMAP_INFERNO):
    arr = np.array(event_frame, dtype=np.float32)
    vmax = arr.max()
    if vmax > 0:
        arr = (arr / vmax * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    return cv2.applyColorMap(arr, colormap)


def make_heatmap_overlay(event_bgr, heatmaps, alpha=0.5):
    h, w = event_bgr.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    for name, hmap in heatmaps.items():
        if name not in HEATMAP_COLORS_RGB:
            continue
        hmap = np.array(hmap, dtype=np.float32)
        vmax = hmap.max()
        if vmax <= 0:
            continue
        hmap_norm = hmap / vmax
        hmap_resized = cv2.resize(hmap_norm, (w, h), interpolation=cv2.INTER_LINEAR)
        color_bgr = HEATMAP_COLORS_RGB[name][::-1].astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = np.maximum(overlay[:, :, c],
                                           hmap_resized * color_bgr[c])

    result = event_bgr.astype(np.float32)
    mask = overlay.max(axis=2) > 10
    result[mask] = result[mask] * (1 - alpha) + overlay[mask] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def find_recording_dir(base_path):
    if os.path.isdir(os.path.join(base_path, "event_frames")):
        return base_path
    for pattern in [os.path.join(base_path, "*_recordings"),
                    os.path.join(base_path, "*", "*_recordings")]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return base_path


def find_all_recordings(base_path):
    recordings = []
    for week_dir in sorted(glob.glob(os.path.join(base_path, "week_*"))):
        for box_dir in sorted(glob.glob(os.path.join(week_dir, "box_*"))):
            rec_dir = find_recording_dir(box_dir)
            if os.path.isdir(os.path.join(rec_dir, "event_frames")):
                recordings.append(rec_dir)
    return recordings


def get_rec_name(recording_dir):
    parts = recording_dir.rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p.startswith("week_"):
            box_part = parts[i + 1] if i + 1 < len(parts) else ""
            return f"{p}-{box_part}"
    return "recording"


def load_labels(labels_path):
    with h5py.File(labels_path, "r") as f:
        labels_ds = f["labels"]
        labels = []
        for i in range(len(labels_ds)):
            entry = labels_ds[i]
            if entry is not None and len(entry) > 0:
                labels.append(np.array(entry, dtype=np.float32))
            else:
                labels.append(np.array([], dtype=np.float32))
    return labels


def load_event_frames(event_path):
    print(f"  Loading event frames...")
    data = torch.load(event_path, map_location="cpu")
    if data.is_sparse:
        print("    (sparse -> dense)")
        data = data.to_dense()
    return data.numpy()


def load_training_data(path):
    print(f"  Loading training .pt (heatmaps)...")
    data = torch.load(path, map_location="cpu")
    return [t.to_dense().numpy() if t.is_sparse else t.numpy() for t in data]


def generate_heatmap_video(recording_dir, output_dir, fps, training_data_dir):
    event_path = os.path.join(recording_dir, "event_frames", "event_frames_000.pt")
    labels_path = os.path.join(recording_dir, "labels", "clip_000_labels.h5")
    rec_name = get_rec_name(recording_dir)

    print(f"\n{'='*60}")
    print(f"Processing: {rec_name}")
    print(f"{'='*60}")

    if not os.path.isfile(event_path):
        print("  ERROR: event frames not found. Skipping.")
        return
    if not os.path.isfile(labels_path):
        print("  ERROR: labels not found. Skipping.")
        return

    training_pt = None
    if rec_name in RECORDING_TO_TRAINING:
        tp = os.path.join(training_data_dir, RECORDING_TO_TRAINING[rec_name], "clip_000.pt")
        if os.path.isfile(tp):
            training_pt = tp
    if not training_pt:
        print(f"  ERROR: training .pt not found for {rec_name}. Skipping.")
        print(f"         Looked in: {training_data_dir}")
        return

    labels = load_labels(labels_path)
    events = load_event_frames(event_path)
    train_data = load_training_data(training_pt)

    heatmap_names = ["person", "car", "bus", "truck"]
    heatmap_arrays = {heatmap_names[i]: train_data[i + 1] for i in range(4)}

    n_frames = min(len(labels), events.shape[0])
    source_fps = 90
    frame_step = max(1, source_fps // fps)
    frame_indices = list(range(0, n_frames, frame_step))
    actual_fps = source_fps / frame_step

    EVENT_H, EVENT_W = 480, 640
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{rec_name}_heatmap_overlay.mp4")
    print(f"  Output: {os.path.basename(out_path)} | {actual_fps:.0f} fps | {len(frame_indices)} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (EVENT_W, EVENT_H))

    t0 = time.time()
    for count, fi in enumerate(frame_indices):
        bgr = event_frame_to_bgr(events[fi])

        frame_heatmaps = {}
        for name in heatmap_names:
            if fi < heatmap_arrays[name].shape[0]:
                frame_heatmaps[name] = heatmap_arrays[name][fi]

        bgr = make_heatmap_overlay(bgr, frame_heatmaps, alpha=0.6)

        dets = parse_labels_for_frame(labels[fi], EVENT_W, EVENT_H)
        draw_detections(bgr, dets)

        # Heatmap legend
        y = EVENT_H - 110
        for name, color_rgb in HEATMAP_COLORS_RGB.items():
            color_bgr = tuple(int(c) for c in color_rgb[::-1])
            cv2.rectangle(bgr, (10, y), (24, y + 14), color_bgr, -1)
            cv2.putText(bgr, f"{name} heatmap", (30, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
            y += 20

        # Timestamp
        t = fi / source_fps
        cv2.putText(bgr, f"t={t:.2f}s  f={fi}", (EVENT_W - 210, EVENT_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        writer.write(bgr)
        if (count + 1) % 500 == 0:
            print(f"    {count+1}/{len(frame_indices)} ({time.time()-t0:.1f}s)")

    writer.release()
    print(f"  Done: {os.path.getsize(out_path)/1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="DA150X Heatmap Overlay Visualizer")
    parser.add_argument("input_path",
                        help="Recording dir (week_XX/box_X) or dataset root (with --all)")
    parser.add_argument("--output_dir", default="data/visualizations/",
                        help="Output directory (default: data/visualizations/)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output framerate (default: 30)")
    parser.add_argument("--all", action="store_true",
                        help="Process ALL recordings")
    parser.add_argument("--training_data", default=None,
                        help="Path to training data dir (default: auto-detect)")

    args = parser.parse_args()

    training_data = args.training_data
    if training_data is None:
        for candidate in ["data/training_output_scaled/", "data/training_output/"]:
            if os.path.isdir(candidate):
                training_data = candidate
                break
    if not training_data or not os.path.isdir(training_data):
        print("ERROR: Could not find training data directory.")
        print("       Use --training_data <path> to specify it.")
        sys.exit(1)

    print(f"DA150X Heatmap Overlay Visualizer")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Output FPS:     {args.fps}")
    print(f"  Training data:  {training_data}")

    if args.all:
        recordings = find_all_recordings(args.input_path)
        if not recordings:
            print(f"\nERROR: No recordings found under {args.input_path}")
            sys.exit(1)
        print(f"  Found {len(recordings)} recordings")
        for rec_dir in recordings:
            generate_heatmap_video(rec_dir, args.output_dir, args.fps, training_data)
    else:
        rec_dir = find_recording_dir(args.input_path)
        generate_heatmap_video(rec_dir, args.output_dir, args.fps, training_data)

    print(f"\nDone! Videos in: {args.output_dir}")


if __name__ == "__main__":
    main()
