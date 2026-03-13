"""
convert_new_data.py

Converts the new pre-processed traffic dataset into the format expected by
SNN_final_model.py / data_loading.py.

Input (new dataset):
  - event_frames/event_frames_000.pt  -> dense tensor [5400, 480, 640]
  - labels/clip_000_labels.h5         -> YOLO format [class, cx, cy, w, h, ...] normalized

Output (model format):
  - A single .pt file containing a list of 5 sparse tensors:
    [0] event frames:   sparse [5400, 256, 256]
    [1] person heatmap: sparse [5400, 64, 64]  (class 0)
    [2] car heatmap:    sparse [5400, 64, 64]  (class 2)
    [3] bus heatmap:    sparse [5400, 64, 64]  (class 5)
    [4] truck heatmap:  sparse [5400, 64, 64]  (class 7)

Usage:
  python3 convert_new_data.py <input_data_root> <output_dir>

  Example:
  python3 convert_new_data.py \
    "data/2026-03-10 da150x-trafficdata" \
    data/training_output/
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import h5py


# YOLO class ID -> heatmap index mapping
# Model has 4 heads: person, car, bus, truck
CLASS_TO_INDEX = {
    0: 0,  # person -> heatmap index 0
    2: 1,  # car    -> heatmap index 1
    5: 2,  # bus    -> heatmap index 2
    7: 3,  # truck  -> heatmap index 3
}

EVENT_SIZE = 256    # resize event frames to this
HEATMAP_SIZE = 64   # heatmap resolution


def make_gaussian_heatmap(cx, cy, w, h, size=64):
    """
    Generate a 2D Gaussian heatmap blob for one detection.
    cx, cy, w, h are in normalized [0,1] coordinates.
    Returns a (size, size) numpy array.
    """
    # Convert normalized coords to pixel coords in heatmap space
    cx_px = cx * size
    cy_px = cy * size
    w_px = w * size
    h_px = h * size

    # Sigma proportional to object size, with a minimum
    sigma = max(w_px, h_px) / 6.0
    sigma = max(sigma, 1.0)

    # Create coordinate grids
    y = np.arange(size, dtype=np.float32)
    x = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian
    gaussian = np.exp(-((xx - cx_px) ** 2 + (yy - cy_px) ** 2) / (2 * sigma ** 2))

    return gaussian


def labels_to_heatmaps(label_array, num_frames=5400):
    """
    Convert YOLO labels from h5 file to 4 heatmap tensors.

    Args:
        label_array: h5py dataset with shape (num_frames,), each entry is a
                     flat array [class, cx, cy, w, h, class, cx, cy, w, h, ...]
        num_frames: number of frames

    Returns:
        List of 4 tensors, each [num_frames, 64, 64], one per class
    """
    heatmaps = [
        np.zeros((num_frames, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for _ in range(4)
    ]

    for frame_idx in range(num_frames):
        entry = np.array(label_array[frame_idx], dtype=np.float32)

        if len(entry) == 0:
            continue

        # Reshape to (num_detections, 5): [class, cx, cy, w, h]
        num_dets = len(entry) // 5
        if num_dets == 0:
            continue

        dets = entry[:num_dets * 5].reshape(num_dets, 5)

        for det in dets:
            class_id = int(det[0])
            cx, cy, w, h = det[1], det[2], det[3], det[4]

            if class_id not in CLASS_TO_INDEX:
                continue

            hmap_idx = CLASS_TO_INDEX[class_id]
            gaussian = make_gaussian_heatmap(cx, cy, w, h, HEATMAP_SIZE)

            # Take max (don't add) so overlapping detections don't blow up values
            gaussian = gaussian * 0.03
            heatmaps[hmap_idx][frame_idx] = np.maximum(
                heatmaps[hmap_idx][frame_idx], gaussian
            )

    return heatmaps


def resize_event_frames(frames_tensor):
    """
    Resize event frames from [N, 480, 640] to [N, 256, 256].
    Uses bilinear interpolation.
    """
    # F.interpolate expects [N, C, H, W]
    frames = frames_tensor.to_dense().float().unsqueeze(1)  # [N, 1, 480, 640]
    resized = F.interpolate(frames, size=(EVENT_SIZE, EVENT_SIZE), mode='bilinear',
                            align_corners=False)
    return resized.squeeze(1)  # [N, 256, 256]


def process_sample(event_frames_path, labels_path, output_path):
    """
    Process a single recording sample and save in model format.
    """
    print(f"  Loading event frames from {event_frames_path}")
    events = torch.load(event_frames_path, map_location='cpu')

    num_frames = events.shape[0]
    print(f"  Event frames shape: {events.shape}, resizing to [{num_frames}, 256, 256]")

    # Resize events
    events_resized = resize_event_frames(events)

    # Convert to sparse
    events_sparse = events_resized.to_sparse()

    print(f"  Loading labels from {labels_path}")
    with h5py.File(labels_path, 'r') as f:
        label_data = f['labels']
        print(f"  Converting {num_frames} frames of labels to Gaussian heatmaps...")
        heatmaps = labels_to_heatmaps(label_data, num_frames)

    # Convert heatmaps to sparse tensors
    heatmap_tensors = []
    for i, class_name in enumerate(['person', 'car', 'bus', 'truck']):
        t = torch.from_numpy(heatmaps[i]).to_sparse()
        nonzero_frames = (heatmaps[i].sum(axis=(1, 2)) > 0.01).sum()
        print(f"    {class_name}: {nonzero_frames}/{num_frames} frames with detections")
        heatmap_tensors.append(t)

    # Bundle: [events, person, car, bus, truck]
    bundle = [events_sparse] + heatmap_tensors

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(bundle, output_path)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert new dataset to SNN model training format"
    )
    parser.add_argument("input_dir", help="Root of new dataset (e.g. 'data/2026-03-10 da150x-trafficdata')")
    parser.add_argument("output_dir", help="Output directory (e.g. 'data/training_output')")
    args = parser.parse_args()

    input_root = args.input_dir
    output_root = args.output_dir

    # Discover all samples
    samples = []
    for week_dir in sorted(os.listdir(input_root)):
        week_path = os.path.join(input_root, week_dir)
        if not os.path.isdir(week_path) or not week_dir.startswith("week_"):
            continue

        for box_dir in sorted(os.listdir(week_path)):
            box_path = os.path.join(week_path, box_dir)
            if not os.path.isdir(box_path):
                continue

            for rec_dir in sorted(os.listdir(box_path)):
                rec_path = os.path.join(box_path, rec_dir)
                if not os.path.isdir(rec_path):
                    continue

                event_pt = os.path.join(rec_path, "event_frames", "event_frames_000.pt")
                labels_h5 = os.path.join(rec_path, "labels", "clip_000_labels.h5")

                if os.path.exists(event_pt) and os.path.exists(labels_h5):
                    # Output subdir name: e.g. "week_32-box_3"
                    subdir_name = f"{week_dir}-{box_dir}"
                    output_path = os.path.join(
                        output_root, subdir_name, "clip_000.pt"
                    )
                    samples.append((event_pt, labels_h5, output_path))
                else:
                    print(f"  WARNING: Missing files in {rec_path}")

    print(f"Found {len(samples)} samples to convert:\n")

    for i, (event_pt, labels_h5, output_path) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Processing: {os.path.dirname(event_pt).replace('/event_frames', '')}")
        process_sample(event_pt, labels_h5, output_path)
        print()

    print("=" * 60)
    print(f"Done! Output in: {output_root}/")
    print(f"Subdirectories: {[os.path.basename(os.path.dirname(s[2])) for s in samples]}")
    print(f"\nTo train:")
    print(f"  CUDA_VISIBLE_DEVICES=2 python3 SNN_final_model.py \\")
    print(f"    {output_root}/ data/model_output/ --gpu 0")


if __name__ == "__main__":
    main()
