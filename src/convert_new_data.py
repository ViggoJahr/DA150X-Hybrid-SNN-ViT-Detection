"""
convert_new_data.py
 
Converts the new pre-processed traffic dataset into the format expected by
SNN_final_model.py / data_loading.py.
 
Input (new dataset structure):
  week_XX/box_Y/<timestamp>_recordings/
    event_frames/clip_XXX_frames.h5   -> dataset 'frames' [5400, 480, 640]
    event_labels/clip_XXX.h5          -> dataset 'labels' [5400,] YOLO flat format
 
Output (model format):
  <output_dir>/week_XX-box_Y/clip_XXX.pt
  Each .pt file is a list of 5 sparse tensors:
    [0] event frames:   sparse [5400, 256, 256]
    [1] person heatmap: sparse [5400, 64, 64]  (class 0)
    [2] car heatmap:    sparse [5400, 64, 64]  (class 2)
    [3] bus heatmap:    sparse [5400, 64, 64]  (class 5)
    [4] truck heatmap:  sparse [5400, 64, 64]  (class 7)
 
Usage:
  python3 convert_new_data.py <input_data_root> <output_dir>
 
  Example:
  python3 convert_new_data.py \
    data/raw/raw-v2 \
    data/raw_scaled/
"""
 
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import h5py
 
 
# YOLO class ID -> heatmap index mapping
CLASS_TO_INDEX = {
    0: 0,  # person -> heatmap index 0
    2: 1,  # car    -> heatmap index 1
    5: 2,  # bus    -> heatmap index 2
    7: 3,  # truck  -> heatmap index 3
}
 
EVENT_SIZE   = 256  # resize event frames to this
HEATMAP_SIZE = 64   # heatmap resolution
 
 
def make_gaussian_heatmap(cx, cy, w, h, size=64):
    cx_px = cx * size
    cy_px = cy * size
    w_px  = w  * size
    h_px  = h  * size
    sigma = max(max(w_px, h_px) / 6.0, 1.0)
    y  = np.arange(size, dtype=np.float32)
    x  = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - cx_px)**2 + (yy - cy_px)**2) / (2 * sigma**2))
 
 
def labels_to_heatmaps(label_dataset, num_frames=5400):
    """
    Convert YOLO flat labels from h5 dataset to 4 heatmap tensors.
    label_dataset: h5py dataset shape (num_frames,), each entry is
                   [class, cx, cy, w, h, class, cx, cy, w, h, ...]
    """
    heatmaps = [
        np.zeros((num_frames, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        for _ in range(4)
    ]
 
    for frame_idx in range(num_frames):
        entry = np.array(label_dataset[frame_idx], dtype=np.float32)
        if len(entry) == 0:
            continue
        num_dets = len(entry) // 6
        if num_dets == 0:
            continue
        dets = entry[:num_dets * 6].reshape(num_dets, 6)
        for det in dets:
            class_id = int(det[0])
            cx, cy, w, h, conf = det[1], det[2], det[3], det[4], det[5]
            if conf < 0.7:
                continue
            if class_id not in CLASS_TO_INDEX:
                continue
            hmap_idx = CLASS_TO_INDEX[class_id]
            gaussian  = make_gaussian_heatmap(cx, cy, w, h, HEATMAP_SIZE) * 0.03
            heatmaps[hmap_idx][frame_idx] = np.maximum(
                heatmaps[hmap_idx][frame_idx], gaussian
            )
    return heatmaps
 
 
def resize_event_frames(frames_np):
    """
    Resize event frames from [N, 480, 640] (numpy) to [N, 256, 256] (tensor).
    """
    frames = torch.from_numpy(frames_np).float().unsqueeze(1)  # [N, 1, H, W]
    resized = F.interpolate(frames, size=(EVENT_SIZE, EVENT_SIZE),
                            mode='bilinear', align_corners=False)
    return resized.squeeze(1)  # [N, 256, 256]
 
 
def process_clip(event_frames_path, event_labels_path, output_path):
    """Process a single clip and save as .pt bundle."""
 
    # --- Event frames ---
    with h5py.File(event_frames_path, 'r') as f:
        frames_np = f['frames'][:]          # [5400, 480, 640]
    num_frames = frames_np.shape[0]
 
    events_resized = resize_event_frames(frames_np)
    events_sparse  = events_resized.to_sparse()
 
    # --- Labels ---
    with h5py.File(event_labels_path, 'r') as f:
        heatmaps = labels_to_heatmaps(f['labels'], num_frames)
 
    # --- Convert heatmaps to sparse tensors ---
    heatmap_tensors = []
    for i, class_name in enumerate(['person', 'car', 'bus', 'truck']):
        t = torch.from_numpy(heatmaps[i]).to_sparse()
        nonzero = (heatmaps[i].sum(axis=(1, 2)) > 0.01).sum()
        print(f"      {class_name}: {nonzero}/{num_frames} frames with detections")
        heatmap_tensors.append(t)
 
    # --- Save ---
    bundle = [events_sparse] + heatmap_tensors
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(bundle, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Saved: {output_path} ({size_mb:.1f} MB)")
 
 
def discover_samples(input_root, output_root):
    """
    Walk input_root and find all (event_frames_h5, event_labels_h5, output_pt) triples.
    Expected structure:
      input_root/week_XX/box_Y/<timestamp>_recordings/
        event_frames/clip_XXX_frames.h5
        event_labels/clip_XXX.h5
    """
    samples = []
 
    for week_dir in sorted(os.listdir(input_root)):
        week_path = os.path.join(input_root, week_dir)
        if not os.path.isdir(week_path) or not week_dir.startswith("week_"):
            continue
 
        for box_dir in sorted(os.listdir(week_path)):
            box_path = os.path.join(week_path, box_dir)
            if not os.path.isdir(box_path):
                continue
 
            subdir_name = f"{week_dir}-{box_dir}"   # e.g. week_32-box_3
            output_subdir = os.path.join(output_root, subdir_name)
 
            for rec_dir in sorted(os.listdir(box_path)):
                rec_path = os.path.join(box_path, rec_dir)
                if not os.path.isdir(rec_path):
                    continue
 
                ef_dir = os.path.join(rec_path, "event_frames")
                el_dir = os.path.join(rec_path, "event_labels")
 
                if not os.path.isdir(ef_dir) or not os.path.isdir(el_dir):
                    print(f"  WARNING: Missing event_frames or event_labels in {rec_path}")
                    continue
 
                # Match clips: clip_XXX_frames.h5 <-> clip_XXX.h5
                for ef_file in sorted(os.listdir(ef_dir)):
                    if not ef_file.endswith("_frames.h5"):
                        continue
                    clip_id = ef_file.replace("_frames.h5", "")   # e.g. clip_000
                    el_file = f"{clip_id}.h5"
 
                    ef_path = os.path.join(ef_dir, ef_file)
                    el_path = os.path.join(el_dir, el_file)
 
                    if not os.path.exists(el_path):
                        print(f"  WARNING: No matching label file for {ef_file} (expected {el_path})")
                        continue
 
                    out_path = os.path.join(output_subdir, f"{clip_id}.pt")
                    samples.append((ef_path, el_path, out_path))
 
    return samples
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Convert new dataset to SNN model training format"
    )
    parser.add_argument("input_dir",  help="Root of new dataset (e.g. data/raw/raw-v2)")
    parser.add_argument("output_dir", help="Output directory   (e.g. data/raw_scaled/)")
    args = parser.parse_args()
 
    samples = discover_samples(args.input_dir, args.output_dir)
    print(f"Found {len(samples)} clips to convert.\n")
 
    # Group by output subdir for nicer progress output
    subdirs = {}
    for ef, el, out in samples:
        key = os.path.basename(os.path.dirname(out))
        subdirs.setdefault(key, []).append((ef, el, out))
 
    total = len(samples)
    done  = 0
 
    for subdir, clips in subdirs.items():
        print(f"\n=== {subdir} ({len(clips)} clips) ===")
        for ef_path, el_path, out_path in clips:
            clip_name = os.path.basename(out_path)
            done += 1
            print(f"  [{done}/{total}] {clip_name}")
            process_clip(ef_path, el_path, out_path)
 
    print("\n" + "=" * 60)
    print(f"Done! Output in: {args.output_dir}")
    print(f"\nTo train:")
    print(f"  CUDA_VISIBLE_DEVICES=2 python3 SNN_final_model.py \\")
    print(f"    {args.output_dir} data/model_output/ --gpu 0")
 
 
if __name__ == "__main__":
    main()
