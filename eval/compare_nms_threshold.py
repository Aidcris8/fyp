"""
Compares NMS IoU threshold 0.45 (YOLO default) vs 0.7 (project value).
Saves side-by-side images to data/nms_comparison/.

Green boxes = detections present at BOTH thresholds.
Red boxes   = only present at 0.45 (merged/suppressed at 0.7 — genuine separate players).
Blue boxes  = only present at 0.7  (shouldn't happen — 0.7 is more permissive).

A red box at 0.45 that disappears at 0.7 means two overlapping boxes were kept at 0.45
but correctly merged into one at 0.7 — OR a real player was suppressed at 0.45.
The visual tells you which.
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/nms_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolo11x-seg.pt")

THRESHOLD_A = 0.45   # YOLO default
THRESHOLD_B = 0.70   # project value

def get_boxes(results):
    boxes_obj = results[0].boxes
    if boxes_obj is None:
        return []
    return [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for b in boxes_obj if int(b.cls[0]) == 0]

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def match_boxes(set_a, set_b, threshold=0.4):
    """Return (shared_a, only_a, only_b) index sets."""
    matched_a, matched_b = set(), set()
    for i, a in enumerate(set_a):
        for j, b in enumerate(set_b):
            if j not in matched_b and iou(a, b) > threshold:
                matched_a.add(i)
                matched_b.add(j)
                break
    only_a = [set_a[i] for i in range(len(set_a)) if i not in matched_a]
    only_b = [set_b[j] for j in range(len(set_b)) if j not in matched_b]
    shared = [set_a[i] for i in matched_a]
    return shared, only_a, only_b

frames = sorted(os.listdir(KEYFRAMES_DIR))
summary = []

for fname in frames:
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, fname))
    if img is None:
        continue
    height, width = img.shape[:2]

    r_a = model(img, classes=[0], conf=0.3, iou=THRESHOLD_A,
                retina_masks=False, verbose=False)
    r_b = model(img, classes=[0], conf=0.3, iou=THRESHOLD_B,
                retina_masks=False, verbose=False)

    boxes_a = get_boxes(r_a)   # 0.45
    boxes_b = get_boxes(r_b)   # 0.70

    shared, only_a, only_b = match_boxes(boxes_a, boxes_b)

    summary.append((fname, len(boxes_a), len(boxes_b), len(only_a), len(only_b)))

    left  = img.copy()
    right = img.copy()

    # Left panel: iou=0.45
    for box in shared:
        cv2.rectangle(left, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 2)
    for box in only_a:
        cv2.rectangle(left, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(left, "extra", (box[0], box[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Right panel: iou=0.70
    for box in shared:
        cv2.rectangle(right, (box[0], box[1]), (box[2], box[3]), (0, 200, 0), 2)
    for box in only_b:
        cv2.rectangle(right, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    for panel, title in [
        (left,  f"NMS iou=0.45 (default)  —  {len(boxes_a)} detections  ({len(only_a)} extra vs 0.70)"),
        (right, f"NMS iou=0.70 (project)  —  {len(boxes_b)} detections"),
    ]:
        cv2.rectangle(panel, (0, 0), (width, 36), (0, 0, 0), -1)
        cv2.putText(panel, title, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined = np.hstack([left, right])
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), combined)

# Print summary
diff_frames = [(f, a, b, oa, ob) for f, a, b, oa, ob in summary if oa > 0 or ob > 0]
total_extra_045 = sum(oa for _, _, _, oa, _ in summary)
total_extra_070 = sum(ob for _, _, _, _, ob in summary)

print(f"\n{'Frame':<65} {'0.45':>5} {'0.70':>5} {'extra@0.45':>10} {'extra@0.70':>10}")
print("-" * 100)
for fname, a, b, oa, ob in summary:
    marker = "  <--" if oa > 0 else ""
    print(f"{fname[-64:]:<65} {a:>5} {b:>5} {oa:>10} {ob:>10}{marker}")

print(f"\n{'─'*60}")
print(f"Total frames                    : {len(summary)}")
print(f"Frames where 0.45 finds more    : {sum(1 for _,_,_,oa,_ in summary if oa > 0)}")
print(f"Frames where 0.70 finds more    : {sum(1 for _,_,_,_,ob in summary if ob > 0)}")
print(f"Total extra detections at 0.45  : {total_extra_045}")
print(f"Total extra detections at 0.70  : {total_extra_070}")
print(f"\nImages saved to {OUTPUT_DIR}/")
