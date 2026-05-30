#Compares single-pass vs two-pass YOLO detection on keyframes.
#Saves side-by-side images to data/detection_comparison/.
#Green boxes = detected in both passes.
#Red boxes   = only found by the second low-confidence pass (what you'd miss with 1 pass).

import cv2
import numpy as np
import os
from ultralytics import YOLO

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/detection_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolo11x-seg.pt")

# pick a spread of frames, edit this list to focus on specific games
FRAMES = sorted(os.listdir(KEYFRAMES_DIR))[:20]

# extract all class-0 (person) bounding boxes from a YOLO result
def get_boxes(results, img_shape):
    boxes_obj = results[0].boxes
    if boxes_obj is None:
        return []
    H, W = img_shape[:2]
    out = []
    for b in boxes_obj:
        if int(b.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        out.append((x1, y1, x2, y2))
    return out

# compute intersection over union between two boxes — used to detect duplicate detections
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

# return True if the box overlaps an existing detection above the IoU threshold
def is_duplicate(box, existing, threshold=0.4):
    return any(iou(box, e) > threshold for e in existing)

# draw coloured bounding boxes on the image with an optional text label above each box
def draw_boxes(img, boxes, colour, label_prefix=""):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        if label_prefix:
            cv2.putText(img, label_prefix, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

# running total of extra detections contributed by the second pass across all frames
extra_found_total = 0

for fname in FRAMES:
    img_path = os.path.join(KEYFRAMES_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    height, width = img.shape[:2]

    # ── Probe run to set adaptive thresholds (same logic as team_identification.py) ──
    results_probe = model(img, classes=[0], conf=0.3, iou=0.7,
                          retina_masks=False, verbose=False)
    probe_boxes = get_boxes(results_probe, img.shape)
    probe_heights = [b[3] - b[1] for b in probe_boxes]

    # set adaptive thresholds based on median detection height — same logic as team_identification.py
    # small median height means the camera is zoomed out, so thresholds are relaxed
    if len(probe_heights) > 3:
        median_h = sorted(probe_heights)[len(probe_heights) // 2]
        min_height = 20 if median_h < 45 else 40
        top_zone   = 0.65 if median_h < 45 else 0.45
    else:
        min_height, top_zone = 25, 0.55

    # single-pass: just the conf=0.3 detections above min_height 
    single_pass = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in probe_boxes
                   if (y2 - y1) > min_height]

    # two-pass: add low-conf detections in the upper zone 
    results_low = model(img, classes=[0], conf=0.05, iou=0.7,
                        retina_masks=False, verbose=False)
    low_boxes = get_boxes(results_low, img.shape)

    # collect low-confidence detections that are in the upper zone and not already detected
    extra_boxes = []
    for box in low_boxes:
        x1, y1, x2, y2 = box
        in_upper_zone = y2 < height * top_zone and (y2 - y1) > 20
        if in_upper_zone and not is_duplicate(box, single_pass):
            extra_boxes.append(box)

    extra_found_total += len(extra_boxes)

    # build comparison image 
    left  = img.copy()   # single-pass
    right = img.copy()   # two-pass

    draw_boxes(left,  single_pass, (0, 200, 0))   # green = single pass detections
    draw_boxes(right, single_pass, (0, 200, 0))   # green = shared detections
    draw_boxes(right, extra_boxes, (0, 0, 255))   # red   = extra from 2nd pass

    # labels
    for panel, title, count in [
        (left,  f"1-pass  ({len(single_pass)} players)", len(single_pass)),
        (right, f"2-pass  ({len(single_pass)+len(extra_boxes)} players,  +{len(extra_boxes)} from 2nd pass)", 0),
    ]:
        cv2.rectangle(panel, (0, 0), (width, 36), (0, 0, 0), -1)
        cv2.putText(panel, title, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # top-zone boundary line
    zone_y = int(height * top_zone)
    cv2.line(left,  (0, zone_y), (width, zone_y), (255, 255, 0), 1)
    cv2.line(right, (0, zone_y), (width, zone_y), (255, 255, 0), 1)
    cv2.putText(right, "upper zone", (4, zone_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # stack both panels side by side and save to disk
    combined = np.hstack([left, right])
    out_path  = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, combined)

    marker = "  <<< EXTRA DETECTIONS" if extra_boxes else ""
    print(f"{fname[-60:]:60s}  1-pass={len(single_pass):2d}  extra={len(extra_boxes)}{marker}")

print(f"\nTotal extra detections from 2nd pass across {len(FRAMES)} frames: {extra_found_total}")
print(f"Images saved to {OUTPUT_DIR}/")
