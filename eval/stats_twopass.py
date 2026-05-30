# Measures how many additional player detections the second low-confidence YOLO pass
# contributes across all keyframes compared to the single-pass baseline.
# Outputs per-frame counts and summary statistics used in Chapter 5 to justify
# the two-pass detection approach.
# Run from the project root: python eval/stats_twopass.py

import cv2, os
from ultralytics import YOLO

KEYFRAMES_DIR = "data/keyframes"
model = YOLO("yolo11x-seg.pt")

# extract all class-0 (person) bounding boxes from a YOLO result
def get_boxes(results):
    boxes_obj = results[0].boxes
    if boxes_obj is None: return []
    return [(int(b.xyxy[0][0]), int(b.xyxy[0][1]), int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for b in boxes_obj if int(b.cls[0]) == 0]

# compute intersection over union between two boxes, used to avoid counting duplicates
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

# run the two-pass detection on every keyframe and record single-pass and extra counts
stats = []
for fname in sorted(os.listdir(KEYFRAMES_DIR)):
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, fname))
    if img is None: continue
    height = img.shape[0]

    r_probe = model(img, classes=[0], conf=0.3, iou=0.7, retina_masks=False, verbose=False)
    probe_boxes = get_boxes(r_probe)
    probe_heights = [b[3]-b[1] for b in probe_boxes]

    # set adaptive thresholds from the probe run, same logic as team_identification.py
    if len(probe_heights) > 3:
        median_h = sorted(probe_heights)[len(probe_heights)//2]
        min_height = 20 if median_h < 45 else 40
        top_zone   = 0.65 if median_h < 45 else 0.45
    else:
        min_height, top_zone = 25, 0.55

    # filter probe detections down to those above the adaptive minimum height
    single_pass = [(x1,y1,x2,y2) for (x1,y1,x2,y2) in probe_boxes if (y2-y1) > min_height]

    r_low = model(img, classes=[0], conf=0.05, iou=0.7, retina_masks=False, verbose=False)
    low_boxes = get_boxes(r_low)

    # count low-confidence detections in the upper zone that are not already detected
    extra = 0
    for box in low_boxes:
        x1, y1, x2, y2 = box
        if y2 < height * top_zone and (y2-y1) > 20:
            if not any(iou(box, e) > 0.4 for e in single_pass):
                extra += 1

    stats.append((len(single_pass), extra))
    print(f"  {len(single_pass):3d} +{extra:2d}  {fname[-55:]}")

# aggregate stats across all frames
total_frames   = len(stats)
frames_w_extra = sum(1 for s,e in stats if e > 0)
total_extra    = sum(e for s,e in stats)
avg_all        = total_extra / total_frames
avg_affected   = total_extra / frames_w_extra
max_extra      = max(e for s,e in stats)
avg_single     = sum(s for s,e in stats) / total_frames

print(f"\nFrames tested          : {total_frames}")
print(f"Frames with extra det. : {frames_w_extra}  ({100*frames_w_extra//total_frames}%)")
print(f"Total extra detections : {total_extra}")
print(f"Avg extra / all frames : {avg_all:.2f}")
print(f"Avg extra / affected   : {avg_affected:.2f}")
print(f"Max extra in one frame : {max_extra}")
print(f"Avg single-pass count  : {avg_single:.1f}")
