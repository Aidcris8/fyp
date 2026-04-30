import cv2
import json
import os

CLIPS_DIR = "data/clips"
KEYFRAMES_DIR = "data/keyframes"
KEYFRAMES_JSON = "data/keyframes.json"

os.makedirs(KEYFRAMES_DIR, exist_ok=True)

with open(KEYFRAMES_JSON) as f:
    data = json.load(f)

# Handle both old format (flat dict) and new format
if "keyframes" in data:
    keyframes = data["keyframes"]
else:
    keyframes = data

print(f"Exporting {len(keyframes)} keyframes...\n")

for clip_name, frame_num in keyframes.items():
    clip_path = os.path.join(CLIPS_DIR, clip_name)

    if not os.path.exists(clip_path):
        print(f"  Clip not found: {clip_name}")
        continue

    output_name = clip_name.replace(".mp4", f"_frame{frame_num}.jpg")
    output_path = os.path.join(KEYFRAMES_DIR, output_name)

    if os.path.exists(output_path):
        print(f"  Already exists, skipping: {output_name}")
        continue

    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  Could not read frame {frame_num} from {clip_name}")
        continue

    cv2.imwrite(output_path, frame)
    print(f"  Saved: {output_name}")

print(f"\nDone! Keyframes saved to {KEYFRAMES_DIR}")