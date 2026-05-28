import cv2
import json
import os

CLIPS_DIR = "data/clips"
KEYFRAMES_DIR = "data/keyframes"
KEYFRAMES_JSON = "data/keyframes.json"

os.makedirs(KEYFRAMES_DIR, exist_ok=True) #create folder if it doesnt exist, dont crash if it does

#reading the json files saved from previous step
#if/else which handles both formats so the script works regardless
with open(KEYFRAMES_JSON) as f:
    data = json.load(f)


if "keyframes" in data:
    keyframes = data["keyframes"]
else:
    keyframes = data

print(f"Exporting {len(keyframes)} keyframes...\n")

#iterate over every entry, with frame num being the frame number the user selects in the annotator
for clip_name, frame_num in keyframes.items():
    clip_path = os.path.join(CLIPS_DIR, clip_name)

    #check if clip exists
    #if it doesnt will warn you that clip not found
    if not os.path.exists(clip_path):
        print(f"  Clip not found: {clip_name}")
        continue

    #build the output filename, replacing .mp4 with .jpg and include the frame number
    output_name = clip_name.replace(".mp4", f"_frame{frame_num}.jpg")
    output_path = os.path.join(KEYFRAMES_DIR, output_name)

    #skip if frame already exported 
    if os.path.exists(output_path):
        print(f"  Already exists, skipping: {output_name}")
        continue

    #tell OpenCV to jump straight to desired frame without decoding anything before it, fast and memory efficient
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