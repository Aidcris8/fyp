# This file implements a browser-based UI for users to pick pixel-to-world correspondence points on each keyframe
# The user clicks on a pitch landmark on the actual keyframe and then selects its real world FIFA coordinates on the 2d displayed pitch 
# These pairs are then saved to homography_points.json and used by compute_homography.py to fit homography matrix per frame
import cv2
import json
import os
import base64
import http.server
import threading
import numpy as np
from urllib.parse import urlparse

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_FILE = "data/homography_points.json"
SKIPPED_FILE = "data/homography_skipped.json"
MIN_POINTS = 4   #mathematical minimum, minimum 4 points to solve homography, the more points the more accurate
SUBPIX_WIN = 7   #search window size used when refining clicks to subpixel accuracy, 7x7 pixel neighbourhood around your click

def refine_click(img_bgr, x, y, win=SUBPIX_WIN):
    # refine a raw pixel click to subpixel accuracy using cornerSubPix
    # falls back to the original click if refinement fails or the point
    # jumps further than the search window (spurious convergence)
    try:
        #converts greyscale since pixel refinement works on intensity gradients, no need to know about colour
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  
        h, w = gray.shape
        #if click is too close to image edge, 7x7 search window would fall outside image bounds and crash, return original click unchanged instead
        if x < win + 1 or y < win + 1 or x > w - win - 2 or y > h - win - 2:
            return float(x), float(y)
        #cornerSubPix requires specific shape a 2D Array with one row per point
        pt = np.array([[float(x), float(y)]], dtype=np.float32)
        #tells algorithm to stop after either 30 iterations or when point moves less than 0.01 pixels between iterations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        #use cv2.cornerSubPix which looks at the image gradients in a 7x7 window around your click
        #and mathematically finds the exact sub pixel location of strongest corner or edge
        cv2.cornerSubPix(gray, pt, (win, win), (-1, -1), criteria)
        rx, ry = float(pt[0][0]), float(pt[0][1])
        #sanity check - if refined point jumped more than 7 pixels from click, it converged on the wrong feature entirely
        #discard refinement and return your original click
        if abs(rx - x) > win or abs(ry - y) > win:
            return float(x), float(y)
        return rx, ry
    except Exception:
        return float(x), float(y)

#load existing points
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        all_points = json.load(f)
else:
    all_points = {}  #dictionary keyed by keyframe filename, value list of point pairs

# Frames the user has given up on (insufficient usable landmarks). Excluded
# from PENDING so they do not reappear next session.
if os.path.exists(SKIPPED_FILE):
    with open(SKIPPED_FILE) as f:
        skipped = set(json.load(f))
else:
    skipped = set()

# persist the skipped set so frames with insufficient landmarks are not re-queud next session
def save_skipped():
    with open(SKIPPED_FILE, "w") as f:
        json.dump(sorted(skipped), f, indent=2)

# get all keyframes that need annotation (fewer than MIN_POINTS)
ALL_KEYFRAMES = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])
#keyframes with fewer than 4 points and not already skipped are added to pending
PENDING = [kf for kf in ALL_KEYFRAMES
           if len(all_points.get(kf, [])) < MIN_POINTS
           and kf not in skipped]

print(f"Total keyframes: {len(ALL_KEYFRAMES)}")
print(f"Already have enough points: {len(ALL_KEYFRAMES) - len(PENDING) - len(skipped)}")
print(f"Previously skipped: {len(skipped)}")
print(f"Remaining to annotate: {len(PENDING)}")

if len(PENDING) == 0:
    print("All keyframes already have enough points!")
    exit(0)

#state dictionary
state = {
    "index": 0,                                         #which frame in pending youre currently on 
    "points": all_points.get(PENDING[0], []).copy(),    #list of points collected so far for current frame
    "done": False                                       #still working on the frame so not done
}

# load a keyframe from disk called on every /iamge and /add_point request
def load_image(kf_name):
    img_path = os.path.join(KEYFRAMES_DIR, kf_name)
    img = cv2.imread(img_path)
    return img


def frame_to_base64(img, points):
    vis = img.copy() #never draw onto original image- draw a copy so base image stays clean 
    for i, p in enumerate(points):
        px, py = int(round(p["image"][0])), int(round(p["image"][1]))
        cv2.circle(vis, (px, py), 8, (0, 255, 255), -1) #filled yellow circle, radius 8 , thickness -1 means filled solid
        cv2.circle(vis, (px, py), 9, (0, 0, 0), 2)      #black outline ring radius 9, to make dot more visible against backgrounds
        cv2.putText(vis, str(i + 1), (px + 12, py - 8), #number label placed 12px to right and 8px above dot center to not overlap, start points from 1
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 88]) #encodes jpeg at quality 88, higher than keyframe annotator
    return base64.b64encode(buffer).decode('utf-8') #converts bytes to ASCII text so it can travel inside a JSON string, decode converts from bytes to a python string

#updates all points with current frames points then writes the whole dict to disk
def save_points():
    kf = PENDING[state["index"]]
    all_points[kf] = state["points"]
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_points, f, indent=2)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Point Picker</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/ui.css">
    <style>
        /* UI-specific styles for the point picker. Palette, fonts and
           top-bar live in /static/ui.css. */
        .layout {
            display: grid; grid-template-columns: 1fr 340px;
            gap: 20px; padding: 16px 24px 24px;
            max-width: 1600px; margin: 0 auto;
        }
        .canvas-wrap {
            position: relative;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 12px 32px rgba(0,0,0,0.4);
            cursor: crosshair;
        }
        #canvas { display: block; width: 100%; height: auto; }
        .canvas-toolbar {
            position: absolute; top: 14px; left: 14px;
            display: flex; gap: 6px;
        }
        .tool-chip {
            background: rgba(10, 11, 16, 0.85);
            backdrop-filter: blur(8px);
            color: var(--text);
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 7px;
            font-size: 11px;
            display: flex; align-items: center; gap: 6px;
        }
        .tool-chip .label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
        .tool-chip .mono-val { font-family: 'JetBrains Mono', monospace; color: var(--warning); }
        .sidebar { display: flex; flex-direction: column; gap: 14px; }
        .card {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 14px 14px 12px;
        }
        .card h3 {
            font-size: 11px; font-weight: 600;
            color: var(--text-dim); text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 10px;
            display: flex; align-items: center; justify-content: space-between;
        }
        .card h3 .title-inner { display: flex; align-items: center; gap: 8px; }
        .card h3 .title-inner::before {
            content: ''; width: 3px; height: 12px;
            background: var(--accent); border-radius: 2px;
        }
        .count-pill {
            font-size: 11px; font-weight: 600;
            padding: 2px 9px; border-radius: 12px;
            background: var(--success-soft); color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.25);
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0; text-transform: none;
        }
        .count-pill.warn {
            background: var(--warning-soft); color: var(--warning);
            border-color: rgba(245, 158, 11, 0.3);
        }
        /* Pixel display */
        .pixel-display {
            display: flex; align-items: center; gap: 10px;
            padding: 8px 10px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .pixel-display .dot {
            width: 10px; height: 10px; background: var(--warning);
            border-radius: 50%;
            box-shadow: 0 0 0 3px var(--warning-soft);
            flex-shrink: 0;
        }
        .pixel-display .label {
            font-size: 10px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
        }
        .pixel-display .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; color: var(--warning); margin-left: auto;
        }
        .pixel-display .value.empty { color: var(--text-muted); font-family: inherit; font-style: italic; font-size: 11px; }
        /* Section label */
        .section-label {
            font-size: 10px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
            margin-bottom: 6px;
        }
        /* Pitch SVG */
        .pitch-svg {
            width: 100%; height: auto;
            display: block;
            border-radius: 8px;
            border: 1px solid var(--border-soft);
            overflow: visible;
        }
        .pitch-line { stroke: #3d6b4a; stroke-width: 0.45; fill: none; }
        .lm {
            fill: #f59e0b;
            cursor: pointer;
            transition: fill 0.12s;
        }
        .lm:hover { fill: #fde68a; }
        .lm.picked { fill: #10b981; }
        .pitch-hint {
            font-size: 10.5px; color: var(--text-muted);
            text-align: center;
            padding: 5px 4px 8px;
            font-family: 'JetBrains Mono', monospace;
            min-height: 28px;
            transition: color 0.1s;
        }
        .pitch-hint.active { color: var(--warning); }
        /* Coord inputs */
        .coord-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
        .coord-grid label {
            display: block;
            font-size: 10px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.7px; font-weight: 600;
            margin-bottom: 3px;
        }
        input[type="number"] {
            width: 100%;
            background: var(--surface-2);
            border: 1px solid var(--border);
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            padding: 7px 8px;
            border-radius: 7px;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }
        /* Status */
        .status {
            padding: 6px 10px;
            font-size: 11px;
            border-radius: 7px;
            margin-bottom: 8px;
        }
        .status.hint { color: var(--text-dim); background: var(--surface-2); border: 1px solid var(--border); }
        .status.ok   { color: var(--success); background: var(--success-soft); border: 1px solid rgba(16,185,129,0.2); }
        .status.warn { color: var(--warning); background: var(--warning-soft); border: 1px solid rgba(245,158,11,0.25); }
        /* Buttons */
        .btn {
            font-family: inherit;
            font-size: 13px; font-weight: 500;
            padding: 8px 14px;
            border: 1px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex; align-items: center; justify-content: center;
            gap: 6px; width: 100%;
        }
        .btn + .btn { margin-top: 7px; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-primary:hover:not(:disabled) { background: #2563eb; }
        .btn-success { background: var(--success); color: #fff; }
        .btn-success:hover:not(:disabled) { background: #0ea368; }
        .btn-ghost { background: transparent; border-color: var(--border); color: var(--text-dim); }
        .btn-ghost:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
        .btn-danger-ghost { background: transparent; border-color: rgba(239,68,68,0.3); color: var(--danger); }
        .btn-danger-ghost:hover:not(:disabled) { background: var(--danger-soft); }
        .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .btn-row .btn { margin-top: 0; }
        /* Points list */
        .point-list {
            display: flex; flex-direction: column; gap: 4px;
            max-height: 200px; overflow-y: auto;
            margin-bottom: 8px;
        }
        .point-list::-webkit-scrollbar { width: 5px; }
        .point-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        .point-item {
            display: grid; grid-template-columns: 22px 1fr;
            align-items: center; gap: 8px;
            padding: 6px 8px;
            background: var(--surface-2);
            border-radius: 7px;
            font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
        }
        .point-item .idx {
            width: 22px; height: 22px;
            background: var(--accent-soft); color: var(--accent);
            border-radius: 50%; display: grid; place-items: center;
            font-weight: 600; font-size: 10px;
        }
        .point-item .coords .px { color: var(--text-muted); font-size: 9.5px; }
        .point-item .coords .rw { color: var(--text); }
        .empty-state { padding: 14px; text-align: center; font-size: 11.5px; color: var(--text-muted); }
        .min-hint { margin-top: 4px; font-size: 10.5px; color: var(--text-muted); text-align: center; }
        .min-hint .good { color: var(--success); }
        .min-hint .warn { color: var(--warning); }
    </style>
</head>
<body>
    <div class="appbar">
        <div class="brand">
            <div class="brand-mark">&#8903;</div>
            <div class="brand-text">
                <h1>Point Picker</h1>
                <p>Football CV &middot; Final Year Project</p>
            </div>
        </div>
        <div class="progress-bar">
            <div class="progress-meta">
                <span class="label">Frame progress</span>
                <span class="count" id="progress-count">&mdash; / &mdash;</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
        </div>
        <div class="clip-chip mono" id="clip-name">&mdash;</div>
    </div>

    <div class="layout">
        <div>
            <div class="canvas-wrap">
                <canvas id="canvas"></canvas>
                <div class="canvas-toolbar">
                    <div class="tool-chip">
                        <span class="label">Cursor</span>
                        <span class="mono-val" id="cursor-coords">&mdash;</span>
                    </div>
                    <div class="tool-chip">
                        <span style="color: var(--accent); font-size:11px;">Click to pick</span>
                    </div>
                </div>
            </div>
        </div>

        <aside class="sidebar">
            <!-- Selection card -->
            <div class="card">
                <h3><span class="title-inner">Point selection</span></h3>

                <div class="pixel-display">
                    <div class="dot"></div>
                    <span class="label">Pixel</span>
                    <span class="value empty" id="pixel-coords">Click on the frame</span>
                </div>

                <div class="section-label">Select landmark on pitch</div>

                <!-- Interactive SVG pitch -->
                <svg class="pitch-svg" id="pitch-svg" viewBox="-57 -37 114 74" xmlns="http://www.w3.org/2000/svg">
                    <!-- Pitch background -->
                    <rect x="-52.5" y="-34" width="105" height="68" fill="#0c1f13"/>
                    <!-- Pitch outline -->
                    <rect class="pitch-line" x="-52.5" y="-34" width="105" height="68"/>
                    <!-- Halfway line -->
                    <line class="pitch-line" x1="0" y1="-34" x2="0" y2="34"/>
                    <!-- Centre circle & spot -->
                    <circle class="pitch-line" cx="0" cy="0" r="9.15"/>
                    <circle cx="0" cy="0" r="0.55" fill="#3d6b4a"/>
                    <!-- Right penalty box -->
                    <rect class="pitch-line" x="36" y="-20.16" width="16.5" height="40.32"/>
                    <!-- Right 6-yard box -->
                    <rect class="pitch-line" x="47" y="-9.16" width="5.5" height="18.32"/>
                    <!-- Right penalty arc -->
                    <path class="pitch-line" d="M 36,-7.313 A 9.15,9.15 0 0 0 36,7.313"/>
                    <!-- Right penalty spot -->
                    <circle cx="41.5" cy="0" r="0.45" fill="#3d6b4a"/>
                    <!-- Right goal -->
                    <rect class="pitch-line" x="52.5" y="-3.66" width="2.2" height="7.32" stroke-dasharray="0"/>
                    <!-- Left penalty box -->
                    <rect class="pitch-line" x="-52.5" y="-20.16" width="16.5" height="40.32"/>
                    <!-- Left 6-yard box -->
                    <rect class="pitch-line" x="-52.5" y="-9.16" width="5.5" height="18.32"/>
                    <!-- Left penalty arc -->
                    <path class="pitch-line" d="M -36,-7.313 A 9.15,9.15 0 0 1 -36,7.313"/>
                    <!-- Left penalty spot -->
                    <circle cx="-41.5" cy="0" r="0.45" fill="#3d6b4a"/>
                    <!-- Left goal -->
                    <rect class="pitch-line" x="-54.7" y="-3.66" width="2.2" height="7.32"/>
                    <!-- "Near" label (bottom = camera side = +Y) -->
                    <text x="0" y="37" text-anchor="middle" fill="#3d6b4a" font-size="2.6" font-family="Inter,sans-serif">near (camera side)</text>

                    <!-- Landmark dots — Centre / Halfway -->
                    <circle class="lm" cx="0"    cy="0"     r="1.4" data-x="0"     data-y="0"      data-name="Centre spot"/>
                    <circle class="lm" cx="0"    cy="34"    r="1.4" data-x="0"     data-y="34"     data-name="Halfway · near touchline"/>
                    <circle class="lm" cx="0"    cy="-34"   r="1.4" data-x="0"     data-y="-34"    data-name="Halfway · far touchline"/>
                    <!-- Right penalty box -->
                    <circle class="lm" cx="36"   cy="20.16" r="1.4" data-x="36"    data-y="20.16"  data-name="R pen. box · near midfield edge"/>
                    <circle class="lm" cx="36"   cy="-20.16"r="1.4" data-x="36"    data-y="-20.16" data-name="R pen. box · far midfield edge"/>
                    <circle class="lm" cx="52.5" cy="20.16" r="1.4" data-x="52.5"  data-y="20.16"  data-name="R pen. box · near goal line"/>
                    <circle class="lm" cx="52.5" cy="-20.16"r="1.4" data-x="52.5"  data-y="-20.16" data-name="R pen. box · far goal line"/>
                    <!-- Left penalty box -->
                    <circle class="lm" cx="-36"  cy="20.16" r="1.4" data-x="-36"   data-y="20.16"  data-name="L pen. box · near midfield edge"/>
                    <circle class="lm" cx="-36"  cy="-20.16"r="1.4" data-x="-36"   data-y="-20.16" data-name="L pen. box · far midfield edge"/>
                    <circle class="lm" cx="-52.5"cy="20.16" r="1.4" data-x="-52.5" data-y="20.16"  data-name="L pen. box · near goal line"/>
                    <circle class="lm" cx="-52.5"cy="-20.16"r="1.4" data-x="-52.5" data-y="-20.16" data-name="L pen. box · far goal line"/>
                    <!-- Right 6-yard box -->
                    <circle class="lm" cx="47"   cy="9.16"  r="1.4" data-x="47"    data-y="9.16"   data-name="R 6-yard · near edge"/>
                    <circle class="lm" cx="47"   cy="-9.16" r="1.4" data-x="47"    data-y="-9.16"  data-name="R 6-yard · far edge"/>
                    <!-- Left 6-yard box -->
                    <circle class="lm" cx="-47"  cy="9.16"  r="1.4" data-x="-47"   data-y="9.16"   data-name="L 6-yard · near edge"/>
                    <circle class="lm" cx="-47"  cy="-9.16" r="1.4" data-x="-47"   data-y="-9.16"  data-name="L 6-yard · far edge"/>
                    <!-- Penalty spots -->
                    <circle class="lm" cx="41.5" cy="0"     r="1.4" data-x="41.5"  data-y="0"      data-name="Right penalty spot"/>
                    <circle class="lm" cx="-41.5"cy="0"     r="1.4" data-x="-41.5" data-y="0"      data-name="Left penalty spot"/>
                    <!-- Goal posts — right -->
                    <circle class="lm" cx="52.5" cy="3.66"  r="1.4" data-x="52.5"  data-y="3.66"   data-name="Right near post (52.5, 3.66)"/>
                    <circle class="lm" cx="52.5" cy="-3.66" r="1.4" data-x="52.5"  data-y="-3.66"  data-name="Right far post (52.5, -3.66)"/>
                    <!-- Goal posts — left -->
                    <circle class="lm" cx="-52.5"cy="3.66"  r="1.4" data-x="-52.5" data-y="3.66"   data-name="Left near post (-52.5, 3.66)"/>
                    <circle class="lm" cx="-52.5"cy="-3.66" r="1.4" data-x="-52.5" data-y="-3.66"  data-name="Left far post (-52.5, -3.66)"/>
                    <!-- Pitch corners -->
                    <circle class="lm" cx="52.5" cy="34"    r="1.4" data-x="52.5"  data-y="34"     data-name="Near-right corner"/>
                    <circle class="lm" cx="52.5" cy="-34"   r="1.4" data-x="52.5"  data-y="-34"    data-name="Far-right corner"/>
                    <circle class="lm" cx="-52.5"cy="34"    r="1.4" data-x="-52.5" data-y="34"     data-name="Near-left corner"/>
                    <circle class="lm" cx="-52.5"cy="-34"   r="1.4" data-x="-52.5" data-y="-34"    data-name="Far-left corner"/>
                </svg>

                <div class="pitch-hint" id="pitch-hint">Hover a landmark</div>

                <div class="section-label" style="margin-top:2px;">Real-world coords</div>
                <div class="coord-grid">
                    <div>
                        <label>X (m)</label>
                        <input type="number" id="real-x" step="0.01" placeholder="auto">
                    </div>
                    <div>
                        <label>Y (m)</label>
                        <input type="number" id="real-y" step="0.01" placeholder="auto">
                    </div>
                </div>

                <div class="status hint" id="status">Click the frame, then click a landmark</div>

                <div class="btn-row">
                    <button class="btn btn-success" onclick="addPoint()">Add point</button>
                    <button class="btn btn-danger-ghost" onclick="deleteLastPoint()">Remove last</button>
                </div>
            </div>

            <!-- Points list card -->
            <div class="card">
                <h3>
                    <span class="title-inner">Points</span>
                    <span class="count-pill warn" id="count-pill">0 / 4+</span>
                </h3>

                <div class="point-list" id="point-list">
                    <div class="empty-state">No points yet</div>
                </div>

                <div class="min-hint" id="min-hint"><span class="warn">Need at least 4 points</span></div>

                <div class="btn-row" style="margin-top: 10px;">
                    <button class="btn btn-primary" onclick="nextFrame()">Save &amp; next &rarr;</button>
                    <button class="btn btn-ghost" onclick="skipFrame()">Skip frame</button>
                </div>
            </div>
        </aside>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let selectedPixel = null;
        let imgWidth = 0, imgHeight = 0;

        // fetch the current keyframe from the server with any existing points drawn on it
        function loadImage() {
            fetch('/image').then(r => r.json()).then(data => {
                const img = new Image();
                img.onload = function() {
                    imgWidth = data.width;
                    imgHeight = data.height;
                    canvas.width = imgWidth;
                    canvas.height = imgHeight;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.frame;
                updateUI(data);
            });
        }
        // update the top bar, point list, and status hint to reflect the latest server response
        function updateUI(data) {
            document.getElementById('clip-name').textContent = data.clip_name;
            const pct = data.total > 0 ? (data.clip_index / data.total) * 100 : 0;
            document.getElementById('progress-fill').style.width = pct + '%';
            document.getElementById('progress-count').textContent = data.clip_index + ' / ' + data.total;

            const count = data.points.length;
            const pill = document.getElementById('count-pill');
            pill.textContent = count + ' / 4+';
            pill.className = 'count-pill' + (count >= 4 ? '' : ' warn');

            const list = document.getElementById('point-list');
            if (count === 0) {
                list.innerHTML = '<div class="empty-state">No points yet</div>';
            } else {
                list.innerHTML = data.points.map((p, i) =>
                    '<div class="point-item">' +
                      '<span class="idx">' + (i + 1) + '</span>' +
                      '<div class="coords">' +
                        '<div class="rw">(' + p.real[0] + ', ' + p.real[1] + ') m</div>' +
                        '<div class="px">px (' + Math.round(p.image[0]) + ', ' + Math.round(p.image[1]) + ')</div>' +
                      '</div>' +
                    '</div>'
                ).join('');
            }

            const hint = document.getElementById('min-hint');
            if (count >= 4) {
                hint.innerHTML = '<span class="good">&#10003; Minimum met</span> &middot; add more for accuracy';
            } else {
                const n = 4 - count;
                hint.innerHTML = '<span class="warn">Need ' + n + ' more point' + (n > 1 ? 's' : '') + '</span>';
            }
        }

        // track cursor position over the canvas and display pixel coordinates in real time
        canvas.addEventListener('mousemove', function(e) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * (imgWidth / rect.width));
            const y = Math.round((e.clientY - rect.top) * (imgHeight / rect.height));
            document.getElementById('cursor-coords').textContent = '(' + x + ', ' + y + ')';
        });

        // clear the coordinate display when the cursor leaves the canvas
        canvas.addEventListener('mouseleave', function() {
            document.getElementById('cursor-coords').textContent = '&mdash;';
        });

        // on canvas click, store the pixel coordinate and wait for the user to pick a landmark
        canvas.addEventListener('click', function(e) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * (imgWidth / rect.width));
            const y = Math.round((e.clientY - rect.top) * (imgHeight / rect.height));
            selectedPixel = [x, y];
            const pd = document.getElementById('pixel-coords');
            pd.textContent = '(' + x + ', ' + y + ')';
            pd.className = 'value';
            setStatus('Now click a landmark on the pitch', 'hint');
        });

        /* Pitch SVG landmark interaction */
        let activeLm = null;

        document.querySelectorAll('.lm').forEach(function(el) {
            el.addEventListener('mouseenter', function() {
                el.setAttribute('r', '2.0');
                const hint = document.getElementById('pitch-hint');
                hint.textContent = el.dataset.name + '  (' + el.dataset.x + ', ' + el.dataset.y + ')';
                hint.className = 'pitch-hint active';
            });

            el.addEventListener('mouseleave', function() {
                el.setAttribute('r', el === activeLm ? '1.8' : '1.4');
                const hint = document.getElementById('pitch-hint');
                hint.textContent = 'Hover a landmark';
                hint.className = 'pitch-hint';
            });

            // when a landmark is clicked, populate the real-world coordinate fields
            // and auto-submit the point if the pixel was already selected
            el.addEventListener('click', function() {
                const rx = parseFloat(el.dataset.x);
                const ry = parseFloat(el.dataset.y);
                document.getElementById('real-x').value = rx;
                document.getElementById('real-y').value = ry;

                /* Visual: mark this one as active */
                if (activeLm) {
                    activeLm.classList.remove('picked');
                    activeLm.setAttribute('r', '1.4');
                }
                activeLm = el;
                el.classList.add('picked');
                el.setAttribute('r', '1.8');

                /* Auto-add if pixel is already selected */
                if (selectedPixel) {
                    addPoint();
                } else {
                    setStatus('Click on the frame first', 'warn');
                }
            });
        });

        function setStatus(msg, kind) {
            const s = document.getElementById('status');
            s.textContent = msg;
            s.className = 'status ' + (kind || 'hint');
        }

        // send the current pixel + real-world pair to the server and refresh the canvas
        function addPoint() {
            if (!selectedPixel) { setStatus('Click on the frame first', 'warn'); return; }
            const rx = parseFloat(document.getElementById('real-x').value);
            const ry = parseFloat(document.getElementById('real-y').value);
            if (isNaN(rx) || isNaN(ry)) { setStatus('Select a landmark or enter coords', 'warn'); return; }

            fetch('/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: selectedPixel, real: [rx, ry]})
            }).then(r => r.json()).then(data => {
                updateUI(data);
                loadImage();
                /* Reset for next point */
                selectedPixel = null;
                const pd = document.getElementById('pixel-coords');
                pd.textContent = 'Click on the frame';
                pd.className = 'value empty';
                document.getElementById('real-x').value = '';
                document.getElementById('real-y').value = '';
                if (activeLm) { activeLm.classList.remove('picked'); activeLm.setAttribute('r', '1.4'); activeLm = null; }
                document.getElementById('pitch-hint').textContent = 'Hover a landmark';
                document.getElementById('pitch-hint').className = 'pitch-hint';
                const n = data.points.length;
                setStatus('Point ' + n + ' added. Click next pixel on frame.', 'ok');
            });
        }

        // remove the most recently added point and refresh the canvas
        function deleteLastPoint() {
            fetch('/delete_point', {method: 'POST'}).then(r => r.json()).then(data => {
                updateUI(data); loadImage();
            });
        }

        // replace the page with a simple done screen when all frames have been processed
        function showDone(title, color, sub) {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;background:#0a0b10;">' +
                '<div style="text-align:center">' +
                '<h2 style="color:' + color + ';font-size:26px;margin-bottom:12px;font-weight:600;">' + title + '</h2>' +
                '<p style="color:#9aa0b0;font-size:14px;">' + sub + '</p>' +
                '</div></div>';
        }

        // advance to the next frame — server handles the skip/save logic based on point count
        function nextFrame() {
            fetch('/next', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.done) {
                    showDone('All frames annotated', '#10b981', 'Points saved to data/homography_points.json');
                } else {
                    selectedPixel = null;
                    activeLm = null;
                    document.getElementById('pixel-coords').textContent = 'Click on the frame';
                    document.getElementById('pixel-coords').className = 'value empty';
                    document.getElementById('real-x').value = '';
                    document.getElementById('real-y').value = '';
                    document.getElementById('pitch-hint').textContent = 'Hover a landmark';
                    document.getElementById('pitch-hint').className = 'pitch-hint';
                    document.querySelectorAll('.lm.picked').forEach(function(e) {
                        e.classList.remove('picked'); e.setAttribute('r', '1.4');
                    });
                    setStatus('Click the frame, then click a landmark', 'hint');
                    loadImage();
                }
            });
        }

        // hard skip — permanently removes the keyframe from the pipeline
        function skipFrame() {
            fetch('/skip', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.done) {
                    showDone('All frames processed', '#f59e0b', 'Points saved to data/homography_points.json');
                } else {
                    selectedPixel = null; activeLm = null;
                    loadImage();
                }
            });
        }

        loadImage();
    </script>
</body>
</html>
"""

# HTTP request handler - GET serves the UI pafe and assets 
# POST handles all useri nteractions in the browser
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def log_error(self, format, *args): pass

    def do_GET(self):
        parsed = urlparse(self.path)
        #serves the main HTML page
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())

        # serves shared stylesheet , hardcoded path 
        elif parsed.path == '/static/ui.css':
            try:
                with open('static/ui.css', 'rb') as f:
                    css_bytes = f.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/css; charset=utf-8')
                self.send_header('Content-length', str(len(css_bytes)))
                self.end_headers()
                self.wfile.write(css_bytes)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()

        # returns current keuframe with existing points drawn on it
        elif parsed.path == '/image':
            kf = PENDING[state["index"]]
            img = load_image(kf)
            h, w = img.shape[:2]
            response = {
                "frame": frame_to_base64(img, state["points"]),
                "width": w, "height": h,
                "points": state["points"],
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(PENDING)
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        #helper to send JSON response, used by every POST route
        def respond(data):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        # moves to next pending frame
        def advance():
            state["index"] += 1
            if state["index"] >= len(PENDING):
                state["done"] = True
                #returns true if all frames are done
                return True
            state["points"] = all_points.get(PENDING[state["index"]], []).copy()
            return False

        # receives a pixel click and real world coordinate pair, refine click to subpixel
        # accuracy and adds it to current frames point list
        if parsed.path == '/add_point':
            data = json.loads(body)
            kf = PENDING[state["index"]]
            x_in, y_in = data["image"][0], data["image"][1]
            img = load_image(kf)
            x_ref, y_ref = refine_click(img, x_in, y_in)
            print(f"  Click ({x_in},{y_in}) refined to ({x_ref:.2f},{y_ref:.2f})")
            state["points"].append({
                "image": [x_ref, y_ref],
                "real": data["real"]
            })
            save_points()
            respond({
                "points": state["points"],
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(PENDING)
            })

        # removes most recently added point from current frame
        elif parsed.path == '/delete_point':
            if state["points"]:
                state["points"].pop()
            save_points()
            kf = PENDING[state["index"]]
            respond({
                "points": state["points"],
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(PENDING)
            })

        # advance to next frame
        elif parsed.path == '/next':
            kf = PENDING[state["index"]]
            if len(state["points"]) < MIN_POINTS:
                # if user clicks next without enough points to fit a homography, the frame is 
                # marked as skipped so it doesn't re-queue next session, and evict any partial points so homography_points.json stays clean
                skipped.add(kf)
                save_skipped()
                if kf in all_points:
                    all_points.pop(kf)
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(all_points, f, indent=2)
                print(f"  Skipped (insufficient: {len(state['points'])}/{MIN_POINTS}): {kf}")
            else:
                save_points()
                print(f"  Saved {len(state['points'])} points for {kf}")
            done = advance()
            respond({"done": done})

        # hard skip permandently deletes the keyframe JPG and removes it from the pipeline
        # done in cases where image either low quality or doesnt have enough landmarks visible
        elif parsed.path == '/skip':
            kf = PENDING[state["index"]]
            img_path = os.path.join(KEYFRAMES_DIR, kf)
            if os.path.exists(img_path):
                os.remove(img_path)
            all_points.pop(kf, None)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_points, f, indent=2)
            print(f"  Skipped and removed {kf}")
            done = advance()
            respond({"done": done})

#server binds to all interfaces so it is reachable from windows browsers over WSL2, main thread waits until all frames are done or user clicks CTRL+c
PORT = 5001
print(f"\nStarting point picker at http://localhost:{PORT}")
print("Open that URL in your Windows browser")
print(f"Frames to annotate: {len(PENDING)}\n")

#server startup, binds all network interfaces
server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

#spin main thread until all frames are done or ctrl + c 
try:
    while not state["done"]:
        pass
except KeyboardInterrupt:
    pass

print("\nDone! Points saved to", OUTPUT_FILE)
