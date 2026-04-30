import cv2
import json
import os
import base64
import http.server
import threading
from urllib.parse import parse_qs, urlparse

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_FILE = "data/homography_points.json"
MIN_POINTS = 4

# Load existing points
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        all_points = json.load(f)
else:
    all_points = {}

# Get all keyframes that need annotation (fewer than MIN_POINTS)
ALL_KEYFRAMES = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])
PENDING = [kf for kf in ALL_KEYFRAMES
           if len(all_points.get(kf, [])) < MIN_POINTS]

print(f"Total keyframes: {len(ALL_KEYFRAMES)}")
print(f"Already have enough points: {len(ALL_KEYFRAMES) - len(PENDING)}")
print(f"Remaining to annotate: {len(PENDING)}")

if len(PENDING) == 0:
    print("All keyframes already have enough points!")
    exit(0)

state = {
    "index": 0,
    "points": all_points.get(PENDING[0], []).copy(),
    "done": False
}

def load_image(kf_name):
    img_path = os.path.join(KEYFRAMES_DIR, kf_name)
    img = cv2.imread(img_path)
    return img

def frame_to_base64(img, points):
    vis = img.copy()
    for i, p in enumerate(points):
        px, py = p["image"]
        cv2.circle(vis, (px, py), 8, (0, 255, 255), -1)
        cv2.circle(vis, (px, py), 9, (0, 0, 0), 2)
        cv2.putText(vis, str(i + 1), (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buffer).decode('utf-8')

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
    <style>
        :root {
            --bg: #0a0b10;
            --bg-elev: #12141c;
            --surface: #171a24;
            --surface-2: #1e2230;
            --border: #262b3d;
            --border-soft: #1e2230;
            --text: #e6e8ee;
            --text-dim: #9aa0b0;
            --text-muted: #6b7085;
            --accent: #3b82f6;
            --accent-soft: rgba(59, 130, 246, 0.12);
            --success: #10b981;
            --success-soft: rgba(16, 185, 129, 0.12);
            --danger: #ef4444;
            --danger-soft: rgba(239, 68, 68, 0.12);
            --warning: #f59e0b;
            --warning-soft: rgba(245, 158, 11, 0.12);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            min-height: 100%;
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px; line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }
        .mono { font-family: 'JetBrains Mono', 'Roboto Mono', monospace; }
        .appbar {
            display: flex; align-items: center; gap: 20px;
            padding: 12px 24px;
            background: var(--bg-elev);
            border-bottom: 1px solid var(--border);
        }
        .brand { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .brand-mark {
            width: 30px; height: 30px; border-radius: 9px;
            background: linear-gradient(135deg, #3b82f6, #10b981);
            display: grid; place-items: center;
            font-weight: 700; color: #fff; font-size: 15px;
        }
        .brand-text h1 { font-size: 14px; font-weight: 600; letter-spacing: -0.1px; }
        .brand-text p { font-size: 10.5px; color: var(--text-muted); letter-spacing: 0.3px; }
        .progress-bar { flex: 1; max-width: 520px; margin-left: 12px; }
        .progress-meta { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; }
        .progress-meta .label { color: var(--text-dim); }
        .progress-meta .count { color: var(--accent); font-weight: 600; }
        .progress-track { height: 4px; background: var(--surface); border-radius: 2px; overflow: hidden; }
        .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #3b82f6, #10b981); border-radius: 2px; transition: width 0.3s ease; }
        .clip-chip {
            margin-left: auto;
            font-size: 10.5px; color: var(--text-muted);
            padding: 5px 10px;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 6px;
            max-width: 360px;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
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
            padding: 16px 16px 14px;
        }
        .card h3 {
            font-size: 11px; font-weight: 600;
            color: var(--text-dim); text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 12px;
            display: flex; align-items: center; justify-content: space-between;
        }
        .card h3 .title-inner { display: flex; align-items: center; gap: 8px; }
        .card h3 .title-inner::before {
            content: ''; width: 3px; height: 12px;
            background: var(--accent); border-radius: 2px;
        }
        .count-pill {
            font-size: 11px; font-weight: 600;
            padding: 2px 9px;
            border-radius: 12px;
            background: var(--success-soft);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.25);
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0; text-transform: none;
        }
        .count-pill.warn {
            background: var(--warning-soft);
            color: var(--warning);
            border-color: rgba(245, 158, 11, 0.3);
        }
        .pixel-display {
            display: flex; align-items: center; gap: 10px;
            padding: 10px 12px;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .pixel-display .dot {
            width: 10px; height: 10px; background: var(--warning);
            border-radius: 50%;
            box-shadow: 0 0 0 3px var(--warning-soft);
        }
        .pixel-display .label {
            font-size: 10px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
        }
        .pixel-display .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px; color: var(--warning); margin-left: auto;
        }
        .pixel-display .value.empty { color: var(--text-muted); font-family: inherit; font-style: italic; }
        label {
            display: block;
            font-size: 10.5px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.7px; font-weight: 600;
            margin-bottom: 4px; margin-top: 10px;
        }
        select, input[type="number"] {
            width: 100%;
            background: var(--surface-2);
            border: 1px solid var(--border);
            color: var(--text);
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            padding: 9px 10px;
            border-radius: 7px;
            transition: border-color 0.15s, box-shadow 0.15s;
        }
        select { font-family: 'Inter', sans-serif; font-size: 13px; }
        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
        }
        optgroup { color: var(--accent); font-weight: 600; background: var(--surface); }
        option { color: var(--text); background: var(--surface-2); }
        .coord-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .status {
            margin-top: 10px; padding: 7px 10px;
            font-size: 11.5px;
            color: var(--success);
            background: var(--success-soft);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 7px;
        }
        .status.hint { color: var(--text-dim); background: var(--surface-2); border-color: var(--border); }
        .status.warn { color: var(--warning); background: var(--warning-soft); border-color: rgba(245, 158, 11, 0.25); }
        .btn {
            font-family: inherit;
            font-size: 13px; font-weight: 500;
            padding: 9px 14px;
            border: 1px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex; align-items: center; justify-content: center;
            gap: 6px; width: 100%;
        }
        .btn + .btn { margin-top: 8px; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-primary { background: var(--accent); color: #fff; }
        .btn-primary:hover:not(:disabled) { background: #2563eb; }
        .btn-success { background: var(--success); color: #fff; }
        .btn-success:hover:not(:disabled) { background: #0ea368; }
        .btn-ghost {
            background: transparent;
            border-color: var(--border);
            color: var(--text-dim);
        }
        .btn-ghost:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
        .btn-danger-ghost {
            background: transparent;
            border-color: rgba(239, 68, 68, 0.3);
            color: var(--danger);
        }
        .btn-danger-ghost:hover:not(:disabled) { background: var(--danger-soft); }
        .point-list {
            display: flex; flex-direction: column; gap: 4px;
            max-height: 240px; overflow-y: auto;
            margin-bottom: 10px;
        }
        .point-list::-webkit-scrollbar { width: 6px; }
        .point-list::-webkit-scrollbar-track { background: transparent; }
        .point-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        .point-item {
            display: grid;
            grid-template-columns: 22px 1fr;
            align-items: center;
            gap: 10px;
            padding: 7px 10px;
            background: var(--surface-2);
            border-radius: 7px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }
        .point-item .idx {
            width: 22px; height: 22px;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 50%;
            display: grid; place-items: center;
            font-weight: 600; font-size: 10.5px;
        }
        .point-item .coords { color: var(--text-dim); line-height: 1.35; }
        .point-item .coords .px { color: var(--text-muted); font-size: 10px; }
        .point-item .coords .rw { color: var(--text); }
        .empty-state {
            padding: 18px 10px; text-align: center;
            font-size: 11.5px; color: var(--text-muted);
        }
        .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .btn-row .btn { margin-top: 0; }
        .min-hint {
            margin-top: 6px; font-size: 10.5px;
            color: var(--text-muted); text-align: center;
        }
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
                <p>Homography calibration &middot; FYP</p>
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
                        <span class="label">Mode</span>
                        <span style="color: var(--accent);">Click to pick</span>
                    </div>
                </div>
            </div>
        </div>

        <aside class="sidebar">
            <div class="card">
                <h3><span class="title-inner">Selected pixel</span></h3>
                <div class="pixel-display">
                    <div class="dot"></div>
                    <span class="label">Pixel</span>
                    <span class="value empty" id="pixel-coords">Click on the image</span>
                </div>

                <label>Landmark preset</label>
                <select id="landmark-preset" onchange="applyLandmark()">
                    <option value="">&mdash; Custom (type coords below) &mdash;</option>
                    <optgroup label="Centre / Halfway">
                        <option value="0,0">Centre spot (0, 0)</option>
                        <option value="0,34">Halfway, near touchline (0, 34)</option>
                        <option value="0,-34">Halfway, far touchline (0, -34)</option>
                    </optgroup>
                    <optgroup label="Right penalty box">
                        <option value="36,20.16">Midfield edge, near side (36, 20.16)</option>
                        <option value="36,-20.16">Midfield edge, far side (36, -20.16)</option>
                        <option value="52.5,20.16">Goal line, near side (52.5, 20.16)</option>
                        <option value="52.5,-20.16">Goal line, far side (52.5, -20.16)</option>
                    </optgroup>
                    <optgroup label="Left penalty box">
                        <option value="-36,20.16">Midfield edge, near side (-36, 20.16)</option>
                        <option value="-36,-20.16">Midfield edge, far side (-36, -20.16)</option>
                        <option value="-52.5,20.16">Goal line, near side (-52.5, 20.16)</option>
                        <option value="-52.5,-20.16">Goal line, far side (-52.5, -20.16)</option>
                    </optgroup>
                    <optgroup label="Right 6-yard box">
                        <option value="47,9.16">Midfield edge, near side (47, 9.16)</option>
                        <option value="47,-9.16">Midfield edge, far side (47, -9.16)</option>
                    </optgroup>
                    <optgroup label="Left 6-yard box">
                        <option value="-47,9.16">Midfield edge, near side (-47, 9.16)</option>
                        <option value="-47,-9.16">Midfield edge, far side (-47, -9.16)</option>
                    </optgroup>
                    <optgroup label="Penalty spots">
                        <option value="41.5,0">Right penalty spot (41.5, 0)</option>
                        <option value="-41.5,0">Left penalty spot (-41.5, 0)</option>
                    </optgroup>
                    <optgroup label="Right goal posts">
                        <option value="52.5,3.66">Near post (52.5, 3.66)</option>
                        <option value="52.5,-3.66">Far post (52.5, -3.66)</option>
                    </optgroup>
                    <optgroup label="Left goal posts">
                        <option value="-52.5,3.66">Near post (-52.5, 3.66)</option>
                        <option value="-52.5,-3.66">Far post (-52.5, -3.66)</option>
                    </optgroup>
                    <optgroup label="Pitch corners">
                        <option value="52.5,34">Near-right corner (52.5, 34)</option>
                        <option value="52.5,-34">Far-right corner (52.5, -34)</option>
                        <option value="-52.5,34">Near-left corner (-52.5, 34)</option>
                        <option value="-52.5,-34">Far-left corner (-52.5, -34)</option>
                    </optgroup>
                </select>

                <div class="coord-grid" style="margin-top: 4px;">
                    <div>
                        <label>Real X (m)</label>
                        <input type="number" id="real-x" step="0.01" placeholder="auto-filled">
                    </div>
                    <div>
                        <label>Real Y (m)</label>
                        <input type="number" id="real-y" step="0.01" placeholder="auto-filled">
                    </div>
                </div>

                <div class="status hint" id="status">Click image &rarr; pick landmark &rarr; Add</div>

                <div class="btn-row" style="margin-top: 12px;">
                    <button class="btn btn-success" onclick="addPoint()">Add point</button>
                    <button class="btn btn-danger-ghost" onclick="deleteLastPoint()">Remove last</button>
                </div>
            </div>

            <div class="card">
                <h3>
                    <span class="title-inner">Points</span>
                    <span class="count-pill warn" id="count-pill">0 / 4+</span>
                </h3>

                <div class="point-list" id="point-list">
                    <div class="empty-state">No points yet</div>
                </div>

                <div class="min-hint" id="min-hint"><span class="warn">Need at least 4 points</span></div>

                <div class="btn-row" style="margin-top: 12px;">
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
                        '<div class="px">px (' + p.image[0] + ', ' + p.image[1] + ')</div>' +
                      '</div>' +
                    '</div>'
                ).join('');
            }

            const hint = document.getElementById('min-hint');
            if (count >= 4) {
                hint.innerHTML = '<span class="good">&check; Minimum met</span> &middot; add more for accuracy';
            } else {
                hint.innerHTML = '<span class="warn">Need at least ' + (4 - count) + ' more point' + (4 - count === 1 ? '' : 's') + '</span>';
            }
        }

        canvas.addEventListener('mousemove', function(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = imgWidth / rect.width;
            const scaleY = imgHeight / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);
            document.getElementById('cursor-coords').textContent = '(' + x + ', ' + y + ')';
        });

        canvas.addEventListener('mouseleave', function() {
            document.getElementById('cursor-coords').textContent = '—';
        });

        canvas.addEventListener('click', function(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = imgWidth / rect.width;
            const scaleY = imgHeight / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);
            selectedPixel = [x, y];
            const pd = document.getElementById('pixel-coords');
            pd.textContent = '(' + x + ', ' + y + ')';
            pd.className = 'value';
            setStatus('Now enter real-world coordinates', 'hint');
        });

        function setStatus(msg, kind) {
            const s = document.getElementById('status');
            s.textContent = msg;
            s.className = 'status' + (kind ? ' ' + kind : '');
        }

        function applyLandmark() {
            const sel = document.getElementById('landmark-preset');
            if (!sel.value) return;
            const parts = sel.value.split(',');
            document.getElementById('real-x').value = parseFloat(parts[0]);
            document.getElementById('real-y').value = parseFloat(parts[1]);
            setStatus(selectedPixel ? 'Ready — click Add point' : 'Now click that landmark on the image', 'hint');
        }

        function addPoint() {
            if (!selectedPixel) { setStatus('Click on the image first', 'warn'); return; }
            const rx = parseFloat(document.getElementById('real-x').value);
            const ry = parseFloat(document.getElementById('real-y').value);
            if (isNaN(rx) || isNaN(ry)) { setStatus('Pick a landmark or enter coords', 'warn'); return; }
            fetch('/add_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: selectedPixel, real: [rx, ry]})
            })
            .then(r => r.json()).then(data => {
                updateUI(data);
                loadImage();
                selectedPixel = null;
                const pd = document.getElementById('pixel-coords');
                pd.textContent = 'Click on the image';
                pd.className = 'value empty';
                document.getElementById('real-x').value = '';
                document.getElementById('real-y').value = '';
                document.getElementById('landmark-preset').value = '';
                if (data.points.length >= 4) {
                    setStatus('Good — ' + data.points.length + ' points. Add more or click Save & next.', '');
                } else {
                    setStatus(data.points.length + ' point(s) — need at least 4', 'warn');
                }
            });
        }

        function deleteLastPoint() {
            fetch('/delete_point', {method: 'POST'})
                .then(r => r.json()).then(data => { updateUI(data); loadImage(); });
        }

        function showDone(title, color, sub) {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;background:#0a0b10;">' +
                '<div style="text-align:center">' +
                '<h2 style="color:' + color + ';font-size:26px;margin-bottom:12px;font-weight:600;">' + title + '</h2>' +
                '<p style="color:#9aa0b0;font-size:14px;">' + sub + '</p>' +
                '</div></div>';
        }

        function nextFrame() {
            fetch('/next', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.done) {
                    showDone('All frames annotated', '#10b981', 'Points saved to data/homography_points.json');
                } else {
                    selectedPixel = null;
                    const pd = document.getElementById('pixel-coords');
                    pd.textContent = 'Click on the image';
                    pd.className = 'value empty';
                    document.getElementById('real-x').value = '';
                    document.getElementById('real-y').value = '';
                    document.getElementById('landmark-preset').value = '';
                    setStatus('Click image → pick landmark → Add', 'hint');
                    loadImage();
                }
            });
        }

        function skipFrame() {
            fetch('/skip', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.done) {
                    showDone('All frames processed', '#f59e0b', 'Points saved to data/homography_points.json');
                } else {
                    selectedPixel = null;
                    loadImage();
                }
            });
        }

        loadImage();
    </script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def log_error(self, format, *args): pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == '/image':
            kf = PENDING[state["index"]]
            img = load_image(kf)
            h, w = img.shape[:2]
            response = {
                "frame": frame_to_base64(img, state["points"]),
                "width": w,
                "height": h,
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

        def respond(data):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def advance():
            state["index"] += 1
            if state["index"] >= len(PENDING):
                state["done"] = True
                return True
            state["points"] = all_points.get(PENDING[state["index"]], []).copy()
            return False

        if parsed.path == '/add_point':
            data = json.loads(body)
            state["points"].append({
                "image": data["image"],
                "real": data["real"]
            })
            kf = PENDING[state["index"]]
            all_points[kf] = state["points"]
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_points, f, indent=2)
            respond({
                "points": state["points"],
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(PENDING)
            })

        elif parsed.path == '/delete_point':
            if state["points"]:
                state["points"].pop()
            kf = PENDING[state["index"]]
            all_points[kf] = state["points"]
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_points, f, indent=2)
            respond({
                "points": state["points"],
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(PENDING)
            })

        elif parsed.path == '/next':
            save_points()
            print(f"  Saved {len(state['points'])} points for {PENDING[state['index']]}")
            done = advance()
            respond({"done": done})

        elif parsed.path == '/skip':
            print(f"  Skipped {PENDING[state['index']]}")
            done = advance()
            respond({"done": done})

PORT = 5001
print(f"\nStarting point picker at http://localhost:{PORT}")
print("Open that URL in your Windows browser")
print(f"Frames to annotate: {len(PENDING)}\n")

server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

try:
    while not state["done"]:
        pass
except KeyboardInterrupt:
    pass

print("\nDone! Points saved to", OUTPUT_FILE)
