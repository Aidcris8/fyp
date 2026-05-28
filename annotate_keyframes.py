import cv2
import json
import os
import base64
import http.server
import threading
from urllib.parse import parse_qs, urlparse

CLIPS_DIR = "data/clips"
OUTPUT_FILE = "data/keyframes.json"

# detect all clips in data/clips/
ALL_CLIPS = sorted([f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")])
print(f"Found {len(ALL_CLIPS)} total clips in {CLIPS_DIR}")

# load existing clips
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        data = json.load(f)
    if "keyframes" in data:
        annotations = data["keyframes"]
        skipped = set(data.get("skipped", []))
    else:
        annotations = data
        skipped = set()
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"keyframes": annotations, "skipped": list(skipped)}, f, indent=2)
        print("  Migrated keyframes.json to new format")
else:
    annotations = {}
    skipped = set()

# keyframes.json evolves, old format was simply clip name.mp4: 247, new format wraps that in a keyframe key and adds a skipped list
VERIFIED_CLIPS = [c for c in ALL_CLIPS if c not in annotations and c not in skipped]
print(f"Already annotated: {len(annotations)}")
print(f"Already skipped: {len(skipped)}")
print(f"Remaining to annotate: {len(VERIFIED_CLIPS)}")

#only clips you have not touched yet are annotated, re running always resumes exactly where you left off 

#if the length of clips not yet annotated is 0, it means you have annotated every clip
if len(VERIFIED_CLIPS) == 0:
    print("All clips already annotated or skipped!")
    exit(0)

#state dictionary, shows which clip youre currently on, which frame within that clip, all decoded frames of current clip in RAM
state = {
    "clip_index": 0,
    "frame_index": 0,
    "frames": [],
    "fps": 25,
    "done": False
}

#read entire clip into RAM as a list of NumPy arrays
#resize frames to 960x540, deliberate downscale, raking less memory
def load_clip(clip_name):
    print(f"Loading {clip_name}...")
    cap = cv2.VideoCapture(os.path.join(CLIPS_DIR, clip_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960, 540))
        frames.append(frame)
    cap.release()
    print(f"  Loaded {len(frames)} frames at {fps}fps")
    return frames, fps

def save():
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"keyframes": annotations, "skipped": list(skipped)}, f, indent=2)

#converts a frame to a jpeg since browsers cant receive raw numpy arrays
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode('utf-8')

#tries clips one by one until it finds one that loads, if file is corrupt, return 0 frames
#automatically add clip to skipped set and try the next one 
def load_next_valid_clip():
    """Advance clip_index until a clip with at least 1 frame is loaded, auto-skipping unreadable ones."""
    while state["clip_index"] < len(VERIFIED_CLIPS):
        clip_name = VERIFIED_CLIPS[state["clip_index"]]
        frames, fps = load_clip(clip_name)
        if len(frames) > 0:
            state["frames"] = frames
            state["fps"] = fps
            return True
        print(f"  WARNING: {clip_name} loaded 0 frames — auto-skipping (corrupt/unreadable)")
        skipped.add(clip_name)
        save()
        state["clip_index"] += 1
    return False

if not load_next_valid_clip():
    print("No readable clips found. Exiting.")
    exit(0)

#HTML/CSS/JavaScript Block
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Keyframe Annotator</title>
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
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            min-height: 100%;
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            line-height: 1.5;
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
        .container { max-width: 1280px; margin: 0 auto; padding: 14px 20px 24px; }
        .timeline {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 12px 16px 10px;
            margin-bottom: 12px;
        }
        .timeline-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
        .timeline-title {
            font-size: 10.5px; color: var(--text-dim);
            text-transform: uppercase; letter-spacing: 1px; font-weight: 600;
            display: flex; align-items: center; gap: 8px;
        }
        .timeline-title::before { content: ''; width: 3px; height: 11px; background: var(--accent); border-radius: 2px; }
        .timeline-readout {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; color: var(--text);
            display: flex; gap: 8px; align-items: center;
        }
        .timeline-readout .sep { color: var(--text-muted); }
        .timeline-readout .label {
            color: var(--text-muted); text-transform: uppercase;
            font-size: 9.5px; letter-spacing: 0.8px;
            font-family: 'Inter', sans-serif;
        }
        .scrubber-track-wrap { position: relative; padding: 6px 0; cursor: pointer; }
        .scrubber-track { height: 6px; background: var(--surface-2); border-radius: 3px; position: relative; overflow: hidden; }
        .scrubber-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #3b82f6, #10b981); border-radius: 3px; transition: width 0.1s; }
        .scrubber-handle {
            position: absolute; top: 50%; left: 0%;
            width: 13px; height: 13px;
            background: #fff; border: 2px solid var(--accent);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 2px 6px rgba(0,0,0,0.4);
            transition: left 0.1s;
        }
        .scrubber-ticks { display: flex; justify-content: space-between; margin-top: 4px; font-size: 10px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
        .frame-row {
            display: grid; grid-template-columns: auto 1fr auto;
            gap: 14px; align-items: center; margin-bottom: 14px;
        }
        .nav-group { display: flex; flex-direction: row; gap: 6px; }
        .nav-btn {
            font-family: inherit;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            color: var(--text-dim);
            border-radius: 10px;
            padding: 10px 12px;
            cursor: pointer;
            transition: all 0.15s;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            gap: 3px; min-width: 58px;
        }
        .nav-btn:hover { background: var(--bg-elev); border-color: var(--accent); color: var(--text); transform: translateY(-1px); }
        .nav-btn:active { transform: translateY(0); }
        .nav-btn .arrow { font-family: 'JetBrains Mono', monospace; font-size: 16px; line-height: 1; color: var(--text); }
        .nav-btn .step-label { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--text-muted); letter-spacing: 0.3px; }
        .nav-btn.fine { background: var(--accent-soft); border-color: rgba(59, 130, 246, 0.3); }
        .nav-btn.fine .arrow, .nav-btn.fine .step-label { color: var(--accent); }
        .canvas-wrap {
            position: relative;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(0,0,0,0.4);
            display: flex; justify-content: center;
            max-height: 60vh;
        }
        .canvas-wrap img { display: block; max-width: 100%; max-height: 60vh; width: auto; height: auto; }
        .frame-overlay {
            position: absolute; bottom: 12px; left: 12px;
            background: rgba(10, 11, 16, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 7px;
            padding: 5px 11px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px; color: var(--text-dim);
            display: flex; align-items: center; gap: 8px;
        }
        .frame-overlay .label { color: var(--text-muted); text-transform: uppercase; font-size: 9.5px; letter-spacing: 0.8px; font-family: 'Inter', sans-serif; }
        .frame-overlay .sep { color: var(--text-muted); }
        .frame-overlay .value { color: var(--text); font-weight: 500; }
        .primary-actions { display: flex; justify-content: center; gap: 12px; margin-bottom: 14px; }
        .btn-mark {
            background: var(--success); color: #fff; border: none;
            font-family: inherit; font-size: 14px; font-weight: 600;
            padding: 13px 36px; border-radius: 10px;
            cursor: pointer; transition: all 0.15s;
            display: inline-flex; align-items: center; gap: 8px;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
            min-width: 240px; justify-content: center;
        }
        .btn-mark:hover { background: #0ea368; transform: translateY(-1px); box-shadow: 0 6px 18px rgba(16, 185, 129, 0.35); }
        .btn-skip {
            background: transparent; color: var(--danger);
            border: 1px solid rgba(239, 68, 68, 0.35);
            font-family: inherit; font-size: 13.5px; font-weight: 500;
            padding: 13px 28px; border-radius: 10px;
            cursor: pointer; transition: all 0.15s;
            min-width: 180px;
        }
        .btn-skip:hover { background: var(--danger-soft); border-color: var(--danger); }
        .dash { display: grid; grid-template-columns: 0.9fr 1.3fr 1.3fr; gap: 12px; }
        .card {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 10px 14px 12px;
        }
        .card h3 {
            font-size: 10.5px; font-weight: 600;
            color: var(--text-dim); text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 8px;
            display: flex; align-items: center; gap: 8px;
        }
        .card h3::before { content: ''; width: 3px; height: 11px; background: var(--accent); border-radius: 2px; }
        .timestamp-display { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
        .timestamp-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px; font-weight: 600; color: var(--text);
            letter-spacing: -0.5px; line-height: 1;
        }
        .timestamp-value .unit { font-size: 12px; color: var(--text-dim); font-weight: 500; margin-left: 3px; }
        .timestamp-sub { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
        .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
        .info-chip { display: flex; flex-direction: column; gap: 2px; padding: 6px 10px; background: var(--surface-2); border-radius: 7px; }
        .info-chip .label { color: var(--text-muted); text-transform: uppercase; font-size: 9.5px; letter-spacing: 0.7px; font-weight: 600; }
        .info-chip .value { font-family: 'JetBrains Mono', monospace; color: var(--text); font-size: 12px; }
        .kbd-list-h { display: flex; flex-wrap: wrap; gap: 8px 16px; font-size: 12px; color: var(--text-dim); }
        .kbd-list-h .row { display: flex; align-items: center; gap: 6px; }
        .kbd {
            font-family: 'JetBrains Mono', monospace;
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-bottom-width: 2px;
            border-radius: 4px;
            padding: 2px 7px;
            color: var(--text);
            font-size: 10.5px;
            min-width: 22px; text-align: center; display: inline-block;
        }
        .kbd + .kbd { margin-left: 2px; }
        @media (max-width: 1100px) {
            .frame-row { grid-template-columns: 1fr; gap: 10px; }
            .nav-group { justify-content: center; }
            .dash { grid-template-columns: 1fr 1fr; }
            .dash .card:last-child { grid-column: span 2; }
            .btn-mark, .btn-skip { min-width: 0; }
        }
    </style>
</head>
<body>
    <div class="appbar">
        <div class="brand">
            <div class="brand-mark">&#9656;</div>
            <div class="brand-text">
                <h1>Keyframe Annotator</h1>
                <p>Football CV &middot; Final Year Project</p>
            </div>
        </div>
        <div class="progress-bar">
            <div class="progress-meta">
                <span class="label">Clip progress</span>
                <span class="count" id="progress-count">&mdash; / &mdash;</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" id="progress-fill"></div>
            </div>
        </div>
        <div class="clip-chip mono" id="clip-name">&mdash;</div>
    </div>

    <div class="container">
        <div class="timeline">
            <div class="timeline-head">
                <div class="timeline-title">Clip timeline</div>
                <div class="timeline-readout">
                    <span class="label">At</span>
                    <span class="value" id="ts-at">&mdash;</span>
                    <span class="sep">&middot;</span>
                    <span class="label">Frame</span>
                    <span class="value" id="ts-frame">&mdash;</span>
                    <span class="sep">/</span>
                    <span id="ts-total">&mdash;</span>
                </div>
            </div>
            <div class="scrubber-track-wrap" id="scrubber">
                <div class="scrubber-track">
                    <div class="scrubber-fill" id="scrubber-fill"></div>
                </div>
                <div class="scrubber-handle" id="scrubber-handle"></div>
            </div>
            <div class="scrubber-ticks" id="scrubber-ticks">
                <span>00:00</span><span>&mdash;</span><span>&mdash;</span><span>&mdash;</span><span>&mdash;</span>
            </div>
        </div>

        <div class="frame-row">
            <div class="nav-group">
                <button class="nav-btn" onclick="prevFrame(25)"><span class="arrow">&#10218;</span><span class="step-label">&minus;25</span></button>
                <button class="nav-btn" onclick="prevFrame(10)"><span class="arrow">&laquo;</span><span class="step-label">&minus;10</span></button>
                <button class="nav-btn fine" onclick="prevFrame(1)"><span class="arrow">&larr;</span><span class="step-label">&minus;1</span></button>
            </div>

            <div class="canvas-wrap">
                <img id="frame-img" src="" alt="frame">
                <div class="frame-overlay">
                    <span class="label">Frame</span>
                    <span class="value" id="ov-frame">&mdash;</span>
                    <span class="sep">/</span>
                    <span id="ov-total">&mdash;</span>
                    <span class="sep">&middot;</span>
                    <span class="value" id="ov-time">&mdash;</span>
                </div>
            </div>

            <div class="nav-group">
                <button class="nav-btn fine" onclick="nextFrame(1)"><span class="arrow">&rarr;</span><span class="step-label">+1</span></button>
                <button class="nav-btn" onclick="nextFrame(10)"><span class="arrow">&raquo;</span><span class="step-label">+10</span></button>
                <button class="nav-btn" onclick="nextFrame(25)"><span class="arrow">&#10219;</span><span class="step-label">+25</span></button>
            </div>
        </div>

        <div class="primary-actions">
            <button class="btn-mark" onclick="markFrame()">&check; Mark as keyframe</button>
            <button class="btn-skip" onclick="skipClip()">Skip this clip</button>
        </div>

        <div class="dash">
            <div class="card">
                <h3>Current frame</h3>
                <div class="timestamp-display">
                    <div class="timestamp-value" id="ts-big">&mdash;<span class="unit">s</span></div>
                    <div class="timestamp-sub" id="ts-sub">&mdash; / &mdash; &middot; &mdash; fps</div>
                </div>
            </div>

            <div class="card">
                <h3>Clip details</h3>
                <div class="info-grid">
                    <div class="info-chip"><span class="label">Length</span><span class="value" id="clip-length">&mdash;</span></div>
                    <div class="info-chip"><span class="label">Done</span><span class="value" id="clip-done">&mdash;</span></div>
                    <div class="info-chip"><span class="label">Skipped</span><span class="value" id="clip-skipped">&mdash;</span></div>
                </div>
            </div>

            <div class="card">
                <h3>Shortcuts</h3>
                <div class="kbd-list-h">
                    <div class="row"><span>Prev / next</span><span class="kbd">&larr;</span><span class="kbd">&rarr;</span></div>
                    <div class="row"><span>Jump +25</span><span class="kbd">Space</span></div>
                    <div class="row"><span>Mark</span><span class="kbd">Enter</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFrame = 0;
        let totalFrames = 0;
        let fps = 25;

        function fmtTick(seconds) {
            const m = Math.floor(seconds / 60);
            const s = seconds - m * 60;
            if (m > 0) return m + ':' + Math.floor(s).toString().padStart(2, '0');
            return s.toFixed(2) + 's';
        }
        
        function updateFrame(data) {
            currentFrame = data.frame_index;
            totalFrames = data.total_frames;
            fps = data.fps || 25;
            const clipIndex = data.clip_index;
            const totalClips = data.total_clips;

            document.getElementById('frame-img').src = 'data:image/jpeg;base64,' + data.frame;
            document.getElementById('clip-name').textContent = data.clip_name;

            const pct = totalClips > 0 ? (clipIndex / totalClips) * 100 : 0;
            document.getElementById('progress-fill').style.width = pct + '%';
            document.getElementById('progress-count').textContent = clipIndex + ' / ' + totalClips;

            const durS = totalFrames / fps;
            const atS = data.timestamp;
            document.getElementById('ts-at').textContent = atS.toFixed(2) + 's';
            document.getElementById('ts-frame').textContent = currentFrame;
            document.getElementById('ts-total').textContent = (totalFrames - 1);

            const handlePct = totalFrames > 1 ? (currentFrame / (totalFrames - 1)) * 100 : 0;
            document.getElementById('scrubber-fill').style.width = handlePct + '%';
            document.getElementById('scrubber-handle').style.left = handlePct + '%';

            const ticks = document.getElementById('scrubber-ticks').children;
            for (let i = 0; i < 5; i++) {
                ticks[i].textContent = fmtTick((i / 4) * durS);
            }

            document.getElementById('ov-frame').textContent = currentFrame;
            document.getElementById('ov-total').textContent = (totalFrames - 1);
            document.getElementById('ov-time').textContent = atS.toFixed(2) + 's';

            document.getElementById('ts-big').innerHTML = atS.toFixed(2) + '<span class="unit">s</span>';
            document.getElementById('ts-sub').textContent = currentFrame + ' / ' + (totalFrames - 1) + ' · ' + Math.round(fps) + ' fps';

            document.getElementById('clip-length').textContent = durS.toFixed(2) + 's · ' + totalFrames + 'f';
            document.getElementById('clip-done').textContent = (data.done_count || 0) + ' / ' + totalClips;
            document.getElementById('clip-skipped').textContent = (data.skipped_count || 0);
        }

        function fetchFrame(idx) {
            fetch('/frame?idx=' + idx).then(r => r.json()).then(updateFrame);
        }

        function nextFrame(n) { fetchFrame(Math.min(currentFrame + n, totalFrames - 1)); }
        function prevFrame(n) { fetchFrame(Math.max(currentFrame - n, 0)); }

        function showDone(title, color, sub) {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;background:#0a0b10;">' +
                '<div style="text-align:center">' +
                '<h2 style="color:' + color + ';font-size:26px;margin-bottom:12px;font-weight:600;">' + title + '</h2>' +
                '<p style="color:#9aa0b0;font-size:14px;">' + sub + '</p>' +
                '</div></div>';
        }

        function markFrame() {
            fetch('/mark?idx=' + currentFrame).then(r => r.json()).then(data => {
                if (data.done) showDone('All clips annotated', '#10b981', 'You can close this tab.');
                else { currentFrame = 0; fetchFrame(0); }
            });
        }

        function skipClip() {
            fetch('/skip').then(r => r.json()).then(data => {
                if (data.done) showDone('All clips processed', '#f59e0b', 'You can close this tab.');
                else { currentFrame = 0; fetchFrame(0); }
            });
        }

        document.getElementById('scrubber').addEventListener('click', function(e) {
            if (totalFrames < 2) return;
            const rect = this.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            fetchFrame(Math.round(pct * (totalFrames - 1)));
        });

        document.addEventListener('keydown', e => {
            if (e.key === 'ArrowRight') nextFrame(1);
            else if (e.key === 'ArrowLeft') prevFrame(1);
            else if (e.key === ' ') { e.preventDefault(); nextFrame(25); }
            else if (e.key === 'Enter') markFrame();
        });

        fetchFrame(0);
    </script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

   
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        #returns HTML page
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())

        #reads idx, clamps it to valid range, encodes fram and returns the response
        elif parsed.path == '/frame':
            if not state["frames"]:
                self.send_response(503)
                self.end_headers()
                return
            idx = int(params.get('idx', [0])[0])
            idx = max(0, min(idx, len(state["frames"]) - 1))
            state["frame_index"] = idx
            frame = state["frames"][idx]
            clip_name = VERIFIED_CLIPS[state["clip_index"]]
            response = {
                "frame": frame_to_base64(frame),
                "frame_index": idx,
                "total_frames": len(state["frames"]),
                "timestamp": round(idx / state["fps"], 2),
                "clip_name": clip_name,
                "clip_index": state["clip_index"] + 1,
                "total_clips": len(VERIFIED_CLIPS),
                "fps": round(state["fps"], 0),
                "done_count": len(annotations),
                "skipped_count": len(skipped)
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        #saves immediately so crash doesnt lose work, then load next clip
        elif parsed.path == '/mark':
            idx = int(params.get('idx', [0])[0])
            clip_name = VERIFIED_CLIPS[state["clip_index"]]
            annotations[clip_name] = idx
            save()
            print(f"  Marked {clip_name} → frame {idx} ({round(idx/state['fps'], 2)}s)")

            state["clip_index"] += 1
            has_more = load_next_valid_clip()
            state["done"] = not has_more
            state["frame_index"] = 0
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"done": not has_more}).encode())

        #same thing but adds to skipped set
        elif parsed.path == '/skip':
            clip_name = VERIFIED_CLIPS[state["clip_index"]]
            skipped.add(clip_name)
            save()
            print(f"  Skipped {clip_name}")
            state["clip_index"] += 1
            has_more = load_next_valid_clip()
            state["done"] = not has_more
            state["frame_index"] = 0
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"done": not has_more}).encode())

PORT = 5000
print(f"\nStarting annotator at http://localhost:{PORT}")
print("Open that URL in your Windows browser")
print("Press Ctrl+C to stop the server\n")

#server startup, server running on a background thread meaning it dies automatically when main thread exits
#main thread loops until either all clips done or user presses Ctrl + C
# 0.0.0 done so it listens on all network interfaces
server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

try:
    while not state["done"]:
        pass
except KeyboardInterrupt:
    pass

print("\n Annotations saved to", OUTPUT_FILE)
