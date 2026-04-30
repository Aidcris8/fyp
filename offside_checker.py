import cv2
import json
import os
import base64
import http.server
import threading
import numpy as np
from urllib.parse import parse_qs, urlparse

KEYFRAMES_DIR = "data/keyframes"
DETECTIONS_FILE = "data/player_detections.json"
HOMOGRAPHY_FILE = "data/homography_matrices.json"
POINTS_FILE = "data/homography_points.json"
OUTPUT_DIR = "data/offside_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DETECTIONS_FILE) as f:
    all_detections = json.load(f)
with open(HOMOGRAPHY_FILE) as f:
    all_homographies = json.load(f)
with open(POINTS_FILE) as f:
    all_corr_points = json.load(f)

def has_reliable_slope(kf):
    return kf in all_homographies

USABLE_FRAMES = sorted([
    kf for kf in all_detections
    if kf in all_homographies and has_reliable_slope(kf)
])
print(f"Usable frames: {len(USABLE_FRAMES)}")
for kf in USABLE_FRAMES:
    print(f"  {kf.split('_frame')[0][-30:]}...frame{kf.split('_frame')[1]}")

state = {
    "index": 0,
    "attacking_team": None,
    "attacking_right": None,
    "selected_attacker": None,
    "auto_defender": None,
    "result": None,
}

def find_auto_defender(kf, attacking_team, attacking_right):
    """Return index of the most-advanced outfield defender. Each defender's
    leading bbox edge (x2 if attacking right, else x1) at foot level y2 is
    projected through H into world space, then we pick the extreme world-X.
    This avoids the perspective bug where image-space x1/x2 picks the defender
    nearest the camera rather than the one genuinely furthest up the pitch."""
    players = all_detections[kf]["players"]
    H = np.array(all_homographies[kf]["H"])
    defending_team = 1 - attacking_team
    defending = [(i, p) for i, p in enumerate(players) if p["team"] == defending_team]
    if not defending:
        return None

    def world_x(p):
        x1, y1, x2, y2 = p["box"]
        edge_px = x2 if attacking_right else x1
        wx, _ = project_point(H, edge_px, y2)
        return wx

    ranked = sorted(defending, key=lambda ip: world_x(ip[1]), reverse=attacking_right)
    direction = "max world-X" if attacking_right else "min world-X"
    print(f"  Defending team T{defending_team} ranked by {direction}:")
    for i, p in ranked:
        x1, y1, x2, y2 = p["box"]
        edge_px = x2 if attacking_right else x1
        print(f"    P{i}: edge_px={edge_px} y2={y2}  world_x={world_x(p):.2f}m")

    chosen = ranked[0][0]
    print(f"  → Chosen: P{chosen}")
    return chosen

def get_frame_data():
    kf = USABLE_FRAMES[state["index"]]
    players = all_detections[kf]["players"]
    H = np.array(all_homographies[kf]["H"])
    return kf, players, H

def project_point(H, px, py):
    pt = np.array([px, py, 1.0])
    proj = H @ pt
    proj /= proj[2]
    return float(proj[0]), float(proj[1])

def is_valid_projection(rx, ry):
    return -60 < rx < 60 and -40 < ry < 40

def get_line_slope(kf, offside_line_x, defender_y, H):
    H_inv = np.linalg.inv(H)

    def real_to_image(rx, ry):
        pt = np.array([rx, ry, 1.0])
        img_pt = H_inv @ pt
        img_pt /= img_pt[2]
        return float(img_pt[0]), float(img_pt[1])

    dy = 5.0
    px1 = real_to_image(offside_line_x, defender_y - dy)
    px2 = real_to_image(offside_line_x, defender_y + dy)
    dx_img = px2[0] - px1[0]
    dy_img = px2[1] - px1[1]
    slope = dx_img / dy_img if abs(dy_img) > 1e-6 else 0.0
    print(f"  Slope from inverse homography: {slope:.3f}")
    return slope

def render_frame():
    kf, players, H = get_frame_data()
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf))
    team_colours = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 255)}

    for i, p in enumerate(players):
        x1, y1, x2, y2 = p["box"]
        team = p["team"]
        colour = team_colours.get(team, (128, 128, 128))
        thickness = 1

        if i == state["selected_attacker"]:
            colour = (0, 255, 0)
            thickness = 2
        elif i == state["auto_defender"]:
            colour = (0, 165, 255)
            thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
        label = "OTH" if team == 2 else f"T{team}"
        cv2.putText(img, f"{i}:{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    if state["result"] is not None:
        res = state["result"]
        line_pts = res.get("line_image_pts")
        if line_pts:
            pt1 = tuple(map(int, line_pts[0]))
            pt2 = tuple(map(int, line_pts[1]))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

        sa = state["selected_attacker"]
        if sa is not None and sa < len(players):
            x1, y1, x2, y2 = players[sa]["box"]
            # OFFSIDE → magenta (distinct from team-0 red); ONSIDE → green
            verdict_colour = (255, 0, 255) if res["offside"] else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), verdict_colour, 2)
            cv2.putText(img, "OFFSIDE" if res["offside"] else "ONSIDE",
                        (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, verdict_colour, 2)

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buffer).decode('utf-8'), img.shape[1], img.shape[0]
def compute_offside():
    kf, players, H = get_frame_data()
    attacking_team = state["attacking_team"]
    attacking_right = state["attacking_right"]
    attacker_idx = state["selected_attacker"]

    if attacking_team is None or attacking_right is None or attacker_idx is None:
        return {"error": "Please complete all steps first"}

    defender_idx = state["auto_defender"]
    if defender_idx is None:
        return {"error": "No defending team players found"}

    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf))
    img_h, img_w = img.shape[:2]

    att_box = players[attacker_idx]["box"]
    x1_att, y1_att, x2_att, y2_att = att_box

    def_box = players[defender_idx]["box"]
    x1_def, y1_def, x2_def, y2_def = def_box
    print(f"  Auto-selected defender: P{defender_idx} team={players[defender_idx]['team']} box={def_box}")

    # use bounding box edges based on user-specified attacking direction
    if attacking_right:
        attacker_x, attacker_y = project_point(H, x2_att, y2_att)
        defender_x, defender_y = project_point(H, x2_def, y2_def)
        anchor_px = x2_def
    else:
        attacker_x, attacker_y = project_point(H, x1_att, y2_att)
        defender_x, defender_y = project_point(H, x1_def, y2_def)
        anchor_px = x1_def

    offside_line_x = defender_x

    if attacking_right:
        is_offside = attacker_x > offside_line_x
    else:
        is_offside = attacker_x < offside_line_x

    margin = abs(attacker_x - offside_line_x)

    slope = get_line_slope(kf, offside_line_x, defender_y, H)

    # anchor line at defender's actual foot pixel position
    anchor_y = players[defender_idx]["foot_px"][1]
    dy_to_top = anchor_y
    dy_to_bottom = img_h - anchor_y

    pt1_x = int(anchor_px - slope * dy_to_top)
    pt2_x = int(anchor_px + slope * dy_to_bottom)

    img_pt1 = [pt1_x, 0]
    img_pt2 = [pt2_x, img_h]

    print(f"\n--- Offside Check ---")
    print(f"  Attacking right: {attacking_right}")
    print(f"  Attacker edge real: ({attacker_x:.2f}, {attacker_y:.2f})")
    print(f"  Defender edge real: ({defender_x:.2f}, {defender_y:.2f})")
    print(f"  Offside line x: {offside_line_x:.2f}m")
    print(f"  Is offside: {is_offside}, margin: {margin:.2f}m")
    print(f"  Anchor px: {anchor_px}, line: {img_pt1} -> {img_pt2}")

    return {
        "offside": True if is_offside else False,
        "attacker_x": round(float(attacker_x), 2),
        "offside_line_x": round(float(offside_line_x), 2),
        "margin_metres": round(float(margin), 2),
        "line_image_pts": [img_pt1, img_pt2],
        "attacking_right": True if attacking_right else False
    }

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Offside Checker</title>
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
            --team-red: #ef4444;
            --team-blue: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            height: 100%;
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px; line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }
        .mono { font-family: 'JetBrains Mono', 'Roboto Mono', monospace; }
        .appbar {
            display: flex; align-items: center; gap: 20px;
            padding: 14px 24px;
            background: var(--bg-elev);
            border-bottom: 1px solid var(--border);
        }
        .brand { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
        .brand-mark {
            width: 32px; height: 32px; border-radius: 9px;
            background: linear-gradient(135deg, #3b82f6, #10b981);
            display: grid; place-items: center;
            font-weight: 700; color: #fff; font-size: 14px;
        }
        .brand-text h1 { font-size: 14px; font-weight: 600; letter-spacing: -0.1px; }
        .brand-text p { font-size: 11px; color: var(--text-muted); letter-spacing: 0.3px; }
        .progress-bar { flex: 1; max-width: 480px; margin-left: 12px; }
        .progress-meta { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 11px; }
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
            gap: 20px; padding: 20px 24px;
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
        .legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .verdict {
            margin-top: 18px;
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 22px 26px;
            display: none;
            grid-template-columns: auto 1fr auto;
            gap: 28px; align-items: center;
        }
        .verdict.offside { border-color: rgba(239,68,68,0.35); background: linear-gradient(135deg, rgba(239,68,68,0.06), var(--surface) 60%); }
        .verdict.onside  { border-color: rgba(16,185,129,0.35); background: linear-gradient(135deg, rgba(16,185,129,0.06), var(--surface) 60%); }
        .verdict-badge { display: flex; flex-direction: column; gap: 2px; align-items: flex-start; }
        .verdict-label {
            font-size: 10.5px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
        }
        .verdict-value { font-size: 34px; font-weight: 700; letter-spacing: -0.5px; line-height: 1; }
        .verdict.offside .verdict-value { color: var(--danger); }
        .verdict.onside .verdict-value { color: var(--success); }
        .verdict-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
        .stat { display: flex; flex-direction: column; gap: 4px; }
        .stat-label {
            font-size: 10.5px; color: var(--text-muted);
            text-transform: uppercase; letter-spacing: 0.8px;
        }
        .stat-value {
            font-size: 17px; font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text);
        }
        .stat-value .unit { color: var(--text-dim); font-size: 13px; font-weight: 500; margin-left: 2px; }
        .margin-meter { width: 190px; display: flex; flex-direction: column; gap: 6px; }
        .margin-meter-track {
            height: 8px;
            background: var(--surface-2);
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        .margin-meter-fill {
            position: absolute; top: 0; bottom: 0; left: 50%;
            background: var(--danger);
            border-radius: 4px;
            width: 0%;
        }
        .margin-meter-fill.onside { background: var(--success); left: auto; right: 50%; }
        .margin-meter-center {
            position: absolute; left: 50%; top: -2px; bottom: -2px;
            width: 1px; background: var(--text-muted);
        }
        .margin-legend {
            display: flex; justify-content: space-between;
            font-size: 10px; color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
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
            display: flex; align-items: center; gap: 8px;
        }
        .card h3::before { content: ''; width: 3px; height: 12px; background: var(--accent); border-radius: 2px; }
        .stepper { display: flex; flex-direction: column; gap: 0; }
        .step {
            display: grid;
            grid-template-columns: 28px 1fr;
            gap: 12px;
            padding: 6px 0;
            position: relative;
        }
        .step::after {
            content: '';
            position: absolute;
            left: 13px; top: 32px; bottom: -6px;
            width: 2px;
            background: var(--border);
        }
        .step:last-child::after { display: none; }
        .step-bullet {
            width: 28px; height: 28px; border-radius: 50%;
            background: var(--surface-2);
            border: 1.5px solid var(--border);
            color: var(--text-muted);
            display: grid; place-items: center;
            font-size: 12px; font-weight: 600;
            transition: all 0.2s;
            z-index: 1;
        }
        .step.active .step-bullet {
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent);
        }
        .step.done .step-bullet {
            background: var(--success);
            border-color: var(--success);
            color: #fff;
        }
        .step.done::after { background: var(--success); }
        .step-body { padding-top: 4px; }
        .step-title { font-size: 13px; font-weight: 500; color: var(--text-dim); }
        .step.active .step-title { color: var(--text); }
        .step.done .step-title { color: var(--text-dim); }
        .step-hint { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
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
        .btn-team {
            background: var(--surface-2);
            border: 1px solid var(--border);
            color: var(--text-dim);
            justify-content: flex-start;
            padding-left: 12px;
        }
        .btn-team .team-dot { width: 10px; height: 10px; border-radius: 50%; }
        .btn-team.t0 .team-dot { background: var(--team-red); }
        .btn-team.t1 .team-dot { background: var(--team-blue); }
        .btn-team.active { color: var(--text); background: var(--bg-elev); }
        .btn-team.t0.active { border-color: var(--team-red); box-shadow: 0 0 0 1px var(--team-red); }
        .btn-team.t1.active { border-color: var(--team-blue); box-shadow: 0 0 0 1px var(--team-blue); }
        .btn-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .btn-row .btn { margin-top: 0; }
        .info-row {
            display: flex; align-items: center; justify-content: space-between;
            padding: 8px 10px;
            background: var(--surface-2);
            border-radius: 7px;
            margin-bottom: 6px;
            font-size: 12px;
        }
        .info-row .label {
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 10px; letter-spacing: 0.8px; font-weight: 600;
        }
        .info-row .value {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text); font-size: 12px;
        }
        .info-row .value.empty { color: var(--text-muted); font-style: italic; font-family: inherit; }
    </style>
</head>
<body>
    <div class="appbar">
        <div class="brand">
            <div class="brand-mark">&Oslash;</div>
            <div class="brand-text">
                <h1>Offside Detection</h1>
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
        <div class="canvas-col">
            <div class="canvas-wrap">
                <canvas id="canvas"></canvas>
                <div class="canvas-toolbar">
                    <div class="tool-chip"><span class="legend-dot" style="background: var(--team-red)"></span>Team 0</div>
                    <div class="tool-chip"><span class="legend-dot" style="background: var(--team-blue)"></span>Team 1</div>
                    <div class="tool-chip"><span class="legend-dot" style="background: #10b981"></span>Attacker</div>
                    <div class="tool-chip"><span class="legend-dot" style="background: #f59e0b"></span>Defender</div>
                </div>
            </div>

            <div class="verdict" id="verdict">
                <div class="verdict-badge">
                    <span class="verdict-label">Verdict</span>
                    <span class="verdict-value" id="v-value">&mdash;</span>
                </div>
                <div class="verdict-stats">
                    <div class="stat">
                        <span class="stat-label">Attacker X</span>
                        <span class="stat-value" id="v-attacker-x">&mdash;</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Offside line</span>
                        <span class="stat-value" id="v-line-x">&mdash;</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Direction</span>
                        <span class="stat-value" id="v-direction" style="font-size: 15px;">&mdash;</span>
                    </div>
                </div>
                <div class="margin-meter">
                    <div class="stat-label" id="v-margin-label">Margin</div>
                    <div class="margin-meter-track">
                        <div class="margin-meter-center"></div>
                        <div class="margin-meter-fill" id="v-margin-fill"></div>
                    </div>
                    <div class="margin-legend">
                        <span>onside</span>
                        <span>offside</span>
                    </div>
                </div>
            </div>
        </div>

        <aside class="sidebar">
            <div class="card">
                <h3>Workflow</h3>
                <div class="stepper">
                    <div class="step active" id="step1">
                        <div class="step-bullet">1</div>
                        <div class="step-body">
                            <div class="step-title">Select attacking team</div>
                            <div class="step-hint">Team 0 or Team 1</div>
                        </div>
                    </div>
                    <div class="step" id="step2">
                        <div class="step-bullet">2</div>
                        <div class="step-body">
                            <div class="step-title">Attacking direction</div>
                            <div class="step-hint">Left or Right</div>
                        </div>
                    </div>
                    <div class="step" id="step3">
                        <div class="step-bullet">3</div>
                        <div class="step-body">
                            <div class="step-title">Click receiver</div>
                            <div class="step-hint">Click a player on the canvas</div>
                        </div>
                    </div>
                    <div class="step" id="step4">
                        <div class="step-bullet">4</div>
                        <div class="step-body">
                            <div class="step-title">Check offside</div>
                            <div class="step-hint">Verdict appears below canvas</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Attacking team</h3>
                <button class="btn btn-team t0" id="btn-t0" onclick="setAttacking(0)">
                    <span class="team-dot"></span>
                    <span>Team 0 &middot; Red</span>
                </button>
                <button class="btn btn-team t1" id="btn-t1" onclick="setAttacking(1)">
                    <span class="team-dot"></span>
                    <span>Team 1 &middot; Blue</span>
                </button>
            </div>

            <div class="card">
                <h3>Attacking direction</h3>
                <div class="btn-row">
                    <button class="btn btn-ghost" id="btn-left" onclick="setDirection(false)" disabled>&larr; Left</button>
                    <button class="btn btn-ghost" id="btn-right" onclick="setDirection(true)" disabled>Right &rarr;</button>
                </div>
            </div>

            <div class="card">
                <h3>Selection</h3>
                <div class="info-row">
                    <span class="label">Attacker</span>
                    <span class="value empty" id="att-info">none</span>
                </div>
                <div class="info-row">
                    <span class="label">Defender</span>
                    <span class="value empty" id="def-info">none</span>
                </div>
                <button class="btn btn-success" id="btn-check" onclick="checkOffside()" disabled style="margin-top: 10px;">Check offside</button>
                <button class="btn btn-ghost" onclick="resetFrame()">Reset frame</button>
            </div>

            <button class="btn btn-primary" onclick="nextFrame()" style="padding: 11px; font-size: 13.5px;">Next frame &rarr;</button>
        </aside>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let imgWidth = 0, imgHeight = 0;
        let attackingTeam = null;
        let attackingRight = null;
        let canClick = false;
        let selectedAttacker = null;

        function loadFrame() {
            fetch('/frame').then(r => r.json()).then(data => {
                const img = new Image();
                img.onload = function() {
                    imgWidth = data.width;
                    imgHeight = data.height;
                    canvas.width = imgWidth;
                    canvas.height = imgHeight;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.frame;
                document.getElementById('clip-name').textContent = data.clip_name;
                const pct = data.total > 0 ? (data.clip_index / data.total) * 100 : 0;
                document.getElementById('progress-fill').style.width = pct + '%';
                document.getElementById('progress-count').textContent = data.clip_index + ' / ' + data.total;
            });
        }

        function setStepState(n, stateCls, hint) {
            const el = document.getElementById('step' + n);
            el.className = 'step' + (stateCls ? ' ' + stateCls : '');
            const bullet = el.querySelector('.step-bullet');
            bullet.textContent = stateCls === 'done' ? '✓' : n;
            if (hint !== undefined) {
                const h = el.querySelector('.step-hint');
                if (h) h.textContent = hint;
            }
        }

        function setAttacking(team) {
            attackingTeam = team;
            document.getElementById('btn-t0').classList.toggle('active', team === 0);
            document.getElementById('btn-t1').classList.toggle('active', team === 1);
            setStepState(1, 'done', 'Team ' + team + ' (' + (team === 0 ? 'Red' : 'Blue') + ')');
            setStepState(2, 'active', 'Pick left or right');
            document.getElementById('btn-left').disabled = false;
            document.getElementById('btn-right').disabled = false;
            fetch('/set_attacking', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({team: team})
            });
        }

        function setDirection(isRight) {
            attackingRight = isRight;
            document.getElementById('btn-left').classList.toggle('active', !isRight);
            document.getElementById('btn-right').classList.toggle('active', isRight);
            setStepState(2, 'done', isRight ? 'Left → Right' : 'Right → Left');
            setStepState(3, 'active', 'Click a player on the canvas');
            canClick = true;
            fetch('/set_direction', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({attacking_right: isRight})
            })
            .then(r => r.json()).then(data => {
                const defEl = document.getElementById('def-info');
                if (data.auto_defender) {
                    const d = data.auto_defender;
                    defEl.textContent = 'P' + d.player_idx + ' · T' + d.team + ' · (' + d.foot_px[0] + ', ' + d.foot_px[1] + ')';
                    defEl.className = 'value';
                } else {
                    defEl.textContent = 'none found';
                    defEl.className = 'value empty';
                }
                loadFrame();
            });
        }

        canvas.addEventListener('click', function(e) {
            if (!canClick) return;
            const rect = canvas.getBoundingClientRect();
            const scaleX = imgWidth / rect.width;
            const scaleY = imgHeight / rect.height;
            const cx = Math.round((e.clientX - rect.left) * scaleX);
            const cy = Math.round((e.clientY - rect.top) * scaleY);
            fetch('/select_attacker', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x: cx, y: cy})
            })
            .then(r => r.json()).then(data => {
                if (data.player_idx !== null) {
                    selectedAttacker = data.player_idx;
                    const attEl = document.getElementById('att-info');
                    attEl.textContent = 'P' + data.player_idx + ' · T' + data.team + ' · (' + data.foot_px[0] + ', ' + data.foot_px[1] + ')';
                    attEl.className = 'value';
                    setStepState(3, 'done', 'Player P' + data.player_idx);
                    setStepState(4, 'active', 'Ready — click Check offside');
                    document.getElementById('btn-check').disabled = false;
                    loadFrame();
                }
            });
        });

        function checkOffside() {
            fetch('/check_offside', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.error) return;
                showVerdict(data);
                setStepState(4, 'done', 'See verdict below');
                loadFrame();
                fetch('/save_result', {method: 'POST'});
            });
        }

        function showVerdict(data) {
            const verdict = document.getElementById('verdict');
            verdict.style.display = 'grid';
            verdict.className = 'verdict ' + (data.offside ? 'offside' : 'onside');
            document.getElementById('v-value').textContent = data.offside ? 'OFFSIDE' : 'ONSIDE';
            document.getElementById('v-attacker-x').innerHTML = data.attacker_x + '<span class="unit">m</span>';
            document.getElementById('v-line-x').innerHTML = data.offside_line_x + '<span class="unit">m</span>';
            document.getElementById('v-direction').textContent = data.attacking_right ? 'L → R' : 'R → L';
            document.getElementById('v-margin-label').textContent =
                'Margin · ' + data.margin_metres + ' m ' + (data.offside ? 'ahead' : 'behind');
            const fill = document.getElementById('v-margin-fill');
            const maxM = 5;
            const mag = Math.min(Math.abs(data.margin_metres), maxM);
            const pct = (mag / maxM) * 50;
            fill.style.width = pct + '%';
            fill.className = 'margin-meter-fill' + (data.offside ? '' : ' onside');
        }

        function resetUI() {
            attackingTeam = null;
            attackingRight = null;
            canClick = false;
            selectedAttacker = null;
            document.getElementById('btn-t0').classList.remove('active');
            document.getElementById('btn-t1').classList.remove('active');
            document.getElementById('btn-left').classList.remove('active');
            document.getElementById('btn-right').classList.remove('active');
            document.getElementById('btn-left').disabled = true;
            document.getElementById('btn-right').disabled = true;
            const attEl = document.getElementById('att-info');
            attEl.textContent = 'none';
            attEl.className = 'value empty';
            const defEl = document.getElementById('def-info');
            defEl.textContent = 'none';
            defEl.className = 'value empty';
            document.getElementById('btn-check').disabled = true;
            document.getElementById('verdict').style.display = 'none';
            setStepState(1, 'active', 'Team 0 or Team 1');
            setStepState(2, '', 'Left or Right');
            setStepState(3, '', 'Click a player on the canvas');
            setStepState(4, '', 'Verdict appears below canvas');
        }

        function resetFrame() {
            resetUI();
            fetch('/reset', {method: 'POST'}).then(() => loadFrame());
        }

        function showDone() {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;background:#0a0b10;">' +
                '<div style="text-align:center">' +
                '<h2 style="color:#10b981;font-size:26px;margin-bottom:12px;font-weight:600;">All frames checked</h2>' +
                '<p style="color:#9aa0b0;font-size:14px;">Results saved to data/offside_results/</p>' +
                '</div></div>';
        }

        function nextFrame() {
            fetch('/next', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.done) showDone();
                else { resetUI(); loadFrame(); }
            });
        }

        loadFrame();
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

        elif parsed.path == '/frame':
            kf = USABLE_FRAMES[state["index"]]
            frame_b64, w, h = render_frame()
            response = {
                "frame": frame_b64, "width": w, "height": h,
                "clip_name": kf,
                "clip_index": state["index"] + 1,
                "total": len(USABLE_FRAMES)
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

        if parsed.path == '/set_attacking':
            data = json.loads(body)
            state["attacking_team"] = data["team"]
            state["attacking_right"] = None
            state["selected_attacker"] = None
            state["auto_defender"] = None
            state["result"] = None
            respond({"ok": True})

        elif parsed.path == '/set_direction':
            data = json.loads(body)
            state["attacking_right"] = bool(data["attacking_right"])
            kf = USABLE_FRAMES[state["index"]]
            def_idx = find_auto_defender(kf, state["attacking_team"], state["attacking_right"])
            state["auto_defender"] = def_idx
            def_info = None
            if def_idx is not None:
                p = all_detections[kf]["players"][def_idx]
                def_info = {"player_idx": def_idx, "team": p["team"],
                            "box": p["box"], "foot_px": p["foot_px"]}
                print(f"  Auto-defender: P{def_idx} team={p['team']} box={p['box']}")
            respond({"ok": True, "auto_defender": def_info})

        elif parsed.path == '/select_attacker':
            data = json.loads(body)
            cx, cy = data["x"], data["y"]
            kf = USABLE_FRAMES[state["index"]]
            players = all_detections[kf]["players"]
            best_idx = None
            for i, p in enumerate(players):
                x1, y1, x2, y2 = p["box"]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    best_idx = i
                    break
            if best_idx is not None:
                state["selected_attacker"] = best_idx
                p = players[best_idx]
                print(f"  Attacker selected: P{best_idx} team={p['team']} box={p['box']} foot={p['foot_px']}")
                respond({"player_idx": best_idx, "team": p["team"], "foot_px": p["foot_px"]})
            else:
                print(f"  Attacker click miss: clicked ({cx},{cy}) — no player found")
                respond({"player_idx": None})

        elif parsed.path == '/check_offside':
            result = compute_offside()
            state["result"] = result
            respond(result)

        elif parsed.path == '/save_result':
            frame_b64, w, h = render_frame()
            import base64 as b64
            img_data = b64.b64decode(frame_b64)
            img_array = np.frombuffer(img_data, np.uint8)
            img_decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            kf = USABLE_FRAMES[state["index"]]
            cv2.imwrite(os.path.join(OUTPUT_DIR, kf), img_decoded)
            respond({"ok": True})

        elif parsed.path == '/reset':
            state["attacking_team"] = None
            state["attacking_right"] = None
            state["selected_attacker"] = None
            state["auto_defender"] = None
            state["result"] = None
            respond({"ok": True})

        elif parsed.path == '/next':
            state["index"] += 1
            state["attacking_team"] = None
            state["attacking_right"] = None
            state["selected_attacker"] = None
            state["auto_defender"] = None
            state["result"] = None
            if state["index"] >= len(USABLE_FRAMES):
                state["index"] = len(USABLE_FRAMES) - 1
                respond({"done": True})
            else:
                respond({"done": False})

PORT = 5002
print(f"\nStarting offside checker at http://localhost:{PORT}")
print("Open that URL in your Windows browser\n")

server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

try:
    while True:
        pass
except KeyboardInterrupt:
    pass
