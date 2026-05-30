# This file implements a browser-based UI for reviewing each keyframe and determining the verdict. 
# The user selects the attacking team, attacking direction, and the player recieving the ball
# The system then automatically finds the second-to-last defender and computes the offside line using the homography matrix for that frame
# verdicts are all saved to data/offside_verdicts.json
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
OUTPUT_DIR    = "data/offside_results"
VERDICTS_FILE = "data/offside_verdicts.json"
SKIPPED_FILE  = "data/offside_skipped.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load player detections
with open(DETECTIONS_FILE) as f:
    all_detections = json.load(f)
#load homography matrices
with open(HOMOGRAPHY_FILE) as f:
    all_homographies = json.load(f)

# load persisted verdicts so completed frames are skipped on restart.
if os.path.exists(VERDICTS_FILE):
    with open(VERDICTS_FILE) as f:
        all_verdicts = json.load(f)
else:
    all_verdicts = {}

def save_verdicts():
    with open(VERDICTS_FILE, "w") as f:
        json.dump(all_verdicts, f, indent=2)

# frames the user passed on without saving a verdict. Excluded from
# USABLE_FRAMES so they do not reappear next session.
if os.path.exists(SKIPPED_FILE):
    with open(SKIPPED_FILE) as f:
        skipped_verdicts = set(json.load(f))
else:
    skipped_verdicts = set()

def save_skipped_verdicts():
    with open(SKIPPED_FILE, "w") as f:
        json.dump(sorted(skipped_verdicts), f, indent=2)

INVALID_ZONE_THRESHOLD = 0.15  # skip frames where >15% of the player zone has H[2]<=0

def has_reliable_homography(kf):
    if kf not in all_homographies:
        return False
    frac = all_homographies[kf].get("invalid_zone_frac", 0.0)
    if frac > INVALID_ZONE_THRESHOLD:
        print(f"  [SKIP] {kf[-60:]}  — {frac:.0%} of player zone invalid (re-pick points)")
        return False
    return True

# build the list of eligible frames i.e frames with reliable homography
ALL_FRAMES = sorted([
    kf for kf in all_detections
    if has_reliable_homography(kf)
])
USABLE_FRAMES = [kf for kf in ALL_FRAMES
                 if kf not in all_verdicts and kf not in skipped_verdicts]
print(f"Total eligible frames : {len(ALL_FRAMES)}")
print(f"Already done          : {len(all_verdicts)}")
print(f"Previously skipped    : {len(skipped_verdicts)}")
print(f"Remaining this session: {len(USABLE_FRAMES)}")
for kf in USABLE_FRAMES:
    parts = kf.split('_frame')
    if len(parts) >= 2:
        print(f"  {parts[0][-30:]}...frame{parts[1]}")
    else:
        print(f"  {kf[-50:]}")

# sessions starts , reset on each new frame
state = {
    "index": 0,
    "attacking_team": None,
    "attacking_right": None,
    "selected_attacker": None,
    "auto_defender": None,
    "result": None,
}

def attacking_larger_wx(H, attacking_right):
    # returns True if the attacking direction corresponds to increasing world_x
    # samples the homography at two image x-positions to detect whether the
    # homography axis is flipped relative to image-space left/right
    left_wx,  _ = project_point(H, 320, 360)
    right_wx, _ = project_point(H, 960, 360)
    image_right_is_world_right = right_wx > left_wx
    return attacking_right == image_right_is_world_right

def find_auto_defender(kf, attacking_team, attacking_right):
    # returns the index of the most-advanced outfield defender using world-space coordinates
    # detects homography axis orientation so image-space attacking_right is correctly
    # translated to world-space direction before ranking defenders
    players = all_detections[kf]["players"]
    H = np.array(all_homographies[kf]["H"])
    # defending team always opposite to attacking time
    defending_team = 1 - attacking_team
    defending = [(i, p) for i, p in enumerate(players) if p["team"] == defending_team]
    if not defending:
        return None

    going_larger = attacking_larger_wx(H, attacking_right)

    # project player's foot center into world space and return X coordinate
    def world_x_center(p):
        x1, y1, x2, y2 = p["box"]
        cx = (x1 + x2) / 2
        wx, _ = project_point(H, cx, y2)
        return wx

    # reject detections whose foot projection lands outside physical pitch boundary
    def within_pitch(p):
        x1, y1, x2, y2 = p["box"]
        cx = (x1 + x2) / 2
        wx, wy = project_point(H, cx, y2)
        return abs(wx) <= 55.0 and abs(wy) <= 36.0

    # exclude defenders whose foot projection falls outside the pitch boundary
    # these are almost always stewards or pitch-side personnel mis-detected as
    # outfield players (a 105x68 m pitch means |X|>55 or |Y|>36 is impossible)
    on_pitch = [(i, p) for i, p in defending if within_pitch(p)]
    if not on_pitch:
        on_pitch = defending  # fallback if filter removes everyone
    else:
        removed = len(defending) - len(on_pitch)
        if removed:
            print(f"  Pitch-boundary filter removed {removed} out-of-bounds defender(s)")

    # Most advanced = smallest world_x if pushing toward larger wx, else largest
    ranked = sorted(on_pitch, key=lambda ip: world_x_center(ip[1]), reverse=going_larger)
    direction = "min world-X" if going_larger else "max world-X"
    print(f"  Defending team T{defending_team} ranked by {direction} (going_larger_wx={going_larger}):")
    for i, p in ranked:
        print(f"    P{i}: center_world_x={world_x_center(p):.2f}m")

    #pick first player in ranked list , most advance defender
    chosen = ranked[0][0]
    print(f"  → Chosen: P{chosen}")
    return chosen

# convenience helper to pull current frame's players and homography matrix
def get_frame_data():
    kf = USABLE_FRAMES[state["index"]]
    players = all_detections[kf]["players"]
    H = np.array(all_homographies[kf]["H"])
    return kf, players, H

# project an image pixel (px, py) into world coordinates through H 
def project_point(H, px, py):
    # convert pixel to homogenoues coordinates, apply H, then divide by third component
    pt = np.array([px, py, 1.0])
    proj = H @ pt
    proj /= proj[2]
    return float(proj[0]), float(proj[1])

# compute the image space slope of the offside line at the defender's feet position
# line is vertical in world space but appears tilted in imge due to perspective
def get_line_slope(kf, offside_line_x, defender_y, H):
    H_inv = np.linalg.inv(H)

    # inner helper, project world point back into image space using H_inv
    def real_to_image(rx, ry):
        pt = np.array([rx, ry, 1.0])
        img_pt = H_inv @ pt
        img_pt /= img_pt[2]
        return float(img_pt[0]), float(img_pt[1])
    # sample two points 5m apart along the pitch and measure the tilt via H_inv
    dy = 5.0
    px1 = real_to_image(offside_line_x, defender_y - dy)
    px2 = real_to_image(offside_line_x, defender_y + dy)
    # measure how far line shifts horizontally per unit of vertical image movement
    dx_img = px2[0] - px1[0]
    dy_img = px2[1] - px1[1]
    slope = dx_img / dy_img if abs(dy_img) > 1e-6 else 0.0
    print(f"  Slope from inverse homography: {slope:.3f}")
    return slope

def _draw_offside_zone(img, line_pts, attacking_right, alpha=0.38):
    # darken the half of the image beyond the offside line to replicate the broadcast VAR look
    h, w = img.shape[:2]
    (x_top, y_top), (x_bot, y_bot) = line_pts
    if attacking_right:
        polygon = np.array([[x_top, y_top], [w, y_top], [w, y_bot], [x_bot, y_bot]], np.int32)
    else:
        polygon = np.array([[0, y_top], [x_top, y_top], [x_bot, y_bot], [0, y_bot]], np.int32)
    # build a filled back polygon covering offside half, then blend with frame 
    # alpha = 0.38 means the zone is darkened but still very visible
    overlay = img.copy()
    cv2.fillPoly(overlay, [polygon], (0, 0, 0), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_verdict_button(img, is_offside):
    # draw a rounded-pill verdict button at the bottom centre of the frame
    # red for OFFSIDE, green for ONSIDE
    h, w = img.shape[:2]

    if is_offside:
        bg_color = (60, 60, 235)     # bright red (BGR)
        text = "OFFSIDE"
    else:
        bg_color = (95, 195, 95)     # bright TV green
        text = "ONSIDE"

    button_h = 60
    radius = button_h // 2           # full pill — semicircle caps

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.05
    thick = 2
    text_size = cv2.getTextSize(text, font, scale, thick)[0]
    button_w = text_size[0] + 80     # horizontal padding around the text

    cx = w // 2
    bottom_pad = 28

    x1 = cx - button_w // 2
    y2 = h - bottom_pad
    y1 = y2 - button_h
    x2 = cx + button_w // 2

    overlay = img.copy()

    # Pill body: middle rectangle + two semicircle caps
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, bg_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, bg_color, -1, lineType=cv2.LINE_AA)

    # Centred text
    text_x = x1 + (button_w - text_size[0]) // 2
    text_y = y1 + (button_h + text_size[1]) // 2 - 2
    cv2.putText(overlay, text, (text_x, text_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)

# render current frame with player boxes, offside line, and verdict overlay drawn on it
# returns a string for th ebrowser plus dimensions and raw numpy image for saving
def render_frame():
    kf, players, H = get_frame_data()
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf))
    team_colours = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 255)}

    # draw each players bounding box 
    for i, p in enumerate(players):
        x1, y1, x2, y2 = p["box"]
        team = p["team"]
        colour = team_colours.get(team, (128, 128, 128))
        thickness = 1

        # attacker in highlighted green
        if i == state["selected_attacker"]:
            colour = (0, 255, 0)
            thickness = 2
        # defender in orange
        elif i == state["auto_defender"]:
            colour = (0, 165, 255)
            thickness = 2

        # others by team colour unless OTH 
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
        label = "OTH" if team == 2 else f"T{team}"
        cv2.putText(img, f"{i}:{label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # if a verdict has been computed, overlay the offside line, zone darkening and result button
    if state["result"] is not None:
        res = state["result"]
        line_pts = res.get("line_image_pts")

        if line_pts:
            # Darken the offside zone (everything beyond the line in the attacking direction)
            _draw_offside_zone(img, line_pts, res.get("attacking_right", True))

            # Offside line itself, drawn over the dimmed zone
            pt1 = tuple(map(int, line_pts[0]))
            pt2 = tuple(map(int, line_pts[1]))
            cv2.line(img, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        sa = state["selected_attacker"]
        if sa is not None and sa < len(players):
            x1, y1, x2, y2 = players[sa]["box"]
            # OFFSIDE → magenta (distinct from team-0 red); ONSIDE → green
            verdict_colour = (255, 0, 255) if res["offside"] else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), verdict_colour, 2)
            # Verdict text now lives in the bottom banner (no putText here)

        # Verdict button (red OFFSIDE / green ONSIDE)
        _draw_verdict_button(img, res["offside"])

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buffer).decode('utf-8'), img.shape[1], img.shape[0], img

# computes offside verdict for current frame using selected attacker and auto selected defender
# projects both players' relevant bounding box edges into world space and compares X positions
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

    # extract attacker bounding box
    att_box = players[attacker_idx]["box"]
    x1_att, y1_att, x2_att, y2_att = att_box

    # extract defenedr bounding box
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

    # offside line is set at defenders world - x position 
    offside_line_x = defender_x

    # determine which direction counts as ahead in world space given this homography
    going_larger = attacking_larger_wx(H, attacking_right)
    if going_larger:
        is_offside = attacker_x > offside_line_x
    else:
        is_offside = attacker_x < offside_line_x

    # margin is always positive (hence abs) this measures how far ahead or behing line attacker is
    margin = abs(attacker_x - offside_line_x)

    slope = get_line_slope(kf, offside_line_x, defender_y, H)

    # anchor line at defender's actual foot pixel position
    anchor_y = players[defender_idx]["foot_px"][1]
    dy_to_top = anchor_y
    dy_to_bottom = img_h - anchor_y

    # extend line from defenders foot to top and bottom of image frame
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

# HTML/CSS/JS served to the browser. Palette and top-bar styles come from
# static/ui.css (shared with the other two UIs). The JavaScript calls the
# HTTP routes defined in class Handler below.
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Offside Checker</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/ui.css">
    <style>
        /* UI-specific styles for the offside checker. Palette, fonts and
           top-bar live in /static/ui.css. */
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

        // fetch the current keyframe from the server and draw it on the canvas
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

        // update a workflow step indicator — stateCls is 'active', 'done', or empty string
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

        // step 1: user selected the attacking team — enable direction buttons and notify server
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

        // step 2: user selected attacking direction — server auto-selects the second-to-last defender
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

        // step 3: user clicks a player on the canvas — find which bounding box was hit
        // and send the selected attacker index to the server
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

        // step 4: trigger the offside computation, display the verdict, and auto-save the result
        function checkOffside() {
            fetch('/check_offside', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.error) return;
                showVerdict(data);
                setStepState(4, 'done', 'See verdict below');
                loadFrame();
                fetch('/save_result', {method: 'POST'});
            });
        }

        // populate the verdict panel below the canvas with the result and margin meter
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

        // clear all step selections and UI state without advancing to the next frame
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

        // reset the UI and re-fetch the frame so the user can start the workflow again
        function resetFrame() {
            resetUI();
            fetch('/reset', {method: 'POST'}).then(() => loadFrame());
        }

        // replace the page with a completion screen once all frames have been checked
        function showDone() {
            document.body.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:Inter,sans-serif;background:#0a0b10;">' +
                '<div style="text-align:center">' +
                '<h2 style="color:#10b981;font-size:26px;margin-bottom:12px;font-weight:600;">All frames checked</h2>' +
                '<p style="color:#9aa0b0;font-size:14px;">Results saved to data/offside_results/</p>' +
                '</div></div>';
        }

        // advance to the next frame — server records a skip if no verdict was saved
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

# HTTP handler, GET serves UI, POST handles stp by stp workflow
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

        elif parsed.path == '/frame':
            kf = USABLE_FRAMES[state["index"]]
            frame_b64, w, h, _ = render_frame()
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

        # user selected attacking team - reset all downstream state
        if parsed.path == '/set_attacking':
            data = json.loads(body)
            state["attacking_team"] = data["team"]
            state["attacking_right"] = None
            state["selected_attacker"] = None
            state["auto_defender"] = None
            state["result"] = None
            respond({"ok": True})

        # user selected attacking direction - auto select defender
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

        # user clicked on player on canvas, find his respective bounding box
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

        # run the offside computation and store result in state
        elif parsed.path == '/check_offside':
            result = compute_offside()
            state["result"] = result
            respond(result)

        elif parsed.path == '/save_result':
            _, w, h, img = render_frame()
            kf = USABLE_FRAMES[state["index"]]
            cv2.imwrite(os.path.join(OUTPUT_DIR, kf), img)
            # Persist the verdict so this frame is skipped on restart.
            if state["result"] is not None:
                all_verdicts[kf] = {
                    "offside":        state["result"].get("offside"),
                    "margin_metres":  state["result"].get("margin_metres"),
                    "attacking_team": state["attacking_team"],
                    "attacking_right":state["attacking_right"],
                }
                save_verdicts()
            respond({"ok": True})

        # clear all step selections for current frame without advancing
        elif parsed.path == '/reset':
            state["attacking_team"] = None
            state["attacking_right"] = None
            state["selected_attacker"] = None
            state["auto_defender"] = None
            state["result"] = None
            respond({"ok": True})

        elif parsed.path == '/next':
            current_kf = USABLE_FRAMES[state["index"]]
            if current_kf not in all_verdicts:
                # User advanced without saving a verdict for this frame.
                # Mark it skipped so it doesn't re-queue next session.
                skipped_verdicts.add(current_kf)
                save_skipped_verdicts()
                print(f"  Skipped (no verdict): {current_kf}")
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
