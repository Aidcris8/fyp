import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/team_identification_v2"
DETECTIONS_FILE = "data/player_detections_v2.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolo11m.pt")

# HSV histogram bins
N_H_BINS = 18   # 10 degrees per bin across 0-180 (OpenCV hue range)
N_S_BINS = 4    # low / medium / high / vivid saturation
N_BINS = N_H_BINS * N_S_BINS  # 72 total bins

# Grass HSV range - excluded from player histograms
GRASS_H_LO, GRASS_H_HI = 30, 85
GRASS_S_MIN, GRASS_V_MIN = 40, 40


def get_grass_colour_lab(img):
    """Return mean grass colour in Lab — used for outlier detection (same as v1)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    px = img[mask > 0]
    if len(px) == 0:
        bgr = np.array([[[50, 100, 50]]], dtype=np.uint8)
    else:
        bgr = np.mean(px, axis=0).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[0][0].astype(np.float32)


def get_player_histogram(img, box):
    """
    Extract a normalised HSV colour histogram from the jersey region only.

    Key differences vs v1:
    - Skips the top 20% of the bounding box (head/hair excluded)
    - Crops the next 40% (upper jersey only, not shorts)
    - Grass-coloured pixels explicitly removed before histogramming
    - Returns a raw 72-element frequency vector; tf-idf applied later across
      all players in the frame so common colours (grass residual, white numbers)
      are down-weighted automatically.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1

    if h < 20 or w < 10:
        return None

    # Skip head: top 20% of box; jersey region: next 40%
    head_skip  = max(5, int(h * 0.20))
    jersey_end = min(h, head_skip + max(8, int(h * 0.40)))

    crop = img[y1 + head_skip: y1 + jersey_end, x1:x2]
    if crop.size == 0:
        return None

    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels   = crop_hsv.reshape(-1, 3).astype(np.float32)
    H_ch, S_ch, V_ch = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Remove grass-coloured pixels
    grass_mask = ((H_ch >= GRASS_H_LO) & (H_ch <= GRASS_H_HI) &
                  (S_ch >= GRASS_S_MIN) & (V_ch >= GRASS_V_MIN))
    pixels = pixels[~grass_mask]

    if len(pixels) < 4:
        return None

    H_ch, S_ch = pixels[:, 0], pixels[:, 1]

    # 2-D histogram: H x S  (18 * 4 = 72 bins)
    hist   = np.zeros(N_BINS, dtype=np.float32)
    h_step = 180.0 / N_H_BINS
    s_step = 256.0 / N_S_BINS

    for hi in range(N_H_BINS):
        for si in range(N_S_BINS):
            mask = ((H_ch >= hi * h_step) & (H_ch < (hi + 1) * h_step) &
                    (S_ch >= si * s_step) & (S_ch < (si + 1) * s_step))
            hist[hi * N_S_BINS + si] = float(np.sum(mask))

    total = hist.sum()
    if total == 0:
        return None

    return hist / total   # normalised term frequency


def apply_tfidf(histograms):
    """
    Apply tf-idf weighting across all player histograms in the frame.
    Colours that appear in many player crops (grass bleed, white shirt numbers)
    receive a low IDF weight; colours unique to one team's kit are amplified.
    Returns L2-normalised tf-idf vectors ready for K-means.
    """
    hists = np.array(histograms, dtype=np.float32)   # (N, 72)
    N     = len(hists)

    # Document frequency: players whose bin exceeds a small threshold
    df  = np.sum(hists > 0.005, axis=0).astype(np.float32)

    # Smoothed IDF
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0

    tfidf = hists * idf

    # L2 normalise each player's vector
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return tfidf / norms


def get_player_colour_lab(img, box, grass_lab):
    """
    Lab 6-feature vector — used ONLY for outlier detection (identical to v1).
    Keeps the original top-40% crop so outlier thresholds remain calibrated.
    """
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width  = x2 - x1

    if height < 20 or width < 10:
        return None

    crop_y2 = y1 + int(height * 0.40)
    if crop_y2 - y1 < 8:
        crop_y2 = y1 + 8
    crop = img[y1:crop_y2, x1:x2]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, (30, 20))
    pixels = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)

    if len(pixels) < 4:
        return None

    grass_dists = np.linalg.norm(pixels - grass_lab, axis=1)
    non_grass   = pixels[grass_dists > 25]
    if len(non_grass) < 4:
        non_grass = pixels

    L, a, b    = non_grass[:, 0], non_grass[:, 1], non_grass[:, 2]
    sat_proxy  = np.median(np.sqrt((a - 128) ** 2 + (b - 128) ** 2))
    size_factor = min(1.0, (height * width) / 8000)

    return np.array([
        np.median(L),
        np.median(a),
        np.median(b),
        np.mean(L < 80) * 100 * size_factor,
        np.mean(L > 150) * 100,
        sat_proxy,
    ])


def identify_teams(img, boxes, grass_lab):
    # ------------------------------------------------------------------ #
    # Step 0 — build both feature sets per player                         #
    # ------------------------------------------------------------------ #
    histograms   = []
    lab_features = []
    valid_boxes  = []

    for box in boxes:
        hist = get_player_histogram(img, box)
        lab  = get_player_colour_lab(img, box, grass_lab)
        if hist is not None and lab is not None:
            histograms.append(hist)
            lab_features.append(lab)
            valid_boxes.append(box)

    if len(histograms) < 4:
        print("  Not enough players for classification")
        return np.array([]), []

    lab_array = np.array(lab_features)   # (N, 6)

    # ------------------------------------------------------------------ #
    # Step 1 — pre-flag dark / achromatic players (referee logic, v1)    #
    # ------------------------------------------------------------------ #
    pre_flagged = set()
    for i, lab in enumerate(lab_array):
        L, sat = lab[0], lab[5]
        if L < 50:
            pre_flagged.add(i)
            print(f"  Pre-flagged P{i}: very dark (L={L:.1f})")
        elif L < 85 and sat < 10:
            pre_flagged.add(i)
            print(f"  Pre-flagged P{i}: dark+achromatic (L={L:.1f} sat={sat:.1f})")

    filtered_idx = [i for i in range(len(histograms)) if i not in pre_flagged]
    filt_hists   = [histograms[i]  for i in filtered_idx]
    filt_lab     = lab_array[filtered_idx]
    filt_boxes   = [valid_boxes[i] for i in filtered_idx]

    if len(filt_hists) < 4:
        print("  Not enough players after pre-flagging")
        return np.array([]), []

    # ------------------------------------------------------------------ #
    # Step 2 — tf-idf weighting then K-means on HSV histograms           #
    # ------------------------------------------------------------------ #
    tfidf_matrix = apply_tfidf(filt_hists)   # (N_filtered, 72)

    best_labels  = None
    best_balance = -1

    for seed in [42, 0, 7, 13, 99]:
        km  = KMeans(n_clusters=2, random_state=seed, n_init=10)
        lbl = km.fit_predict(tfidf_matrix)
        c0, c1  = np.sum(lbl == 0), np.sum(lbl == 1)
        balance = min(c0, c1) / max(c0, c1) if max(c0, c1) > 0 else 0
        if balance > best_balance:
            best_balance = balance
            best_labels  = lbl

    team_labels = np.array(best_labels)

    # Lab centroids for outlier detection
    centres = np.array([
        np.mean(filt_lab[team_labels == 0], axis=0),
        np.mean(filt_lab[team_labels == 1], axis=0),
    ])

    t0, t1 = np.sum(team_labels == 0), np.sum(team_labels == 1)
    print(f"  Team 0: {t0} players, avg Lab: "
          f"({int(centres[0][0])},{int(centres[0][1])},{int(centres[0][2])})")
    print(f"  Team 1: {t1} players, avg Lab: "
          f"({int(centres[1][0])},{int(centres[1][1])},{int(centres[1][2])})")

    # ------------------------------------------------------------------ #
    # Step 2b — degenerate-split refit on Lab chrominance (same as v1)   #
    # ------------------------------------------------------------------ #
    if min(t0, t1) < 3:
        print(f"  Degenerate split ({t0} vs {t1}), refitting on Lab chrominance...")
        minority = 0 if t0 < t1 else 1
        maj_idx  = [i for i, l in enumerate(team_labels) if l != minority]
        min_idx  = [i for i, l in enumerate(team_labels) if l == minority]
        maj_lab  = filt_lab[maj_idx]

        if len(maj_lab) >= 6:
            sc2     = StandardScaler()
            maj_ab  = sc2.fit_transform(maj_lab[:, [1, 2]])
            km2     = KMeans(n_clusters=2, random_state=42, n_init=10)
            new_lbl = km2.fit_predict(maj_ab)
            centres = np.array([
                np.mean(maj_lab[new_lbl == 0], axis=0),
                np.mean(maj_lab[new_lbl == 1], axis=0),
            ])
            new_team_labels = np.full(len(team_labels), 2)
            for arr_i, lbl in zip(maj_idx, new_lbl):
                new_team_labels[arr_i] = lbl
            for arr_i in min_idx:
                d0 = np.linalg.norm(filt_lab[arr_i] - centres[0])
                d1 = np.linalg.norm(filt_lab[arr_i] - centres[1])
                new_team_labels[arr_i] = 0 if d0 < d1 else 1
            team_labels = new_team_labels
            print(f"  After refit — Team 0: {np.sum(team_labels==0)}, "
                  f"Team 1: {np.sum(team_labels==1)}")

    # ------------------------------------------------------------------ #
    # Step 3 — outlier detection on Lab features (identical to v1)       #
    # ------------------------------------------------------------------ #
    final_labels = list(team_labels)

    dist_to_c0 = np.linalg.norm(filt_lab - centres[0], axis=1)
    dist_to_c1 = np.linalg.norm(filt_lab - centres[1], axis=1)
    d_nearest  = np.minimum(dist_to_c0, dist_to_c1)

    # Pass 1: global outliers
    team_01 = [i for i, l in enumerate(final_labels) if l in (0, 1)]
    if len(team_01) >= 4:
        d01      = d_nearest[team_01]
        med_d    = np.median(d01)
        mad      = max(1.4826 * np.median(np.abs(d01 - med_d)), 8.0)
        g_thresh = med_d + 3.5 * mad
        print(f"  Global outlier threshold: {g_thresh:.1f} (median={med_d:.1f} MAD={mad:.1f})")
        for i in team_01:
            if d_nearest[i] > g_thresh:
                d_other = max(dist_to_c0[i], dist_to_c1[i])
                if d_nearest[i] < 0.4 * d_other:
                    continue
                final_labels[i] = 2

    # Pass 2: per-team outliers
    for tid in [0, 1]:
        t_idx   = [i for i, l in enumerate(final_labels) if l == tid]
        if len(t_idx) < 4:
            continue
        t_dists  = np.linalg.norm(filt_lab[t_idx] - centres[tid], axis=1)
        t_med    = np.median(t_dists)
        t_mad    = max(min(1.4826 * np.median(np.abs(t_dists - t_med)), 10.0), 5.0)
        t_thresh = t_med + 3.0 * t_mad
        for arr_i, dist in zip(t_idx, t_dists):
            if dist > t_thresh:
                final_labels[arr_i] = 2

    # Pass 3: spatial GK detection
    if len(filt_boxes) >= 4:
        cx_py = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in filt_boxes],
                         dtype=np.float32)
        nn_dists = np.zeros(len(cx_py))
        for i in range(len(cx_py)):
            ds     = np.linalg.norm(cx_py - cx_py[i], axis=1)
            ds[i]  = np.inf
            nn_dists[i] = np.min(ds)

        x_pos = cx_py[:, 0]
        x_med = np.median(x_pos)
        x_mad = max(1.4826 * np.median(np.abs(x_pos - x_med)), 50.0)
        x_dev = np.abs(x_pos - x_med)

        y_pos = cx_py[:, 1]
        y_med = np.median(y_pos)
        y_mad = max(1.4826 * np.median(np.abs(y_pos - y_med)), 50.0)
        top_y = np.maximum(0.0, y_med - y_pos)

        for tid in [0, 1]:
            t_idx    = [i for i, l in enumerate(final_labels) if l == tid]
            if len(t_idx) < 4:
                continue
            t_col_d   = np.linalg.norm(filt_lab[t_idx] - centres[tid], axis=1)
            t_col_med = max(np.median(t_col_d), 1.0)
            spatial   = (x_dev[t_idx] / x_mad + top_y[t_idx] / y_mad) * nn_dists[t_idx]
            best      = int(np.argmax(spatial))
            best_idx  = t_idx[best]
            if (spatial[best] > 250 and nn_dists[best_idx] > 100
                    and t_col_d[best] / t_col_med >= 0.8):
                final_labels[best_idx] = 2
                print(f"  GK (spatial) T{tid}: combined={spatial[best]:.0f} "
                      f"nn={nn_dists[best_idx]:.0f}px")

    # Pass 4: hard OTH cap
    MAX_OTH = 6
    oth_idx = [i for i, l in enumerate(final_labels) if l == 2]
    if len(oth_idx) > MAX_OTH:
        print(f"  OTH cap: {len(oth_idx)} -> {MAX_OTH}")
        oth_by_d = sorted(oth_idx, key=lambda i: d_nearest[i], reverse=True)
        for arr_i in oth_by_d[MAX_OTH:]:
            final_labels[arr_i] = 0 if dist_to_c0[arr_i] < dist_to_c1[arr_i] else 1

    # ------------------------------------------------------------------ #
    # Step 4 — rebuild full label list including pre-flagged              #
    # ------------------------------------------------------------------ #
    all_labels = []
    all_boxes  = []
    ptr = 0
    for i in range(len(histograms)):
        if i in pre_flagged:
            all_labels.append(2)
            all_boxes.append(valid_boxes[i])
        else:
            all_labels.append(final_labels[ptr])
            all_boxes.append(filt_boxes[ptr])
            ptr += 1

    oth_count = sum(1 for l in all_labels if l == 2)
    print(f"  Officials/GK detected: {oth_count}")
    return np.array(all_labels), all_boxes


def draw_teams(img, labels, boxes):
    colours = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]
    result  = img.copy()
    counts  = {0: 0, 1: 0, 2: 0}
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        col  = colours[label]
        text = "OTH" if label == 2 else f"T{label}"
        cv2.rectangle(result, (x1, y1), (x2, y2), col, 2)
        cv2.putText(result, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        counts[label] += 1
    print(f"  Team 0: {counts[0]}  Team 1: {counts[1]}  OTH: {counts[2]}")
    return result


# ======================================================================= #
# Main loop                                                                #
# ======================================================================= #
detections = {}
keyframes  = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])

for kf in keyframes:
    print(f"\nTesting on: {kf}")
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf))
    frame_h, frame_w = img.shape[:2]

    grass_lab = get_grass_colour_lab(img)

    # Probe run
    probe = model(img, classes=[0], conf=0.3, iou=0.7, verbose=False)[0].boxes
    probe_heights = [(b.xyxy[0][3] - b.xyxy[0][1]).item()
                     for b in probe if int(b.cls[0]) == 0]

    if len(probe_heights) > 3:
        med_h      = sorted(probe_heights)[len(probe_heights) // 2]
        min_height = 20 if med_h < 45 else 40
        top_zone   = 0.65 if med_h < 45 else 0.45
    else:
        min_height, top_zone = 25, 0.55

    player_boxes = [
        b.xyxy[0].tolist() for b in probe
        if int(b.cls[0]) == 0 and (b.xyxy[0][3] - b.xyxy[0][1]) > min_height
    ]

    # Second pass — low confidence, upper zone
    low_boxes = model(img, classes=[0], conf=0.05, iou=0.7, verbose=False)[0].boxes
    for b in low_boxes:
        if int(b.cls[0]) != 0:
            continue
        box = b.xyxy[0].tolist()
        x1, y1, x2, y2 = box
        if y2 < frame_h * top_zone and (y2 - y1) > 20:
            dup = any(
                max(0, min(x2, ex2) - max(x1, ex1)) > 20 and
                max(0, min(y2, ey2) - max(y1, ey1)) > 20
                for ex1, ey1, ex2, ey2 in player_boxes
            )
            if not dup:
                player_boxes.append(box)

    # Dynamic bottom cutoff
    centres_y = sorted([(b[1] + b[3]) / 2 for b in player_boxes])
    dynamic_bottom = frame_h * 0.88
    if len(centres_y) > 4:
        gaps = [(centres_y[i+1] - centres_y[i], i) for i in range(len(centres_y) - 1)]
        max_gap, max_gap_idx = max(gaps)
        bot_count = len(centres_y) - (max_gap_idx + 1)
        bot_dets  = centres_y[max_gap_idx + 1:]
        if max_gap > 50 and bot_count <= 3 and all(cy > frame_h * 0.80 for cy in bot_dets):
            dynamic_bottom = (centres_y[max_gap_idx] + centres_y[max_gap_idx + 1]) / 2

    non_player, pitch_boxes = [], []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        cy, cx, bh = (y1+y2)/2, (x1+x2)/2, y2-y1
        if ((cx < 80 and bh < 50) or (cx > frame_w - 80 and bh < 50) or
                (cy > dynamic_bottom and bh < 70) or
                (y1 < frame_h * 0.20 and bh < 40)):
            non_player.append(box)
        else:
            pitch_boxes.append(box)

    print(f"  Players detected: {len(pitch_boxes)}")

    labels, valid_boxes = identify_teams(img, pitch_boxes, grass_lab)
    if len(labels) > 0:
        all_labels = list(labels) + [2] * len(non_player)
        all_boxes  = list(valid_boxes) + non_player
        out_img    = draw_teams(img, np.array(all_labels), all_boxes)
        cv2.imwrite(os.path.join(OUTPUT_DIR, kf), out_img)

        players_data = []
        for box, lbl in zip(all_boxes, all_labels):
            x1, y1, x2, y2 = map(int, box)
            players_data.append({
                "box": [x1, y1, x2, y2],
                "team": int(lbl),
                "foot_px": [(x1+x2)//2, y2],
            })
        detections[kf] = {"players": players_data}
        print(f"  Saved")
    else:
        print("  Skipped - insufficient players")

with open(DETECTIONS_FILE, "w") as f:
    json.dump(detections, f, indent=2)
print(f"\nSaved to {DETECTIONS_FILE}")
print("Done!")
