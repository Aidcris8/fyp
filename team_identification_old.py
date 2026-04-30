import cv2
import numpy as np
import os
import json
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/team_identification"
DETECTIONS_FILE = "data/player_detections.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolo11m.pt")

def get_grass_colour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_pixels = img[mask > 0]
    if len(grass_pixels) == 0:
        grass_bgr = np.array([[[50, 100, 50]]], dtype=np.uint8)
    else:
        mean_bgr = np.mean(grass_pixels, axis=0).astype(np.uint8)
        grass_bgr = mean_bgr.reshape(1, 1, 3)
    grass_lab = cv2.cvtColor(grass_bgr, cv2.COLOR_BGR2Lab)[0][0].astype(np.float32)
    return grass_lab

def get_player_colour(img, box, grass_colour):
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width = x2 - x1

    if height < 20 or width < 10:
        return None

    head_skip = max(5, int(height * 0.20))
    crop_y1 = y1 + head_skip
    crop_y2 = y1 + min(height, head_skip + max(8, int(height * 0.40)))
    crop = img[crop_y1:crop_y2, x1:x2]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, (30, 20))
    crop_lab = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2Lab)
    pixels = crop_lab.reshape(-1, 3).astype(np.float32)

    if len(pixels) < 4:
        return None

    grass_dists = np.linalg.norm(pixels - grass_colour, axis=1)
    non_grass = pixels[grass_dists > 25]
    if len(non_grass) < 4:
        non_grass = pixels

    L = non_grass[:, 0]
    a = non_grass[:, 1]
    b = non_grass[:, 2]

    dark_ratio = np.mean(L < 80)
    bright_ratio = np.mean(L > 150)
    median_L = np.median(L)
    median_a = np.median(a)
    median_b = np.median(b)
    sat_proxy = np.median(np.sqrt((a - 128)**2 + (b - 128)**2))

    size_factor = min(1.0, (height * width) / 8000)
    scaled_dark = dark_ratio * 100 * size_factor

    return np.array([median_L, median_a, median_b, scaled_dark, bright_ratio * 100, sat_proxy])

def identify_teams(img, boxes, grass_colour, debug=False):
    colours = []
    valid_boxes = []

    for box in boxes:
        colour = get_player_colour(img, box, grass_colour)
        if colour is not None:
            colours.append(colour)
            valid_boxes.append(box)

    if len(colours) < 4:
        print("  Not enough players for classification")
        return np.array([]), []

    colours_array = np.array(colours)

    # Step 1 — Pre-flag players that would disrupt K-Means: very dark (L<50) or
    # dark+achromatic (referee in black: L<85 and near-neutral colour saturation<10).
    # FIX 3: skip the dark+achromatic rule if >= 4 players fit the "dark desaturated"
    # profile (L<95 AND sat<15) — that's a TEAM wearing a dark kit (e.g. PSG navy),
    # not isolated refs.
    pre_flagged = set()
    dark_team_count = sum(1 for c in colours_array if c[0] < 95 and c[5] < 15)
    dark_team_present = dark_team_count >= 4
    if dark_team_present:
        print(f"  Dark-team detected ({dark_team_count} dark-desaturated players) — "
              f"skipping dark+achromatic pre-flag")

    for i, colour in enumerate(colours_array):
        L, sat = colour[0], colour[5]
        if L < 50:
            pre_flagged.add(i)
            print(f"  Pre-flagged player {i+1}: very dark (L={L:.1f})")
        elif L < 85 and sat < 10 and not dark_team_present:
            pre_flagged.add(i)
            print(f"  Pre-flagged player {i+1}: dark+achromatic referee (L={L:.1f} sat={sat:.1f})")

    # FIX 1: Colour-isolated pre-flag — catches refs/GKs/linesmen whose kit colour
    # is chromatically distinct. A player is isolated if fewer than 2 others share
    # their (a, b) chromaticity within radius 15. Uses chromaticity only (not L)
    # so that team players in shadow/highlight are still seen as teammates.
    # Only active with >= 8 players to ensure a reliable neighbour count.
    if len(colours_array) >= 8:
        for i, c in enumerate(colours_array):
            if i in pre_flagged:
                continue
            similar = 0
            for j, cj in enumerate(colours_array):
                if j == i or j in pre_flagged:
                    continue
                if np.hypot(cj[1]-c[1], cj[2]-c[2]) < 15:
                    similar += 1
            if similar < 2:
                pre_flagged.add(i)
                print(f"  Pre-flagged player {i+1}: colour-isolated "
                      f"(only {similar} chroma-similar; L={c[0]:.0f} a={c[1]:.0f} b={c[2]:.0f} sat={c[5]:.1f})")

    # Step 2 — K-Means on non-pre-flagged players
    filtered_indices = [i for i in range(len(colours)) if i not in pre_flagged]
    filtered_colours = colours_array[filtered_indices]
    filtered_boxes = [valid_boxes[i] for i in filtered_indices]

    if len(filtered_colours) < 4:
        print("  Not enough players after referee removal")
        return np.array([]), []

    # Normalise features before K-Means so all dimensions contribute equally
    scaler = StandardScaler()
    filtered_colours_scaled = scaler.fit_transform(filtered_colours)

    # Run K-Means multiple times on normalised features, pick most balanced split
    best_labels = None
    best_centres = None
    best_balance = -1

    for seed in [42, 0, 7, 13, 99]:
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        lbl = km.fit_predict(filtered_colours_scaled)
        c0 = np.sum(lbl == 0)
        c1 = np.sum(lbl == 1)
        balance = min(c0, c1) / max(c0, c1)
        if balance > best_balance:
            best_balance = balance
            best_labels = lbl
            best_centres = km.cluster_centers_

    team_labels = list(best_labels)

    # Compute centres in original (unscaled) space for L-axis correction
    team_labels_arr = np.array(team_labels)
    centres = np.array([
        np.mean(filtered_colours[team_labels_arr == 0], axis=0),
        np.mean(filtered_colours[team_labels_arr == 1], axis=0)
    ])

    # Step 2b — L-axis post-correction (using original unscaled colours)
    L_centre0 = centres[0][0]
    L_centre1 = centres[1][0]
    L_separation = abs(L_centre0 - L_centre1)

    if L_separation > 30:
        print(f"  L-axis correction active (separation={L_separation:.0f})")
        for i, label in enumerate(team_labels):
            player_L = filtered_colours[i][0]
            own_L = L_centre0 if label == 0 else L_centre1
            other_L = L_centre1 if label == 0 else L_centre0
            dist_to_own_L = abs(player_L - own_L)
            dist_to_other_L = abs(player_L - other_L)
            if dist_to_other_L < dist_to_own_L * 0.6:
                team_labels[i] = 1 - label

    team_labels = np.array(team_labels)
    centres = np.array([
        np.mean(filtered_colours[team_labels == 0], axis=0),
        np.mean(filtered_colours[team_labels == 1], axis=0)
    ])

    team0_count = np.sum(team_labels == 0)
    team1_count = np.sum(team_labels == 1)
    print(f"  Team 0: {team0_count} players, avg Lab: ({int(centres[0][0])},{int(centres[0][1])},{int(centres[0][2])})")
    print(f"  Team 1: {team1_count} players, avg Lab: ({int(centres[1][0])},{int(centres[1][1])},{int(centres[1][2])})")

    # FIX 2: Bimodal chroma refit — if the two K-Means clusters have wildly
    # different chromatic spreads (one tight, one dispersed), the split was
    # probably driven by brightness (L) rather than kit colour. Common cause:
    # white-vs-yellow teams where both are bright, so K-Means on 6D picks up
    # incidental L variation. Re-fit on (a, b) only to recover the true kit
    # separation.
    refit_done = False
    if team0_count >= 3 and team1_count >= 3:
        b0_std = float(np.std(filtered_colours[team_labels == 0, 2]))
        b1_std = float(np.std(filtered_colours[team_labels == 1, 2]))
        a0_std = float(np.std(filtered_colours[team_labels == 0, 1]))
        a1_std = float(np.std(filtered_colours[team_labels == 1, 1]))
        chroma0, chroma1 = a0_std + b0_std, a1_std + b1_std
        mx = max(chroma0, chroma1)
        mn = max(min(chroma0, chroma1), 3.0)
        if mx > 2.5 * mn and max(b0_std, b1_std) > 22:
            print(f"  Bimodal chroma refit: chroma_std=({chroma0:.1f},{chroma1:.1f}) — refitting on (a,b)")
            ab_scaled = StandardScaler().fit_transform(filtered_colours[:, [1, 2]])
            best_ab_labels = None
            best_ab_balance = -1
            for seed in [42, 0, 7, 13, 99]:
                km_ab = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(ab_scaled)
                c0, c1 = np.sum(km_ab.labels_ == 0), np.sum(km_ab.labels_ == 1)
                bal = min(c0, c1) / max(c0, c1)
                if bal > best_ab_balance:
                    best_ab_balance = bal
                    best_ab_labels = km_ab.labels_
            team_labels = best_ab_labels
            centres = np.array([
                np.mean(filtered_colours[team_labels == 0], axis=0),
                np.mean(filtered_colours[team_labels == 1], axis=0)
            ])
            team0_count = np.sum(team_labels == 0)
            team1_count = np.sum(team_labels == 1)
            refit_done = True

    # Reject degenerate splits
    min_team_size = 3
    if not refit_done and (team0_count < min_team_size or team1_count < min_team_size):
        print(f"  Degenerate split detected ({team0_count} vs {team1_count}), refitting...")
        minority = 0 if team0_count < team1_count else 1
        majority_indices = [i for i, l in enumerate(team_labels) if l != minority]
        minority_indices = [i for i, l in enumerate(team_labels) if l == minority]
        majority_colours = filtered_colours[majority_indices]

        if len(majority_colours) >= 6:
            # Refit using only [a, b] chrominance — the initial split was dominated by a
            # brightness outlier (L/bright dimension), so clustering on hue gives a better
            # team-colour separation than repeating the full 6D fit.
            scaler2 = StandardScaler()
            majority_ab_scaled = scaler2.fit_transform(majority_colours[:, [1, 2]])
            kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
            new_labels = kmeans2.fit_predict(majority_ab_scaled)
            centres = np.array([
                np.mean(majority_colours[new_labels == 0], axis=0),
                np.mean(majority_colours[new_labels == 1], axis=0)
            ])

            new_team_labels = np.full(len(filtered_colours), 2)
            for idx, label in zip(majority_indices, new_labels):
                new_team_labels[idx] = label
            # Assign minority to nearest new centroid instead of OTH
            for idx in minority_indices:
                d0 = np.linalg.norm(filtered_colours[idx] - centres[0])
                d1 = np.linalg.norm(filtered_colours[idx] - centres[1])
                new_team_labels[idx] = 0 if d0 < d1 else 1
            team_labels = new_team_labels

            team0_count = np.sum(team_labels == 0)
            team1_count = np.sum(team_labels == 1)
            print(f"  After refit — Team 0: {team0_count}, Team 1: {team1_count}")
        else:
            print("  Not enough players for refit")

    # Step 3 — Outlier detection using robust MAD-based thresholds
    final_labels = team_labels.copy().tolist()

    # Compute distance from each player to both team centroids
    dist_to_c0 = np.linalg.norm(filtered_colours - centres[0], axis=1)
    dist_to_c1 = np.linalg.norm(filtered_colours - centres[1], axis=1)
    d_nearest = np.minimum(dist_to_c0, dist_to_c1)
    d_farther = np.maximum(dist_to_c0, dist_to_c1)

    # Per-team intra-team median distance — how tight is each team's colour cluster?
    intra_med = np.array([20.0, 20.0])
    for team_id in (0, 1):
        t_mask = np.array([l == team_id for l in final_labels])
        if np.sum(t_mask) >= 2:
            t_dists = np.linalg.norm(filtered_colours[t_mask] - centres[team_id], axis=1)
            intra_med[team_id] = max(float(np.median(t_dists)), 5.0)
    nearer_team = (dist_to_c1 < dist_to_c0).astype(int)
    nearer_med = intra_med[nearer_team]

    # Unified veto: a player is "clearly one team" only if they are BOTH ratio-close
    # to one team (team_ratio < 0.6) AND absolutely close to it (d_nearest within
    # 2.5x that team's intra-team spread). GKs/refs wearing kits different from both
    # outfield teams end up far from both in absolute terms — even if they happen to
    # be slightly nearer one team, the absolute-distance check un-protects them.
    team_ratio = d_nearest / np.maximum(d_farther, 1e-6)
    clearly_one_team = (team_ratio < 0.6) & (d_nearest < 2.5 * nearer_med)

    # Pass 1: Global outliers — players far from BOTH centroids
    team_01_indices = [i for i, l in enumerate(final_labels) if l in (0, 1)]
    if len(team_01_indices) >= 4:
        d_nearest_01 = d_nearest[team_01_indices]
        median_d = np.median(d_nearest_01)
        mad = max(1.4826 * np.median(np.abs(d_nearest_01 - median_d)), 8.0)
        global_threshold = median_d + 3.5 * mad
        print(f"  Global outlier threshold: {global_threshold:.1f} (median={median_d:.1f} MAD={mad:.1f})")

        for i in team_01_indices:
            if d_nearest[i] > global_threshold and not clearly_one_team[i]:
                final_labels[i] = 2

    # Pass 2: Per-team outliers — tighter thresholds to catch GKs with moderately different kits
    # MAD is floored at 5 and capped at 20 to prevent inflation from high-variance teams.
    # Ratio guard (dist > 1.3 * threshold) avoids flagging players barely over the limit;
    # genuine GK outliers are typically 1.5-2x beyond their team's normal spread.
    for team_id in [0, 1]:
        team_indices = [i for i, l in enumerate(final_labels) if l == team_id]
        if len(team_indices) < 4:
            continue

        team_dists = np.linalg.norm(filtered_colours[team_indices] - centres[team_id], axis=1)
        team_median = np.median(team_dists)
        team_mad_raw = 1.4826 * np.median(np.abs(team_dists - team_median))
        team_mad = max(min(team_mad_raw, 20.0), 5.0)  # floor=5, cap=20
        team_threshold = team_median + 3.0 * team_mad

        for idx, dist in zip(team_indices, team_dists):
            if dist > 1.3 * team_threshold and not clearly_one_team[idx]:
                final_labels[idx] = 2

    # Pass 3: Spatial GK detection — catches GKs whose kit colour is clearly different from team.
    # GKs are either at the X-extreme (left/right goal) or at the Y-top-extreme (far-end goal,
    # which appears at the top of the frame when the camera is near one end of the pitch).
    # Combined score = (x_dev/x_mad + top_y_dev/y_mad) * nn_dist — normalised so both axes
    # contribute fairly. A score > 250 with nn > 100 and colour_ratio >= 1.5 flags a GK.
    # colour_ratio >= 1.5 means the player's colour is at least 50% further from the team
    # centroid than the median — only genuinely different kits (real GKs) pass this bar.
    if len(filtered_boxes) >= 4:
        centers_px = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in filtered_boxes],
                               dtype=np.float32)
        # Nearest-neighbour distance to any other player (regardless of team)
        nn_dists = np.zeros(len(centers_px))
        for i in range(len(centers_px)):
            dists = np.linalg.norm(centers_px - centers_px[i], axis=1)
            dists[i] = np.inf
            nn_dists[i] = np.min(dists)
        # X-deviation and top-Y-deviation (only above-median y counts: top of frame = far end)
        x_pos = centers_px[:, 0]
        x_median = np.median(x_pos)
        x_mad = max(1.4826 * np.median(np.abs(x_pos - x_median)), 50.0)
        x_devs = np.abs(x_pos - x_median)
        y_pos = centers_px[:, 1]
        y_median = np.median(y_pos)
        y_mad = max(1.4826 * np.median(np.abs(y_pos - y_median)), 50.0)
        top_y_devs = np.maximum(0.0, y_median - y_pos)  # positive only when above median (far end)

        for team_id in [0, 1]:
            team_indices = [i for i, l in enumerate(final_labels) if l == team_id]
            if len(team_indices) < 4:
                continue
            team_colour_dists = np.linalg.norm(
                filtered_colours[team_indices] - centres[team_id], axis=1)
            team_colour_med = max(np.median(team_colour_dists), 1.0)
            # Score ranks players by how spatially extreme AND isolated they are
            team_spatial = (x_devs[team_indices] / x_mad +
                            top_y_devs[team_indices] / y_mad) * nn_dists[team_indices]
            best = int(np.argmax(team_spatial))
            best_idx = team_indices[best]
            best_combined = team_spatial[best]
            best_nn = nn_dists[best_idx]
            best_colour_ratio = team_colour_dists[best] / team_colour_med
            if (best_combined > 250 and best_nn > 100 and best_colour_ratio >= 1.5
                    and not clearly_one_team[best_idx]):
                final_labels[best_idx] = 2
                print(f"  GK (spatial) T{team_id}: combined={best_combined:.0f} "
                      f"nn={best_nn:.0f}px colour_ratio={best_colour_ratio:.2f}")

    # FIX 5: Officials mini-cluster detection. When 3–5 officials/GKs share a
    # similar low-sat colour profile they mutually validate each other under the
    # FIX 1 isolation check. A k=3 K-Means on chromaticity reveals the third
    # mini-cluster; if it is small, tight, chromatically distinct from both
    # teams, AND less saturated than both, its members are reclassified OTH.
    if len(filtered_colours) >= 10:
        try:
            ab_scaled_k3 = StandardScaler().fit_transform(filtered_colours[:, [1, 2]])
            km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(ab_scaled_k3)
            k3_labels = km3.labels_
            counts_k3 = [int(np.sum(k3_labels == c)) for c in range(3)]
            smallest = int(np.argmin(counts_k3))
            s_size = counts_k3[smallest]
            other_sizes = [counts_k3[c] for c in range(3) if c != smallest]

            if 3 <= s_size <= 5 and min(other_sizes) >= 4:
                cen3 = np.array([filtered_colours[k3_labels == c].mean(axis=0)
                                 for c in range(3)])
                s_cen = cen3[smallest]
                o_cens = [cen3[c] for c in range(3) if c != smallest]
                d_to_others = [np.hypot(s_cen[1]-o[1], s_cen[2]-o[2]) for o in o_cens]
                s_members = filtered_colours[k3_labels == smallest]
                s_ab_std = float(np.std(s_members[:, 1]) + np.std(s_members[:, 2]))
                s_sat = float(np.median(s_members[:, 5]))
                o_sats = [float(np.median(filtered_colours[k3_labels == c, 5]))
                          for c in range(3) if c != smallest]

                if min(d_to_others) > 15 and s_ab_std < 16 and s_sat < min(o_sats):
                    s_indices = np.where(k3_labels == smallest)[0]
                    print(f"  Officials mini-cluster: size={s_size} ab_std={s_ab_std:.1f} "
                          f"sat={s_sat:.1f} vs {[f'{x:.1f}' for x in o_sats]} "
                          f"chroma_dist_min={min(d_to_others):.1f}")
                    for idx in s_indices:
                        final_labels[idx] = 2
        except Exception as e:
            print(f"  Officials mini-cluster check skipped ({e})")

    # Pass 4: Hard cap — safety net; raised to 6 to allow referee + 2 GKs + linesmen
    MAX_OTH_PITCH = 6
    oth_pitch_indices = [i for i, l in enumerate(final_labels) if l == 2]
    if len(oth_pitch_indices) > MAX_OTH_PITCH:
        print(f"  OTH cap triggered: {len(oth_pitch_indices)} detected, capping at {MAX_OTH_PITCH}")
        oth_by_dist = sorted(oth_pitch_indices, key=lambda i: d_nearest[i], reverse=True)
        to_reassign = oth_by_dist[MAX_OTH_PITCH:]
        for idx in to_reassign:
            final_labels[idx] = 0 if dist_to_c0[idx] < dist_to_c1[idx] else 1
        print(f"  Reassigned {len(to_reassign)} players to nearest team centroid")

    # Step 4 — Rebuild full labels including pre-flagged
    all_labels = []
    all_boxes = []
    filtered_ptr = 0

    for i in range(len(colours)):
        if i in pre_flagged:
            all_labels.append(2)
            all_boxes.append(valid_boxes[i])
        else:
            all_labels.append(final_labels[filtered_ptr])
            all_boxes.append(filtered_boxes[filtered_ptr])
            filtered_ptr += 1

    oth_count = sum(1 for l in all_labels if l == 2)
    print(f"  Officials/GK detected: {oth_count}")

    return np.array(all_labels), all_boxes

def draw_teams(img, labels, boxes):
    display_colours = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 255),
    ]
    result = img.copy()
    counts = {0: 0, 1: 0, 2: 0}

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        colour = display_colours[label]
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)
        label_text = "OTH" if label == 2 else f"T{label}"
        cv2.putText(result, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        counts[label] += 1

    print(f"  Team 0: {counts[0]} players")
    print(f"  Team 1: {counts[1]} players")
    print(f"  Officials/GK: {counts[2]} players")
    return result

# Output JSON for player detections
detections = {}

# Loop through all keyframes
keyframes = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])

for test_kf in keyframes:
    print(f"\nTesting on: {test_kf}")
    img_path = os.path.join(KEYFRAMES_DIR, test_kf)
    img = cv2.imread(img_path)
    height = img.shape[0]
    frame_height = img.shape[0]
    frame_width = img.shape[1]

    grass_colour = get_grass_colour(img)

    # Probe run to determine adaptive thresholds
    results_probe = model(img, classes=[0], conf=0.3, iou=0.7, verbose=False)
    probe_boxes = results_probe[0].boxes
    probe_heights = [
        (b.xyxy[0][3] - b.xyxy[0][1]).item()
        for b in probe_boxes if int(b.cls[0]) == 0
    ]

    if len(probe_heights) > 3:
        median_h = sorted(probe_heights)[len(probe_heights) // 2]
        if median_h < 45:
            min_height = 20
            top_zone = 0.65
        else:
            min_height = 40
            top_zone = 0.45
    else:
        min_height = 25
        top_zone = 0.55

    print(f"  Adaptive threshold: min_h={min_height}, top_zone={top_zone}")

    # First pass using adaptive threshold
    player_boxes = [
        b.xyxy[0].tolist() for b in probe_boxes
        if int(b.cls[0]) == 0
        and (b.xyxy[0][3] - b.xyxy[0][1]) > min_height
    ]

    # Second pass - low confidence for upper portion
    results_low = model(img, classes=[0], conf=0.05, iou=0.7, verbose=False)
    boxes_low = results_low[0].boxes
    for b in boxes_low:
        if int(b.cls[0]) == 0:
            box = b.xyxy[0].tolist()
            x1, y1, x2, y2 = box
            if y2 < height * top_zone and (y2 - y1) > 20:
                duplicate = False
                for existing in player_boxes:
                    ex1, ey1, ex2, ey2 = existing
                    overlap_x = max(0, min(x2, ex2) - max(x1, ex1))
                    overlap_y = max(0, min(y2, ey2) - max(y1, ey1))
                    if overlap_x > 20 and overlap_y > 20:
                        duplicate = True
                        break
                if not duplicate:
                    player_boxes.append(box)

    # Gap-based dynamic bottom cutoff
    all_centres_y = sorted([(b[1]+b[3])/2 for b in player_boxes])
    dynamic_bottom = frame_height * 0.88
    if len(all_centres_y) > 4:
        gaps = [(all_centres_y[i+1] - all_centres_y[i], i)
                for i in range(len(all_centres_y)-1)]
        max_gap, max_gap_idx = max(gaps)
        bottom_count = len(all_centres_y) - (max_gap_idx + 1)
        bottom_detections = all_centres_y[max_gap_idx + 1:]
        all_near_bottom = all(cy > frame_height * 0.80 for cy in bottom_detections)
        if max_gap > 50 and bottom_count <= 3 and all_near_bottom:
            dynamic_bottom = (all_centres_y[max_gap_idx] + all_centres_y[max_gap_idx+1]) / 2

    # FIX 4: Extended touchline (50→55) + corner filter. Catches ballboys/staff
    # sitting in the upper or lower image corners outside the playable pitch
    # area but within YOLO's detection zone.
    non_player_boxes = []
    pitch_boxes = []
    for box in player_boxes:
        x1, y1, x2, y2 = box
        centre_y = (y1 + y2) / 2
        centre_x = (x1 + x2) / 2
        box_h = y2 - y1
        on_touchline = (centre_x < 80 and box_h < 55) or \
                       (centre_x > frame_width - 80 and box_h < 55)
        in_bottom_zone = centre_y > dynamic_bottom and box_h < 70
        near_top_hoarding = y1 < frame_height * 0.12 and box_h < 22
        in_corner_x = (centre_x < frame_width * 0.18) or (centre_x > frame_width * 0.82)
        in_corner_y = (centre_y < frame_height * 0.25) or (centre_y > frame_height * 0.85)
        in_corner = in_corner_x and in_corner_y and box_h < 55
        if on_touchline or in_bottom_zone or near_top_hoarding or in_corner:
            non_player_boxes.append(box)
        else:
            pitch_boxes.append(box)

    print(f"  Dynamic bottom cutoff: {dynamic_bottom:.0f}px")
    print(f"  Touchline/bottom officials removed: {len(non_player_boxes)}")
    player_boxes = pitch_boxes

    print(f"  Players detected: {len(player_boxes)}")

    labels, valid_boxes = identify_teams(img, player_boxes, grass_colour)
    if len(labels) > 0:
        all_labels = list(labels) + [2] * len(non_player_boxes)
        all_boxes = list(valid_boxes) + non_player_boxes
        output_img = draw_teams(img, np.array(all_labels), all_boxes)
        cv2.imwrite(os.path.join(OUTPUT_DIR, test_kf), output_img)

        # Save player data to JSON
        players_data = []
        for box, label in zip(all_boxes, all_labels):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            players_data.append({
                "box": [x1, y1, x2, y2],
                "team": int(label),
                "foot_px": [cx, y2]
            })
        detections[test_kf] = {"players": players_data}
        print(f"  Saved")
    else:
        print("  Skipped - insufficient players")

# Save all detections to JSON
with open(DETECTIONS_FILE, "w") as f:
    json.dump(detections, f, indent=2)
print(f"\nPlayer detections saved to {DETECTIONS_FILE}")
print("\nDone!")