# This file runs both player detection and team assignments. It runs YOLO player detections on each
# keyframe and assigns every detected player to either Team 0 or Team 1 or OTH through K-Means clustering in Lab
# colour space. It saves bounding boxes team labels and foot positions to data/player_detections.json for 
# use later on in pick_points.py and offside_checker.py
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
# When True, draws P{idx} above each box on the output image and writes a
# per-player feature dump (Lab, saturation, distances) to DEBUG_FILE. The
# evaluation script eval/test_chroma_weight.py reads that dump. Leave False
# for production runs.
DEBUG_LABELS = False
DEBUG_FILE = "data/team_identification_debug.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#loads model weights into memory
#segmentation model x, produces pixel level mask showing exactly which pixels belong to that person
model = YOLO("yolo11x-seg.pt")

#converts image to hsv colour space, finds all pixels that fall in the green hue range
#averages them, to get typical grass colour then coverts that colour to lab
#this colour is then removed from identification since a large portion of players bounding box is actually the grass and not the kit
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

#takes one players box and returns a 6 number feature vector 
def get_player_colour(img, box, grass_colour, mask=None):
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    width = x2 - x1

    # reject detections too small to give reliable colour sample
    if height < 20 or width < 10:
        return None

    
    #skip the top 20% of the box (head/hair/face — not kit)
    #min of 5px so very small boxes still skip something
    head_skip = max(5, int(height * 0.20))
    crop_y1 = y1 + head_skip
    #take next 40% of the box height after the head skip 
    #min(height, ...) stops the crop going below the bottom of the box.
    #max(8, ...) guarantees at least 8px of torso even on tiny detections.
    crop_y2 = y1 + min(height, head_skip + max(8, int(height * 0.40)))

    #skip outer 18% on each side
    arm_skip = int(width * 0.18)
    crop_x1 = x1 + arm_skip
    crop_x2 = x2 - arm_skip
    #if the box is so narrow that skipping arms leaves < 5px, use the full width.
    if crop_x2 - crop_x1 < 5:
        crop_x1, crop_x2 = x1, x2

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None
    
    #sampling only the pixels INSIDE the YOLO segmentation mask so that
    #background pitch, advertising boards, and overlapping players are excluded.
    #fall back to the plain rectangular crop when the mask contributes < 4 pixels
    #inside the torso ROI (e.g. small distant players where the mask doesn't reach).
    if mask is not None:
        #slice the mask to the same torso region as the crop.
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        #convert torso crop to Lab before masking
        crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        #keep only pixels where the mask is non-zero
        pixels = crop_lab[mask_crop > 0].astype(np.float32)
        if len(pixels) < 4:
            #mask gave too few pixels, fall back to rectangular one
            #resize to a fixed 30×20 thumbnail so sample size is consistent
            #regardless of how large the original box was.
            crop_resized = cv2.resize(crop, (30, 20))
            crop_lab_fb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2Lab)
            pixels = crop_lab_fb.reshape(-1, 3).astype(np.float32)
    else:
        #no mask available at all — use the 30×20 rectangular fallback directly
        crop_resized = cv2.resize(crop, (30, 20))
        crop_lab_fb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2Lab)
        pixels = crop_lab_fb.reshape(-1, 3).astype(np.float32)

    #bail out if we still have fewer than 4 pixels i.e not enough to compute a reliable colour.
    if len(pixels) < 4:
        return None

    #removes the grass pixels
    grass_dists = np.linalg.norm(pixels - grass_colour, axis=1)
    non_grass = pixels[grass_dists > 25]
    if len(non_grass) < 4:
        non_grass = pixels

    #build the 6 dimensional feature vector
    L = non_grass[:, 0]  #brightness
    a = non_grass[:, 1]  #red green axis
    b = non_grass[:, 2]  #blue yellow axis

    dark_ratio = np.mean(L < 80)   #fraction of dark pixels
    bright_ratio = np.mean(L > 150)#fraction of very bright pixels
    median_L = np.median(L)
    median_a = np.median(a)
    median_b = np.median(b)
    #Lab values from OpenCV shifted, pure grey is (128,128) in a/b space ant not (0,0)
    sat_proxy = np.median(np.sqrt((a - 128)**2 + (b - 128)**2))

    # scale dark_ratio down for small boxes, tiny distant player with one dark pixel
    # would otherwise get the same dark score as a close player fully dressed in black
    size_factor = min(1.0, (height * width) / 8000)
    scaled_dark = dark_ratio * 100 * size_factor

    return np.array([median_L, median_a, median_b, scaled_dark, bright_ratio * 100, sat_proxy])

#where team classification happens
def identify_teams(img, boxes, masks, grass_colour, debug=False):
    colours = []
    valid_boxes = []

    #call get_player_colour for every detected player
    #player returning None are dropped 
    #any player who passes gets added to colours_array
    for box, mask in zip(boxes, masks):
        colour = get_player_colour(img, box, grass_colour, mask)
        if colour is not None:
            colours.append(colour)
            valid_boxes.append(box)

    #minimum player check, K-means needs at least 4 players to split into 2 teams
    if len(colours) < 4:
        print("  Not enough players for classification")
        return np.array([]), [], {"centres": None, "players": []}

    colours_array = np.array(colours) #convert plain python list of 6 element vectors into a 2D NumPy array

    #pre-flag setup and dark team check
    pre_flagged = set() #set of player indices that will be excluded from K-Means
    pre_flag_reasons = {}
    
    #before flagging anyone as dark = referee you first check if a whole team is wearing a dark kit
    #first threshold used is a linient one (L<95) which catches dark kit teams 
    dark_team_count = sum(1 for c in colours_array if c[0] < 95 and c[5] < 15)
    #second threshold (L<80) stricter catches cases where only 3 clearly dark players are visible
    strict_dark_count = sum(1 for c in colours_array if c[0] < 80 and c[5] < 15)
    dark_team_present = dark_team_count >= 4 or strict_dark_count >= 3

    #if either condition is met, skip the dark pre flagging to avoid misidentifying real players
    if dark_team_present:
        print(f"  Dark-team detected (L<95:{dark_team_count}, L<80:{strict_dark_count}) — "
              f"skipping dark pre-flags")

    # Pre-compute chroma neighbour counts — used to protect players that are dark
    # due to shadow but chromatically match a kit colour (fix for very_dark false positives)
    chroma_neighbour_count = [
        sum(1 for j, cj in enumerate(colours_array)
            if j != i and np.hypot(cj[1]-c[1], cj[2]-c[2]) < 15)
        for i, c in enumerate(colours_array)
    ]

    #dark and achromatic pre flags
    for i, colour in enumerate(colours_array):
        L, sat = colour[0], colour[5]
        #rule 1 -kit nearly black, almost certainly a referee
        # Guard: if this player's hue matches ≥2 others they're in shadow, not a ref
        if L < 50 and not dark_team_present:
            if chroma_neighbour_count[i] >= 2:
                print(f"  Skipping very_dark pre-flag player {i+1}: "
                      f"chroma-consistent with {chroma_neighbour_count[i]} others (L={L:.1f})")
            else:
                pre_flagged.add(i)
                pre_flag_reasons[i] = "very_dark"
                print(f"  Pre-flagged player {i+1}: very dark (L={L:.1f})")
        #rule 2 - L<100 and sat <13, together means dark and colourless, referee pattern
        elif L < 100 and sat < 13 and not dark_team_present:
            pre_flagged.add(i)
            pre_flag_reasons[i] = "dark_achromatic_ref"
            print(f"  Pre-flagged player {i+1}: dark+achromatic referee (L={L:.1f} sat={sat:.1f})")
    #both rules skipped when a dark team is present 

    #for each non flagged player, count how many other non flagged players have similar colour
    #by similar meaning within a radius 18 in a,b colour space
    if len(colours_array) >= 8:
        for i, c in enumerate(colours_array):
            if i in pre_flagged:
                continue
            similar = 0
            for j, cj in enumerate(colours_array):
                if j == i or j in pre_flagged:
                    continue
                if np.hypot(cj[1]-c[1], cj[2]-c[2]) < 18:
                    similar += 1
            #if fewer than 2 others share colour, you're wearing a unique kit
            if similar < 2:
                pre_flagged.add(i)
                pre_flag_reasons[i] = "colour_isolated"
                print(f"  Pre-flagged player {i+1}: colour-isolated "
                      f"(only {similar} chroma-similar; L={c[0]:.0f} a={c[1]:.0f} b={c[2]:.0f} sat={c[5]:.1f})")

    #mini team rescue
    #isolation check has a flaw, if a team only has 3 players visible , all 3 get 
    #flagged as isolated because none has 2 colour similar neighbouts

    #fix is to check if isolated players are similar to each other 
    isolated_idx = [i for i, r in pre_flag_reasons.items() if r == "colour_isolated"]
    if len(isolated_idx) >= 3:
        adj = {k: set() for k in range(len(isolated_idx))}
        for k1 in range(len(isolated_idx)):
            for k2 in range(k1 + 1, len(isolated_idx)):
                c1 = colours_array[isolated_idx[k1]]
                c2 = colours_array[isolated_idx[k2]]
                #two nodes connected if colours are within radius 22 of each other
                if np.hypot(c1[1] - c2[1], c1[2] - c2[2]) <= 22:
                    adj[k1].add(k2); adj[k2].add(k1)
        visited = set()
        #depth first search finds connected groups
        for start in range(len(isolated_idx)):
            if start in visited:
                continue
            comp = set()
            stack = [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u); visited.add(u)
                stack.extend(adj[u] - comp)
            #any group of 3+ players gets unflagged because they form a real team and not scattered officials
            if len(comp) >= 3:
                for u in comp:
                    i = isolated_idx[u]
                    pre_flagged.discard(i)
                    pre_flag_reasons.pop(i, None)
                    print(f"  Un-flagged player {i+1}: mini-team chroma cluster "
                          f"({len(comp)} members)")

    #build a filtered version of the player list with pre flagged players removed
    filtered_indices = [i for i in range(len(colours)) if i not in pre_flagged]
    filtered_colours = colours_array[filtered_indices]
    filtered_boxes = [valid_boxes[i] for i in filtered_indices]

    #bail out again if fewer than 4 remain
    if len(filtered_colours) < 4:
        print("  Not enough players after referee removal")
        return np.array([]), [], {"centres": None, "players": []}

    #normalise features before K-Means so all dimensions contribute equally
    #standard scaler rescales all 6 features to mean=0, std=1, without this features 
    #with bigger raw ranges would odminate features with smaller ranges
    scaler = StandardScaler()
    filtered_colours_scaled = scaler.fit_transform(filtered_colours)

    #after scaling, a and b colour channels are multiplied by 1.8 to give kit colour more
    #influence than brightness when K-Means runs
    filtered_colours_scaled[:, 1] *= 1.8
    filtered_colours_scaled[:, 2] *= 1.8

    # run K-Means multiple times on normalised features, pick most balanced split
    best_labels = None
    best_balance = -1

    # K-means ran 5 times with different random seeds
    #each run produces split,balance measures how even it is
    #perfect 50/50 gives a 1
    #10v2 split gives a 0.2
    for seed in [42, 0, 7, 13, 99]:
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        lbl = km.fit_predict(filtered_colours_scaled)
        c0 = np.sum(lbl == 0)
        c1 = np.sum(lbl == 1)
        balance = min(c0, c1) / max(c0, c1)
        #best balance is kept
        if balance > best_balance:
            best_balance = balance
            best_labels = lbl

    team_labels = list(best_labels)

    #recompute centres in original (unscaled) space for L-axis correction
    team_labels_arr = np.array(team_labels)
    centres = np.array([
        np.mean(filtered_colours[team_labels_arr == 0], axis=0),
        np.mean(filtered_colours[team_labels_arr == 1], axis=0)
    ])

    #L-axis post-correction 
    L_centre0 = centres[0][0]
    L_centre1 = centres[1][0]
    L_separation = abs(L_centre0 - L_centre1)

    #if two team centroids differ by more than 30 in brightness, cheack each individual player
    if L_separation > 30:
        print(f"  L-axis correction active (separation={L_separation:.0f})")
        for i, label in enumerate(team_labels):
            player_L = filtered_colours[i][0]
            own_L = L_centre0 if label == 0 else L_centre1
            other_L = L_centre1 if label == 0 else L_centre0
            dist_to_own_L = abs(player_L - own_L)
            dist_to_other_L = abs(player_L - other_L)
            
            #if a players brightness is much closer to other teams average
            #flip their assignment
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

    #bimodal chroma refit
    #classic failure case , white vs yellow kits for example 
    #both bright, K-means splits by brightness variation instead of kit colour 
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

    # A cluster of fewer than 4 players is suspicious. But only refit if the
    # majority cluster has HIGH internal colour variance — that indicates it
    # contains two merged teams. If the majority is colour-uniform, the small
    # cluster is a genuine small team and the original split should be kept.
    min_team_size = 4
    if not refit_done and (team0_count < min_team_size or team1_count < min_team_size):
        minority = 0 if team0_count < team1_count else 1
        majority_indices = [i for i, l in enumerate(team_labels) if l != minority]
        minority_indices = [i for i, l in enumerate(team_labels) if l == minority]
        majority_colours = filtered_colours[majority_indices]

        # Mean std across L, a, b of the majority cluster.
        # One uniform team gives ~5-12; two merged teams give ~14+.
        majority_lab_std = np.std(majority_colours[:, :3], axis=0).mean()

        minority_count = min(team0_count, team1_count)
        if (majority_lab_std > 13 or minority_count == 1) and len(majority_colours) >= 6:
            print(f"  Degenerate split detected ({team0_count} vs {team1_count}), "
                  f"majority variance={majority_lab_std:.1f} → refitting...")
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

            # If a minority player is farther from both new centroids than
            # 1.2× the centroid separation, their colour doesn't match either
            # team → OTH. Otherwise assign to nearest centroid.
            centroid_sep = np.linalg.norm(centres[0] - centres[1])
            oth_threshold = max(centroid_sep * 1.2, 30.0)
            for idx in minority_indices:
                d0 = np.linalg.norm(filtered_colours[idx] - centres[0])
                d1 = np.linalg.norm(filtered_colours[idx] - centres[1])
                if min(d0, d1) > oth_threshold:
                    new_team_labels[idx] = 2
                    print(f"    Minority P{idx}: d={min(d0,d1):.1f} > {oth_threshold:.1f} → OTH")
                else:
                    new_team_labels[idx] = 0 if d0 < d1 else 1
            team_labels = new_team_labels

            team0_count = np.sum(team_labels == 0)
            team1_count = np.sum(team_labels == 1)
            print(f"  After refit — Team 0: {team0_count}, Team 1: {team1_count}")
        else:
            print(f"  Degenerate split ({team0_count} vs {team1_count}) but majority "
                  f"is colour-uniform (std={majority_lab_std:.1f}) and minority>1 — keeping original split")

    # Outlier detection using robust MAD-based thresholds
    final_labels = team_labels.copy().tolist()

    # compute distance from each player to both team centroids
    dist_to_c0 = np.linalg.norm(filtered_colours - centres[0], axis=1)
    dist_to_c1 = np.linalg.norm(filtered_colours - centres[1], axis=1)
    d_nearest = np.minimum(dist_to_c0, dist_to_c1)
    d_farther = np.maximum(dist_to_c0, dist_to_c1)

    # chromatic distance (a, b only) — used to rescue players whose 6D distance
    # is inflated by L/sat extremes (sun on jersey, broadcast graphics) but who
    # are clearly chromatically consistent with their team kit.
    centres_ab = centres[:, [1, 2]]
    filt_ab = filtered_colours[:, [1, 2]]
    ab_to_c0 = np.linalg.norm(filt_ab - centres_ab[0], axis=1)
    ab_to_c1 = np.linalg.norm(filt_ab - centres_ab[1], axis=1)
    ab_nearest = np.minimum(ab_to_c0, ab_to_c1)

    # per-team intra-team median distance — how tight is each team's colour cluster?
    intra_med = np.array([20.0, 20.0])
    for team_id in (0, 1):
        t_mask = np.array([l == team_id for l in final_labels])
        if np.sum(t_mask) >= 2:
            t_dists = np.linalg.norm(filtered_colours[t_mask] - centres[team_id], axis=1)
            intra_med[team_id] = max(float(np.median(t_dists)), 5.0)
    nearer_team = (dist_to_c1 < dist_to_c0).astype(int)
    nearer_med = intra_med[nearer_team]

    # unified veto: a player is "clearly one team" only if they are BOTH ratio-close
    # to one team (team_ratio < 0.6) AND absolutely close to it (d_nearest within
    # 2.5x that team's intra-team spread). GKs/refs wearing kits different from both
    # outfield teams end up far from both in absolute terms — even if they happen to
    # be slightly nearer one team, the absolute-distance check un-protects them.
    team_ratio = d_nearest / np.maximum(d_farther, 1e-6)
    clearly_one_team = (team_ratio < 0.6) & (d_nearest < 2.5 * nearer_med)

    #pass 1: Global outliers — players far from BOTH centroids
    team_01_indices = [i for i, l in enumerate(final_labels) if l in (0, 1)]
    if len(team_01_indices) >= 4:
        d_nearest_01 = d_nearest[team_01_indices]
        median_d = np.median(d_nearest_01)
        mad = max(1.4826 * np.median(np.abs(d_nearest_01 - median_d)), 8.0)
        global_threshold = median_d + 3.5 * mad
        print(f"  Global outlier threshold: {global_threshold:.1f} (median={median_d:.1f} MAD={mad:.1f})")

        for i in team_01_indices:
            if d_nearest[i] > global_threshold and not clearly_one_team[i]:
                if ab_nearest[i] > 15:
                    final_labels[i] = 2
                else:
                    # 6D-far but chromatically consistent with one team — likely
                    # a player with extreme L/sat (sun, broadcast graphic) put
                    # in the wrong K-means bucket. Reassign by chroma instead
                    # of dropping to OTH.
                    new_label = 0 if ab_to_c0[i] < ab_to_c1[i] else 1
                    if new_label != final_labels[i]:
                        final_labels[i] = new_label

    #pass 2: Per-team outliers — tighter thresholds to catch GKs with moderately different kits
    #MAD is floored at 5 and capped at 20 to prevent inflation from high-variance teams.
    #Ratio guard (dist > 1.3 * threshold) avoids flagging players barely over the limit;
    #genuine GK outliers are typically 1.5-2x beyond their team's normal spread.
    for team_id in [0, 1]:
        team_indices = [i for i, l in enumerate(final_labels) if l == team_id]
        if len(team_indices) < 3:
            continue

        team_dists = np.linalg.norm(filtered_colours[team_indices] - centres[team_id], axis=1)
        team_median = np.median(team_dists)
        team_mad_raw = 1.4826 * np.median(np.abs(team_dists - team_median))
        team_mad = max(min(team_mad_raw, 20.0), 5.0)  # floor=5, cap=20
        team_threshold = team_median + 3.0 * team_mad

        for idx, dist in zip(team_indices, team_dists):
            if (dist > 1.3 * team_threshold and ab_nearest[idx] > 15
                    and not clearly_one_team[idx]):
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
            if len(team_indices) < 3:
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

    # Officials mini-cluster detection. When 3–5 officials/GKs share a similar
    # low-sat colour profile they mutually validate each other under the per-
    # player isolation check earlier in this function. A k=3 K-Means on
    # chromaticity reveals the third mini-cluster; if it is small, tight,
    # chromatically distinct from both teams, AND less saturated than both,
    # its members are reclassified OTH.
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

                # Officials must be DISTINCTLY less saturated than the closest
                # team — a tiny gap (e.g. 32.8 vs 33) was previously enough to
                # flag a sub-cluster of a real team as officials.
                if min(d_to_others) > 15 and s_ab_std < 16 and (min(o_sats) - s_sat) >= 8:
                    s_indices = np.where(k3_labels == smallest)[0]
                    print(f"  Officials mini-cluster: size={s_size} ab_std={s_ab_std:.1f} "
                          f"sat={s_sat:.1f} vs {[f'{x:.1f}' for x in o_sats]} "
                          f"chroma_dist_min={min(d_to_others):.1f}")
                    for idx in s_indices:
                        final_labels[idx] = 2
        except Exception as e:
            print(f"  Officials mini-cluster check skipped ({e})")

    # Per-team chroma outlier: a player whose (a, b) is markedly far from the
    # mean (a, b) of their assigned team is wearing a kit different from the
    # team — almost always a goalkeeper. Pass 2's 6D outlier check misses them
    # because GKs often sit near their team in L space (clearly_one_team holds).
    CHROMA_OUTLIER_THRESHOLD = 24
    for team_id in [0, 1]:
        team_idx_in_filt = [k for k, l in enumerate(final_labels) if l == team_id]
        if len(team_idx_in_filt) < 3:
            continue
        team_ab = filtered_colours[team_idx_in_filt][:, [1, 2]]
        team_ab_mean = team_ab.mean(axis=0)
        for k in team_idx_in_filt:
            cd = float(np.linalg.norm(filtered_colours[k, [1, 2]] - team_ab_mean))
            if cd > CHROMA_OUTLIER_THRESHOLD:
                print(f"  Chroma-outlier P{k+1} (T{team_id}) → OTH (cd={cd:.1f})")
                final_labels[k] = 2

    # reassemble full player list, re inster preflagged players alongside filtered playuers
    # whose labels were determined by K-means pipeline above
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

    # Per-player feature dump for the debug overlay / eval scripts. Skipped
    # entirely when debug=False so production runs don't pay for it.
    debug_info = None
    if debug:
        filtered_idx_for = {orig: pos for pos, orig in enumerate(filtered_indices)}
        players_dbg = []
        for i in range(len(colours)):
            c = colours[i]
            entry = {
                "box": [int(v) for v in valid_boxes[i]],
                "label": int(all_labels[i]),
                "lab": [float(c[0]), float(c[1]), float(c[2])],
                "dark_score": float(c[3]),
                "bright_pct": float(c[4]),
                "sat": float(c[5]),
            }
            if i in pre_flag_reasons:
                entry["pre_flag"] = pre_flag_reasons[i]
                entry["dist_to_c0"] = None
                entry["dist_to_c1"] = None
            else:
                fi = filtered_idx_for[i]
                entry["pre_flag"] = None
                entry["dist_to_c0"] = float(dist_to_c0[fi])
                entry["dist_to_c1"] = float(dist_to_c1[fi])
            players_dbg.append(entry)
        debug_info = {
            "centres": [[float(v) for v in centres[0]], [float(v) for v in centres[1]]],
            "players": players_dbg,
        }
    return np.array(all_labels), all_boxes, debug_info

#take image and label+box lists from identify teams and draw them on a copy of the image 
def draw_teams(img, labels, boxes, draw_indices=False):
    display_colours = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 255),
    ]
    result = img.copy()
    counts = {0: 0, 1: 0, 2: 0}

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        colour = display_colours[label]
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)
        label_text = "OTH" if label == 2 else f"T{label}"
        if draw_indices:
            cv2.putText(result, f"P{i}", (x1, y1 - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        cv2.putText(result, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        counts[label] += 1

    print(f"  Team 0: {counts[0]} players")
    print(f"  Team 1: {counts[1]} players")
    print(f"  Officials/GK: {counts[2]} players")
    return result


def get_box_mask_pairs(results, target_shape):
    """For class-0 detections in a YOLO-seg result, return list of
    (xyxy_list, mask_2d_uint8 or None) — one entry per detection.
    Mask is resized to target_shape (H, W) of the original image if needed."""
    boxes_obj = results[0].boxes
    masks_obj = results[0].masks
    pairs = []
    if boxes_obj is None or len(boxes_obj) == 0:
        return pairs

    masks_data = None
    if masks_obj is not None:
        masks_data = masks_obj.data.cpu().numpy()  # (N, H, W) float

    H_t, W_t = target_shape
    for i, b in enumerate(boxes_obj):
        if int(b.cls[0]) != 0:
            continue
        box = b.xyxy[0].tolist()
        mask = None
        if masks_data is not None and i < masks_data.shape[0]:
            m = masks_data[i].astype(np.uint8)
            if m.shape[0] != H_t or m.shape[1] != W_t:
                m = cv2.resize(m, (W_t, H_t), interpolation=cv2.INTER_NEAREST)
            mask = m
        pairs.append((box, mask))
    return pairs


# Load existing detections so re-runs only process new keyframes
detections = {}
debug_data = {}
if os.path.exists(DETECTIONS_FILE):
    with open(DETECTIONS_FILE) as f:
        detections = json.load(f)
if DEBUG_LABELS and os.path.exists(DEBUG_FILE):
    with open(DEBUG_FILE) as f:
        debug_data = json.load(f)

# Loop through all keyframes
keyframes = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])

for kf in keyframes:
    if kf in detections:
        print(f"  Skipping (already processed): {kf}")
        continue
    print(f"\nProcessing: {kf}")
    img_path = os.path.join(KEYFRAMES_DIR, kf)
    img = cv2.imread(img_path)
    frame_height = img.shape[0]
    frame_width = img.shape[1]

    grass_colour = get_grass_colour(img)

    # first YOLO pass at conf = 0.3 to measure typical player box height in this frame
    # players appear smaller when they are on top of the pitch and camera is zoomed out so minimum detectoin
    # height and upper zone boundaries are set adaptively based on the median box height
    results_probe = model(img, classes=[0], conf=0.3, iou=0.7,
                          retina_masks=True, verbose=False)
    probe_pairs = get_box_mask_pairs(results_probe, img.shape[:2])
    probe_heights = [b[3] - b[1] for b, _ in probe_pairs]

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
    player_pairs = [(b, m) for b, m in probe_pairs if (b[3] - b[1]) > min_height]

    # second pass at conf =0.05 to catch distant players at far end of pitch
    # restricted to upper portion of frame to avoid flooding with false positives near camera 
    results_low = model(img, classes=[0], conf=0.05, iou=0.7,
                        retina_masks=True, verbose=False)
    low_pairs = get_box_mask_pairs(results_low, img.shape[:2])
    for box, mask in low_pairs:
        x1, y1, x2, y2 = box
        if y2 < frame_height * top_zone and (y2 - y1) > 20:
            duplicate = False
            for ex_box, _ in player_pairs:
                ex1, ey1, ex2, ey2 = ex_box
                overlap_x = max(0, min(x2, ex2) - max(x1, ex1))
                overlap_y = max(0, min(y2, ey2) - max(y1, ey1))
                intersection = overlap_x * overlap_y
                area_new = (x2 - x1) * (y2 - y1)
                area_ex  = (ex2 - ex1) * (ey2 - ey1)
                iou = intersection / (area_new + area_ex - intersection + 1e-6)
                if iou > 0.4:
                    duplicate = True
                    break
            # prevents adding duplicate boxes
            if not duplicate:
                player_pairs.append((box, mask))

    # detect a large vertical gap between main player cluster and small group at the bottom edge
    # usually advertising board personnel or cameramen, if gap is big enough and bottom group
    # are 3 or fewer detections, cutoff is moved up to exclude them
    all_centres_y = sorted([(b[1] + b[3]) / 2 for b, _ in player_pairs])
    dynamic_bottom = frame_height * 0.88
    if len(all_centres_y) > 4:
        gaps = [(all_centres_y[i+1] - all_centres_y[i], i)
                for i in range(len(all_centres_y) - 1)]
        max_gap, max_gap_idx = max(gaps)
        bottom_count = len(all_centres_y) - (max_gap_idx + 1)
        bottom_detections = all_centres_y[max_gap_idx + 1:]
        all_near_bottom = all(cy > frame_height * 0.80 for cy in bottom_detections)
        if max_gap > 50 and bottom_count <= 3 and all_near_bottom:
            dynamic_bottom = (all_centres_y[max_gap_idx] + all_centres_y[max_gap_idx+1]) / 2

    # touchline + corner filter, this catches ballboys, stewards and other staff
    # sitting in the image corners outside the playable pitch area but still
    # within YOLO's person-detection zone
    non_player_pairs = []
    pitch_pairs = []
    for box, mask in player_pairs:
        x1, y1, x2, y2 = box
        centre_y = (y1 + y2) / 2
        centre_x = (x1 + x2) / 2
        box_h = y2 - y1
        on_touchline = (centre_x < 80 and box_h < 55) or \
                       (centre_x > frame_width - 80 and box_h < 55)
        in_bottom_zone = centre_y > dynamic_bottom and box_h < 70
        # any detection whose feet (y2) are in the top 12% of the frame is
        # above the far-touchline hoarding — stewards, not outfield players.
        near_top_hoarding = y2 < frame_height * 0.12
        # people near the top of the frame AND within 20% of the left/right
        # edge are almost always near-goal stewards or photographers.
        near_goal_zone = (centre_x < frame_width * 0.20 or
                          centre_x > frame_width * 0.80) and \
                         centre_y < frame_height * 0.25 and box_h < 65
        # corner filter — only bottom corners. Upper corners are the FAR end
        # of the pitch on broadcast cameras, where real defenders legitimately
        # appear small (bh < 55) due to perspective.
        in_corner_x = (centre_x < frame_width * 0.18) or (centre_x > frame_width * 0.82)
        in_corner_y = centre_y > frame_height * 0.85
        in_corner = in_corner_x and in_corner_y and box_h < 55
        if on_touchline or in_bottom_zone or near_top_hoarding or near_goal_zone or in_corner:
            non_player_pairs.append((box, mask))
        else:
            pitch_pairs.append((box, mask))

    print(f"  Dynamic bottom cutoff: {dynamic_bottom:.0f}px")
    print(f"  Touchline/bottom officials removed: {len(non_player_pairs)}")

    # seperate the filtered detections into boxes and masks for identify_teams
    # keeping non_player_boxes aside so they can be appended as OTH afterwards
    pitch_boxes = [b for b, _ in pitch_pairs]
    pitch_masks = [m for _, m in pitch_pairs]
    non_player_boxes = [b for b, _ in non_player_pairs]

    print(f"  Players detected: {len(pitch_boxes)}")

    # run full team classification pipelien on the onpitch players only (after filtering out off pitch people)
    labels, valid_boxes, dbg = identify_teams(img, pitch_boxes, pitch_masks, grass_colour,
                                              debug=DEBUG_LABELS)
    if len(labels) > 0:
        # merge K-means results with touchline/corner detections already classified
        # as OTH - the final list covers every detection in the frame
        all_labels = list(labels) + [2] * len(non_player_boxes)
        all_boxes = list(valid_boxes) + non_player_boxes
        # draw coloured bounding boxes on a copy of the frame and save to disk 
        # so team assignments can be visually verified for debugging
        output_img = draw_teams(img, np.array(all_labels), all_boxes, draw_indices=DEBUG_LABELS)
        cv2.imwrite(os.path.join(OUTPUT_DIR, kf), output_img)

        # when debugging, add touchline/corner detections to per-player debug dump so eval scripts have a complete picture of the frmae
        if DEBUG_LABELS:
            for nb in non_player_boxes:
                x1, y1, x2, y2 = map(int, nb)
                dbg["players"].append({
                    "box": [x1, y1, x2, y2],
                    "label": 2,
                    "filtered_as": "touchline_or_corner",
                })
            debug_data[kf] = dbg

        # save player data to JSON
        players_data = []
        for box, label in zip(all_boxes, all_labels):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            players_data.append({
                "box": [x1, y1, x2, y2],
                "team": int(label),
                #foot position is bottom center of bounding box used as ground contact point when projecting to world coordinates later on
                "foot_px": [cx, y2]
            })
        detections[kf] = {"players": players_data}
        print(f"  Saved")
    else:
        print("  Skipped - insufficient players")

# save all detections to JSON at the end rather then per frame so a crash whilst running
# wouldnt corrupt output file
with open(DETECTIONS_FILE, "w") as f:
    json.dump(detections, f, indent=2)
print(f"\nPlayer detections saved to {DETECTIONS_FILE}")

if DEBUG_LABELS:
    with open(DEBUG_FILE, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"Debug data saved to {DEBUG_FILE}")

print("\nDone!")
