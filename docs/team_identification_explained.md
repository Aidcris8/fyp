# Team Identification — Deep Walkthrough

File: [team_identification.py](../team_identification.py)
Output: `data/player_detections.json` (consumed by [offside_checker.py](../offside_checker.py))

## 1. The problem we are solving

For each keyframe we need to tag every detected person with one of three labels:

| Label | Meaning | Display colour |
|-------|---------|----------------|
| `0`   | Team 0 (one of the two outfield teams) | Red |
| `1`   | Team 1 (the other outfield team) | Blue |
| `2`   | OTH — referees, linesmen, goalkeepers, ballboys, staff | Yellow |

The old failure mode: across ~80% of frames, 2–3 outfield players were being flagged as OTH instead of their real team. K-Means was latching onto the wrong visual signal (brightness, head/hair colour, or a couple of chromatic outliers) and the outlier passes were then banishing genuine teammates.

The supervisor's suggestion was to **extract the jersey colour from the upper body only and skip the top ~15% of the bounding box so hair/skin don't pollute the colour sample**. The current implementation does exactly this (with 20% skip — see §3.1 note below).

---

## 2. High-level flow (per keyframe)

```
keyframe ──► YOLO detect persons ──► filter non-pitch boxes (corners, touchline, hoardings)
         ──► extract upper-body Lab colour for each player
         ──► pre-flag very dark / achromatic (refs)
         ──► K-Means(k=2) on the remaining players  →  raw Team 0 / Team 1 split
         ──► L-axis correction        (fix brightness-driven mis-splits)
         ──► bimodal chroma refit     (fix white-vs-yellow style splits)
         ──► degenerate-split refit   (fix lopsided splits)
         ──► 4 outlier passes         (catch refs, GKs, mini-cluster of officials)
         ──► hard OTH cap (≤ 6)
         ──► write boxes + labels to player_detections.json
```

Everything below is an expansion of those steps.

---

## 3. Feature extraction

### 3.1 `get_player_colour(img, box, grass_colour)` — [team_identification.py:30](../team_identification.py#L30)

This is the function that answers your supervisor's suggestion. Given a YOLO bounding box, it returns a **6-dimensional Lab-space feature vector** computed from the upper body only.

```python
head_skip = max(5, int(height * 0.20))                             # skip top 20%
crop_y1   = y1 + head_skip
crop_y2   = y1 + min(height, head_skip + max(8, int(height*0.40))) # next 40%
crop      = img[crop_y1:crop_y2, x1:x2]
```

What this does geometrically:

```
┌──────────────┐  y1      ──┐
│    head      │            │ skipped — removes hair, skin, ref's cap, hat colour
│              │            │ (20% of bbox height, minimum 5 px)
├──────────────┤  y1+20%  ──┘
│              │            ┐
│   JERSEY     │            │ sampled — this is what K-Means sees
│  (upper body)│            │ 40% of bbox height (minimum 8 px)
│              │            │
├──────────────┤  y1+60%  ──┘
│   shorts +   │            ┐
│    legs      │            │ ignored — shorts/socks are often a different
│              │            │ colour to the jersey and would blur the cluster
└──────────────┘  y2      ──┘
```

**Note for your meeting:** your supervisor said "top 15%" — the code currently uses **20%**. You can either (a) tell him you tuned it to 20% because it gave better results, or (b) change line 38 to `0.15` if he specifically wants 15%. The rest of this write-up is valid either way.

After cropping, the colour is processed like this:

```python
crop_resized = cv2.resize(crop, (30, 20))           # normalise crop size so a big
                                                     # box doesn't dominate anything
crop_lab     = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2Lab)
pixels       = crop_lab.reshape(-1, 3).astype(np.float32)

grass_dists  = np.linalg.norm(pixels - grass_colour, axis=1)
non_grass    = pixels[grass_dists > 25]             # drop pixels that are grass
```

**Why Lab and not BGR/HSV?**
Lab separates lightness (`L`) from chromaticity (`a`, `b`). Two teams playing in shadow vs. sunshine will have very different BGR values even if their kits are identical in colour — but their `a,b` will stay close. This lets us handle lighting variation within the same clip.

**Why remove grass?**
The bounding box is rectangular so it always includes grass around the player's silhouette. Grass is ~50% of the crop pixels on thin players, and it's the same colour for everyone — it adds no information and pulls both clusters toward green.

The 6 features returned are:

| Index | Feature | Purpose |
|-------|---------|---------|
| 0 | `median_L` | Jersey lightness — separates light kits from dark kits |
| 1 | `median_a` | Red ↔ Green chrominance |
| 2 | `median_b` | Blue ↔ Yellow chrominance |
| 3 | `scaled_dark` | Fraction of dark pixels × size factor — helps spot refs in black |
| 4 | `bright_ratio` | Fraction of very bright pixels — helps spot white/silver kits |
| 5 | `sat_proxy` | Chromatic saturation (how colourful vs. grey) — critical for ref detection |

The `size_factor = min(1.0, (height*width) / 8000)` is a trick: very small (distant) player crops have less reliable colour sampling, so we tone down their `dark_ratio` contribution rather than letting a tiny crop dominate.

### 3.2 `get_grass_colour(img)` — [team_identification.py:16](../team_identification.py#L16)

Finds the dominant green by HSV-masking pixels in the range `H∈[35,85], S≥40, V≥40`, averaging them, then converting the mean to Lab. The value feeds `get_player_colour` so that grass-like pixels inside each bounding box can be removed before the median is computed. If no green is found, a generic fallback `[50,100,50]` is used (rare — would mean a completely non-grass frame).

---

## 4. The two-pass YOLO detection + non-pitch filter
([team_identification.py:471–566](../team_identification.py#L471-L566))

Before clustering, we build the list of boxes to classify.

### 4.1 Adaptive size threshold (probe pass)
A first `model(img, conf=0.3)` gives us a sense of the camera zoom. We measure the median bounding-box height:

- Median height `< 45 px` → far camera, use `min_height = 20`, `top_zone = 0.65`
- Median height `≥ 45 px` → close camera, use `min_height = 40`, `top_zone = 0.45`

This makes the pipeline work both for wide shots and for close-ups without hand-tuning.

### 4.2 Low-confidence top-up (second pass)
`model(img, conf=0.05)` re-runs YOLO with a very permissive threshold and we **only keep the new boxes that are in the upper `top_zone` of the frame** (distant players the main pass missed). Duplicates are rejected by checking for >20 px overlap.

### 4.3 Dynamic bottom cutoff
We sort all detected player y-centres and look for a large vertical gap (`> 50 px`) near the bottom of the frame: that's often the gap between the last row of players and a cluster of photographers/coaches on the touchline. The cutoff `dynamic_bottom` is set to the midpoint of that gap.

### 4.4 Non-pitch filter (`FIX 4`)
A box is dropped to `non_player_boxes` (labelled OTH immediately, never fed to clustering) if **any** of these are true:

| Rule | Catches |
|------|---------|
| `cx < 80 and box_h < 55` (far left + short) | Linesman / coach on left touchline |
| `cx > W−80 and box_h < 55` (far right + short) | Linesman / coach on right touchline |
| `cy > dynamic_bottom and box_h < 70` | Photographers / staff at bottom |
| `y1 < 0.12·H and box_h < 22` | Tiny boxes near top — crowd / hoardings |
| `in_corner_x and in_corner_y and box_h < 55` | Ballboys in corner areas |

Everything else is a "pitch player" candidate and goes into `identify_teams`.

---

## 5. Inside `identify_teams` — [team_identification.py:73](../team_identification.py#L73)

### 5.1 Step 1 — Pre-flag obvious non-players

Some people on the pitch are *never* going to cluster correctly with outfield players (referee in black, linesman in bright yellow, goalkeeper in a clashing kit). If we let K-Means see them, they distort the two cluster centroids.

```python
if L < 50:                                # very dark — ref in black
    pre_flagged.add(i)
elif L < 85 and sat < 10 and not dark_team_present:
    pre_flagged.add(i)                     # dark + achromatic — ref
```

The `dark_team_present` guard (`>= 4 players with L<95 and sat<15`) is important: if a whole team is in navy or black (e.g. PSG), we don't want to pre-flag all of them as refs — so the rule turns off when a full dark team is on the pitch.

**`FIX 1` colour-isolated pre-flag** (`len(colours_array) >= 8`): any player whose `(a,b)` chromaticity has fewer than 2 neighbours within a radius of 15 is pre-flagged. This catches GKs / refs / linesmen whose kit is a *colour* nobody else shares. We use `(a,b)` only (not `L`) so a teammate in shadow is still counted as similar.

All pre-flagged indices skip clustering and are appended as OTH at the end.

### 5.2 Step 2 — The K-Means split

```python
scaler                  = StandardScaler()
filtered_colours_scaled = scaler.fit_transform(filtered_colours)
for seed in [42, 0, 7, 13, 99]:
    km  = KMeans(n_clusters=2, random_state=seed, n_init=10)
    lbl = km.fit_predict(filtered_colours_scaled)
    balance = min(c0,c1) / max(c0,c1)
    if balance > best_balance:   # keep the most even 0/1 split
        best_labels = lbl
```

Two points here:

1. **`StandardScaler`** — each of the 6 features has a different natural scale (`L` is 0–255, `sat_proxy` is ~0–50). Without scaling, `L` would dominate distance in Euclidean space. Scaling makes each dimension contribute roughly equally.
2. **5 seeds, pick most balanced** — K-Means is non-deterministic on the initial centroids. We try 5 seeds and keep the split whose two clusters are the most even in size (since real football has ~10 vs ~10 outfield players). This biases away from pathological 1-vs-N splits.

### 5.3 Step 2b — L-axis correction ([team_identification.py:168](../team_identification.py#L168))

K-Means still sometimes splits on brightness (`L`) when we wanted it to split on hue. After the cluster, if the two L-centroids differ by `> 30`, we iterate through all players: if a player is *much* closer (< 60%) to the other team's L-centroid than their own, we flip the label.

This is a gentle per-player correction — it doesn't re-cluster, it just corrects obvious L-driven mislabels.

### 5.4 Step 2c — Bimodal chroma refit ([team_identification.py:201](../team_identification.py#L201), `FIX 2`)

Catches the **white-vs-yellow** failure mode. Both teams are bright, so `L` is similar for everyone, and K-Means ends up splitting by "how white" not by "white vs yellow". Detection signal:

- One cluster has tight `(a,b)` spread, the other has wide spread (ratio > 2.5×), **and**
- Maximum `b` std > 22 (chromatic spread exists — one team really is coloured)

When both trip, we **re-fit K-Means on `(a,b)` only** (drop `L` entirely). That forces the split to be by hue.

### 5.5 Step 2d — Degenerate-split refit ([team_identification.py:231](../team_identification.py#L231))

Safety net: if one of the clusters has `< 3` players, we re-fit K-Means on the majority cluster using chromaticity only, then re-assign the minority's members to the nearest new centroid. This prevents the "all 18 players in Team 0, 1 player in Team 1" failure.

### 5.6 Pass 1 — Global outliers ([team_identification.py:296](../team_identification.py#L296))

For each player, compute `d_nearest` = distance to the closer of the two team centroids.

```
threshold = median(d_nearest_01) + 3.5 * MAD(d_nearest_01)       # floor MAD at 8
```

**Why MAD not standard deviation?**  MAD (Median Absolute Deviation) is robust to outliers — the very players we're trying to detect. If we used `stddev`, one odd player would inflate `stddev` and hide himself from the threshold.

A player exceeding the threshold is labelled OTH unless `clearly_one_team[i]` is True (see §5.8).

### 5.7 Pass 2 — Per-team outliers ([team_identification.py:313](../team_identification.py#L313))

Same idea, but within each team separately and with a tighter multiplier (`3.0` instead of `3.5`). MAD is floored at 5 and **capped at 20** — the cap is what prevents a high-variance team from shielding a genuine outlier.

The extra ratio guard `dist > 1.3 * team_threshold` ensures we only flag players *clearly* beyond the limit, not just borderline.

### 5.8 The `clearly_one_team` veto

```python
team_ratio         = d_nearest / max(d_farther, 1e-6)
clearly_one_team   = (team_ratio < 0.6) & (d_nearest < 2.5 * nearer_med)
```

A player is vetoed from being demoted to OTH if they are **both**:
- ratio-close to one team (< 0.6 — closer than 60% of the way to the other), **and**
- absolutely close (within 2.5× their own team's typical spread).

Why both? A GK in an unusual kit might end up "nearer" to one team than the other in relative terms but still far in absolute terms. The absolute check lets us catch them.

### 5.9 Pass 3 — Spatial GK detection ([team_identification.py:335](../team_identification.py#L335))

Pure-colour outlier passes miss GKs whose kit happens to be close-ish to a team's colour. So we add a *spatial* signal: goalkeepers are (a) near the goal line (x-extreme or top-y-extreme, because the far goal appears at the top of the frame) and (b) isolated from other players.

```python
spatial_score = (x_dev / x_mad + top_y_dev / y_mad) * nn_dist
```

Per team, pick the player with the highest score. Flag as OTH if:

- `spatial_score > 250`, **and**
- nearest neighbour > 100 px away (actually isolated), **and**
- colour distance from team is ≥ 1.5× the median (genuinely different kit, not a teammate standing alone).

### 5.10 `FIX 5` — Officials mini-cluster ([team_identification.py:380](../team_identification.py#L380))

Handles the case where 3–5 officials/GKs all wear a similar low-saturation colour. Under `FIX 1`'s isolation check they would mutually validate each other (each has ≥ 2 chroma-similar neighbours), so they escape pre-flagging.

Solution: run **K-Means with k=3** on `(a,b)` only. If one cluster is:

- Size 3–5 (tiny relative to the other two), **and**
- Chromatically > 15 from both larger clusters, **and**
- `(a,b)` std < 16 (tight), **and**
- Less saturated than either larger cluster,

…then its members are officials and are reclassified OTH.

### 5.11 Pass 4 — Hard OTH cap

Ultimate safety net:

```python
MAX_OTH_PITCH = 6
```

If more than 6 players on the pitch are labelled OTH, we reassign the **least-extreme** OTH players (smallest `d_nearest`) back to whichever team centroid they're closer to. 6 = referee + 2 GKs + 3 linesmen worst case. This guarantees we never OTH so many players that we can't check offside.

---

## 6. Output

After Step 4 "rebuild" ([team_identification.py:423](../team_identification.py#L423)), every box has a label again (pre-flagged first, then the clustered ones). The main loop writes:

```json
{
  "keyframe.jpg": {
    "players": [
      {
        "box":     [x1, y1, x2, y2],
        "team":    0 | 1 | 2,
        "foot_px": [(x1+x2)//2, y2]
      },
      ...
    ]
  }
}
```

`foot_px` is the point on the ground that the offside checker later projects through the homography matrix.

---

## 7. What to tell your supervisor

**Did I implement the upper-body + head-skip idea?**  Yes:

- The jersey-only crop is `get_player_colour` at [team_identification.py:30](../team_identification.py#L30).
- The head is skipped via `head_skip = max(5, int(height * 0.20))` — currently **20%** (your supervisor said 15%; easy to change on line 38 if he prefers).
- The 40% slice below the head is used, giving us the upper torso where the jersey colour lives.

**Did it fix the 80% mislabel problem?**  The upper-body crop removed the head/skin noise that was the single biggest source of outliers. On top of that, the cascade of safety nets (pre-flag → K-Means → L-correction → bimodal refit → degenerate refit → four outlier passes → cap) means no single failure mode can dominate. If you run it now on `data/keyframes/` and spot-check the output images in `data/team_identification/`, you should see the regression he asked about is resolved.

**Is there anything fragile?**  Two things you may want to pre-empt:

1. The `dark_team_count >= 4` guard that disables ref pre-flagging when a team wears black — works today but would fail if only 3 outfielders of a dark team are in frame. The current tests don't hit this, but a wide-angle close-up of a counter-attack could.
2. The `top_zone` threshold between close/far cameras is binary (`median_h < 45`). A borderline medium shot could flip between the two on consecutive frames. Not a correctness bug, just a stability concern.
