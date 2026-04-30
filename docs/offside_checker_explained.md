# Offside Checker — Deep Walkthrough

File: [offside_checker.py](../offside_checker.py)
UI: runs a small HTTP server on `http://localhost:5002` (keyframes rendered to canvas, buttons call JSON endpoints).
Inputs: `data/player_detections.json` (from team ID), `data/homography_matrices.json` + `data/homography_points.json` (from [compute_homography.py](../compute_homography.py)).
Output: annotated JPGs in `data/offside_results/`.

## 1. What the offside rule actually requires

A player is offside when, at the moment the ball is played to them, they are **nearer to the opponents' goal line than both the ball AND the second-to-last defender**. For our system, the relevant part is:

> The attacker's leading body edge must be level with or behind the second-to-last defender's leading edge.

Two practical consequences for this code:
1. We need the **most-advanced defender** (keeper usually counts as the last, so the second-to-last defender is typically the most-advanced outfielder — but the laws treat "defender" as *any* opponent, so the auto-detection just picks the defending player furthest up the pitch).
2. The reference is the **defender's leading bounding-box edge projected to world space**, not the image-space pixel. That distinction is the "perspective bug" fix described in §4.

---

## 2. Flow of one check

```
usable frames (have both detections + homography)
          │
          ▼
  ┌──────────────────────────────┐
  │ Step 1: user picks team      │  button — sets state["attacking_team"]
  │ Step 2: user picks direction │  button — sets state["attacking_right"],
  │                              │           triggers find_auto_defender()
  │ Step 3: user clicks attacker │  canvas click — sets state["selected_attacker"]
  │ Step 4: "Check offside"      │  compute_offside()  → verdict + yellow line
  └──────────────────────────────┘
```

The server keeps one `state` dict (reset between frames via `/next` and `/reset`):

```python
state = {
    "index":              0,     # which keyframe we are on
    "attacking_team":     None,  # 0 or 1
    "attacking_right":    None,  # True = attacking →, False = attacking ←
    "selected_attacker":  None,  # index into players[]
    "auto_defender":      None,  # index into players[]
    "result":             None,  # dict returned by compute_offside()
}
```

Each UI action hits an HTTP endpoint; the server mutates `state` and returns JSON so the browser can re-render.

---

## 3. Loading and filtering frames
([offside_checker.py:17–33](../offside_checker.py#L17-L33))

```python
USABLE_FRAMES = sorted([
    kf for kf in all_detections
    if kf in all_homographies and has_reliable_slope(kf)
])
```

A frame is usable only if it has **both** a team-identification record and a computed homography. `has_reliable_slope(kf)` currently returns `kf in all_homographies` — it's a placeholder for filtering out frames whose H is too noisy to trust. Today both checks are effectively the same; the separate function is kept as a future hook.

---

## 4. `find_auto_defender` — the modification you made
([offside_checker.py:44](../offside_checker.py#L44))

This is the piece you weren't sure about. It's the right change; here is why:

### 4.1 What it does

```python
def find_auto_defender(kf, attacking_team, attacking_right):
    players         = all_detections[kf]["players"]
    H               = np.array(all_homographies[kf]["H"])
    defending_team  = 1 - attacking_team
    defending       = [(i, p) for i, p in enumerate(players)
                       if p["team"] == defending_team]

    def world_x(p):
        x1, y1, x2, y2 = p["box"]
        edge_px        = x2 if attacking_right else x1   # leading edge in pixels
        wx, _          = project_point(H, edge_px, y2)    # project foot through H
        return wx

    ranked  = sorted(defending, key=lambda ip: world_x(ip[1]), reverse=attacking_right)
    return ranked[0][0]
```

**Which edge?** If the attack moves →, the defender's *leading* edge is `x2` (the right side of their bbox). If the attack moves ←, it's `x1`. That's the offside-relevant edge by the same logic we apply to the attacker later.

**Which y?** `y2` — the bottom of the bbox, i.e. the player's feet on the grass. That's the point that actually lies on the pitch plane (the homography only maps the ground plane, not the player's head).

**Why project through `H` first, then rank?**
Pure image-space `x2` gives the wrong answer when the camera is angled. Consider a right-sided attack: a defender standing at the near sideline might have a larger pixel `x2` than another defender who is actually further up the pitch but on the far sideline. Ranking by pixel would pick the near-camera defender (wrong). Projecting to world coordinates first gives us metres along the length of the pitch, which is the actual offside axis. The inline comment at [offside_checker.py:46–49](../offside_checker.py#L46-L49) documents exactly this.

**The `reverse=attacking_right` trick.**
If attacking right, the most-advanced defender is the one with the **largest** world-X → `reverse=True` (descending sort). If attacking left, smallest world-X → `reverse=False`. Same `sorted` call, one line.

### 4.2 Why this is correct per the laws (and your supervisor's alternative would be over-complication)

Your supervisor's proposal:
> Don't auto-detect the defender. Draw the line at the attacker instead, then count whether there are defenders ahead of or behind that line.

You are right that that's not the rule. The rule places the line at the **second-to-last defender** and asks whether the attacker is ahead of that. Counting defenders vs. the attacker's line would answer a related but different question ("how many defenders has the attacker beaten?") — it doesn't tell you whether the attacker was ahead of the second-to-last defender at the moment of play.

Your current implementation is the standard refereeing interpretation:

1. Find the most-advanced defending outfielder (`find_auto_defender`).
2. Draw the offside line through them.
3. Check whether the attacker's leading edge is beyond that line.

That's what a VAR operator does with their drawn line. The only thing your code *doesn't* do is also compare against "level with the ball" — because you're working from a keyframe where the ball-play moment is already chosen by `annotate_keyframes.py`. The ball-level constraint is effectively baked in.

### 4.3 One caveat to raise with him

`find_auto_defender` will pick a **goalkeeper** if the keeper happens to be labelled as team 0 or 1 rather than OTH. Normally team identification tags GKs as OTH (see §5.9 of the team-ID doc), so they're excluded — but on frames where the GK was mis-classified, the auto-defender will be the keeper (who is almost always the most advanced in the other direction, so this usually just makes the offside line extremely deep and the check trivially says "onside"). It's a reason to keep the team-ID OTH pass strict.

---

## 5. `project_point` and `is_valid_projection`
([offside_checker.py:81–88](../offside_checker.py#L81-L88))

```python
def project_point(H, px, py):
    pt     = np.array([px, py, 1.0])
    proj   = H @ pt
    proj  /= proj[2]
    return float(proj[0]), float(proj[1])
```

Standard homogeneous-coordinates projection through the 3×3 homography. `H` was fitted in [compute_homography.py](../compute_homography.py) from the correspondence points the user clicked in `pick_points.py`. The result is (x, y) in **metres on the pitch plane**, origin at the centre spot, per the project coordinate system.

`is_valid_projection` is a sanity check (`|x|<60, |y|<40`) — useful for filtering obviously bad projections, but not currently called from the main flow. It's left in place for diagnostics.

---

## 6. `get_line_slope` — drawing the offside line correctly
([offside_checker.py:90](../offside_checker.py#L90))

A perpendicular-to-the-pitch-length line in **world coordinates** does *not* project to a vertical pixel line on screen — it tilts according to how the camera sees the pitch. We have to compute that tilt.

```python
def get_line_slope(kf, offside_line_x, defender_y, H):
    H_inv = np.linalg.inv(H)

    def real_to_image(rx, ry):
        pt      = np.array([rx, ry, 1.0])
        img_pt  = H_inv @ pt
        img_pt /= img_pt[2]
        return float(img_pt[0]), float(img_pt[1])

    dy  = 5.0
    px1 = real_to_image(offside_line_x, defender_y - dy)
    px2 = real_to_image(offside_line_x, defender_y + dy)
    dx_img = px2[0] - px1[0]
    dy_img = px2[1] - px1[1]
    return dx_img / dy_img if abs(dy_img) > 1e-6 else 0.0
```

**What this computes:**
- Two world points 10 m apart in the `y` direction, both at `x = offside_line_x`. They lie along the line perpendicular to the pitch length (i.e. along the pitch width) that passes through the defender.
- Project both back to the image with `H_inv`.
- Measure the pixel slope `dx_img / dy_img`.

**Why `dx/dy` and not `dy/dx`?** Because we use the slope as "for each pixel down the screen, move this many pixels sideways" — the line is drawn from top to bottom of the frame (§7).

---

## 7. `compute_offside` — the actual verdict
([offside_checker.py:149](../offside_checker.py#L149))

This is the heart of the check. Step by step:

```python
att_box          = players[attacker_idx]["box"]
x1_att, y1_att, x2_att, y2_att = att_box
def_box          = players[defender_idx]["box"]
x1_def, y1_def, x2_def, y2_def = def_box

if attacking_right:
    attacker_x, attacker_y = project_point(H, x2_att, y2_att)  # leading edge
    defender_x, defender_y = project_point(H, x2_def, y2_def)
    anchor_px              = x2_def
else:
    attacker_x, attacker_y = project_point(H, x1_att, y2_att)
    defender_x, defender_y = project_point(H, x1_def, y2_def)
    anchor_px              = x1_def
```

We project **each player's leading edge at foot level (`y2`)** into world space. The choice of edge matches the attacking direction, so we're comparing the same side of each player's body — the one the offside rule cares about.

```python
offside_line_x = defender_x
is_offside     = (attacker_x > offside_line_x) if attacking_right \
                 else (attacker_x < offside_line_x)
margin         = abs(attacker_x - offside_line_x)
```

In world coordinates, offside is a one-dimensional comparison: is the attacker's world-X beyond the defender's? Attacking right → "beyond" means greater X; attacking left → smaller X. `margin` is the distance in metres; useful for close calls and for reporting in the UI.

### 7.1 Drawing the line in image coordinates

```python
slope        = get_line_slope(kf, offside_line_x, defender_y, H)
anchor_y     = players[defender_idx]["foot_px"][1]       # defender's feet, pixels
dy_to_top    = anchor_y
dy_to_bottom = img_h - anchor_y

pt1_x        = int(anchor_px - slope * dy_to_top)        # top of frame
pt2_x        = int(anchor_px + slope * dy_to_bottom)     # bottom of frame

img_pt1 = [pt1_x, 0]
img_pt2 = [pt2_x, img_h]
```

We anchor the line at the defender's foot pixel (`anchor_px`, `anchor_y`) and extend it upward to `y=0` and downward to `y=img_h` using the slope from §6. The result is a full-height line that visually passes through the defender and is geometrically correct in world space.

The `line_image_pts` are returned to the UI and drawn in `render_frame` as a yellow line on the keyframe.

### 7.2 The returned verdict JSON

```python
return {
    "offside":         bool(is_offside),
    "attacker_x":      round(attacker_x, 2),
    "offside_line_x":  round(offside_line_x, 2),
    "margin_metres":   round(margin, 2),
    "line_image_pts":  [img_pt1, img_pt2],
    "attacking_right": bool(attacking_right),
}
```

That payload drives both the on-screen line and the "OFFSIDE / ONSIDE" label with the verdict details in the side panel.

---

## 8. The HTTP endpoints ([offside_checker.py:546–660](../offside_checker.py#L546-L660))

For completeness, here is what each endpoint does:

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/`                   | Serve the HTML UI |
| GET  | `/frame`              | Return base64 JPG of the current keyframe + width/height + clip name |
| POST | `/set_attacking`      | Store `attacking_team`, reset everything downstream |
| POST | `/set_direction`      | Store `attacking_right`, run `find_auto_defender`, return defender info |
| POST | `/select_attacker`    | Find the player whose bbox contains the click (first hit wins) |
| POST | `/check_offside`      | Run `compute_offside`, store result, return verdict payload |
| POST | `/save_result`        | Re-render the frame (with line + verdict) and save to `data/offside_results/` |
| POST | `/reset`              | Clear state for the current frame |
| POST | `/next`               | Advance to the next frame (or respond `done: True` at the end) |

The `Handler` class silences its default logging (`log_message`, `log_error` do nothing), so the console stays readable for the pipeline's own prints.

---

## 9. `render_frame` — what you see on screen
([offside_checker.py:108](../offside_checker.py#L108))

Each call rebuilds the keyframe image with current state overlays:

- Every player gets a bbox + `"i:Tx"` label in their team colour (red/blue/yellow).
- Selected attacker: bright green box, thickness 4.
- Auto-defender: orange box, thickness 4.
- If a result exists: draws the yellow offside line and, around the attacker, a red box + "OFFSIDE" label, or green box + "ONSIDE".

The image is JPG-encoded at quality 88 and base64-ed into the JSON response so the browser can render it as a data URL.

---

## 10. What to say in your meeting

### 10.1 On the auto-defender approach

You kept the auto-detection instead of switching to "line at attacker, count defenders ahead/behind". Reasons you can give him:

1. **The rule is about the defender's line, not the attacker's.** The second-to-last defender defines the offside line; the attacker is what we compare against it. Swapping the reference point changes what's being asked.
2. **His alternative answers a different question.** "Are there defenders ahead of the attacker?" tells you *how many defenders the attacker has beaten*, not *whether the attacker was in an offside position*. Two scenarios where they diverge:
   - All defenders are behind the attacker → his method says "offside" → but if the attacker is *level* with the last defender, the rule says *onside*.
   - One defender is ahead → his method says "onside" → but if that one defender is the keeper and the attacker is ahead of every outfielder, the rule still says *offside* (for VAR, level with the last outfielder is the threshold in most contexts).
3. **Auto-detection is doing the work a VAR operator does manually.** Projecting through the homography and picking the most-advanced defender is just the automated equivalent of drawing a perpendicular line at the correct point on the pitch.

### 10.2 The modification you weren't sure about

What you changed (or kept): `find_auto_defender` ranks defenders by **world-space X after projecting their leading edge through the homography**, not by pixel x1/x2. See [offside_checker.py:48–63](../offside_checker.py#L48-L63). That is the *correct* fix to the perspective bug where image-space ranking can pick the defender closest to the camera rather than the one furthest up the pitch. If you previously had it ranking on pixel x2, the change to world-X is the right move and matches the logic you already use for the attacker/defender comparison inside `compute_offside`.

### 10.3 Edge cases he may ask about

- **Keeper mis-classified as team 1:** auto-defender might pick the keeper. Mitigation: strict OTH tagging in team identification (the pre-flag / spatial GK / mini-cluster passes in the team-ID file are all aimed at this). A manual override in the UI could be added as a follow-up if he wants one.
- **Attacker's leading edge at the bounding box, not the actual body part:** YOLO gives us a rectangle, and we use `x2` / `x1` as the leading edge. That's an approximation — a player leaning forward would have their shoulder ahead of `x2`. Acceptable for a thesis-scale demo; VAR uses pose estimation to find the actual body part.
- **Homography noise:** if `compute_homography.py` produced an H with high reprojection error, the world-X ranking becomes unreliable. `has_reliable_slope` is the place to add a stricter filter (today it only checks that `kf` has an H at all — you could tighten it to a max reprojection-error threshold).
