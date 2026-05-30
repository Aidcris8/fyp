# Reads pixel-to-world correspondence points from homography_points.json and fits
# a homography matrix H per keyframe using RANSAC followed by a least-squares refit
# on the inliers. Saves matrices and reprojection-error metrics to
# data/homography_matrices.json and writes a validation overlay image per frame.

import cv2
import numpy as np
import json
import os

POINTS_FILE = "data/homography_points.json"
OUTPUT_FILE = "data/homography_matrices.json"
KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/homography_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# pitch landmarks with their respective FIFA-standard pitch coordinates in metres. 
# Lines are drawn by projecting their endpoints back into the image via H_inv and should align with the
# real pitch markings when H is well-fitted
PITCH_LANDMARKS = [
    (0, 0), (0, 34), (0, -34),
    (41.5, 0), (-41.5, 0),
    (36, 20.16), (36, -20.16), (52.5, 20.16), (52.5, -20.16),
    (-36, 20.16), (-36, -20.16), (-52.5, 20.16), (-52.5, -20.16),
    (47, 9.16), (47, -9.16), (-47, 9.16), (-47, -9.16),
    (52.5, 3.66), (52.5, -3.66), (-52.5, 3.66), (-52.5, -3.66),
    (52.5, 34), (52.5, -34), (-52.5, 34), (-52.5, -34),
]

PITCH_LINES = [
    # halfway line
    ((0, 34), (0, -34)),
    # right penalty box
    ((36, 20.16), (52.5, 20.16)),
    ((36, -20.16), (52.5, -20.16)),
    ((36, 20.16), (36, -20.16)),
    # left penalty box
    ((-36, 20.16), (-52.5, 20.16)),
    ((-36, -20.16), (-52.5, -20.16)),
    ((-36, 20.16), (-36, -20.16)),
    # right 6-yard box
    ((47, 9.16), (52.5, 9.16)),
    ((47, -9.16), (52.5, -9.16)),
    ((47, 9.16), (47, -9.16)),
    # left 6-yard box
    ((-47, 9.16), (-52.5, 9.16)),
    ((-47, -9.16), (-52.5, -9.16)),
    ((-47, 9.16), (-47, -9.16)),
    # touchlines
    ((52.5, 34), (-52.5, 34)),
    ((52.5, -34), (-52.5, -34)),
    # goal lines
    ((52.5, 34), (52.5, -34)),
    ((-52.5, 34), (-52.5, -34)),
]


def project_inv(H_inv, rx, ry):
    """Project world (rx, ry) back into image space via H_inv."""
    #convert homogenous coordinates back to 2d by dividing by third component
    #if p[2] is near zero the point is at infinity therefore cant be projected
    p = H_inv @ np.array([rx, ry, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])

#load correspondence points saved by pick points.py
#each entry maps a keyframe filename to list of {image: [px,py], real: [rx,ry]} pairs
with open(POINTS_FILE) as f:
    all_points = json.load(f)

# load existing matrices so re-runs only compute new keyframes
results = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        results = json.load(f)

print("Computing homography matrices (improved)...\n")

#process each keyframe, skipping any that already in results
#also skip frames with less than 4 points since homogrpahy requires min 4 points
for kf_name, points in sorted(all_points.items()):
    if kf_name in results:
        print(f"  Skipping (already computed): {kf_name}")
        continue
    if len(points) < 4:
        print(f"  SKIP {kf_name} — only {len(points)} points")
        continue

    image_pts = np.array([p["image"] for p in points], dtype=np.float32)
    real_pts = np.array([p["real"] for p in points], dtype=np.float32)

    # RANSAC threshold in world-space metres, 0.5 m sounds tight but at broadcast-camera scale 1 pixel near the center circle is about 0.1 m, 
    # so 0.5 m = only about 5 pixels of allowed click error which is too strict. Manual clicks on pitch markings are typically +-10-15 px, 
    # which at the far touchline scale is +-2-3 m. 2.0 m keeps correctly-picked points as inliers while still rejecting genuinely wrong click
    H_ransac, mask = cv2.findHomography(
        image_pts, real_pts, cv2.RANSAC, ransacReprojThreshold=2.0
    )

    if H_ransac is None:
        print(f"  FAILED {kf_name} — homography could not be computed")
        continue

    inliers_mask = mask.ravel().astype(bool) if mask is not None \
        else np.ones(len(points), dtype=bool)
    inlier_image = image_pts[inliers_mask]
    inlier_real = real_pts[inliers_mask]
    n_inliers = int(inliers_mask.sum())

    # refit on RANSAC inliers with ordinary least-squares for the tightest fit
    H = H_ransac
    if n_inliers >= 4:
        H_refit, _ = cv2.findHomography(inlier_image, inlier_real, method=0)
        if H_refit is not None:
            H = H_refit

    # sign normalisation: H and -H represent the same projective transform, but only one sign gives H[2]·pt > 0 for points on the physical pitch.
    # normalise so the centroid of the pick points always lands on the positive side, which makes the invalid_zone_frac check below meaningful.
    centroid = image_pts.mean(axis=0)
    centroid_denom = float(H[2] @ np.array([centroid[0], centroid[1], 1.0]))
    if centroid_denom < 0:
        H = -H
        print(f"      [sign-norm] negated H so pick-point region has H[2]>0")

    H_inv = np.linalg.inv(H)

    # pixel-space reprojection error- The user's labelled real-world coordinates are projected via H_inv back into image space and compared
    # to where the user actually clicked. A value of N means "the fitted H places each labelled landmark at a location that is N pixels away from
    # where the user clicked it." This is preferred over a world-space residual on the same fit points because 1 px of image error projects to many
    # metres at the far touchline, so a world-space number would be dominated by perspective rather than fit quality
    real_h = np.hstack([real_pts, np.ones((len(real_pts), 1))])
    img_pred = (H_inv @ real_h.T).T
    img_pred = img_pred[:, :2] / img_pred[:, 2:3]
    pixel_errors = np.linalg.norm(img_pred - image_pts, axis=1)
    mean_pixel_error = float(np.mean(pixel_errors))
    mean_pixel_error_inliers = float(np.mean(pixel_errors[inliers_mask])) \
        if n_inliers > 0 else float("nan")

    # world-space error on inliers, reported alongside the pixel-space
    # metric and used as the headline mean-error number quoted in Chapter 5
    world_errors_m = []
    for i, (img_pt, real_pt) in enumerate(zip(image_pts, real_pts)):
        if not inliers_mask[i]:
            continue
        pt_h = np.array([img_pt[0], img_pt[1], 1.0])
        proj = H @ pt_h
        proj /= proj[2]
        err = np.sqrt((proj[0] - real_pt[0]) ** 2 + (proj[1] - real_pt[1]) ** 2)
        world_errors_m.append(err)
    mean_world_error_m = float(np.mean(world_errors_m)) if world_errors_m else 999.0

    # sample a grid of image points and count how many have a negative projective denominator
    # negative denominator means the point is projected to the wrong side of the world plane — any world_x
    # derived from it is sign-flipped garbage. frames with many invalid pixels in the player zone need their pick_points redone
    grid_xs = np.linspace(100, 1180, 12)
    grid_ys = np.linspace(200, 580, 8)
    bad = 0
    total_grid = 0
    for gx in grid_xs:
        for gy in grid_ys:
            denom = H[2, 0]*gx + H[2, 1]*gy + H[2, 2]
            total_grid += 1
            if denom <= 0:
                bad += 1
    invalid_frac = bad / total_grid
    validity_warn = f"  *** WARNING: {bad}/{total_grid} grid cells have H[2]<=0 ({invalid_frac:.0%} of player zone invalid) — re-pick points ***" if invalid_frac > 0.05 else ""

    print(f"  OK  {kf_name}")
    print(f"      Inliers: {n_inliers}/{len(points)}")
    print(f"      Symmetric pixel error (all):     {mean_pixel_error:.2f} px")
    print(f"      Symmetric pixel error (inliers): {mean_pixel_error_inliers:.2f} px")
    print(f"      Mean world error (inliers):      {mean_world_error_m:.3f} m")
    print(f"      H[2]<=0 in player zone:          {bad}/{total_grid} ({invalid_frac:.0%})")
    if validity_warn:
        print(validity_warn)

    #store matrix an all error metrics 
    #invalid zone frac is read by offside checker to decide whether frame is usable for verdict 
    results[kf_name] = {
        "H": H.tolist(),
        "inliers": n_inliers,
        "total_points": len(points),
        "mean_error_metres": round(mean_world_error_m, 4),
        "mean_pixel_error_all": round(mean_pixel_error, 3),
        "mean_pixel_error_inliers": round(mean_pixel_error_inliers, 3),
        "invalid_zone_frac": round(invalid_frac, 3),
    }

    # projects the FIFA pitch model back into the image so a misaligned homography is visible at a glance
    # the projected lines should sit on the actual white pitch markings in the frame.
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf_name))
    if img is not None:
        # user-annotated points: yellow if inlier, red if outlier under RANSAC.
        for i, p in enumerate(points):
            px = int(round(p["image"][0]))
            py = int(round(p["image"][1]))
            colour = (0, 255, 255) if inliers_mask[i] else (0, 0, 255)
            cv2.circle(img, (px, py), 8, colour, -1)
            cv2.circle(img, (px, py), 9, (0, 0, 0), 2)

        # projected pitch lines — should align with the real pitch markings.
        for (rx1, ry1), (rx2, ry2) in PITCH_LINES:
            p1 = project_inv(H_inv, rx1, ry1)
            p2 = project_inv(H_inv, rx2, ry2)
            if p1 is None or p2 is None:
                continue
            x1, y1 = p1
            x2, y2 = p2
            # Skip lines that explode off the canvas
            if not all(-1e4 < v < 1e4 for v in (x1, y1, x2, y2)):
                continue
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 200, 0), 1, cv2.LINE_AA)

        # Projected landmark dots
        for rx, ry in PITCH_LANDMARKS:
            p = project_inv(H_inv, rx, ry)
            if p is None:
                continue
            x, y = p
            if not (-1e4 < x < 1e4 and -1e4 < y < 1e4):
                continue
            cv2.circle(img, (int(x), int(y)), 4, (255, 200, 0), -1)

        # burn the error metrics into the image as a text header so quality is visible without opening the json
        header = (f"px_err(all)={mean_pixel_error:.2f}  "
                  f"px_err(in)={mean_pixel_error_inliers:.2f}  "
                  f"world={mean_world_error_m:.2f}m  "
                  f"inliers={n_inliers}/{len(points)}")
        # putText is called twice, once with thick black for outline and other with thin white on top
        cv2.putText(img, header, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, header, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(OUTPUT_DIR, kf_name), img)

#write all matrices out once at the end rather than after each frame, so a crash mid run doesnt corrupt output filw with partial write
print(f"\nSaving {len(results)} homography matrices to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

#print one line summary per frame so outlier errors come up instantly
print("\nSummary:")
for kf, r in results.items():
    short = kf.split("_frame")[0].split("_half")[0][-40:]
    print(f"  {short}... px_err(all)={r['mean_pixel_error_all']}px "
          f"px_err(in)={r['mean_pixel_error_inliers']}px "
          f"world={r['mean_error_metres']}m "
          f"inliers={r['inliers']}/{r['total_points']}")

print("\nDone!")
