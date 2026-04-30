import cv2
import numpy as np
import json
import os

POINTS_FILE = "data/homography_points.json"
OUTPUT_FILE = "data/homography_matrices.json"
KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/homography_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMPROVEMENT 17: pitch landmarks/lines for the validation overlay. Each tuple
# is in FIFA-standard pitch coordinates (metres). Lines are drawn by projecting
# their endpoints back into the image via H_inv — they should align with the
# real pitch markings if H is good.
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
    # Halfway line
    ((0, 34), (0, -34)),
    # Right penalty box
    ((36, 20.16), (52.5, 20.16)),
    ((36, -20.16), (52.5, -20.16)),
    ((36, 20.16), (36, -20.16)),
    # Left penalty box
    ((-36, 20.16), (-52.5, 20.16)),
    ((-36, -20.16), (-52.5, -20.16)),
    ((-36, 20.16), (-36, -20.16)),
    # Right 6-yard box
    ((47, 9.16), (52.5, 9.16)),
    ((47, -9.16), (52.5, -9.16)),
    ((47, 9.16), (47, -9.16)),
    # Left 6-yard box
    ((-47, 9.16), (-52.5, 9.16)),
    ((-47, -9.16), (-52.5, -9.16)),
    ((-47, 9.16), (-47, -9.16)),
    # Touchlines
    ((52.5, 34), (-52.5, 34)),
    ((52.5, -34), (-52.5, -34)),
    # Goal lines
    ((52.5, 34), (52.5, -34)),
    ((-52.5, 34), (-52.5, -34)),
]


def project_point(H, x, y):
    """Project image-space (x, y) to world via H, with safe div-by-zero guard."""
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def project_inv(H_inv, rx, ry):
    """Project world (rx, ry) back into image space via H_inv."""
    p = H_inv @ np.array([rx, ry, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


with open(POINTS_FILE) as f:
    all_points = json.load(f)

results = {}
print("Computing homography matrices (improved)...\n")

for kf_name, points in sorted(all_points.items()):
    if len(points) < 4:
        print(f"  SKIP {kf_name} — only {len(points)} points")
        continue

    image_pts = np.array([p["image"] for p in points], dtype=np.float32)
    real_pts = np.array([p["real"] for p in points], dtype=np.float32)

    # IMPROVEMENT 13: tighter RANSAC threshold (0.5 m instead of 2.0 m).
    # 2 m is wider than two strides — almost any clicked point would have been
    # accepted as an inlier. 0.5 m forces genuinely consistent correspondences.
    H_ransac, mask = cv2.findHomography(
        image_pts, real_pts, cv2.RANSAC, ransacReprojThreshold=0.5
    )

    if H_ransac is None:
        print(f"  FAILED {kf_name} — homography could not be computed")
        continue

    inliers_mask = mask.ravel().astype(bool) if mask is not None \
        else np.ones(len(points), dtype=bool)
    inlier_image = image_pts[inliers_mask]
    inlier_real = real_pts[inliers_mask]
    n_inliers = int(inliers_mask.sum())

    # IMPROVEMENT 14: refit on the RANSAC inliers using ordinary
    # least-squares (method=0). RANSAC picks the best fit among random
    # sub-samples; refitting on the inlier set with LM produces the
    # maximum-likelihood estimate, which is usually tighter than the RANSAC
    # fit alone.
    H = H_ransac
    if n_inliers >= 4:
        H_refit, _ = cv2.findHomography(inlier_image, inlier_real, method=0)
        if H_refit is not None:
            H = H_refit

    H_inv = np.linalg.inv(H)

    # IMPROVEMENT 15: pixel-space reprojection error.
    # Old metric: world-space residual on the same points used to fit — 1 px
    # of image error projects to many metres at the far touchline, so the
    # number was dominated by perspective rather than fit quality. New
    # metric: take the user's labelled real-world coordinates, project them
    # via H_inv into image space, compare to where the user actually clicked.
    # A value of N means "the fitted H places each labelled landmark at a
    # location that's N pixels away from where the user clicked it."
    real_h = np.hstack([real_pts, np.ones((len(real_pts), 1))])
    img_pred = (H_inv @ real_h.T).T
    img_pred = img_pred[:, :2] / img_pred[:, 2:3]
    pixel_errors = np.linalg.norm(img_pred - image_pts, axis=1)
    mean_pixel_error = float(np.mean(pixel_errors))
    mean_pixel_error_inliers = float(np.mean(pixel_errors[inliers_mask])) \
        if n_inliers > 0 else float("nan")

    # Legacy world-space error on inliers — kept so the metric is comparable
    # to the original compute_homography.py output number.
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

    print(f"  OK  {kf_name}")
    print(f"      Inliers: {n_inliers}/{len(points)}")
    print(f"      Symmetric pixel error (all):     {mean_pixel_error:.2f} px")
    print(f"      Symmetric pixel error (inliers): {mean_pixel_error_inliers:.2f} px")
    print(f"      Legacy world error (inliers):    {mean_world_error_m:.3f} m")

    results[kf_name] = {
        "H": H.tolist(),
        "inliers": n_inliers,
        "total_points": len(points),
        "mean_error_metres": round(mean_world_error_m, 4),
        "mean_pixel_error_all": round(mean_pixel_error, 3),
        "mean_pixel_error_inliers": round(mean_pixel_error_inliers, 3),
    }

    # IMPROVEMENT 17: validation overlay — projects the FIFA pitch model back
    # into the image so a misaligned homography is visible at a glance. The
    # cyan lines should sit on the actual white pitch markings in the frame.
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf_name))
    if img is not None:
        # User-annotated points: yellow if inlier, red if outlier under RANSAC.
        for i, p in enumerate(points):
            px = int(round(p["image"][0]))
            py = int(round(p["image"][1]))
            colour = (0, 255, 255) if inliers_mask[i] else (0, 0, 255)
            cv2.circle(img, (px, py), 8, colour, -1)
            cv2.circle(img, (px, py), 9, (0, 0, 0), 2)

        # Projected pitch lines — should align with the real pitch markings.
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

        header = (f"px_err(all)={mean_pixel_error:.2f}  "
                  f"px_err(in)={mean_pixel_error_inliers:.2f}  "
                  f"world={mean_world_error_m:.2f}m  "
                  f"inliers={n_inliers}/{len(points)}")
        cv2.putText(img, header, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, header, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(OUTPUT_DIR, kf_name), img)

print(f"\nSaving {len(results)} homography matrices to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\nSummary:")
for kf, r in results.items():
    short = kf.split("_frame")[0].split("_half")[0][-40:]
    print(f"  {short}... px_err(all)={r['mean_pixel_error_all']}px "
          f"px_err(in)={r['mean_pixel_error_inliers']}px "
          f"world={r['mean_error_metres']}m "
          f"inliers={r['inliers']}/{r['total_points']}")

print("\nDone!")
