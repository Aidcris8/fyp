import cv2
import numpy as np
import json
import os

POINTS_FILE = "data/homography_points.json"
OUTPUT_FILE = "data/homography_matrices.json"
KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/homography_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(POINTS_FILE) as f:
    all_points = json.load(f)

results = {}

print("Computing homography matrices...\n")

for kf_name, points in sorted(all_points.items()):
    if len(points) < 4:
        print(f"  SKIP {kf_name} — only {len(points)} points")
        continue

    # Build correspondence arrays
    image_pts = np.array([p["image"] for p in points], dtype=np.float32)
    real_pts  = np.array([p["real"]  for p in points], dtype=np.float32)

    # Compute homography using RANSAC for robustness
    H, mask = cv2.findHomography(image_pts, real_pts, cv2.RANSAC, ransacReprojThreshold=2.0)

    if H is None:
        print(f"  FAILED {kf_name} — homography could not be computed")
        continue

    inliers = int(mask.sum()) if mask is not None else len(points)
    total   = len(points)

    # Compute reprojection error on inliers
    errors = []
    for i, (img_pt, real_pt) in enumerate(zip(image_pts, real_pts)):
        if mask is not None and mask[i] == 0:
            continue
        # Project image point to real world
        pt_h = np.array([img_pt[0], img_pt[1], 1.0])
        projected = H @ pt_h
        projected /= projected[2]
        error = np.sqrt((projected[0] - real_pt[0])**2 + (projected[1] - real_pt[1])**2)
        errors.append(error)

    mean_error = float(np.mean(errors)) if errors else 999.0
    print(f"  OK  {kf_name}")
    print(f"      Points: {inliers}/{total} inliers, mean reprojection error: {mean_error:.3f}m")

    # Save H matrix as list for JSON serialisation
    results[kf_name] = {
        "H": H.tolist(),
        "inliers": inliers,
        "total_points": total,
        "mean_error_metres": round(mean_error, 4)
    }

    # Validation image — project player foot positions and draw on pitch diagram
    img = cv2.imread(os.path.join(KEYFRAMES_DIR, kf_name))
    if img is not None:
        # Draw the annotated correspondence points
        for i, p in enumerate(points):
            px, py = p["image"]
            cv2.circle(img, (px, py), 8, (0, 255, 255), -1)
            cv2.circle(img, (px, py), 9, (0, 0, 0), 2)

            # Project and show reprojection
            pt_h = np.array([px, py, 1.0])
            proj = H @ pt_h
            proj /= proj[2]
            label = f"({proj[0]:.1f},{proj[1]:.1f})"
            cv2.putText(img, label, (px + 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, kf_name), img)

print(f"\nSaving {len(results)} homography matrices to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\nSummary:")
for kf, r in results.items():
    short = kf.split("_frame")[0].split("_half")[0][-40:]
    print(f"  {short}... error={r['mean_error_metres']}m inliers={r['inliers']}/{r['total_points']}")

print("\nDone!")