import cv2
import numpy as np
import os

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_DIR = "data/pitch_detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def segment_lines(lines, height):
    """Separate lines into horizontal and vertical groups."""
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < 30:
            horizontal.append(line[0])
        elif angle > 60:
            vertical.append(line[0])
    return horizontal, vertical

def line_intersection(l1, l2):
    """Find intersection point of two lines."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (int(x), int(y))

def find_intersections(horizontal, vertical, width, height):
    """Find all intersections between horizontal and vertical lines."""
    intersections = []
    margin = 50

    for h in horizontal:
        for v in vertical:
            pt = line_intersection(h, v)
            if pt is not None:
                x, y = pt
                if margin <= x <= width - margin and margin <= y <= height - margin:
                    intersections.append(pt)

    if len(intersections) == 0:
        return []

    clustered = []
    used = [False] * len(intersections)
    min_dist = 20

    for i, pt1 in enumerate(intersections):
        if used[i]:
            continue
        cluster = [pt1]
        for j, pt2 in enumerate(intersections):
            if i != j and not used[j]:
                dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                if dist < min_dist:
                    cluster.append(pt2)
                    used[j] = True
        cx = int(np.mean([p[0] for p in cluster]))
        cy = int(np.mean([p[1] for p in cluster]))
        clustered.append((cx, cy))
        used[i] = True

    for pt in clustered:
        print(f"    Intersection at: {pt}")

    return clustered

def detect_pitch_lines(img_path):
    img = cv2.imread(img_path)
    original = img.copy()
    height, width = img.shape[:2]

    # Step 1 - Mask out top 35% of image
    mask = np.zeros_like(img)
    mask[int(height * 0.35):, :] = 255
    img_masked = cv2.bitwise_and(img, mask)

    # Step 2 - Grayscale
    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # Step 3 - Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 4 - Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=60,
        maxLineGap=15
    )

    # Step 5 - Filter by angle and y position
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if (angle < 30 or angle > 60) and \
               (y1 > height * 0.35 and y2 > height * 0.35):
                filtered_lines.append(line)

    # Step 6 - Separate into horizontal and vertical
    horizontal, vertical = segment_lines(filtered_lines, height)

    # Step 7 - Find intersections
    intersections = find_intersections(horizontal, vertical, width, height)

    print(f"  Total lines detected: {len(lines) if lines is not None else 0}")
    print(f"  Filtered lines kept: {len(filtered_lines)}")
    print(f"  Horizontal lines: {len(horizontal)}")
    print(f"  Vertical lines: {len(vertical)}")
    print(f"  Intersections found: {len(intersections)}")

    # Draw results
    result_img = original.copy()
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for pt in intersections:
        cv2.circle(result_img, pt, 8, (0, 0, 255), -1)

    return edges, result_img, intersections

# Loop through all keyframes
keyframes = sorted([f for f in os.listdir(KEYFRAMES_DIR) if f.endswith(".jpg")])

for test_kf in keyframes:
    print(f"\nTesting on: {test_kf}")
    img_path = os.path.join(KEYFRAMES_DIR, test_kf)
    edges, result_img, intersections = detect_pitch_lines(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, test_kf), result_img)
    print(f"Saved to {OUTPUT_DIR}")