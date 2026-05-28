"""
Tests different chroma weighting values applied after StandardScaler
before K-Means clustering. Uses the already-computed per-player feature
vectors from team_identification_debug.json so YOLO does not need to re-run.

Metric: average cluster balance = min(c0,c1)/max(c0,c1) across all frames.
1.0 = perfect 50/50 split.  Higher is better.
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEBUG_FILE = "data/team_identification_debug.json"
WEIGHTS    = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
SEEDS      = [42, 0, 7, 13, 99]

with open(DEBUG_FILE) as f:
    debug_data = json.load(f)

def run_clustering(features, weight):
    """Run K-Means with given chroma weight, return best balance score."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled[:, 1] *= weight   # a*
    scaled[:, 2] *= weight   # b*

    best_balance = -1
    for seed in SEEDS:
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        lbl = km.fit_predict(scaled)
        c0, c1 = np.sum(lbl == 0), np.sum(lbl == 1)
        balance = min(c0, c1) / max(c0, c1)
        if balance > best_balance:
            best_balance = balance
    return best_balance

# Collect per-frame results for each weight
results = {w: [] for w in WEIGHTS}

for fname, frame_data in debug_data.items():
    players = frame_data["players"]
    # Only use players that were not pre-flagged (same as real pipeline)
    active = [p for p in players if p.get("pre_flag") is None and p["label"] in (0, 1)]
    if len(active) < 4:
        continue

    features = np.array([
        [p["lab"][0], p["lab"][1], p["lab"][2],
         p["dark_score"], p["bright_pct"], p["sat"]]
        for p in active
    ], dtype=np.float32)

    for w in WEIGHTS:
        balance = run_clustering(features, w)
        results[w].append(balance)

# Print summary table
print(f"\n{'Weight':>8}  {'Avg balance':>12}  {'Frames <0.5':>12}  {'Frames <0.7':>12}  {'Min balance':>12}")
print("─" * 70)
for w in WEIGHTS:
    scores = results[w]
    avg    = np.mean(scores)
    n_poor = sum(1 for s in scores if s < 0.5)
    n_ok   = sum(1 for s in scores if s < 0.7)
    worst  = min(scores)
    marker = "  <-- current" if w == 1.8 else ""
    print(f"{w:>8.1f}  {avg:>12.4f}  {n_poor:>12}  {n_ok:>12}  {worst:>12.4f}{marker}")

print(f"\nFrames evaluated: {len(results[WEIGHTS[0]])}")
best_w = max(WEIGHTS, key=lambda w: np.mean(results[w]))
print(f"Best average balance at weight: {best_w}")
