import os
import json

SOCCERNET_DIR  = "data/soccernet"
CLIPS_DIR      = "data/clips"
KEYFRAMES_JSON = "data/keyframes.json"

# Set to a league slug to restrict the scan, or None to search every league.
#   england_epl, spain_laliga, italy_serie-a, germany_bundesliga,
#   france_ligue-1, europe_uefa-champions-league
LEAGUE_FILTER = "europe_uefa-champions-league"

# How many games to list in the output table.
TOP_N = 50

# How many additional events you still need.
TARGET_EXTRA = 50

# ── Load already-annotated keyframes ────────────────────────────────────────
already_done = set()
if os.path.exists(KEYFRAMES_JSON):
    with open(KEYFRAMES_JSON) as f:
        kf_data = json.load(f)
    keyframes = kf_data.get("keyframes", kf_data) if isinstance(kf_data, dict) else kf_data
    for clip_name in keyframes:
        # clip names are like  <game-slug>_half1_offside3.mp4
        already_done.add(clip_name)

# ── Scan every Labels-v2.json under SOCCERNET_DIR ───────────────────────────
offside_games = []

for root, dirs, files in os.walk(SOCCERNET_DIR):
    if "Labels-v2.json" not in files:
        continue

    filepath  = os.path.join(root, "Labels-v2.json")
    game_name = os.path.relpath(root, SOCCERNET_DIR)

    if LEAGUE_FILTER and not game_name.startswith(LEAGUE_FILTER + os.sep):
        continue

    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception:
        continue

    annotations = data.get("annotations", [])
    offsides    = [a for a in annotations if a.get("label") == "Offside"]
    if not offsides:
        continue

    # Check whether this game's clips are already in the clips directory.
    # We match by the date / team fragment in the folder name.
    game_slug = game_name.replace(os.sep, "_").replace(" ", "_").replace("-", "_")
    clips_present = any(
        game_slug[:30] in f or game_name.split(os.sep)[-1][:20] in f
        for f in os.listdir(CLIPS_DIR)
    ) if os.path.isdir(CLIPS_DIR) else False

    # Count how many of this game's offside events are already annotated.
    game_date_team = game_name.split(os.sep)[-1]  # e.g. "2015-05-17 - ..."
    done_count = sum(
        1 for c in already_done
        if game_date_team.replace(" ", "_")[:20] in c
    )

    offside_games.append({
        "game":         game_name,
        "count":        len(offsides),
        "done":         done_count,
        "new":          len(offsides) - done_count,
        "downloaded":   clips_present,
    })

offside_games.sort(key=lambda x: x["count"], reverse=True)

# ── Summary ──────────────────────────────────────────────────────────────────
scope       = f"league={LEAGUE_FILTER}" if LEAGUE_FILTER else "all leagues"
total_games = len(offside_games)
total_ev    = sum(g["count"] for g in offside_games)
total_done  = sum(g["done"]  for g in offside_games)

print(f"Scope          : {scope}")
print(f"Games found    : {total_games}")
print(f"Total offsides : {total_ev}")
print(f"Already done   : {total_done}")
print(f"Still need     : {TARGET_EXTRA} more events")
print()

# ── Table header ─────────────────────────────────────────────────────────────
header = f"{'#':>3}  {'DL':2}  {'Done':>4}  {'New':>4}  {'Total':>5}  {'Cumul':>6}  Game"
print(header)
print("-" * len(header))

cumul = 0
for rank, g in enumerate(offside_games[:TOP_N], start=1):
    cumul += g["new"]
    dl_flag   = "✓" if g["downloaded"] else " "
    done_str  = str(g["done"]) if g["done"] else "-"
    marker    = "  <-- TARGET MET" if cumul >= TARGET_EXTRA and cumul - g["new"] < TARGET_EXTRA else ""
    print(
        f"{rank:>3}  {dl_flag:2}  {done_str:>4}  {g['new']:>4}  "
        f"{g['count']:>5}  {cumul:>6}  {g['game']}{marker}"
    )

print()
print("Columns: # = rank | DL = already downloaded | Done = events already annotated")
print("         New = unannotated events in this game | Cumul = running new-event total")
print(f"\nDownload the games marked above until Cumul >= {TARGET_EXTRA}.")
