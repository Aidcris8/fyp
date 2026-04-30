import os
import json
from collections import defaultdict

SOCCERNET_DIR = "data/soccernet"
# Filter to a single league. SoccerNet directory names use lowercase slugs:
#   england_epl, spain_laliga, italy_serie-a, germany_bundesliga,
#   france_ligue-1, europe_uefa-champions-league
# Set LEAGUE_FILTER = None to scan every league.
LEAGUE_FILTER = "england_epl"

offside_games = []

#go through all game folders
for root, dirs, files in os.walk(SOCCERNET_DIR):
    for file in files:
        if file == "Labels-v2.json":
            filepath = os.path.join(root, file)
            game_name = os.path.relpath(root, SOCCERNET_DIR)

            # Skip games not in the target league
            if LEAGUE_FILTER and not game_name.startswith(LEAGUE_FILTER + os.sep):
                continue

            with open(filepath, "r") as f:
                try:
                    data = json.load(f)
                except:
                    continue

            annotations = data.get("annotations", [])
            offsides = [a for a in annotations if a.get("label") == "Offside"]

            if offsides:
                offside_games.append({
                    "game": game_name,
                    "count": len(offsides),
                    "events": offsides
                })

#sort by highest number of offside events
offside_games.sort(key=lambda x: x["count"], reverse=True)

scope = f"in {LEAGUE_FILTER}" if LEAGUE_FILTER else "(all leagues)"
print(f"Total games with offside events {scope}: {len(offside_games)}")
print(f"Total offside events: {sum(g['count'] for g in offside_games)}")
print(f"\nTop 40 games by offside count {scope}:")
for g in offside_games[:40]:
    print(f"  {g['count']} offsides -- {g['game']}")