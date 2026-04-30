import os
import json
import subprocess
import argparse

SOCCERNET_DIR = "data/soccernet"
OUTPUT_DIR = "data/clips"
CLIP_BEFORE = 10
CLIP_AFTER = 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default=None,
                    help="Game path relative to data/soccernet, e.g. "
                         "'england_epl/2016-2017/2017-01-15 - 19-00 Manchester United 1 - 1 Liverpool'. "
                         "If omitted, all downloaded games are processed.")
args = parser.parse_args()

if args.game:
    GAMES = [args.game]
    print(f"Processing single game: {args.game}")
else:
    # Auto-detect all downloaded games by scanning for Labels-v2.json
    GAMES = []
    for league in os.listdir(SOCCERNET_DIR):
        league_dir = os.path.join(SOCCERNET_DIR, league)
        if not os.path.isdir(league_dir):
            continue
        for season in os.listdir(league_dir):
            season_dir = os.path.join(league_dir, season)
            if not os.path.isdir(season_dir):
                continue
            for match in os.listdir(season_dir):
                match_dir = os.path.join(season_dir, match)
                labels_path = os.path.join(match_dir, "Labels-v2.json")
                if os.path.exists(labels_path):
                    GAMES.append(f"{league}/{season}/{match}")
    print(f"Found {len(GAMES)} games:")
    for g in GAMES:
        print(f"  {g}")

for game in GAMES:
    game_dir = os.path.join(SOCCERNET_DIR, game)
    labels_path = os.path.join(game_dir, "Labels-v2.json")

    with open(labels_path, "r") as f:
        data = json.load(f)

    offsides = [a for a in data["annotations"] if a.get("label") == "Offside"]
    print(f"\n{game}")
    print(f"Found {len(offsides)} offside events")

    for i, event in enumerate(offsides):
        half = int(event["gameTime"].split(" - ")[0])
        position_ms = int(event["position"])
        position_sec = position_ms / 1000.0

        start = max(0, position_sec - CLIP_BEFORE)
        duration = CLIP_BEFORE + CLIP_AFTER

        video_file = os.path.join(game_dir, f"{half}_720p.mkv")
        if not os.path.exists(video_file):
            print(f"  Video not found: {video_file}")
            continue

        game_clean = game.replace("/", "_").replace(" ", "_")
        output_file = os.path.join(OUTPUT_DIR, f"{game_clean}_half{half}_offside{i+1}.mp4")

        if os.path.exists(output_file):
            print(f"  Already exists, skipping: {os.path.basename(output_file)}")
            continue

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_file,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-loglevel", "error",
            output_file
        ]

        print(f"  Extracting offside {i+1} at {position_sec:.1f}s (half {half})...")
        subprocess.run(cmd)
        print(f"  Saved: {os.path.basename(output_file)}")

print(f"\nAll clips saved to {OUTPUT_DIR}")