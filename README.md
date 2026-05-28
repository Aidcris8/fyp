# Football Offside Detection — FYP

Source code for Aidan Cristina's B.Sc.(Hons.) Computing Science Final Year
Project (University of Malta, supervisor: Prof. Carl James Debono).

The full write-up — motivation, related work, design rationale, evaluation,
and discussion — lives in [`report/dissertation/main.pdf`](report/dissertation/main.pdf).
This README only covers what you need to **run** the code.

---

## Requirements

- Python 3.11
- `ffmpeg` on the system PATH (used by `extract_clips.py`)
- GNU `make` (drives the pipeline)
- An NVIDIA GPU is recommended for `team_identification.py`
  (YOLO11x-seg) but not required.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

If you do not have a CUDA 12.1 GPU, install `torch` separately from
<https://pytorch.org/get-started/locally/> before the line above.

---

## Running the pipeline

The whole pipeline is orchestrated by the [Makefile](Makefile). Stages are
idempotent — re-running any stage only processes new or stale items.

```bash
# 1. (manual, one-off) populate data/soccernet/ with match videos:
python download_videos.py

# 2. build everything that is currently stale:
make

# 3. open the offside-verdict UI once the build chain is done:
make offside
```

Three of the seven stages are browser-based human-in-the-loop UIs served on
localhost. After `make` launches them, open the URL printed in the terminal:

| Stage | Script | Port |
|------:|--------|------|
| keyframe selection   | `annotate_keyframes.py` | `:5000` |
| pitch-point picking  | `pick_points.py`        | `:5001` |
| offside verdict      | `offside_checker.py`    | `:5002` |

`make help` lists all targets, including `make dry` (preview without
running), `make clean-derived`, and `make clean-all`.

---

## Repository layout

```
.                         pipeline scripts (one per stage)
eval/                     evaluation scripts that back the Chapter 5 numbers
data/                     all generated artefacts; gitignored
docs/                     extended commentary on the trickier modules
report/dissertation/      LaTeX source for the dissertation
Makefile                  pipeline orchestration
requirements.txt          pinned Python dependencies
```

The first run of `team_identification.py` will auto-download
`yolo11x-seg.pt` (~125 MB) into the project root via Ultralytics. No
manual download step is required.

The pipeline DAG, the design choices behind each stage, and a discussion of
the evaluation results are all in the dissertation — please read it for
anything beyond "how do I get this running."
