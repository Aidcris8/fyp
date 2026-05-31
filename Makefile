# ============================================================
# FYP pipeline — football offside detection
#
# Run `make` to bring everything up to date.
# Run `make <target>` to build one stage and its prerequisites.
# Run `make help` to see all targets.
#
# Note: scan_offsides.py and download_videos.py are NOT part of the
# auto-build chain — they involve large downloads and dataset-selection
# choices, so we keep them as one-off manual steps. Run them by hand once per
# dataset change, then `make` handles everything from clip extraction onward.
# ============================================================

PYTHON := python

# ---- sentinel files (stamps) ------------------------------------------------
# Stages that emit many files (clips/, keyframes/) can't be tracked
# file-by-file, so we use a single timestamp per directory. Touched at the end
# of the stage's recipe.
CLIPS_STAMP     := data/clips/.stamp
KEYFRAMES_STAMP := data/keyframes/.stamp

# Marker touched by download_videos.py after each download run. When it is
# newer than the clips stamp, Make knows new game data arrived and re-runs
# clip extraction (which is incremental — existing clips are skipped).
DOWNLOAD_STAMP  := data/soccernet/.download_stamp

# ---- canonical output files (Make tracks these directly) -------------------
KEYFRAMES_JSON  := data/keyframes.json
DETECTIONS      := data/player_detections.json
POINTS          := data/homography_points.json
MATRICES        := data/homography_matrices.json

# ============================================================
# Default target
# ============================================================
.PHONY: all
all: $(MATRICES) $(DETECTIONS)
	@echo ""
	@echo "Pipeline up to date."
	@echo "  Run 'make offside' to launch the verdict UI."

# ============================================================
# Pipeline stages
# ============================================================

# ---- Stage 1: extract clips -------------------------------------------------
# Re-runs when extract_clips.py is edited OR download_videos.py was run since
# the last clip extraction. Clip extraction is incremental (skips existing
# clips), so only newly downloaded game videos are touched.
$(CLIPS_STAMP): extract_clips.py $(DOWNLOAD_STAMP)
	@mkdir -p data/clips
	$(PYTHON) extract_clips.py
	@touch $@
	@echo "[stamp] $(CLIPS_STAMP)"

# ---- Stage 2: annotate keyframes (INTERACTIVE — browser UI on :5000) -------
# Opens the annotator. You click through clips, Ctrl+C when done.
$(KEYFRAMES_JSON): $(CLIPS_STAMP) annotate_keyframes.py
	$(PYTHON) annotate_keyframes.py

# ---- Stage 3: export selected keyframes as JPGs ----------------------------
$(KEYFRAMES_STAMP): $(KEYFRAMES_JSON) export_keyframes.py
	@mkdir -p data/keyframes
	$(PYTHON) export_keyframes.py
	@touch $@
	@echo "[stamp] $(KEYFRAMES_STAMP)"

# ---- Stage 4: detect players + identify teams ------------------------------
$(DETECTIONS): $(KEYFRAMES_STAMP) team_identification.py
	$(PYTHON) team_identification.py

# ---- Stage 5: pick homography points (INTERACTIVE — browser UI on :5001) ---
$(POINTS): $(KEYFRAMES_STAMP) pick_points.py
	$(PYTHON) pick_points.py

# ---- Stage 6: compute homography matrices ----------------------------------
$(MATRICES): $(POINTS) compute_homography.py
	$(PYTHON) compute_homography.py

# ---- Stage 7: offside checker (INTERACTIVE — browser UI on :5002) ----------
# Manual target — you run this when you want to inspect verdicts. Make builds
# any stale prerequisites first.
.PHONY: offside
offside: $(MATRICES) $(DETECTIONS)
	$(PYTHON) offside_checker.py

# ============================================================
# Convenience targets
# ============================================================

# ---- mark existing artefacts as up-to-date ---------------------------------
# Use after a fresh clone (or after this Makefile is first added) so Make
# doesn't try to rebuild things that already exist on disk.
.PHONY: init-stamps
init-stamps:
	@if [ -d data/clips ] && [ -n "$$(ls data/clips/*.mp4 2>/dev/null)" ]; then \
		touch $(CLIPS_STAMP); echo "[stamp] $(CLIPS_STAMP)"; fi
	@if [ -d data/keyframes ] && [ -n "$$(ls data/keyframes/*.jpg 2>/dev/null)" ]; then \
		touch $(KEYFRAMES_STAMP); echo "[stamp] $(KEYFRAMES_STAMP)"; fi
	@for f in $(KEYFRAMES_JSON) $(DETECTIONS) $(POINTS) $(MATRICES); do \
		if [ -f $$f ]; then touch $$f; echo "[touch] $$f"; fi \
	done
	@echo "Stamps initialised. 'make' will only re-run stages that are genuinely stale."

# ---- inspect what Make would do without doing it ---------------------------
.PHONY: dry
dry:
	@$(MAKE) -n all

# ---- clean everything that can be regenerated ------------------------------
.PHONY: clean-derived
clean-derived:
	rm -rf data/team_identification data/homography_validation data/offside_results
	rm -f $(DETECTIONS) $(MATRICES) data/offside_verdicts.json data/offside_skipped.json
	@echo "Cleaned derived outputs. Next 'make' will rebuild team-id and homography only."

# ---- nuclear: also wipes clips, keyframes, manual annotations -------------
.PHONY: clean-all
clean-all: clean-derived
	rm -rf data/clips data/keyframes
	rm -f $(KEYFRAMES_JSON) $(POINTS) $(CLIPS_STAMP) data/homography_skipped.json
	@echo "Cleaned ALL pipeline artefacts. You will need to re-annotate everything."

# ---- help -------------------------------------------------------------------
.PHONY: help
help:
	@echo "FYP pipeline targets:"
	@echo "  make                  — build everything that's stale (default)"
	@echo "  make offside          — launch offside-checker UI (builds prereqs first)"
	@echo "  make dry              — show what would run without running"
	@echo "  make init-stamps      — mark existing clips/keyframes as up-to-date"
	@echo "  make clean-derived    — wipe team-id, homography, offside outputs"
	@echo "  make clean-all        — wipe ALL pipeline outputs (DESTRUCTIVE)"
	@echo "  make help             — this message"
	@echo ""
	@echo "Individual stage targets (build only that stage and its prereqs):"
	@echo "  make $(CLIPS_STAMP)"
	@echo "  make $(KEYFRAMES_JSON)"
	@echo "  make $(KEYFRAMES_STAMP)"
	@echo "  make $(DETECTIONS)"
	@echo "  make $(POINTS)"
	@echo "  make $(MATRICES)"
