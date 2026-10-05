# Stage 3 paired comparison — 2026-10-02

This is the original Stage 3 experiment before pedestrian-specific tuning. Current results are in `pedestrian_tuning.md`.

Command: `python main.py --source video.mp4 --risk-mode compare --no-display --csv`

Model/device: `yolo26n.pt` / `cuda:0`. Source: `video.mp4` (29.97 FPS, 2560 × 1440). All three modes used the same 396 frames, 4,553 tracked observations, and 137 ByteTrack IDs. `approaching` was true on 2,018 observations spanning 98 track IDs in every mode.

| Mode | Mean score | Population variance | LOW | MEDIUM | HIGH | CRITICAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: bbox approach only | 0.2074 | 0.1107 | 3,376 (74.15%) | 433 (9.51%) | 362 (7.95%) | 382 (8.39%) |
| B: Stage 2 corridor + class | 0.2374 | 0.0425 | 3,326 (73.05%) | 830 (18.23%) | 397 (8.72%) | 0 |
| C: Stage 3 ego-motion aware | 0.2144 | 0.0319 | 3,479 (76.41%) | 852 (18.71%) | 222 (4.88%) | 0 |

The original Stage 2 baseline is preserved at `video_20261002_115959.csv` and `video_20261002_115959_summary.md`. For this paired rerun, all 4,553 Stage 2 rows matched the original exactly for frame index, track ID, risk score/level, approach rate, and path relevance. No Stage 2 thresholds were retuned. The bbox-only ablation uses a different score definition, so its levels are descriptive rather than calibrated against the other two modes.

HIGH observations span 35 distinct track IDs in Stage 2 and 23 in Stage 3. These are observation/ID counts, not validated hazard detections.

Paired CSVs from the final run: `video_bbox_20261002_132805.csv`, `video_stage2_20261002_132805.csv`, and `video_ego_motion_20261002_132805.csv`.

## Background motion and case inspection

Background estimation met the configured confidence criterion on 395/396 frames (99.7%). The observed state was mainly LATERAL or STABLE; the maximum estimated background expansion was +0.045/s, below the +0.08/s forward-expansion state threshold. The clip therefore does not validate forward camera motion compensation. Cases with raw approach ≥0.5/s and residual approach ≤0.12/s: **0**; there are no example frame indices from this real clip. The forward/static and forward/approaching cases were checked using controlled synthetic tests.

A 100-frame annotated video is `video_ego_motion_20261002_131648.mp4` (2560 × 1440, nominal 29.97 FPS). Frames 10, 40, and 80 were visually inspected, with stills saved as `stage3_inspect_10.jpg`, `stage3_inspect_40.jpg`, and `stage3_inspect_80.jpg`. At frame 40, car #6 grows in the image and has raw/residual rates around +0.83/s, path relevance 0.57, and HIGH risk. Person #1 has a negative approach rate and LOW risk. A short-lived very large positive rate on person #9 at frame 10 remains LOW under the persistence/confidence gate. Several labels overlap in the crowded scene; risk labels should not be read as verified hazards.

## Timing

The 396-frame compare run measured mean feature detection 2.037 ms/frame, optical flow 3.460 ms/frame, affine/global estimation 1.160 ms/frame, and combined global motion 6.657 ms/frame (rolling p95 10.305 ms). Mean corridor/risk-engine time was 0.844 ms/frame for **all three modes together**. The rolling last-120-frame total was 108.01 ms (p95 134.23 ms); whole-run end-to-end FPS was 7.62. This compare run also wrote three CSVs and evaluated three scores per track.

Separate 180-frame, no-display, no-CSV runs gave Stage 2 rolling total 106.63 ms/frame (p95 128.11), risk-engine 0.269 ms/frame, and 6.80 whole-run FPS; Stage 3 rolling total 118.99 ms/frame (p95 149.63), global motion 6.888 ms/frame, risk-engine 0.332 ms/frame, and 6.24 whole-run FPS. The directly measured added CPU logic was about 6.95 ms/frame (6.888 global motion + 0.063 risk-engine difference). Total-frame difference was 12.36 ms, but detector time also differed by 6.46 ms between separate runs; these are timing observations, not a controlled edge-device benchmark. First-frame model startup affects whole-run FPS.

## Limits and interpretation

Both raw box-area growth and background radial expansion are image-space signals. `background_alpha=1` is an experimental scaling, not a physical equality between these quantities. Residual near zero does not imply safety: retained raw closing and corridor relevance still contribute. This detector tracks only six COCO classes, so a wall itself is not scored. Bounding-box occlusion, detector jitter, mixed-depth parallax, low texture, blur, night scenes, large camera rotations, crowds, rolling shutter, and moving-object feature contamination can all mislead the estimate. No collision ground truth, depth, metric velocity, or TTC was evaluated. Lower HIGH counts in this clip are a change in heuristic behavior, **not** evidence of improved safety.
