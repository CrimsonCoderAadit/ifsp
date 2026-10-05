# Pedestrian risk tuning — 2026-10-02

Command: `python main.py --source video.mp4 --risk-mode compare --no-display --csv`

This run used the same 396 frames and 4,553 tracked observations as the earlier Stage 3 comparison. The current `ego_motion` mode scales only pedestrian path and motion contributions. The pedestrian motion factor is `0.80 × (0.35 + 0.65 × path_relevance)`; the pedestrian base path term uses `0.80 ×` its previous weight. The path relevance comes from the pedestrian bounding box bottom-center in the normalized trapezoid. This is a continuous gate, not a hard risk cap. Vehicles and `stage2` scoring are unchanged.

| Score set | Mean | Population variance | Population SD | LOW | MEDIUM | HIGH | CRITICAL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stored Stage 2 baseline | 0.2374 | 0.0425 | 0.2062 | 3,326 | 830 | 397 | 0 |
| Original Stage 3 | 0.2144 | 0.0319 | 0.1787 | 3,479 | 852 | 222 | 0 |
| Pedestrian-tuned Stage 3 | 0.1725 | 0.0201 | 0.1417 | 3,857 | 631 | 65 | 0 |

| Class and mode | LOW | MEDIUM | HIGH |
| --- | ---: | ---: | ---: |
| Person, original Stage 3 | 2,903 | 633 | 191 |
| Person, tuned Stage 3 | 3,281 | 412 | 34 |
| Car, original and tuned Stage 3 | 532 | 216 | 31 |

All 4,553 Stage 2 observations still match the stored original row-for-row on risk score and level. Every car risk score and level matches the previous Stage 3 run row-for-row. The remaining 34 pedestrian HIGH observations span seven track IDs and have path relevance 0.84–1.00; for example, person #3 at frame 125 has path relevance 0.90, raw approach +1.13/s, and HIGH risk. A pedestrian outside the corridor remains LOW even with large box growth in synthetic tests. This is a deliberate heuristic behavior change, not evidence of better safety performance.

Current CSVs: `video_bbox_20261002_185849.csv`, `video_stage2_20261002_185849.csv`, and `video_ego_motion_20261002_185849.csv`. Previous Stage 3 CSV: `video_ego_motion_20261002_132805.csv`. Original Stage 2 baseline: `video_20261002_115959.csv`.

Stable experiment copies: `../experiments/stage2_baseline.csv` and `../experiments/stage3_ego_motion.csv`. Full-length annotated videos: `video_stage2.mp4` and `video_ego_motion_20261002_190636.mp4`.

Updated annotated preview: `video_ego_motion_20261002_190041.mp4` (135 frames, source resolution 2560 × 1440 and nominal 29.97 FPS). Frames 40 and 125 were visually inspected and saved as `pedestrian_tuning_frame_40.jpg` and `pedestrian_tuning_frame_125.jpg`. Car #6 remains HIGH at frame 40; person #3 is HIGH at frame 125. Crowded labels still overlap, and a growing box does not prove physical approach.

The compare run measured mean global motion 5.173 ms/frame, rolling p95 7.976 ms, rolling total frame latency 84.13 ms, rolling p95 113.49 ms, and whole-run end-to-end 8.91 FPS. These numbers include three score modes and three CSVs and vary with detector workload. Background motion met the configured reliability criterion on 99.7% of frames. This clip has no raw-high/residual-low cases and does not validate forward camera motion handling.
