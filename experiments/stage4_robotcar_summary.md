# Stage 4 RobotCar stereo baseline — 2026-10-02

Command (from `assistive_nav/`):

```powershell
python main.py --stereo-dataset sample_small.tar --calibration dataset/robotcar_models --risk-mode ego_motion --no-display --csv --save --stereo-debug --max-frames 49
```

Input: Oxford RobotCar `sample_small.tar`, 49 matched wide-stereo Bayer image pairs, 1280 × 960. The corresponding official SDK camera models and LUTs are in `dataset/robotcar_models/`. The LUT remapped each pair to a rectified image. Focal length: 983.044 px; metric baseline: 0.239983 m. Frame IDs matched exactly, and the rectification image was visually inspected for horizontal epipolar alignment. No ground-truth depth was available, so these results do not establish metric accuracy.

| Measure | Result |
|---|---:|
| Frames / tracked observations | 49 / 246 |
| Unique tracked IDs | 22 |
| IDs with valid depth at least once | 22 |
| Valid smoothed-depth observations | 214 / 246 |
| Mean depth confidence on valid observations | 0.767 |
| Mean valid disparity | 70.07% |
| Mean invalid depth | 36.22% |
| Median per-frame scene depth | 15.99 m |
| Mean / p95 SGBM latency | 580.53 / 715.47 ms |
| Mean / p95 total frame latency | 1533.20 / 1578.94 ms |
| End-to-end FPS | 0.65 |

Selected object histories from the CSV (median accepted depth and mean rolling temporal variance):

| Track ID | Class | Frames seen | Median depth | Mean rolling variance |
|---|---|---:|---:|---:|
| 2 | car | 29 | 12.06 m | 1.36 m² |
| 5 | car | 46 | 13.22 m | 1.26 m² |
| 6 | car | 13 | 8.46 m | 0.95 m² |
| 21 | person | 26 | 23.40 m | 1.24 m² |
| 4 | car label on ego hood | 49 | 1.92 m | 0.00 m² |

The final saved video was opened and frame 20 was manually inspected. Parked cars had distinct positive depths and the output frame kept all four panels. Track 4 is a known semantic error: YOLO detects the camera vehicle's hood as a car. Its depth is geometrically plausible but does not describe an external hazard. Far small detections also approach the configured 80 m limit; reflective/low-texture and occluded pixels remain unreliable. Object depth uses box interiors, so background contamination remains possible. No LiDAR or disparity ground truth was used.

The initial sequence run inherited Ultralytics' one-thread OpenCV setting: mean SGBM 1168.28 ms, mean total 2257.59 ms, 0.44 FPS. Restoring a bounded OpenCV thread count reduced mean SGBM to 580.53 ms and raised end-to-end speed to 0.65 FPS. The first YOLO inference took 10659 ms; subsequent frame inference averaged 135.94 ms. Full-resolution stereo is still slower than real time on this machine. The output video has the nominal source rate of 16 FPS; this is playback rate, not live throughput.

Artifacts:

- `experiments/stage4_stereo_20261002_201517.csv`
- `outputs/stage4_stereo_20261002_201517.mp4`
- `outputs/stage4_epipolar_20261002_201517.png`
- `outputs/stereo_pair_20261002_201217_disparity.png`
- `outputs/stereo_pair_20261002_201217_depth.png`

Static pair check: 69.06% valid disparity, 13.88 px median disparity, 14.80 m median scene depth. These are sanity checks, not an accuracy benchmark. The Stage 1–3 risk heuristics were unchanged; stereo only populates `RiskEvent.distance_m` when a smoothed estimate is valid. Velocity and TTC remain unset.
