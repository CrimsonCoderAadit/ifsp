# Assistive navigation: Stages 1–5

A small baseline for video or webcam detection, ByteTrack IDs, temporal box growth, background optical flow, walking-path relevance, heuristic risk events, annotated output, and timing. Stage 4 primarily validates wide-view disparity with WideDepth; the legacy calibrated RobotCar path can attach metric stereo distance. It detects only person, bicycle, car, motorcycle, bus, and truck. **Risk levels are unvalidated image-space heuristics, not collision predictions.** WideDepth's equirectangular disparity is kept in pixels; metric velocity and TTC remain unavailable.

## Setup

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Ultralytics installs a compatible PyTorch build through pip. If you already have a CUDA PyTorch installation, install its matching torchvision version first, then install these requirements without `-U` to preserve that build. On Raspberry Pi, install a PyTorch build compatible with the Pi and its Python version before installing these requirements. The default `yolo26n.pt` weights are downloaded automatically by Ultralytics on first use when available. Change `model_name` in `config/config.yaml` if your installed version does not support them or you have a local `.pt` file.

This repository contains code, tests, configuration, selected experiment CSVs, and summaries. Local input videos, RobotCar/WideDepth dataset imagery, RobotCar distortion LUTs, model weights, and generated preview videos are excluded. To reproduce a video, provide your own input with `--source`; to reproduce the stereo experiments, obtain the corresponding datasets and camera model files from their original sources. The included CSVs are recorded results, so tests and monocular input can be used without downloading either stereo dataset.

## Stage 5: temporal stereo hazard experiment

Run the calibrated, time-ordered RobotCar sample from this directory:

```powershell
python main.py --stereo-dataset sample_small.tar --calibration dataset/robotcar_models --risk-mode stereo_dynamic --csv --save --stereo-debug --max-frames 49
```

The four-panel preview shows left detections and risk, right image, display-only disparity colors, and display-only metric-depth colors. Press `q` to exit; use `--no-display` to save without opening a window. Outputs are `experiments/stage5_robotcar_<timestamp>.csv`, `experiments/stage5_robotcar_summary_<timestamp>.md`, and `outputs/stage5_robotcar_<timestamp>.mp4`. The configured RobotCar hood mask applies only to the raw Bayer/LUT source; it prevents the ego vehicle's own hood from becoming a track hazard.

Stage 5 fits multiple accepted depth observations for each ByteTrack ID. Negative radial relative velocity means distance to the camera is falling; it does **not** prove that the other object moves by itself. TTC is `distance / closing speed` only when metric depth and a confident closing velocity are available. TTC assumes current radial motion continues. It is **not collision probability**: lateral passing, turns, braking, and path changes matter. The configured 4.0, 2.5, and 1.2 second urgency bands are development values, **not validated safety thresholds**. Path relevance reduces urgency for objects away from the estimated walking corridor without forcing it to zero. The CSV records disagreement between 2D box growth and stereo trends. A high or low risk label is still a heuristic, not a verified hazard verdict.

The current WideDepth 20-sample subset has no verified temporal cadence. Its backend explicitly declares no metric conversion for equirectangular object boxes and no temporal sequence. Thus the WideDepth run reports disparity validation only: **no m/s and no TTC**. A future verified sequence could enable pixel-disparity rates; those remain pixels per second, never metres per second. Outside configured stereo overlap, Stage 5 uses the existing ego-motion-aware 2D risk.

Stable copies of the verified 49-frame run are `experiments/stage5_robotcar.csv` and `experiments/stage5_robotcar_summary.md`. The generated `outputs/video_stage5.mp4` is local to the original experiment and is not distributed here.

## Stage 4 primary dataset: WideDepth outdoor train

The run opens a four-panel preview and saves `outputs/widedepth_demo.mp4` when the dataset is available. Press `q` to stop. In PowerShell, play a generated demo with `Invoke-Item .\outputs\widedepth_demo.mp4`. Add `--no-display` for headless processing; the MP4 is still saved. These selected IDs are independent samples, so playback is a labeled contact sheet, not continuous tracked video. Panels show left YOLO detections, right stereo view, sparse released disparity, and SGBM disparity.

The primary Stage 4 command is:

```powershell
python main.py --stereo-backend widedepth --max-frames 20
```

It uses the local 20-sample subset at `data/widedepth_sample/` and the official [WideDepth loader](https://github.com/IlyaInd/WideDepth) cloned at `third_party/WideDepth/`. `python main.py --no-display` also selects WideDepth through `stereo.dataset: widedepth` in the config. The old `--source` monocular route and `--stereo-dataset ... --calibration ...` calibrated route still work. `--widedepth-root` overrides the subset path.

To recreate the bounded subset, first clone the official loader, then review the exact file budget and download:

```powershell
git clone --depth 1 https://github.com/IlyaInd/WideDepth.git third_party/WideDepth
python scripts/download_widedepth_sample.py --count 20 --depth-count 3 --dry-run
python scripts/download_widedepth_sample.py --count 20 --depth-count 3 --max-download-mb 150
```

The download script follows the repository's quickstart method of requesting exact files with `hf_hub_download`. The current subset contains **20 synchronized stereo IDs and 3 matching sparse-depth IDs**, 66 files / 122.4 MB. `data/widedepth_sample/sample_manifest.json` records the exact file names, sizes and dataset revision. The dataset is excluded from version control. Its [Hub dataset card](https://huggingface.co/datasets/IlyaInd/WideDepth-train) states CC BY-NC 4.0 terms.

The official loader requirements are installed separately in `.venv_widedepth` from `third_party/WideDepth/requirements.txt`. This keeps its headless OpenCV `<5` requirement separate from the project's GUI-capable OpenCV installation. The integrated pipeline was also verified with the project's existing runtime. To check the official loader in that isolated environment:

```powershell
python -m venv .venv_widedepth
.\.venv_widedepth\Scripts\python.exe -m pip install -r third_party\WideDepth\requirements.txt
.\.venv_widedepth\Scripts\python.exe -c "import sys; sys.path.insert(0, 'third_party/WideDepth'); from widedepth import WideDepthTrainDataset; print(len(WideDepthTrainDataset('data/widedepth_sample', task='stereo')))"
```

The official stereo task loads `equirect_up`, `equirect_down_virt`, and `disp_equirect`; it returns RGB `uint8` images, float32 disparity **in pixels** (the loader divides encoded values by 100), and a Boolean valid mask (zero is invalid). The provided stereo pair is a **vertical** equirectangular pair, so the optional SGBM experiment rotates downsampled views counter-clockwise before matching. Native 2048×4096 geometry and target values are retained for dataset validation; SGBM operates at quarter resolution for runtime measurement, and comparison errors are rescaled to native pixels. The train release supplies sparse depth in metres for a **different fisheye frame**. Without an official alignment/conversion into the equirectangular stereo frame, WideDepth objects have disparity but `distance_m=None`.

The run writes `experiments/stage4_widedepth.csv`, `experiments/stage4_widedepth_summary.md`, and left/right/ground-truth disparity/SGBM disparity/annotated PNGs in `outputs/`. The CSV has blank track IDs because the selected sample IDs have no verified temporal cadence; ByteTrack persistence and approach risk are not inferred across them. The `StereoOverlapRegion` separates a configurable candidate overlap from left-only and right-only areas; released `valid_mask` determines actual supported pixels. The experiment also compares full available image width with a **central image-column crop proxy** (`25%` to `75%` by default). This is not a calibrated 105-degree angular crop or a model of final camera overlap. WideDepth provides useful wide-view geometry and released disparity but uses vertical stereo, unlike the intended outward-angled horizontal camera pair. Its sparse labels and different rig do not make it a universal accuracy guarantee.

The previous RobotCar sample used perspective stereo and had no released depth/disparity target for an accuracy check: 70.07% valid SGBM disparity and 580.53 ms mean SGBM time at 1280×960 in the saved 49-frame run. WideDepth provides sparse released disparity, so SGBM pixel error can be measured. Its SGBM experiment operates at quarter resolution on 2048×4096 wide-view images; its latency and valid-pixel percentage are **not directly comparable** with the full-resolution RobotCar numbers. The WideDepth sparse fisheye depth is only inspected in its own coordinate frame. WideDepth is closer to the planned cameras in field of view, while the stereo orientation and optical geometry still differ.

## Run

```powershell
python main.py --source path/to/video.mp4
python main.py --source 0
python main.py --source path/to/video.mp4 --save
python main.py --source path/to/video.mp4 --csv
python main.py --source video.mp4 --no-display --save --csv --max-frames 120
python main.py --source path/to/video.mp4 --no-display --max-frames 100
python main.py --source video.mp4 --risk-mode compare --no-display --csv
python main.py --source video.mp4 --risk-mode ego_motion --no-display --save --motion-debug --max-frames 100
python -m unittest discover -s tests -v
```

Press `q` in the window or Ctrl+C to stop. `--save` writes an annotated MP4 into `outputs/` at the source FPS reported by OpenCV, or 30 FPS if the source does not report one. The output is saved at its nominal FPS; processing may run slower than real time. `--no-display` is useful on headless machines. Set `display` and `save_output` in `config/config.yaml` for persistent choices.

`--csv` writes one row per tracked observation. It records the frame index, timestamp, raw and smoothed box area, approach and path scores, direction, risk score and level, and latencies. For files, timestamps are seconds from the video start, based on its FPS. For webcams, they are elapsed wall-clock seconds. `APPROACH EVENTS` and risk event counts in the exit report count track observations, not unique objects or alert transitions.

`--risk-mode` selects `bbox` (box approach alone), `stage2` (the original corridor/class heuristic), `ego_motion` (Stage 3), or `compare` (all three on the same detections and track histories). With `compare --csv`, three mode-named CSVs are written. The saved video in compare mode uses the Stage 3 overlay. The original Stage 2 CSV is included as `experiments/stage2_baseline.csv`; its summary remains in `outputs/video_20261002_115959_summary.md`. Its 4,553 observations have mean score 0.2374, population variance 0.0425, and counts LOW 3,326 / MEDIUM 830 / HIGH 397 / CRITICAL 0. No Stage 2 thresholds were changed for Stage 3.

The original Stage 3 paired comparison and timing notes are stored in `outputs/stage3_comparison.md`. The current pedestrian-tuned comparison is stored in `outputs/pedestrian_tuning.md`.

## Saved experiment results

The two comparable 396-frame result sets are included at stable paths:

```text
experiments/
  stage2_baseline.csv
  stage3_ego_motion.csv
outputs/
  stage3_comparison.md
  pedestrian_tuning.md
```

`stage2_baseline.csv` is an exact copy of the preserved Stage 2 baseline. `stage3_ego_motion.csv` is an exact copy of the pedestrian-tuned Stage 3 run. The original generated videos are excluded from this repository. They were rendered from a local `video.mp4` and contained 396 frames at 2560 × 1440 and nominal 29.97 FPS. Use the commands above with your own source video to generate new annotated output.

The preview shrinks large frames to fit within `display_max_width` and `display_max_height` while preserving the full scene and aspect ratio. Adjust those two values for your monitor. Saved video keeps the input resolution.

## Measurements

Acquisition measures `read()`. Inference measures `YOLO.track()`, including ByteTrack. Total processing measures read, tracking, drawing, and any display/write for each completed frame. The overlay shows source FPS, rolling end-to-end processing FPS, current detector/tracker latency, and active relevant tracks. The FPS overlay uses the preceding completed frames, so it is blank (`0.0`) on the first frame. The exit report shows end-to-end average FPS for the run, plus rolling acquisition and total frame means and rolling mean and p95 inference latency (last 120 frames). These measurements include first-frame startup work. A persistent ID is reported only when the same ID appears in consecutive frames.

`DetectorTracker.track()` returns `Track` dataclasses with IDs, class, confidence, box corners, center, bottom center, width, height, and area. Detections without tracker IDs are omitted until ByteTrack assigns an ID.

## Stage 2 heuristic

Each ByteTrack ID has a bounded history (15 observations by default) that expires after a configurable timeout. Box area and center use an exponential moving average. Approach rate is the change in log smoothed area per video second over five observations, followed by light rate smoothing. A positive rate must persist across at least three intervals before `approaching` becomes true. Negative rates indicate apparent recession. This does not distinguish object motion from camera motion.

The normalized trapezoid in `config/config.yaml` approximates a forward walking corridor. Each object's bottom-center gives continuous path relevance; its horizontally smoothed center gives LEFT, CENTER, or RIGHT image position. Risk combines path relevance, apparent approach, detection confidence, and a small class prior. HIGH and CRITICAL require sustained approach evidence and sufficient confidence. A stationary vehicle outside the corridor should stay LOW. Tune the YAML parameters for a different camera view; the current values are an initial baseline, not calibrated safety thresholds.

The risk report separates history, approach, and corridor/risk computation time. These logic timings exclude visualization, file encoding, and CSV writing; end-to-end frame time includes them. CSV `total_frame_latency_ms` is sampled just before its rows are written, while the console's total frame metric also includes CSV writing. Mean and p95 total-frame latency in the exit report use the last 120 frames, while FPS uses all completed frames.

## Stage 3 motion estimate

At 480-pixel analysis width, OpenCV tracks up to 140 background corners with pyramidal Lucas–Kanade flow. The feature mask excludes detected objects with padding in both adjacent frames. Corners are refreshed every 10 frames or when too few remain. RANSAC fits a partial affine transform; the median inlier radial log change gives background expansion per second after compensating image-center translation. The result also reports image translation, rotation, inlier ratio, confidence, and a coarse motion state. It is **not** metric camera speed or full visual odometry. When features or confidence are insufficient, Stage 3 scoring falls back to Stage 2.

For each tracked object, `raw_approach_rate` is the Stage 2 smoothed log bounding-box-area growth per second. `residual_approach_rate = raw_approach_rate - background_alpha × background_expansion_rate` when the background estimate is reliable. Both rates are logged. `background_alpha` defaults to 1.0 and is experimental: area growth and radial expansion are different image quantities, and a single scene-wide expansion cannot describe all depths. A low residual never declares an object safe; the raw closing signal still contributes, especially inside the corridor.

Stage 3 starts from the Stage 2 base score, gates and thresholds. It blends a retained raw motion term with a confidence-weighted positive residual term. `raw_retention_floor`, `raw_retention_path_weight`, and `residual_weight` in `config/config.yaml` control that blend. The default residual weight is 0.2 so stable-background objects stay close to their Stage 2 scores, while background-shared growth is somewhat reduced outside the path. Scores and levels are heuristic; the three ablation modes have different score definitions and are useful for behavior comparison, not calibrated safety probabilities.

The current `ego_motion` mode also applies a pedestrian-specific, continuous path gate to motion risk: `person_motion_scale × (person_path_floor + (1 - person_path_floor) × path_relevance)`. Its pedestrian path base term is scaled by `person_path_weight_scale`. This reduces frequent MEDIUM/HIGH pedestrian labels in crowded footage while retaining HIGH for a very fast, central pedestrian approach. Vehicle scoring and the historical `stage2` mode are unchanged. These three settings are in the `ego_risk` block; they are heuristic priorities, not measured collision likelihood.

`--motion-debug` displays the motion state, expansion, feature count, inlier ratio, confidence, and at most 12 background flow vectors. Stage 3 CSV rows include raw and residual approach, ego-motion confidence/state, feature and inlier counts, and feature/flow/global-estimation latencies. The run report gives mode distributions plus mean and rolling p95 motion and total-frame latencies. Total-frame latency includes drawing, optional writing/display, and inference; GPU and I/O variation can dominate small CPU overhead differences.

Known limits: sparse texture, blur, night scenes, large rotation, rolling shutter, crowds covering the background, feature contamination from moving objects, parallax at mixed depths, and box changes from occlusion or detector jitter. The motion mask only knows the six configured detector classes. Static walls are not tracked risk objects in this stage. No metric velocity, TTC, SLAM, or collision validation is included.

## Stage 4 legacy calibrated stereo depth: RobotCar

The existing `--source` video/webcam path still works. Stereo mode accepts either a single pair or a sequence of strictly matched image IDs. The supplied `sample_small.tar` is an **Oxford RobotCar** sample with 49 wide-stereo pairs. Its images are raw Bayer and require the corresponding [official camera models and distortion LUTs](https://github.com/ori-mrg/robotcar-dataset-sdk/tree/master/models). The code reads matching pairs directly from the archive, demosaics them, and applies the SDK LUTs. No KITTI data is needed. The [RobotCar documentation](https://robotcar-dataset.robots.ox.ac.uk/documentation/) describes the raw image format and microsecond timestamp IDs.

From this directory, run the independent single-pair sanity check:

```powershell
python main.py --stereo-left dataset/robotcar_pair/left.png --stereo-right dataset/robotcar_pair/right.png --calibration dataset/robotcar_models --stereo-debug --no-display
```

It reports dimensions, focal length, baseline, valid disparity fraction, median disparity and scene depth; it saves disparity, depth and horizontal epipolar-line images in `outputs/`. No detector is loaded. For the full RobotCar sample with tracking, risk, CSV and an annotated four-panel video:

```powershell
python main.py --stereo-dataset sample_small.tar --calibration dataset/robotcar_models --risk-mode ego_motion --csv --save --stereo-debug --no-display
```

Remove `--stereo-debug` for a single annotated left-camera view. Omit `--no-display` to open a preview. `--risk-mode stage2` preserves the older risk baseline while adding distance to each event; stereo distance does not alter the Stage 2/3 risk score. The CSV is saved under `experiments/stage4_stereo_<timestamp>.csv`, the video under `outputs/stage4_stereo_<timestamp>.mp4`. Each object row includes raw/smoothed distance, confidence, valid ROI pixel ratio, existing approach and risk fields, and stereo/total timing. `RiskEvent.distance_m` is set only for a valid stereo estimate. `relative_velocity_ms` and `ttc_s` remain `None`.

For another image sequence, use matching `left/` and `right/` files with identical stems:

```powershell
python main.py --stereo-dataset path/to/sequence --stereo-left-subdir left --stereo-right-subdir right --calibration path/to/calibration.npz --csv --no-display
```

The default subdirectories are `image_02/data` and `image_03/data` for already rectified KITTI color pairs. KITTI calibration accepts `P_rect_02`/`P_rect_03` or `P2`/`P3` projection matrices. Generic `.npz` calibration requires `K1`, `K2`, `D1`, `D2`, `R`, `T`, `translation_unit` (`m` or `mm`), and `rectified` (boolean). It may also contain `P1`, `P2`, `Q`, and `image_size`. Generic raw pairs are rectified with OpenCV; already rectified KITTI images are not remapped. For a generic calibrated camera, verify left/right order and horizontal epipolar alignment before treating its disparity as metric depth.

StereoSGBM returns disparity in pixel units after OpenCV's fixed-point division by 16. Metric depth uses `focal_px × baseline_m / (disparity - principal_point_offset)`. Invalid/zero/negative disparities, nonfinite depths and values outside the configured 0.5–80 m range are unavailable. Each object's depth comes from a trimmed median in an inset lower-middle box ROI; the valid-pixel ratio and median absolute deviation determine confidence. A short per-ID history smooths valid depths and rejects isolated large jumps. These checks cannot prove depth accuracy. Reflective surfaces, textureless regions, occlusion boundaries, background within detection boxes and calibration errors can still yield wrong readings. This stage does not use depth for TTC, speed, or risk thresholds.

For sequence processing, OpenCV threads are restored after loading Ultralytics, which otherwise sets them to one. `stereo.opencv_threads: null` chooses up to eight logical CPUs; set a positive count to benchmark a Pi or another edge device. Full-resolution SGBM can still be slower than real time. Output MP4 uses the source's nominal FPS even when processing takes longer.

## Bus Finder integration (Review 1)

The Rider path reads a bus route from Person 1's `DetectorTracker` output. `transit.BusReader` never creates a second YOLO. Video requires two matching OCR results in the last three reads of a track, and confirmed buses expire after 30 seconds. Single-image uploads use `DetectorTracker.detect()` on the same model instance. The web app has one Rider page and four endpoints: `/ws`, `/api/image`, `/api/current_bus`, and `/api/health` (plus `/` and static assets).

Place the repos side by side. From `Camera-depth-risk-system-main/`, on macOS or Linux:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/download_models.py --only vosk piper rapidocr gtfs
python -m unittest discover -s tests
python -m pytest tests/test_bus_reader.py tests/test_route_matcher.py tests/test_gtfs_loader.py tests/test_scheduler.py
python main.py --source data/real/own/YOUR_VIDEO.mp4 --bus-reader --listen --web --save --csv --no-display
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/download_models.py --only vosk piper rapidocr gtfs
python -m unittest discover -s tests
python -m pytest tests/test_bus_reader.py tests/test_route_matcher.py tests/test_gtfs_loader.py tests/test_scheduler.py
python main.py --source data\real\own\YOUR_VIDEO.mp4 --bus-reader --listen --web --save --csv --no-display
```

Open http://127.0.0.1:8001. `--listen` needs an input microphone; omit it for upload-only use. For the stand-alone page run `python -m uvicorn app.server:app --host 127.0.0.1 --port 8001`. Supply `yolo26n.pt` in this directory or let Ultralytics download it once. The setup script provisions Vosk, Piper, RapidOCR, and the GTFS feed into their configured locations. This checkout currently shares `models/` with the adjacent `voice-transit/models` directory by symlink; the same script provisions a local `models/` directory when moved elsewhere.

Set `ROBOFLOW_API_KEY=...` in `.env` for the optional Roboflow datasets and install the optional `roboflow` package. Run `python scripts/fetch_datasets.py` and `python scripts/label_tool.py`, confirm labels, then `python scripts/evaluate_ocr.py` on dev and `python scripts/evaluate_ocr.py --test` once. See `data/real/README.md`, `DEMO_SCRIPT.md`, and `KNOWN_FAILURES.md` for data rights, the demo sequence, and current evidence limits. The responsive Rider screenshots from real Commons inputs are in `docs/rider-*.png`.

The pinned `requirements.txt` was installed successfully in a clean Python 3.12 macOS arm64 venv on 2026-10-05. Python 3.10/Windows compatibility is based on available package wheels; it has not been run on a teammate's Windows machine. The macOS test host had no audio output device, so the CLI scheduler used a silent sink while the WebSocket speech response still produced a valid Piper WAV.
