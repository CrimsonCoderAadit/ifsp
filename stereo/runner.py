"""Stage 4 pair sanity check and stereo sequence pipeline."""

from collections import Counter, defaultdict, deque
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from input.stereo_source import StereoSource
from motion.ego_motion import EgoMotionEstimator
from perception.track_history import TrackHistoryStore
from risk.approach import ApproachEstimator
from risk.corridor import Corridor
from risk.risk_engine import RiskEngine
from stereo.calibration import load_calibration
from stereo.depth_backend import RobotCarPinholeBackend
from stereo.object_disparity import ObjectDisparity, ObjectDisparityExtractor
from stereo.object_depth import DepthHistoryStore, DepthObservation, ObjectDepthExtractor
from stereo.overlap import StereoOverlapRegion
from stereo.rectification import StereoRectifier, epipolar_debug_image
from stereo.sgbm import StereoMatcher, depth_visualization, disparity_visualization
from utils.drawing import draw_overlay, fit_for_display
from utils.event_logger import StereoEventLogger
from utils.score_summary import ScoreSummary
from stereo.stage5_logic import Stage5Processor


class TimingSummary:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, key: str, seconds: float) -> None:
        self.values[key].append(seconds * 1000)

    def mean(self, key: str) -> float:
        values = self.values[key]
        return float(np.mean(values)) if values else 0.0

    def p95(self, key: str) -> float:
        values = self.values[key]
        return float(np.percentile(values, 95)) if values else 0.0


def _label_tile(image: np.ndarray, label: str) -> np.ndarray:
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 470), 34), (20, 20, 20), -1)
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _depth_overlay(frame: np.ndarray, tracks, depths: dict[int, DepthObservation]) -> np.ndarray:
    for track in tracks:
        depth = depths[track.track_id]
        text = (f"Depth {depth.smoothed_depth_m:.1f} m  conf {depth.depth_confidence:.2f}"
                if depth.smoothed_depth_m is not None else "Depth unavailable")
        x1, _y1, _x2, y2 = track.bbox_xyxy
        x = max(2, min(round(x1), frame.shape[1] - 300))
        y = max(25, min(round(y2) - 5, frame.shape[0] - 8))
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _four_panel(left: np.ndarray, right: np.ndarray, disparity: np.ndarray, depth: np.ndarray) -> np.ndarray:
    tile_size = (800, 600)
    panels = [
        _label_tile(cv2.resize(left, tile_size), "LEFT / DETECTIONS + DEPTH"),
        _label_tile(cv2.resize(right, tile_size), "RIGHT CAMERA"),
        _label_tile(cv2.resize(disparity_visualization(disparity), tile_size), "DISPARITY (DISPLAY ONLY)"),
        _label_tile(cv2.resize(depth_visualization(depth), tile_size), "METRIC DEPTH (DISPLAY ONLY)"),
    ]
    return np.concatenate([np.concatenate(panels[:2], axis=1),
                           np.concatenate(panels[2:], axis=1)], axis=0)


def _basic_stats(disparity: np.ndarray, depth: np.ndarray) -> tuple[float, float, float, float]:
    valid_d = disparity[np.isfinite(disparity)]
    valid_z = depth[np.isfinite(depth)]
    return (
        100 * valid_d.size / disparity.size,
        float(np.median(valid_d)) if valid_d.size else float("nan"),
        100 * valid_z.size / depth.size,
        float(np.median(valid_z)) if valid_z.size else float("nan"),
    )


def _source(args, calibration) -> StereoSource:
    if args.stereo_dataset:
        if args.stereo_left or args.stereo_right:
            raise ValueError("choose a dataset or a single stereo pair")
        return StereoSource.from_dataset(
            args.stereo_dataset, left_subdir=args.stereo_left_subdir,
            right_subdir=args.stereo_right_subdir,
        )
    if not args.stereo_left or not args.stereo_right:
        raise ValueError("single pair requires both --stereo-left and --stereo-right")
    return StereoSource(left_file=args.stereo_left, right_file=args.stereo_right,
                        raw_bayer=calibration.robotcar_models_dir is not None)


def _is_robotcar_hood(track, width: int, height: int, source: StereoSource, calibration, config: dict) -> bool:
    settings = config.get("robotcar_ignore_hood", {})
    if not (settings.get("enabled") and source.raw_bayer and calibration.robotcar_models_dir is not None):
        return False
    x1, y1, x2, _y2 = track.bbox_xyxy
    return (track.class_name == "car" and y1 / height >= settings["min_top_y"]
            and (x2 - x1) / width >= settings["min_width_ratio"])


def run_stereo(args, config: dict, root: Path) -> int:
    source = None
    writer = None
    logger = None
    timing = TimingSummary()
    valid_percentages: list[float] = []
    valid_depth_percentages: list[float] = []
    median_disparities: list[float] = []
    median_depths: list[float] = []
    depth_confidences: list[float] = []
    valid_track_ids: set[int] = set()
    tracked_ids: set[int] = set()
    valid_observations = 0
    frames = 0
    calibration = None
    rectification_status = "unavailable"
    output_paths: list[Path] = []
    warning = None
    mode = "pair"
    stage5_summaries = {name: ScoreSummary() for name in ("stage2", "ego_motion", "stereo_dynamic")}
    stage5_counts: Counter[str] = Counter()
    stage5_examples: dict[int, deque[tuple[str, float | None, float | None, float | None, str, str]]] = defaultdict(lambda: deque(maxlen=20))
    stage5_disagreements: list[tuple[int, int, str]] = []
    velocity_confidences: list[float] = []
    ttc_confidences: list[float] = []
    pair_only = not bool(args.stereo_dataset)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        calibration = load_calibration(args.calibration)
        source = _source(args, calibration)
        if source.raw_bayer and calibration.robotcar_models_dir is None:
            raise ValueError("RobotCar raw Bayer frames require official RobotCar LUT models")
        if calibration.robotcar_models_dir is not None and not source.raw_bayer:
            raise ValueError("RobotCar LUT calibration requires raw Bayer stereo input")
        rectifier = StereoRectifier(calibration)
        matcher = StereoMatcher(**config["stereo"]["sgbm"])
        stereo_cfg = config["stereo"]
        display = bool(config["display"]) and not args.no_display
        output_dir = root / "outputs"
        output_dir.mkdir(exist_ok=True)

        if not pair_only:
            from perception.detector_tracker import DetectorTracker
            mode = args.risk_mode or config["risk_mode"]
            if mode == "compare":
                raise ValueError("stereo sequence supports bbox, stage2, ego_motion, or stereo_dynamic risk mode")
            if mode == "stereo_dynamic" and not source.raw_bayer:
                dataset_path = Path(args.stereo_dataset)
                if not dataset_path.is_dir() or not (dataset_path / "timestamps.txt").is_file():
                    raise ValueError("stereo_dynamic generic sequences require explicit timestamps.txt; synthetic FPS is not sufficient for metric velocity")
            detector = DetectorTracker(
                config["model_name"], config["allowed_classes"],
                config["confidence_threshold"], config["iou_threshold"],
                config["image_size"], config["tracker"],
            )
            # Ultralytics sets OpenCV to one CPU thread on import. StereoSGBM
            # benefits greatly from a bounded number of image-processing threads.
            opencv_threads = stereo_cfg.get("opencv_threads")
            opencv_threads = int(opencv_threads) if opencv_threads is not None else min(8, os.cpu_count() or 4)
            if opencv_threads < 1:
                raise ValueError("stereo opencv_threads must be positive or null")
            cv2.setNumThreads(opencv_threads)
            histories = TrackHistoryStore(**config["history"])
            approach_estimator = ApproachEstimator(**config["approach"])
            corridor = Corridor(**config["corridor"])
            risk_engine = RiskEngine(corridor, config["risk"], config["class_priors"],
                                     config["direction"], config["ego_risk"])
            ego_estimator = EgoMotionEstimator(**config["ego_motion"]) if mode in ("ego_motion", "stereo_dynamic") else None
            depth_extractor = ObjectDepthExtractor(**stereo_cfg["object_depth"])
            depth_histories = DepthHistoryStore(**stereo_cfg["depth_history"])
            if mode == "stereo_dynamic":
                disparity_extractor = ObjectDisparityExtractor(
                    min_valid_pixels=stereo_cfg["object_depth"]["min_valid_pixels"],
                    min_valid_ratio=stereo_cfg["object_depth"]["min_valid_ratio"],
                )
                stage5 = Stage5Processor(
                    risk_engine, RobotCarPinholeBackend(calibration).capabilities,
                    StereoOverlapRegion(**stereo_cfg["overlap"]), config["stereo_dynamic"],
                )
            if args.csv or config["log_csv"]:
                stem = "stage5_robotcar" if mode == "stereo_dynamic" else "stage4_stereo"
                csv_path = root / "experiments" / f"{stem}_{run_stamp}.csv"
                logger = StereoEventLogger(csv_path)
                output_paths.append(csv_path)
                print("Logging stereo observations:", csv_path)
        while not args.max_frames or frames < args.max_frames:
            frame_start = perf_counter()
            item = source.read()
            if item is None:
                break
            timing.add("stereo_input", perf_counter() - frame_start)
            mark = perf_counter()
            left, right, rect_calibration = rectifier.rectify(item.left, item.right)
            rectification_status = rect_calibration.source
            timing.add("rectification", perf_counter() - mark)
            mark = perf_counter()
            disparity = matcher.compute(left, right)
            timing.add("sgbm", perf_counter() - mark)
            mark = perf_counter()
            depth = RobotCarPinholeBackend(
                rect_calibration, stereo_cfg["min_depth_m"], stereo_cfg["max_depth_m"],
            ).metric_depth_from_disparity(disparity)
            timing.add("depth_conversion", perf_counter() - mark)
            valid_d, median_d, valid_z, median_z = _basic_stats(disparity, depth)
            valid_percentages.append(valid_d)
            valid_depth_percentages.append(valid_z)
            if np.isfinite(median_d):
                median_disparities.append(median_d)
            if np.isfinite(median_z):
                median_depths.append(median_z)

            if pair_only:
                stem = f"stereo_pair_{run_stamp}"
                disparity_path = output_dir / f"{stem}_disparity.png"
                depth_path = output_dir / f"{stem}_depth.png"
                cv2.imwrite(str(disparity_path), disparity_visualization(disparity))
                cv2.imwrite(str(depth_path), depth_visualization(depth))
                output_paths.extend((disparity_path, depth_path))
                if args.stereo_debug:
                    epi_path = output_dir / f"{stem}_epipolar.png"
                    cv2.imwrite(str(epi_path), epipolar_debug_image(left, right))
                    output_paths.append(epi_path)
                print("IMAGE DIMENSIONS:", left.shape[1], "x", left.shape[0])
                print(f"VALID DISPARITY: {valid_d:.2f}%")
                print(f"MEDIAN DISPARITY: {median_d:.2f} px")
                print(f"MEDIAN SCENE DEPTH: {median_z:.2f} m")
                frames += 1
                timing.add("total", perf_counter() - frame_start)
                break

            mark = perf_counter()
            tracks = detector.track(left)
            if mode == "stereo_dynamic":
                tracks = [track for track in tracks if not _is_robotcar_hood(
                    track, left.shape[1], left.shape[0], source, calibration, stereo_cfg)]
            timing.add("yolo", perf_counter() - mark)
            tracked_ids.update(track.track_id for track in tracks)
            if ego_estimator is not None:
                ego, _ego_timing = ego_estimator.process(left, detector.last_detection_boxes, item.timestamp)
            else:
                ego = None
            histories.prune(item.timestamp)
            depth_histories.prune(item.timestamp)
            events = []
            depths: dict[int, DepthObservation] = {}
            rows = []
            stage5_frame_times = {key: 0.0 for key in (
                "history_update", "velocity_regression", "disparity_rate", "ttc", "risk_fusion")}
            if mode == "stereo_dynamic":
                stage5.histories.prune(item.timestamp)
            for track in tracks:
                history, history_obs = histories.update(track, item.timestamp)
                approach = approach_estimator.estimate(history, left.shape[1])
                base_mode = "ego_motion" if mode == "stereo_dynamic" else mode
                event = risk_engine.evaluate(
                    track, approach, left.shape[1], left.shape[0], item.timestamp,
                    history_obs.smoothed_bbox_center[0], base_mode, ego,
                )
                if mode == "stereo_dynamic":
                    stage2_event = risk_engine.evaluate(
                        track, approach, left.shape[1], left.shape[0], item.timestamp,
                        history_obs.smoothed_bbox_center[0], "stage2", ego,
                    )
                    stage5_summaries["stage2"].add(stage2_event, item.frame_index)
                    stage5_summaries["ego_motion"].add(event, item.frame_index)
                    if ego is None or not ego.reliable:
                        event = stage2_event
                mark = perf_counter()
                in_stage5_overlap = (mode != "stereo_dynamic" or
                                     stage5.overlap.zone(track.bbox_bottom_center[0], left.shape[1]) == "STEREO_OVERLAP")
                measurement = depth_extractor.extract(track, depth) if in_stage5_overlap else DepthObservation()
                timing.add("object_depth", perf_counter() - mark)
                mark = perf_counter()
                if in_stage5_overlap and measurement.depth_confidence >= stereo_cfg["min_depth_confidence"]:
                    depth_result = depth_histories.update(track.track_id, measurement, item.timestamp)
                else:
                    depth_result = measurement
                timing.add("depth_smoothing", perf_counter() - mark)
                if depth_result.smoothed_depth_m is not None:
                    event = replace(event, distance_m=depth_result.smoothed_depth_m)
                    valid_track_ids.add(track.track_id)
                    valid_observations += 1
                    depth_confidences.append(depth_result.depth_confidence)
                if mode == "stereo_dynamic":
                    mark = perf_counter()
                    disparity_result = disparity_extractor.extract(
                        track.bbox_xyxy, disparity, np.isfinite(disparity) & (disparity > 0))
                    timing.add("object_disparity", perf_counter() - mark)
                    event, logic_timing = stage5.process(
                        track, event, depth_result, disparity_result,
                        item.timestamp, item.frame_index, left.shape[1],
                    )
                    for key in stage5_frame_times:
                        stage5_frame_times[key] += getattr(logic_timing, key + "_s")
                    stage5_summaries["stereo_dynamic"].add(event, item.frame_index)
                    stage5_counts["valid_velocity"] += event.relative_velocity_ms is not None
                    stage5_counts["valid_ttc"] += event.ttc_s is not None
                    stage5_counts["valid_disparity_rate"] += event.disparity_rate_px_s is not None
                    stage5_counts["closing"] += event.closing is True
                    stage5_counts[event.risk_source] += 1
                    if event.relative_velocity_ms is not None:
                        velocity_confidences.append(event.velocity_confidence)
                    if event.ttc_s is not None:
                        ttc_confidences.append(event.ttc_confidence)
                    if "SIGNAL_DISAGREEMENT" in event.risk_reason:
                        stage5_disagreements.append((item.frame_index, track.track_id, event.risk_reason))
                    stage5_examples[track.track_id].append(
                        (track.class_name, event.distance_m, event.relative_velocity_ms,
                         event.ttc_s, event.risk_level.value, event.risk_source))
                events.append(event)
                depths[track.track_id] = depth_result
                rows.append((track, event, depth_result))

            if mode == "stereo_dynamic":
                for key, seconds in stage5_frame_times.items():
                    timing.add(key, seconds)
                timing.add("stage5_logic", sum(stage5_frame_times.values()))

            annotated = draw_overlay(
                left.copy(), tracks, source.fps, 0.0, timing.values["yolo"][-1],
                events, corridor, bool(config["draw_corridor"]), ego,
                bool(config["draw_motion_debug"]) or args.motion_debug,
            )
            if mode != "stereo_dynamic":
                annotated = _depth_overlay(annotated, tracks, depths)
            elif args.stereo_debug:
                # The four-panel tile draws its own heading over this strip.
                annotated[:64] = left[:64]
            frame_out = _four_panel(annotated, right, disparity, depth) if args.stereo_debug else annotated
            if args.save or config["save_output"]:
                if writer is None:
                    stem = "stage5_robotcar" if mode == "stereo_dynamic" else "stage4_stereo"
                    video_path = output_dir / f"{stem}_{run_stamp}.mp4"
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source.fps, (frame_out.shape[1], frame_out.shape[0]))
                    if not writer.isOpened():
                        raise OSError(f"cannot create stereo video: {video_path}")
                    output_paths.append(video_path)
                    print("Saving stereo video:", video_path)
                writer.write(frame_out)
            quit_requested = False
            if display:
                preview = fit_for_display(frame_out, config["display_max_width"], config["display_max_height"])
                cv2.imshow("Assistive navigation - stereo", preview)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            if args.stereo_debug and frames == 0:
                epi_path = output_dir / f"stage4_epipolar_{run_stamp}.png"
                cv2.imwrite(str(epi_path), epipolar_debug_image(left, right))
                output_paths.append(epi_path)
            if logger is not None:
                frame_times = {key: values[-1] for key, values in timing.values.items() if values}
                frame_times["object_depth"] = sum(timing.values["object_depth"][-len(tracks):]) if tracks else 0.0
                frame_times["depth_smoothing"] = sum(timing.values["depth_smoothing"][-len(tracks):]) if tracks else 0.0
                if mode == "stereo_dynamic":
                    frame_times.update({key: seconds * 1000 for key, seconds in stage5_frame_times.items()})
                    frame_times["stage5_logic"] = sum(stage5_frame_times.values()) * 1000
                frame_times["total"] = (perf_counter() - frame_start) * 1000
                for track, event, depth_result in rows:
                    logger.write(item.frame_index, item.frame_id, track, event, depth_result, frame_times)
            frames += 1
            timing.add("total", perf_counter() - frame_start)
            if quit_requested:
                break
        return 0 if frames else 1
    except KeyboardInterrupt:
        warning = "stopped with Ctrl+C"
        return 0
    except Exception as exc:
        warning = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        if writer is not None:
            writer.release()
        if logger is not None:
            logger.close()
        if source is not None:
            source.release()
        cv2.destroyAllWindows()
        print("\nDATASET:", args.stereo_dataset or f"{args.stereo_left} + {args.stereo_right}")
        print("FRAMES:", frames)
        print("LEFT/RIGHT SYNCHRONIZED:", "yes, exact frame IDs" if source is not None and not pair_only else "explicit pair" if pair_only else "unverified")
        print("RECTIFICATION STATUS:", rectification_status)
        print(f"FOCAL LENGTH: {calibration.focal_px:.3f} px" if calibration else "FOCAL LENGTH: unavailable")
        print(f"BASELINE: {calibration.baseline_m:.6f} m" if calibration else "BASELINE: unavailable")
        print(f"MEAN VALID DISPARITY: {np.mean(valid_percentages):.2f}%" if valid_percentages else "MEAN VALID DISPARITY: unavailable")
        print(f"MEDIAN DISPARITY: {np.median(median_disparities):.2f} px" if median_disparities else "MEDIAN DISPARITY: unavailable")
        print(f"MEDIAN SCENE DEPTH: {np.median(median_depths):.2f} m" if median_depths else "MEDIAN SCENE DEPTH: unavailable")
        print(f"INVALID DEPTH: {100 - np.mean(valid_depth_percentages):.2f}%" if valid_depth_percentages else "INVALID DEPTH: unavailable")
        for name in ("stereo_input", "rectification", "sgbm", "depth_conversion", "object_depth", "depth_smoothing", "yolo", "total"):
            print(f"{name.upper()} LATENCY: mean {timing.mean(name):.2f} ms; p95 {timing.p95(name):.2f} ms")
        print("TRACKED OBJECTS WITH VALID DEPTH:", len(valid_track_ids), "IDs;", valid_observations, "observations;", len(tracked_ids), "IDs total")
        print(f"MEAN DEPTH CONFIDENCE: {np.mean(depth_confidences):.3f}" if depth_confidences else "MEAN DEPTH CONFIDENCE: unavailable")
        print(f"FPS: {1000 / timing.mean('total'):.2f}" if timing.mean("total") else "FPS: 0")
        print("KNOWN FAILURE MODES: low texture, reflection, occlusion boundaries, far range, calibration/LUT mismatch, object-box background contamination")
        if mode == "stereo_dynamic" and frames:
            print("TEMPORAL DATASET:", args.stereo_dataset)
            print("TRACK IDS:", len(tracked_ids))
            print("VALID METRIC DEPTH OBSERVATIONS:", valid_observations)
            print("VALID VELOCITY ESTIMATES:", stage5_counts["valid_velocity"])
            print("VALID TTC ESTIMATES:", stage5_counts["valid_ttc"])
            print("VALID DISPARITY-DYNAMIC OBSERVATIONS:", stage5_counts["valid_disparity_rate"])
            print("CLOSING OBSERVATIONS:", stage5_counts["closing"])
            for name, summary in stage5_summaries.items():
                print(f"{name.upper()}: mean {summary.mean:.4f}; variance {summary.population_variance:.4f}; {summary.counts_text()}")
            print("MEAN VELOCITY CONFIDENCE:", f"{np.mean(velocity_confidences):.3f}" if velocity_confidences else "unavailable")
            print("MEAN TTC CONFIDENCE:", f"{np.mean(ttc_confidences):.3f}" if ttc_confidences else "unavailable")
            for source_name in ("metric_stereo", "disparity_stereo", "monocular_ego_motion"):
                print(source_name.upper() + " DECISIONS:", stage5_counts[source_name])
            print(f"MEAN STAGE-5 LOGIC LATENCY: {timing.mean('stage5_logic'):.3f} ms")
            print(f"P95 STAGE-5 LOGIC LATENCY: {timing.p95('stage5_logic'):.3f} ms")
            print("SIGNAL DISAGREEMENT CASES:", len(stage5_disagreements), stage5_disagreements[:10])
            for track_id, examples in list(stage5_examples.items())[:5]:
                valid_distances = [row[1] for row in examples if row[1] is not None]
                last = examples[-1]
                print(f"EXAMPLE TRACK #{track_id} {last[0]}: depth range "
                      f"{(min(valid_distances), max(valid_distances)) if valid_distances else 'unavailable'} m; "
                      f"velocity {last[2]} m/s; TTC {last[3]} s; {last[4]}; {last[5]}")
            summary_path = root / "experiments" / f"stage5_robotcar_summary_{run_stamp}.md"
            lines = [
                "# Stage 5 temporal stereo experiment", "",
                f"Command: `python {' '.join(__import__('sys').argv)}`",
                f"Dataset: `{args.stereo_dataset}`; frames: {frames}; track IDs: {len(tracked_ids)}",
                "RobotCar calibrated rectified stereo provides metric depth. Its frame IDs are microsecond timestamps.",
                "Velocity is radial RELATIVE motion; a parked car can close as the camera moves.",
                "TTC assumes current radial motion continues; it is not collision probability.",
                "The 4.0 / 2.5 / 1.2 second urgency bands are exploratory, not validated human-safety thresholds.",
                "", "| Mode | Mean score | Population variance | LOW | MEDIUM | HIGH | CRITICAL |", "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for name, summary in stage5_summaries.items():
                from common.events import RiskLevel
                counts = [summary.level_counts[level] for level in RiskLevel]
                lines.append(f"| {name} | {summary.mean:.4f} | {summary.population_variance:.4f} | "
                             + " | ".join(str(n) for n in counts) + " |")
            lines += [
                "", f"Valid depth observations: {valid_observations}; velocity: {stage5_counts['valid_velocity']}; "
                f"TTC: {stage5_counts['valid_ttc']}; disparity rate: {stage5_counts['valid_disparity_rate']}.",
                f"Closing observations: {stage5_counts['closing']}.",
                f"Decision sources: {dict(stage5_counts)}.",
                f"Mean velocity / TTC confidence: "
                f"{np.mean(velocity_confidences) if velocity_confidences else float('nan'):.3f} / "
                f"{np.mean(ttc_confidences) if ttc_confidences else float('nan'):.3f}.",
                f"Mean / p95 Stage 5 logic: {timing.mean('stage5_logic'):.3f} / {timing.p95('stage5_logic'):.3f} ms.",
                f"Mean / p95 total frame: {timing.mean('total'):.2f} / {timing.p95('total'):.2f} ms; "
                f"throughput {1000 / timing.mean('total'):.2f} FPS.",
                f"Signal disagreement cases (frame, ID, reason): `{stage5_disagreements[:30]}`.",
                "", "## Example tracks", "",
            ]
            for track_id, examples in list(stage5_examples.items())[:10]:
                distances = [row[1] for row in examples if row[1] is not None]
                last = examples[-1]
                lines.append(f"- #{track_id} {last[0]}: depth range "
                             f"{(min(distances), max(distances)) if distances else 'unavailable'} m; "
                             f"latest velocity {last[2]} m/s; TTC {last[3]} s; {last[4]}, {last[5]}.")
            lines += [
                "", "Known limits: depth from box interiors can include background; monocular and stereo cues can disagree; "
                "velocity uses a short noisy history; TTC ignores lateral paths, turns, braking, and occlusion. "
                "RobotCar has no metric-depth ground truth in this sample. The configured hood mask is RobotCar-specific.",
                "WideDepth's selected samples are not temporal: no velocity or TTC is inferred from their ID order.",
            ]
            summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            output_paths.append(summary_path)
        print("OUTPUTS:", ", ".join(str(path) for path in output_paths) if output_paths else "none")
        print("ERRORS/WARNINGS:", warning or "none")
