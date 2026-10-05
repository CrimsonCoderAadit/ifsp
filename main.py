"""Video tracking with comparable image-space risk baselines."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from input.video_source import VideoSource
from motion.ego_motion import EgoMotionEstimator, EgoMotionResult, EgoMotionTiming
from perception.track_history import TrackHistoryStore
from risk.approach import ApproachEstimator
from risk.corridor import Corridor
from risk.risk_engine import RiskEngine
from utils.drawing import draw_overlay, draw_bus_reads, fit_for_display
from utils.event_logger import EventLogger
from utils.profiler import EgoMotionProfiler, Profiler, RiskProfiler
from utils.score_summary import ScoreSummary


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistive navigation image-space risk baselines")
    parser.add_argument("--source", help="Video path or webcam index, e.g. 0")
    parser.add_argument("--stereo-backend", choices=("widedepth", "robotcar"), help="Stage 4 dataset backend")
    parser.add_argument("--widedepth-root", type=Path, help="WideDepth-train subset root; defaults to config")
    parser.add_argument("--stereo-dataset", type=Path, help="Legacy calibrated stereo sequence or RobotCar .tar")
    parser.add_argument("--stereo-left", type=Path, help="Single stereo left image (sanity mode)")
    parser.add_argument("--stereo-right", type=Path, help="Single stereo right image (sanity mode)")
    parser.add_argument("--stereo-left-subdir", default="image_02/data", help="Left image subdirectory under dataset root")
    parser.add_argument("--stereo-right-subdir", default="image_03/data", help="Right image subdirectory under dataset root")
    parser.add_argument("--calibration", type=Path, help="KITTI .txt, generic .npz, or RobotCar models directory")
    parser.add_argument("--stereo-debug", action="store_true", help="Show four-panel stereo view and save epipolar check")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument("--save", action="store_true", help="Save annotated MP4 in outputs/")
    parser.add_argument("--csv", action="store_true", help="Log every track observation to CSV")
    parser.add_argument("--risk-mode", choices=("bbox", "stage2", "ego_motion", "stereo_dynamic", "compare"), help="Override risk_mode in config")
    parser.add_argument("--motion-debug", action="store_true", help="Show background motion data and a few flow vectors")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for short runs")
    parser.add_argument("--bus-reader", action="store_true", help="Read Chennai bus route boards")
    parser.add_argument("--listen", action="store_true", help="Always listen for the Vosk 'bus' keyword")
    parser.add_argument("--web", action="store_true", help="Serve the Rider page in this process on port 8001")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return config


def print_report(
    model: str, device: str, source: str, mode: str, profiler: Profiler,
    risk_profiler: RiskProfiler, motion_profiler: EgoMotionProfiler,
    summaries: dict[str, ScoreSummary], tracking_working: bool,
    track_ids: set[int], warnings: list[str],
) -> None:
    stats = profiler.snapshot()
    print("\nMODEL:", model)
    print("DEVICE:", device)
    print("INPUT:", source)
    print("RISK MODE:", mode)
    print("FRAMES:", profiler.frames_processed)
    print("TRACKING WORKING:", "yes (same ID on consecutive frames)" if tracking_working else "not demonstrated")
    print("TRACKS OBSERVED:", len(track_ids))
    print("TRACK OBSERVATIONS:", next(iter(summaries.values())).observations if summaries else 0)
    for name, summary in summaries.items():
        print(f"{name.upper()}:", f"mean score {summary.mean:.4f}; variance {summary.population_variance:.4f}; {summary.counts_text()}; approaching {summary.approach_events}")
    print(f"MEAN DETECTOR LATENCY: {stats.mean_inference_ms:.2f} ms (rolling)")
    print(f"MEAN FEATURE DETECTION LATENCY: {motion_profiler.mean_feature_ms:.3f} ms/frame")
    print(f"MEAN OPTICAL-FLOW LATENCY: {motion_profiler.mean_flow_ms:.3f} ms/frame")
    print(f"MEAN GLOBAL ESTIMATION LATENCY: {motion_profiler.mean_estimation_ms:.3f} ms/frame")
    print(f"MEAN GLOBAL MOTION LATENCY: {motion_profiler.mean_total_ms:.3f} ms/frame")
    print(f"P95 GLOBAL MOTION LATENCY: {motion_profiler.p95_total_ms:.3f} ms (rolling)")
    print(f"MEAN HISTORY LATENCY: {risk_profiler.mean_history_ms:.3f} ms/frame")
    print(f"MEAN APPROACH LATENCY: {risk_profiler.mean_approach_ms:.3f} ms/frame")
    print(f"MEAN RISK-ENGINE LATENCY: {risk_profiler.mean_corridor_risk_ms:.3f} ms/frame")
    frame_label = "STAGE 3" if mode in ("ego_motion", "compare") else mode.upper()
    print(f"MEAN TOTAL {frame_label} LATENCY: {stats.mean_total_ms:.2f} ms (rolling)")
    print(f"P95 TOTAL {frame_label} LATENCY: {stats.p95_total_ms:.2f} ms (rolling)")
    print(f"FPS: {profiler.overall_fps:.2f} (end-to-end)")
    print(f"EGO-MOTION RELIABLE FRAMES: {motion_profiler.reliable_percent:.1f}%")
    ego_summary = summaries.get("ego_motion")
    print("NUMBER OF CASES raw high + residual low:", ego_summary.raw_high_residual_low if ego_summary else 0)
    print("EXAMPLE FRAME INDICES:", ego_summary.example_frames if ego_summary else [])
    print("KNOWN FAILURE MODES: low texture, blur, darkness, large rotation, crowd occlusion, rolling shutter, moving-object feature contamination; no depth or TTC")
    print("ERRORS/WARNINGS:", "; ".join(warnings) if warnings else "none")


def run() -> int:
    args = parse_args()
    legacy_stereo_requested = bool(args.stereo_dataset or args.stereo_left or args.stereo_right or args.stereo_backend == "robotcar")
    widedepth_requested = bool(args.widedepth_root or args.stereo_backend == "widedepth")
    if legacy_stereo_requested and widedepth_requested:
        print("Choose WideDepth or legacy calibrated stereo input, not both")
        return 2
    if widedepth_requested or (not args.source and not legacy_stereo_requested and
                               load_config(args.config)["stereo"]["dataset"] == "widedepth"):
        if args.source or args.calibration:
            print("WideDepth uses its official equirectangular loader; omit --source and --calibration")
            return 2
        from stereo.widedepth_runner import run_widedepth
        return run_widedepth(args, load_config(args.config), ROOT)
    if legacy_stereo_requested:
        if args.source or not args.calibration:
            print("Legacy calibrated stereo requires --calibration and cannot be combined with --source")
            return 2
        from stereo.runner import run_stereo
        return run_stereo(args, load_config(args.config), ROOT)
    if not args.source:
        print("Provide --source for video/webcam or stereo input and --calibration")
        return 2
    if args.risk_mode == "stereo_dynamic":
        print("stereo_dynamic requires a calibrated --stereo-dataset; use ego_motion for --source")
        return 2
    from perception.detector_tracker import DetectorTracker
    profiler = Profiler()
    risk_profiler = RiskProfiler()
    motion_profiler = EgoMotionProfiler()
    warnings: list[str] = []
    model_name, device = "unknown", "unknown"
    risk_mode = "unknown"
    summaries: dict[str, ScoreSummary] = {}
    source = None
    detector = None
    writer = None
    event_loggers: dict[str, EventLogger] = {}
    tracking_working = False
    previous_ids: set[int] = set()
    track_ids: set[int] = set()
    bus_reader = None
    bus_reports = []
    scheduler = None
    microphone = None
    web_server = None
    web_runtime = None

    try:
        config = load_config(args.config)
        risk_mode = args.risk_mode or str(config["risk_mode"])
        if risk_mode not in ("bbox", "stage2", "ego_motion", "compare"):
            raise ValueError(f"Unknown risk mode: {risk_mode}")
        active_modes = ("bbox", "stage2", "ego_motion") if risk_mode == "compare" else (risk_mode,)
        display_mode = "ego_motion" if risk_mode == "compare" else risk_mode
        summaries = {name: ScoreSummary() for name in active_modes}
        model_name = str(config["model_name"])
        detector = DetectorTracker(
            model_name=model_name,
            allowed_classes=config["allowed_classes"],
            confidence_threshold=float(config["confidence_threshold"]),
            iou_threshold=float(config["iou_threshold"]),
            image_size=int(config["image_size"]),
            tracker=str(config["tracker"]),
        )
        if args.bus_reader or config.get("bus_reader", {}).get("enabled", False):
            from transit.bus_reader import BusReader
            options = config.get("bus_reader", {})
            bus_reader = BusReader(every_n=int(options.get("every_n", 5)),
                                   threshold=float(options.get("threshold", .70)),
                                   ttl=float(options.get("ttl", 30)),
                                   include_trucks=bool(options.get("include_trucks", False)))
            print("Bus reader enabled; sharing the existing DetectorTracker YOLO instance")
            from speech.scheduler import AudioScheduler, NullSink
            from speech.tts import SpeechSynthesizer
            from speech.adapter import submit_bus
            try:
                scheduler = AudioScheduler(SpeechSynthesizer())
            except Exception as error:
                warnings.append(f"Audio output unavailable; using silent scheduler: {error}")
                scheduler = AudioScheduler(SpeechSynthesizer(), sink=NullSink())
            if args.listen:
                from speech.stt_keyword import BusKeyword
                import sounddevice as sd
                from core.config import SpeechConfig
                keyword = BusKeyword(SpeechConfig().vosk_model)
                keyword.start(16000)
                def on_audio(data, frames, timing, status):
                    if keyword.feed(bytes(data)):
                        submit_bus(scheduler, bus_reader.current())
                microphone = sd.RawInputStream(samplerate=16000, channels=1, dtype="int16",
                                               callback=on_audio, blocksize=1600)
                microphone.start()
            if args.web:
                import threading
                import uvicorn
                from app import server as rider_server
                rider_server.runtime.detector = detector
                rider_server.runtime.reader = bus_reader
                web_runtime = rider_server
                web_server = uvicorn.Server(uvicorn.Config(rider_server.app, host="127.0.0.1", port=8001,
                                                           log_level="warning"))
                threading.Thread(target=web_server.run, daemon=True).start()
                print("Rider page: http://127.0.0.1:8001")
        histories = TrackHistoryStore(**config["history"])
        approach_estimator = ApproachEstimator(**config["approach"])
        corridor = Corridor(**config["corridor"])
        risk_engine = RiskEngine(corridor, config["risk"], config["class_priors"], config["direction"], config["ego_risk"])
        ego_estimator = EgoMotionEstimator(**config["ego_motion"]) if "ego_motion" in active_modes else None
        device = detector.device
        source = VideoSource(args.source)
        display = bool(config["display"]) and not args.no_display
        save = bool(config["save_output"]) or args.save
        log_csv = bool(config["log_csv"]) or args.csv
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"webcam_{args.source}" if str(args.source).isdecimal() else Path(args.source).stem
        if log_csv:
            for name in active_modes:
                csv_path = ROOT / "outputs" / f"{stem}_{name}_{run_stamp}.csv"
                event_loggers[name] = EventLogger(csv_path)
                print(f"Logging {name} observations: {csv_path}")
        source_start = time.perf_counter()
        if source.fps <= 0:
            warnings.append("Input FPS unavailable; output uses 30 FPS")

        while not args.max_frames or profiler.frames_processed < args.max_frames:
            frame_index = profiler.frames_processed
            frame_start = time.perf_counter()
            ok, frame = source.read()
            acquisition_end = time.perf_counter()
            if not ok:
                break

            if web_runtime:
                with web_runtime.runtime.detector_lock:
                    tracks = detector.track(frame)
                    bus_reports = bus_reader.update(frame, tracks, frame_index=frame_index) if bus_reader else []
            else:
                tracks = detector.track(frame)
                bus_reports = bus_reader.update(frame, tracks, frame_index=frame_index) if bus_reader else []
            inference_end = time.perf_counter()
            timestamp = (
                frame_index / source.fps
                if isinstance(source.source, str) and source.fps > 0
                else acquisition_end - source_start
            )
            current_ids = {track.track_id for track in tracks}
            track_ids.update(current_ids)
            if current_ids & previous_ids:
                tracking_working = True
            previous_ids = current_ids

            ego_result = EgoMotionResult()
            ego_timing = EgoMotionTiming()
            if ego_estimator is not None:
                ego_result, ego_timing = ego_estimator.process(
                    frame, detector.last_detection_boxes, timestamp,
                )
                motion_profiler.add(
                    ego_timing.feature_detection_s, ego_timing.optical_flow_s,
                    ego_timing.global_estimation_s,
                    ego_result.reliable and ego_result.motion_confidence >= config["ego_risk"]["min_reliable_confidence"],
                )

            history_seconds = approach_seconds = corridor_risk_seconds = 0.0
            step_start = time.perf_counter()
            histories.prune(timestamp)
            history_seconds += time.perf_counter() - step_start
            events_by_mode = {name: [] for name in active_modes}
            log_rows = []
            height, width = frame.shape[:2]
            for track in tracks:
                step_start = time.perf_counter()
                history, observation = histories.update(track, timestamp)
                history_seconds += time.perf_counter() - step_start

                step_start = time.perf_counter()
                approach = approach_estimator.estimate(history, width)
                approach_seconds += time.perf_counter() - step_start

                step_start = time.perf_counter()
                row_events = {}
                for name in active_modes:
                    event = risk_engine.evaluate(
                        track, approach, width, height, timestamp,
                        observation.smoothed_bbox_center[0], name, ego_result,
                    )
                    row_events[name] = event
                    events_by_mode[name].append(event)
                    summaries[name].add(
                        event, frame_index,
                        config["ego_risk"]["raw_high_threshold"],
                        config["ego_risk"]["residual_low_threshold"],
                        config["ego_risk"]["min_reliable_confidence"],
                    )
                corridor_risk_seconds += time.perf_counter() - step_start
                log_rows.append((track, observation, row_events))
            risk_profiler.add(history_seconds, approach_seconds, corridor_risk_seconds)
            if scheduler is not None:
                from speech.adapter import submit_risk
                for event in events_by_mode[display_mode]:
                    submit_risk(scheduler, event)
                    if web_runtime and event.risk_level.value in {"HIGH", "CRITICAL"}:
                        web_runtime.publish_risk(event)

            previous_stats = profiler.snapshot()
            annotated = draw_overlay(
                frame,
                tracks,
                source.fps,
                previous_stats.average_fps,
                (inference_end - acquisition_end) * 1000,
                events_by_mode[display_mode],
                corridor,
                bool(config["draw_corridor"]),
                ego_result if ego_estimator is not None else None,
                bool(config["draw_motion_debug"]) or args.motion_debug,
            )
            if bus_reports:
                draw_bus_reads(annotated, bus_reports)

            if save:
                if writer is None:
                    output_dir = ROOT / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"{stem}_{display_mode}_{run_stamp}.mp4"
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source.fps if source.fps > 0 else 30.0,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output video: {output_path}")
                    print(f"Saving annotated video: {output_path}")
                writer.write(annotated)

            if display:
                preview = fit_for_display(
                    annotated,
                    int(config["display_max_width"]),
                    int(config["display_max_height"]),
                )
                cv2.imshow(f"Assistive navigation - {display_mode}", preview)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            else:
                quit_requested = False

            if event_loggers:
                total_ms = (time.perf_counter() - frame_start) * 1000
                risk_ms = (history_seconds + approach_seconds + corridor_risk_seconds) * 1000
                for track, observation, row_events in log_rows:
                    for name, logger in event_loggers.items():
                        logger.write(
                            frame_index, track, observation, row_events[name],
                            (inference_end - acquisition_end) * 1000, risk_ms, total_ms,
                            ego_result if ego_estimator is not None else None,
                            ego_timing if ego_estimator is not None else None,
                            bus_reader.last_read.get(track.track_id) if bus_reader else None,
                        )

            profiler.add(acquisition_end - frame_start, inference_end - acquisition_end, time.perf_counter() - frame_start)
            if quit_requested:
                break

        if profiler.frames_processed == 0:
            warnings.append("No frames processed")
        elif not tracking_working:
            warnings.append("No relevant ID was observed on consecutive frames")
        return 0 if profiler.frames_processed else 1
    except KeyboardInterrupt:
        warnings.append("Stopped with Ctrl+C")
        return 0
    except Exception as exc:
        warnings.append(f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if writer is not None:
            writer.release()
        for logger in event_loggers.values():
            logger.close()
        if source is not None:
            source.release()
        if microphone is not None:
            microphone.stop()
            microphone.close()
        if scheduler is not None:
            scheduler.close()
        if web_server is not None:
            web_server.should_exit = True
        cv2.destroyAllWindows()
        if detector is not None:
            device = detector.device
        print_report(
            model_name, device, args.source, risk_mode, profiler, risk_profiler,
            motion_profiler, summaries, tracking_working, track_ids, warnings,
        )
        if bus_reader:
            print("BUS READER:", bus_reader.current() or "No confirmed route")


if __name__ == "__main__":
    raise SystemExit(run())
