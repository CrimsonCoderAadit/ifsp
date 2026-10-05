"""WideDepth outdoor stereo validation without invented metric conversion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from input.widedepth_source import WideDepthSource
from stereo.depth_backend import WideDepthBackend
from stereo.object_disparity import ObjectDisparity, ObjectDisparityExtractor
from stereo.overlap import StereoOverlapRegion
from stereo.sgbm import StereoMatcher, disparity_visualization, depth_visualization
from stereo.widedepth_match import DisparityComparison, compare_sgbm
from utils.drawing import fit_for_display
from utils.widedepth_drawing import VIDEO_SIZE, compose_widedepth_demo


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def _path(root: Path, configured: str | Path) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _detect(detector, image_bgr: np.ndarray, config: dict) -> list[Detection]:
    results = detector.model.predict(
        source=image_bgr, classes=detector.allowed_ids,
        conf=config["confidence_threshold"], iou=config["iou_threshold"],
        imgsz=config["image_size"], verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    boxes = results[0].boxes
    return [
        Detection(detector.names[int(class_id)], float(confidence), tuple(float(v) for v in xyxy))
        for xyxy, class_id, confidence in zip(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(),
        )
    ]


def _preview(image: np.ndarray, max_width: int = 1600) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image.copy()
    ratio = max_width / image.shape[1]
    return cv2.resize(image, (max_width, round(image.shape[0] * ratio)), interpolation=cv2.INTER_AREA)


def _detection_preview(
    image_bgr: np.ndarray, detections: list[Detection],
    measurements: list[ObjectDisparity],
) -> np.ndarray:
    canvas = _preview(image_bgr)
    scale = canvas.shape[1] / image_bgr.shape[1]
    for index, (detection, measurement) in enumerate(zip(detections, measurements), start=1):
        x1, y1, x2, y2 = (round(v * scale) for v in detection.bbox_xyxy)
        color = (70, 205, 255) if measurement.disparity_px is not None else (110, 110, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"#{index}", (max(2, x1), max(24, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], min(canvas.shape[0], 36 + 30 * len(detections))),
                  (15, 15, 15), -1)
    cv2.putText(canvas, "WideDepth released disparity (px) | metric depth unavailable",
                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 2, cv2.LINE_AA)
    for index, (detection, measurement) in enumerate(zip(detections, measurements), start=1):
        value = (f"GT disp {measurement.disparity_px:.1f} px; valid {measurement.valid_ratio:.2%}"
                 if measurement.disparity_px is not None else "GT disparity unavailable")
        label = f"#{index} {detection.class_name} {detection.confidence:.2f} | {value}"
        cv2.putText(canvas, label, (15, 27 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (70, 205, 255), 2, cv2.LINE_AA)
    if not detections:
        cv2.putText(canvas, "No relevant YOLO detections", (20, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def _save_previews(
    output_dir: Path, sample_id: str, variant: str,
    left_bgr: np.ndarray, right_bgr: np.ndarray,
    gt_disparity: np.ndarray, gt_valid: np.ndarray,
    detections: list[Detection], measurements: list[ObjectDisparity],
    comparison: DisparityComparison | None,
) -> None:
    suffix = f"{variant}_{sample_id}"
    cv2.imwrite(str(output_dir / f"widedepth_left_{suffix}.png"), _preview(left_bgr))
    cv2.imwrite(str(output_dir / f"widedepth_right_{suffix}.png"), _preview(right_bgr))
    gt_for_view = np.where(gt_valid, gt_disparity, np.nan).astype(np.float32)
    cv2.imwrite(str(output_dir / f"widedepth_gt_disparity_{suffix}.png"),
                _preview(disparity_visualization(gt_for_view)))
    if comparison is not None:
        estimated = cv2.rotate(comparison.estimated_rotated_px, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(output_dir / f"widedepth_sgbm_disparity_{suffix}.png"),
                    disparity_visualization(estimated))
    cv2.imwrite(str(output_dir / f"widedepth_detection_depth_{suffix}.png"),
                _detection_preview(left_bgr, detections, measurements))


def _stats(values: list[float]) -> tuple[float, float]:
    return (float(np.mean(values)), float(np.percentile(values, 95))) if values else (0.0, 0.0)


def _sparse_metric_depth(root_dir: Path, loader_repo: Path, output_dir: Path) -> tuple[int, float | None]:
    if not all((root_dir / folder).is_dir() for folder in ("left_fisheye", "depth_fisheye")):
        return 0, None
    from widedepth import WideDepthTrainDataset
    dataset = WideDepthTrainDataset(root_dir, task="sparse_depth")
    ratios = []
    for index in range(len(dataset)):
        sample = dataset[index]
        ratios.append(float(sample["valid_mask"].mean()))
        if index == 0:
            depth = np.where(sample["valid_mask"], sample["sparse_depth"], np.nan)
            cv2.imwrite(str(output_dir / f"widedepth_fisheye_depth_{sample['sample_id']}.png"),
                        _preview(depth_visualization(depth)))
    return len(dataset), float(np.mean(ratios)) if ratios else None


def run_widedepth(args, config: dict, root: Path) -> int:
    stereo_cfg = config["stereo"]
    settings = stereo_cfg["widedepth"]
    dataset_root = _path(root, args.widedepth_root or settings["root_dir"])
    loader_repo = _path(root, settings["loader_repo"])
    source = WideDepthSource(
        dataset_root, loader_repo, task=settings["task"],
        crop_equirect=settings["crop_equirect"], rotate_ccw=settings["rotate_ccw"],
    )
    backend = WideDepthBackend()
    capabilities = backend.capabilities
    overlap_cfg = settings["overlap"]
    overlap = StereoOverlapRegion(overlap_cfg["x_min"], overlap_cfg["x_max"])
    central_cfg = settings["central_crop_proxy"]
    central = StereoOverlapRegion(central_cfg["x_min"], central_cfg["x_max"])
    extractor = ObjectDisparityExtractor(**settings["object_disparity"])
    output_dir = root / "outputs"
    experiment_dir = root / "experiments"
    output_dir.mkdir(exist_ok=True)
    experiment_dir.mkdir(exist_ok=True)
    csv_path = experiment_dir / "stage4_widedepth.csv"
    summary_path = experiment_dir / "stage4_widedepth_summary.md"
    manifest_path = dataset_root / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    repo_sha_path = loader_repo / ".git"
    from subprocess import check_output
    repo_sha = (check_output(["git", "-c", f"safe.directory={loader_repo}", "rev-parse", "HEAD"],
                             cwd=loader_repo, text=True).strip()
                if repo_sha_path.exists() else "unknown")

    from perception.detector_tracker import DetectorTracker
    detector = DetectorTracker(
        config["model_name"], config["allowed_classes"],
        config["confidence_threshold"], config["iou_threshold"],
        config["image_size"], config["tracker"],
    )
    opencv_threads = stereo_cfg.get("opencv_threads")
    opencv_threads = int(opencv_threads) if opencv_threads is not None else min(8, os.cpu_count() or 4)
    if opencv_threads < 1:
        raise ValueError("stereo opencv_threads must be positive or null")
    cv2.setNumThreads(opencv_threads)
    matcher = StereoMatcher(**settings["sgbm"]) if settings["use_sgbm"] else None
    processing_scale = float(settings["sgbm_processing_scale"])
    preview_count = int(settings["save_preview_samples"])
    max_samples = args.max_frames or len(source)
    sample_count = min(max_samples, len(source))
    demo_path = output_dir / "widedepth_demo.mp4"
    save_demo = bool(settings.get("save_demo_video", True)) or args.save
    display = bool(config["display"]) and not args.no_display
    demo_writer = None

    fields = [
        "sample_id", "variant", "frame_index", "image_width", "image_height",
        "gt_valid_percent", "detection_count", "class", "confidence", "track_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "overlap_zone",
        "object_gt_disparity_px", "object_depth_m", "object_valid_ratio",
        "object_disparity_mad_px", "object_disparity_confidence",
        "sgbm_estimated_valid_percent", "sgbm_evaluated_gt_pixels",
        "sgbm_mae_native_px", "sgbm_median_absolute_error_native_px",
        "dataset_load_ms", "color_conversion_ms", "yolo_ms",
        "released_object_sampling_ms", "sgbm_ms", "disparity_comparison_ms", "total_sample_ms",
    ]
    metrics: dict[str, dict[str, list]] = {
        variant: {key: [] for key in (
            "gt_valid", "detections", "object_valid", "yolo_ms", "sampling_ms",
            "sgbm_ms", "comparison_ms", "estimated_valid", "errors",
            "gt_sampled_pixels", "gt_evaluated_pixels",
        )} for variant in ("full", "central_proxy")
    }
    loading_ms: list[float] = []
    conversion_ms: list[float] = []
    total_ms: list[float] = []
    first_shape = None
    first_disparity_shape = None
    processed = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        try:
            while processed < min(max_samples, len(source)):
                sample_started = perf_counter()
                mark = perf_counter()
                frame = source.read()
                if frame is None:
                    break
                load_time = (perf_counter() - mark) * 1000
                loading_ms.append(load_time)
                backend.validate_released(frame.left_rgb, frame.right_rgb,
                                          frame.gt_disparity_px, frame.valid_mask)
                first_shape = first_shape or frame.left_rgb.shape
                first_disparity_shape = first_disparity_shape or frame.gt_disparity_px.shape
                mark = perf_counter()
                left_bgr = cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR)
                right_bgr = cv2.cvtColor(frame.right_rgb, cv2.COLOR_RGB2BGR)
                color_time = (perf_counter() - mark) * 1000
                conversion_ms.append(color_time)

                variants = [("full", 0, left_bgr.shape[1], overlap)]
                if settings["run_central_crop_proxy"]:
                    x0 = int(np.floor(central.x_min * left_bgr.shape[1]))
                    x1 = int(np.ceil(central.x_max * left_bgr.shape[1]))
                    variants.append(("central_proxy", x0, x1, StereoOverlapRegion()))
                pending_rows = []
                demo_inputs = None
                for variant, x0, x1, region in variants:
                    left = left_bgr[:, x0:x1]
                    right = right_bgr[:, x0:x1]
                    gt = frame.gt_disparity_px[:, x0:x1]
                    valid = (frame.valid_mask[:, x0:x1] & region.mask(*gt.shape)
                             if settings["use_released_disparity"] else np.zeros(gt.shape, dtype=bool))
                    slot = metrics[variant]
                    slot["gt_valid"].append(float(valid.mean()))
                    mark = perf_counter()
                    detections = _detect(detector, left, config)
                    yolo_time = (perf_counter() - mark) * 1000
                    slot["yolo_ms"].append(yolo_time)
                    slot["detections"].append(len(detections))
                    mark = perf_counter()
                    measurements = [extractor.extract(d.bbox_xyxy, gt, valid) for d in detections]
                    sampling_time = (perf_counter() - mark) * 1000
                    slot["sampling_ms"].append(sampling_time)
                    slot["object_valid"].append(sum(m.disparity_px is not None for m in measurements))
                    comparison = (compare_sgbm(left, right, gt, valid, matcher, processing_scale)
                                  if matcher is not None else None)
                    if comparison is not None:
                        slot["sgbm_ms"].append(comparison.sgbm_ms)
                        slot["comparison_ms"].append(comparison.comparison_ms)
                        slot["estimated_valid"].append(comparison.estimated_valid_ratio)
                        slot["errors"].append(comparison.absolute_errors_native_px)
                        slot["gt_sampled_pixels"].append(comparison.gt_pixels_after_sampling)
                        slot["gt_evaluated_pixels"].append(comparison.evaluated_pixels)
                    if processed < preview_count:
                        _save_previews(output_dir, frame.sample_id, variant, left, right,
                                       gt, valid, detections, measurements, comparison)
                    if variant == "full":
                        demo_inputs = (left, right, gt, valid, detections, measurements, comparison)

                    common = {
                        "sample_id": frame.sample_id, "variant": variant,
                        "frame_index": frame.frame_index,
                        "image_width": left.shape[1], "image_height": left.shape[0],
                        "gt_valid_percent": 100 * valid.mean(),
                        "detection_count": len(detections),
                        "sgbm_estimated_valid_percent": 100 * comparison.estimated_valid_ratio if comparison else "",
                        "sgbm_evaluated_gt_pixels": comparison.evaluated_pixels if comparison else "",
                        "sgbm_mae_native_px": comparison.mae_native_px if comparison else "",
                        "sgbm_median_absolute_error_native_px": comparison.median_absolute_error_native_px if comparison else "",
                        "dataset_load_ms": load_time, "color_conversion_ms": color_time,
                        "yolo_ms": yolo_time, "released_object_sampling_ms": sampling_time,
                        "sgbm_ms": comparison.sgbm_ms if comparison else "",
                        "disparity_comparison_ms": comparison.comparison_ms if comparison else "",
                    }
                    for detection, measure in zip(detections, measurements):
                        x_a, y_a, x_b, y_b = detection.bbox_xyxy
                        pending_rows.append(common | {
                            "class": detection.class_name, "confidence": detection.confidence,
                            "track_id": "",  # Independent sample IDs are not a video timeline.
                            "bbox_x1": x_a, "bbox_y1": y_a, "bbox_x2": x_b, "bbox_y2": y_b,
                            "overlap_zone": region.zone((x_a + x_b) / 2, left.shape[1]),
                            "object_gt_disparity_px": measure.disparity_px,
                            "object_depth_m": "",  # No documented equirect disparity-to-metre conversion.
                            "object_valid_ratio": measure.valid_ratio,
                            "object_disparity_mad_px": measure.mad_px,
                            "object_disparity_confidence": measure.confidence,
                        })
                    if not detections:
                        pending_rows.append(common)
                quit_requested = False
                if save_demo or display:
                    demo = compose_widedepth_demo(frame.sample_id, processed + 1, sample_count,
                                                  *demo_inputs)
                    if save_demo:
                        if demo_writer is None:
                            demo_writer = cv2.VideoWriter(
                                str(demo_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                float(settings.get("demo_fps", 2.0)), VIDEO_SIZE,
                            )
                            if not demo_writer.isOpened():
                                raise RuntimeError(f"Could not create WideDepth demo: {demo_path}")
                            print(f"Saving WideDepth visual demo: {demo_path}")
                        demo_writer.write(demo)
                    if display:
                        preview = fit_for_display(
                            demo, int(config["display_max_width"]),
                            int(config["display_max_height"]),
                        )
                        cv2.imshow("WideDepth stereo samples (q to quit)", preview)
                        quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
                sample_time = (perf_counter() - sample_started) * 1000
                total_ms.append(sample_time)
                for row in pending_rows:
                    writer.writerow(row | {"total_sample_ms": sample_time})
                processed += 1
                print(f"WideDepth {processed}/{min(max_samples, len(source))}: {frame.sample_id}")
                if quit_requested:
                    break
        finally:
            if demo_writer is not None:
                demo_writer.release()
            if display:
                cv2.destroyAllWindows()
            source.release()

    sparse_count, sparse_valid_ratio = _sparse_metric_depth(dataset_root, loader_repo, output_dir)
    lines = [
        "# Stage 4 WideDepth outdoor validation",
        "",
        f"Loader repository commit: `{repo_sha}`",
        f"Dataset revision: `{manifest.get('revision', 'not recorded')}`",
        f"Command: `python {' '.join(sys.argv)}`",
        f"Data root: `{dataset_root}`",
        f"Downloaded files / bytes: {len(manifest.get('files', []))} / {manifest.get('actual_bytes', 'not recorded')}",
        f"Processed stereo samples: {processed}; loader unmatched files: `{source.unmatched_files}`",
        f"RGB shape: {first_shape}; disparity shape: {first_disparity_shape}",
        "The outdoor train set is vertical equirectangular stereo. Released disparity is sparse pixels, not metres.",
        "Fisheye sparse depth is in metres but is not aligned to equirectangular object boxes.",
        "Independent sample IDs have no verified temporal cadence; ByteTrack IDs and approach events are not reported.",
        f"Backend capabilities: metric_depth={capabilities.supports_metric_depth}, "
        f"temporal_sequence={capabilities.supports_temporal_sequence}, "
        f"disparity_closer_sign={capabilities.disparity_increases_when_closer}; "
        f"projection={capabilities.projection_model}.",
        "",
        "| Measure | Full FOV | Central crop proxy |",
        "|---|---:|---:|",
    ]
    for label, key, multiplier in (
        ("Mean valid GT disparity (%)", "gt_valid", 100),
        ("YOLO detections", "detections", 1),
        ("Objects with valid GT disparity", "object_valid", 1),
        ("Mean YOLO latency (ms)", "yolo_ms", 1),
        ("Mean released object sampling (ms)", "sampling_ms", 1),
        ("Mean SGBM latency (ms)", "sgbm_ms", 1),
        ("P95 SGBM latency (ms)", "sgbm_ms", 1),
        ("Mean estimated valid disparity (%)", "estimated_valid", 100),
    ):
        values = []
        for variant in ("full", "central_proxy"):
            data = metrics[variant][key]
            number = (sum(data) if key in ("detections", "object_valid") else
                      np.percentile(data, 95) if label.startswith("P95") and data else
                      np.mean(data) * multiplier if data else None)
            values.append((str(int(number)) if key in ("detections", "object_valid") else f"{number:.3f}")
                          if number is not None else "n/a")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    error_summaries = {}
    for variant in ("full", "central_proxy"):
        arrays = metrics[variant]["errors"]
        errors = np.concatenate(arrays) if arrays else np.empty(0)
        error_summaries[variant] = errors
    for label, function in (
        ("Disparity MAE (native px)", np.mean),
        ("Median absolute error (native px)", np.median),
    ):
        values = [f"{function(error_summaries[v]):.3f}" if error_summaries[v].size else "n/a"
                  for v in ("full", "central_proxy")]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    coverages = []
    bad_three = []
    for variant in ("full", "central_proxy"):
        sampled = sum(metrics[variant]["gt_sampled_pixels"])
        evaluated = sum(metrics[variant]["gt_evaluated_pixels"])
        errors = error_summaries[variant]
        coverages.append(f"{100 * evaluated / sampled:.2f}" if sampled else "n/a")
        bad_three.append(f"{100 * np.mean(errors > 3):.2f}" if errors.size else "n/a")
    lines.append(f"| GT support evaluated by SGBM (%) | {coverages[0]} | {coverages[1]} |")
    lines.append(f"| Bad disparity >3 native px (%) | {bad_three[0]} | {bad_three[1]} |")
    if first_shape and metrics["central_proxy"]["gt_valid"]:
        height, width = first_shape[:2]
        cropped_width = int(np.ceil(central.x_max * width)) - int(np.floor(central.x_min * width))
        all_valid = sum(metrics["full"]["gt_valid"]) * height * width
        central_valid = sum(metrics["central_proxy"]["gt_valid"]) * height * cropped_width
        central_gt_share = 100 * central_valid / all_valid if all_valid else None
    else:
        central_gt_share = None
    lines += [
        "",
        f"Sparse fisheye depth: {sparse_count} samples, mean valid ratio "
        + (f"{100 * sparse_valid_ratio:.3f}%" if sparse_valid_ratio is not None else "unavailable"),
        f"Mean dataset loading: {_stats(loading_ms)[0]:.2f} ms/sample",
        f"Mean RGB→BGR conversion: {_stats(conversion_ms)[0]:.2f} ms/sample",
        f"Mean / p95 total sample processing: {_stats(total_ms)[0]:.2f} / {_stats(total_ms)[1]:.2f} ms",
        "SGBM uses quarter-resolution images rotated CCW to turn vertical disparity into horizontal disparity; errors are rescaled to native pixel units.",
        "The central crop uses configured image columns. It is a narrower-view proxy, not an exact 105° angular crop.",
        "YOLO runs separately on full and cropped images; detection counts are not matched-object gains.",
        (f"Released GT points inside that central crop: {central_gt_share:.3f}% of full-image valid points. "
         "The sparse labels therefore do not validate outer-field SGBM quality."
         if central_gt_share is not None else "Central-crop GT support was not measured."),
        "The released target is sparse; evaluation uses nearest-neighbour sampled valid GT pixels also supported by SGBM.",
        "Released GT lookup and object sampling are validation operations, not runtime depth inference.",
        "Known limits: equirect distortion, sparse LiDAR labels, occlusion, matcher invalid pixels, detector domain shift, no metric conversion or time continuity.",
        "RobotCar pinhole stereo remains available as a legacy backend and is not comparable as an accuracy benchmark without ground truth.",
        "",
        f"CSV: `{csv_path}`",
        f"Preview images: `{output_dir / 'widedepth_*'}`",
        f"Playable sample montage: `{demo_path}` ({processed} independent stereo samples at {settings.get('demo_fps', 2.0)} FPS)." if save_demo else "Playable sample montage: disabled.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nWIDEDEPTH REPOSITORY COMMIT:", repo_sha)
    print("DATASET:", dataset_root)
    print("NUMBER OF SAMPLES DOWNLOADED:", len(manifest.get("stereo_sample_ids", [])))
    print("NUMBER OF VALID STEREO SAMPLES:", processed)
    print("IMAGE SHAPES:", first_shape)
    print("DISPARITY SHAPE:", first_disparity_shape)
    print("TEMPORAL VELOCITY / TTC: unavailable for these independent WideDepth samples")
    print("MEAN VALID GT DISPARITY %:", f"{100 * np.mean(metrics['full']['gt_valid']):.3f}" if processed else "n/a")
    print("YOLO DETECTIONS:", sum(metrics["full"]["detections"]))
    print("OBJECTS WITH VALID DISPARITY:", sum(metrics["full"]["object_valid"]))
    print("MEAN SGBM LATENCY:", f"{_stats(metrics['full']['sgbm_ms'])[0]:.2f} ms")
    print("P95 SGBM LATENCY:", f"{_stats(metrics['full']['sgbm_ms'])[1]:.2f} ms")
    print("DISPARITY MAE:", f"{np.mean(error_summaries['full']):.3f} native px" if error_summaries["full"].size else "n/a")
    print("MEDIAN ABSOLUTE ERROR:", f"{np.median(error_summaries['full']):.3f} native px" if error_summaries["full"].size else "n/a")
    print("EVALUATED GT SUPPORT:", f"{coverages[0]}%" if coverages[0] != "n/a" else "n/a")
    print("BAD >3 PX:", f"{bad_three[0]}%" if bad_three[0] != "n/a" else "n/a")
    print("GT POINTS IN CENTRAL CROP:", f"{central_gt_share:.3f}%" if central_gt_share is not None else "n/a")
    print("MEAN VALID SPARSE FISHEYE DEPTH RATIO:", f"{100 * sparse_valid_ratio:.3f}%" if sparse_valid_ratio is not None else "unavailable")
    print("TOTAL PIPELINE LATENCY:", f"mean {_stats(total_ms)[0]:.2f} ms; p95 {_stats(total_ms)[1]:.2f} ms")
    print("KNOWN FAILURE MODES: sparse GT, equirect distortion, occlusions, invalid SGBM, perspective-detector domain shift; no metric stereo conversion")
    print("OUTPUTS:", csv_path, summary_path, output_dir / "widedepth_*.png")
    if save_demo:
        print("VISUAL DEMO:", demo_path)
    return 0
