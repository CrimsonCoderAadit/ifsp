# Stage 5 temporal stereo experiment

Command: `python main.py --stereo-dataset sample_small.tar --calibration dataset/robotcar_models --risk-mode stereo_dynamic --no-display --csv --save --stereo-debug --max-frames 49`
Dataset: `sample_small.tar`; frames: 49; track IDs: 21
RobotCar calibrated rectified stereo provides metric depth. Its frame IDs are microsecond timestamps.
Velocity is radial RELATIVE motion; a parked car can close as the camera moves.
TTC assumes current radial motion continues; it is not collision probability.
The 4.0 / 2.5 / 1.2 second urgency bands are exploratory, not validated human-safety thresholds.

| Mode | Mean score | Population variance | LOW | MEDIUM | HIGH | CRITICAL |
|---|---:|---:|---:|---:|---:|---:|
| stage2 | 0.2567 | 0.0473 | 120 | 70 | 7 | 0 |
| ego_motion | 0.1864 | 0.0252 | 132 | 65 | 0 | 0 |
| stereo_dynamic | 0.1338 | 0.0083 | 186 | 11 | 0 | 0 |

Valid depth observations: 194; velocity: 138; TTC: 121; disparity rate: 144.
Closing observations: 127.
Decision sources: {'valid_velocity': 138, 'valid_ttc': 121, 'valid_disparity_rate': 144, 'closing': 127, 'monocular_ego_motion': 54, 'metric_stereo': 138, 'disparity_stereo': 5}.
Mean velocity / TTC confidence: 0.713 / 0.701.
Mean / p95 Stage 5 logic: 4.717 / 7.737 ms.
Mean / p95 total frame: 930.67 / 869.36 ms; throughput 1.07 FPS.
Signal disagreement cases (frame, ID, reason): `[(18, 15, 'SIGNAL_DISAGREEMENT+METRIC_NOT_CLOSING'), (24, 15, 'SIGNAL_DISAGREEMENT+METRIC_NOT_CLOSING'), (25, 15, 'SIGNAL_DISAGREEMENT+METRIC_NOT_CLOSING'), (43, 5, 'SIGNAL_DISAGREEMENT+METRIC_CLOSING'), (44, 5, 'SIGNAL_DISAGREEMENT+DISPARITY_CLOSING'), (44, 47, 'SIGNAL_DISAGREEMENT+DISPARITY_NOT_CLOSING')]`.

## Example tracks

- #1 car: depth range (7.20789647102356, 7.445013046264648) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.
- #2 car: depth range (6.316104006767273, 15.18695387840271) m; latest velocity -2.703899575469319 m/s; TTC 2.3359240350747785 s; LOW, metric_stereo.
- #3 car: depth range (5.626503384113311, 6.158579111099243) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.
- #5 car: depth range (5.568491411209107, 10.466928911209106) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.
- #6 car: depth range (6.190438151359558, 12.098904776573182) m; latest velocity -6.521783480478157 m/s; TTC 0.9491940616994685 s; LOW, metric_stereo.
- #7 person: depth range (23.692812728881837, 24.04217529296875) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.
- #8 car: depth range (6.931640148162842, 10.341428756713867) m; latest velocity -6.479139065143673 m/s; TTC 1.0698396929699385 s; LOW, metric_stereo.
- #9 person: depth range (21.446714401245117, 21.446714401245117) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.
- #12 car: depth range (6.1967144250869755, 11.233992576599121) m; latest velocity -5.232615923725836 m/s; TTC 1.1842479011290898 s; LOW, metric_stereo.
- #13 person: depth range (18.594194412231445, 18.594194412231445) m; latest velocity None m/s; TTC None s; LOW, monocular_ego_motion.

Known limits: depth from box interiors can include background; monocular and stereo cues can disagree; velocity uses a short noisy history; TTC ignores lateral paths, turns, braking, and occlusion. RobotCar has no metric-depth ground truth in this sample. The configured hood mask is RobotCar-specific.
WideDepth's selected samples are not temporal: no velocity or TTC is inferred from their ID order.

## Manual review of this run

The saved 49-frame `outputs/video_stage5.mp4` was opened and frame 22 was inspected. Left-camera boxes, metric distance, radial velocity, TTC, and risk source were readable; the right image and disparity/depth panels were present. The broad car box over the ego vehicle hood seen in Stage 4 was absent under the RobotCar-only hood mask. At frame 22, car #2 showed about 6.9 m, -7.7 m/s, and 0.9 s TTC while remaining LOW because its corridor relevance was 0.00. This illustrates the path gate, not a validated safety judgment. Risk labels can still overlap in dense scenes, and depth/velocity lack ground-truth accuracy validation in this sample.
