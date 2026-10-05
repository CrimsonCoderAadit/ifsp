# Stage 4 WideDepth outdoor validation

Loader repository commit: `309704e05001b453fb4a1674b0c9562fd7daf0f9`
Dataset revision: `0fddfc6de2c1fc096ed9249fba969823fff96eaa`
Command: `python main.py --stereo-backend widedepth --max-frames 20`
Data root: `C:\Users\amrit\Documents\IFSP\assistive_nav\data\widedepth_sample`
Downloaded files / bytes: 66 / 122408140
Processed stereo samples: 20; loader unmatched files: `{}`
RGB shape: (2048, 4096, 3); disparity shape: (2048, 4096)
The outdoor train set is vertical equirectangular stereo. Released disparity is sparse pixels, not metres.
Fisheye sparse depth is in metres but is not aligned to equirectangular object boxes.
Independent sample IDs have no verified temporal cadence; ByteTrack IDs and approach events are not reported.

| Measure | Full FOV | Central crop proxy |
|---|---:|---:|
| Mean valid GT disparity (%) | 0.189 | 0.377 |
| YOLO detections | 76 | 158 |
| Objects with valid GT disparity | 76 | 154 |
| Mean YOLO latency (ms) | 309.205 | 93.002 |
| Mean released object sampling (ms) | 3.599 | 1.760 |
| Mean SGBM latency (ms) | 47.336 | 20.617 |
| P95 SGBM latency (ms) | 72.218 | 33.101 |
| Mean estimated valid disparity (%) | 39.232 | 59.251 |
| Disparity MAE (native px) | 2.184 | 2.185 |
| Median absolute error (native px) | 1.800 | 1.800 |
| GT support evaluated by SGBM (%) | 92.20 | 92.18 |
| Bad disparity >3 native px (%) | 13.41 | 13.47 |

Sparse fisheye depth: 3 samples, mean valid ratio 0.683%
Mean dataset loading: 284.64 ms/sample
Mean RGB→BGR conversion: 6.76 ms/sample
Mean / p95 total sample processing: 1226.55 / 2919.48 ms
SGBM uses quarter-resolution images rotated CCW to turn vertical disparity into horizontal disparity; errors are rescaled to native pixel units.
The central crop uses configured image columns. It is a narrower-view proxy, not an exact 105° angular crop.
YOLO runs separately on full and cropped images; detection counts are not matched-object gains.
Released GT points inside that central crop: 99.988% of full-image valid points. The sparse labels therefore do not validate outer-field SGBM quality.
The released target is sparse; evaluation uses nearest-neighbour sampled valid GT pixels also supported by SGBM.
Released GT lookup and object sampling are validation operations, not runtime depth inference.
Known limits: equirect distortion, sparse LiDAR labels, occlusion, matcher invalid pixels, detector domain shift, no metric conversion or time continuity.
RobotCar pinhole stereo remains available as a legacy backend and is not comparable as an accuracy benchmark without ground truth.

CSV: `C:\Users\amrit\Documents\IFSP\assistive_nav\experiments\stage4_widedepth.csv`
Preview images: `C:\Users\amrit\Documents\IFSP\assistive_nav\outputs\widedepth_*`
Playable sample montage: `C:\Users\amrit\Documents\IFSP\assistive_nav\outputs\widedepth_demo.mp4` (20 independent stereo samples at 2.0 FPS).
