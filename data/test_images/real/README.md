# Real Chennai image evaluation

Place labelled JPG/PNG stills from Chennai bus footage in this directory.
Add one `labels.csv` row per image. Required fields: `filename`, `route`.
For a full frame, provide `bbox` as JSON `[x1,y1,x2,y2]` in image pixels;
otherwise the evaluator runs YOLO and uses its highest-confidence bus box.
Use a close-up board crop with `bbox` covering the crop when detector recall
is not being evaluated. Only labelled images are scored.

`failure_type` is a semicolon or pipe-separated set such as
`glare|motion_blur`, `tamil_only`, `occlusion`, `oblique`, `cut_exp`, or
`clean`. `distortions` may instead be a JSON list. The benchmark prints
exact/fuzzy-correct/wrong/none counts and route confusions for each type.
An empty dataset prints N/A; it never counts as success.

Run:

```sh
.venv/bin/python -m bench.benchmark --device-label laptop
```
