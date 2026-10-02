# Person 2: offline voice and transit (Review 1)

Python 3.10+. All setup downloads are explicit; runtime route matching is offline.
The empty speech, pipeline, detector, OCR, web and benchmark files mark interfaces
to implement in subsequent steps. No Person 1 code is imported.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_models.py
.venv/bin/python -m transit.gtfs_loader
.venv/bin/python -m pytest -q
```

For YOLO ONNX export, install optional `ultralytics onnx` inside the venv and rerun
`scripts/download_models.py --only yolo`. Export uses fixed 640px input and opset 17.
The PT weights remain available for the optional Ultralytics backend.

Downloads use temporary files, minimum sizes, Content-Length checks when supplied,
and SHA256 receipts for subsequent cache checks. Failed tasks report separately
and make the command exit nonzero. Use `--only vosk piper yolo rapidocr gtfs` to
select tasks. To refresh the weekly GTFS feed, delete `data/mtc-gtfs.zip` and rerun
the GTFS download and loader. Only `routes.txt` is opened by the loader.

`load_routes()` prefers `data/routes.csv`; if absent it uses the 20-route
`data/routes_review1.csv`. Out-of-shape bases remain exact-only aliases. Raw short
names and up to three long names are retained per base in the generated CSV.

`normalize_route()` preserves already-valid shapes: without a catalogue, `M7O`
could be either a literal route or an OCR error. `match_route()` tries exact,
unique contextual confusion corrections, then RapidFuzz (default threshold 85).
Fuzzy matching never changes the digit block; tied results are rejected.
`extract_route_candidates()` accepts strings, `{text, box}` dictionaries and
RapidOCR `[box, text, confidence]` rows, returning ranked unique route strings.

RapidOCR is explicitly configured for PP-OCRv4 mobile models using ONNX Runtime
([upstream configuration](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/python/rapidocr/config.yaml)).
The setup script warms it once and writes a complete local-path configuration to
`models/rapidocr/runtime_config.yaml`. A future OCR adapter should load that file
with `RapidOCR(config_path=...)` and use local images or arrays. The PP-OCR models
are Chinese/Latin models, not Tamil recognition models; Latin route tokens are
the focus for Review 1. The runtime OCR adapter is still a scaffold.

Models and ZIPs are ignored by Git. Copy the entire `models/` directory and the
route CSV when moving to another machine; regenerate the OCR config there because
its model paths are absolute. All dependencies belong in `.venv`.
