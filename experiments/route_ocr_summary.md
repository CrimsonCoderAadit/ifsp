# Real route OCR evaluation: awaiting confirmed labels

Collection on 2026-10-05: 82 downloaded Wikimedia Commons images, 0 Roboflow images, and 0 team images. `data/real/labels.csv` contains six suggestions and zero confirmed labels. The number of readable MTC buses in the Commons set is not yet known.

The dev and held-out test runs have **not** been performed. `scripts/evaluate_ocr.py` intentionally exits when its selected split has no confirmed labels. There is no defensible route accuracy, per-source breakdown, before/after OCR comparison, or top-ten test failures to report yet. The qualitative examples in `DEMO_SCRIPT.md` are smoke tests only.

Next: provide the Roboflow API key and team photos/video, review labels with `scripts/label_tool.py`, then run the dev evaluation and compare `rapidocr-v4`, `rapidocr-v5-mobile`, and `rapidocr-v5-server`. After dev choices are fixed, run `python scripts/evaluate_ocr.py --test` once. That command replaces this file with the measured per-source, board-type, and lighting summary and writes `experiments/route_ocr_eval.csv`.
