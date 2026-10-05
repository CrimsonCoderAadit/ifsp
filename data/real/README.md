# Real bus evaluation set

This directory holds real photographs and extracted video frames. `labels.csv` distinguishes reviewer suggestions (`suggested=1, confirmed=0`) from confirmed ground truth (`confirmed=1`). Only confirmed rows enter `scripts/evaluate_ocr.py`. A blank route means no readable route or not a city bus.

## Sources and rights

- `commons/`: Wikimedia Commons files discovered from [Metropolitan Transport Corporation (Chennai)](https://commons.wikimedia.org/wiki/Category:Metropolitan_Transport_Corporation_(Chennai)), [Buses in Chennai](https://commons.wikimedia.org/wiki/Category:Buses_in_Chennai), and [Buses in Tamil Nadu](https://commons.wikimedia.org/wiki/Category:Buses_in_Tamil_Nadu), including subcategories to depth 2. **Each file has its own license and author in `commons/manifest.csv`; use that row's URL, author, and license in any slide or publication.** The downloader accepts CC0, CC BY, CC BY-SA, and public-domain files only. Commons inclusion does not establish that a route board is readable or that the bus is MTC.
- `roboflow/tsrtc/`: [AAJ TSRTC](https://universe.roboflow.com/aaj/tsrtc), CC BY 4.0. Attribution: “TSRTC dataset by AAJ, via Roboflow Universe, CC BY 4.0.” Telangana buses provide real number-panel boxes, not Chennai route ground truth. Download requires your `ROBOFLOW_API_KEY` in `.env`.
- `roboflow/bus-route-detection/`: [AAJ Bus route detection](https://universe.roboflow.com/aaj/bus-route-detection), CC BY 4.0. Attribution: “Bus route detection dataset by AAJ, via Roboflow Universe, CC BY 4.0.” Its panel boxes are useful for crop evaluation.
- `own/`: Photos or videos supplied by the project team. Record photographer permission and capture date separately before publishing. `scripts/fetch_datasets.py --source own` extracts one frame per second from videos.

Run `python scripts/fetch_datasets.py` to resume downloads. Run `python scripts/label_tool.py`, then open http://127.0.0.1:8765 and review every image. The downloader keeps an attribution manifest and skips existing files. The 70/30 split is deterministic by source and file; run dev first, tune only on dev, then run `python scripts/evaluate_ocr.py --test` once.

At the 2026-10-05 work session, six Commons labels were prefilled as *unconfirmed suggestions*. Do not quote a route accuracy until labels are reviewed. The Commons category includes non-MTC vehicles and photographs with no visible route board; the number of usable MTC images must be counted after review.

## Current collection snapshot (2026-10-05)

`scripts/dataset_report.py` reported 82 downloaded Commons images, 0 Roboflow images, 0 team images, 6 suggested labels, and 0 confirmed labels. Downloads were stopped after the Chennai categories because intermittent Wikimedia TLS resets made the wider Tamil Nadu category slow; the script can resume. The count of *usable* readable MTC images is **unverified and may be below 30**. Team photos remain important for a defensible demo and test set.
