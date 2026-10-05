"""Download a bounded, synchronized WideDepth-train subset using Hub files.

The exact-file method follows the official WideDepth quickstart notebook. It
never snapshots the full 118 GB repository. Run --dry-run to review the file
list and byte budget before downloading.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from itertools import islice
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "IlyaInd/WideDepth-train"
STEREO_FOLDERS = ("equirect_up", "equirect_down_virt", "disp_equirect")
DEPTH_FOLDERS = ("left_fisheye", "depth_fisheye")


def listed_files(api: HfApi, folder: str, limit: int, revision: str) -> dict[str, tuple[str, int]]:
    found: dict[str, tuple[str, int]] = {}
    tree = api.list_repo_tree(
        REPO_ID, repo_type="dataset", revision=revision,
        path_in_repo=folder, expand=True,
    )
    for entry in islice(tree, limit):
        path = Path(entry.path)
        if path.suffix.lower() == ".png" and path.stem.isdigit() and entry.size is not None:
            found[path.stem] = (entry.path, int(entry.size))
    return found


def plan_download(api: HfApi, count: int, depth_count: int, candidate_limit: int) -> dict:
    revision = api.repo_info(REPO_ID, repo_type="dataset").sha
    folders = STEREO_FOLDERS + (DEPTH_FOLDERS if depth_count else ())
    listings = {folder: listed_files(api, folder, candidate_limit, revision) for folder in folders}
    stereo_ids = sorted(set.intersection(*(set(listings[folder]) for folder in STEREO_FOLDERS)), key=int)
    if len(stereo_ids) < count:
        raise RuntimeError(f"only {len(stereo_ids)} matched stereo IDs in first {candidate_limit} candidates; increase --candidate-limit")
    selected = stereo_ids[:count]
    depth_ids = [sample_id for sample_id in selected if all(sample_id in listings[folder] for folder in DEPTH_FOLDERS)][:depth_count]
    if depth_count and len(depth_ids) < depth_count:
        raise RuntimeError(f"only {len(depth_ids)} selected stereo IDs also have sparse depth; increase --candidate-limit or reduce --depth-count")
    files = [listings[folder][sample_id] for sample_id in selected for folder in STEREO_FOLDERS]
    files += [listings[folder][sample_id] for sample_id in depth_ids for folder in DEPTH_FOLDERS]
    return {
        "repo_id": REPO_ID,
        "revision": revision,
        "stereo_sample_ids": selected,
        "sparse_depth_sample_ids": depth_ids,
        "files": [{"path": name, "expected_bytes": size} for name, size in files],
        "expected_bytes": sum(size for _, size in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "widedepth_sample")
    parser.add_argument("--count", type=int, default=20, help="Matched equirect stereo IDs")
    parser.add_argument("--depth-count", type=int, default=3, help="Matching fisheye sparse-depth samples")
    parser.add_argument("--candidate-limit", type=int, default=500, help="File-list entries queried per modality")
    parser.add_argument("--max-download-mb", type=float, default=300.0, help="Hard cap on total expected file size")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.depth_count < 0 or args.depth_count > args.count or args.candidate_limit < args.count:
        parser.error("require count >= 1, 0 <= depth-count <= count, candidate-limit >= count")
    if args.max_download_mb <= 0:
        parser.error("max-download-mb must be positive")

    api = HfApi()
    plan = plan_download(api, args.count, args.depth_count, args.candidate_limit)
    expected_mb = plan["expected_bytes"] / 1_000_000
    print(f"Revision: {plan['revision']}")
    print(f"Stereo IDs ({len(plan['stereo_sample_ids'])}): {', '.join(plan['stereo_sample_ids'])}")
    print(f"Sparse-depth IDs ({len(plan['sparse_depth_sample_ids'])}): {', '.join(plan['sparse_depth_sample_ids'])}")
    print(f"Exact files: {len(plan['files'])}; expected data: {expected_mb:.1f} MB")
    if expected_mb > args.max_download_mb:
        raise RuntimeError(f"download exceeds {args.max_download_mb:.1f} MB cap; reduce --count")
    if args.dry_run:
        return

    # The official quickstart uses exact hf_hub_download calls and disables Xet
    # for this small-file workflow. Set before starting any downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.root.mkdir(parents=True, exist_ok=True)

    def fetch(item: dict) -> str:
        return hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", revision=plan["revision"],
            filename=item["path"], local_dir=args.root,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(fetch, plan["files"]))
    for item, path in zip(plan["files"], downloaded):
        if not Path(path).is_file():
            raise OSError(f"missing downloaded file: {item['path']}")
    plan["actual_bytes"] = sum(Path(path).stat().st_size for path in downloaded)
    manifest = args.root / "sample_manifest.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(downloaded)} files, {plan['actual_bytes'] / 1_000_000:.1f} MB to {args.root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
