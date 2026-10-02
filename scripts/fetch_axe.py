"""One-time download of axe-core for the UI accessibility tests (never fetched at test time)."""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import download  # noqa: E402

VERSION = "4.13.0"
TARBALL = f"https://registry.npmjs.org/axe-core/-/axe-core-{VERSION}.tgz"
VENDOR = ROOT / "tests/ui/vendor"


def main():
    archive = ROOT / "models" / f"axe-core-{VERSION}.tgz"
    download(TARBALL, archive, 100_000)
    VENDOR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member, target in (("package/axe.min.js", "axe.min.js"), ("package/LICENSE", "axe-LICENSE.txt")):
            (VENDOR / target).write_bytes(bundle.extractfile(member).read())
    (VENDOR / "VERSION").write_text(VERSION + "\n")
    print(f"axe-core {VERSION}: {(VENDOR / 'axe.min.js').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
