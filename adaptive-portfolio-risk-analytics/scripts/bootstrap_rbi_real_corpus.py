"""Bootstrap the governed real-RBI local corpus directory structure."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import REAL_RBI_MANIFEST_COLUMNS  # noqa: E402


DEFAULT_RBI_REAL_DIR = REPO_ROOT / "data" / "sentiment" / "rbi_real"


def bootstrap_rbi_real_corpus(
    *,
    corpus_dir: str | Path = DEFAULT_RBI_REAL_DIR,
) -> dict[str, object]:
    """Ensure the real-RBI corpus directories and empty manifest exist."""
    root = Path(corpus_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    manifest_path = root / "manifest.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / ".gitkeep").touch(exist_ok=True)
    (processed_dir / ".gitkeep").touch(exist_ok=True)
    manifest_created = False
    if not manifest_path.exists():
        pd.DataFrame(columns=REAL_RBI_MANIFEST_COLUMNS).to_csv(
            manifest_path,
            index=False,
        )
        manifest_created = True
    return {
        "corpus_dir": root,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "manifest_path": manifest_path,
        "manifest_created": manifest_created,
    }


def main() -> int:
    result = bootstrap_rbi_real_corpus()
    print("RBI real-corpus bootstrap complete.")
    print(f"Raw directory: {Path(result['raw_dir']).resolve()}")
    print(f"Processed directory: {Path(result['processed_dir']).resolve()}")
    print(f"Manifest: {Path(result['manifest_path']).resolve()}")
    print(
        "Manifest created: "
        + ("yes" if result["manifest_created"] else "no; existing file retained")
    )
    print("")
    print("Manual next steps:")
    print("1. Download or manually extract public RBI text into a local UTF-8 .txt file.")
    print("2. Run scripts/import_rbi_text_document.py with the source URL and dates.")
    print("3. Run scripts/check_rbi_corpus_status.py to verify sufficiency.")
    print("No fake real documents or placeholders were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
