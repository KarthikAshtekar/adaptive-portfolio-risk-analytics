"""Report governed RBI corpus intake readiness."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import validate_nlp_corpus_intake  # noqa: E402


def main() -> int:
    result = validate_nlp_corpus_intake()
    row = result["intake_status"].loc[
        result["intake_status"]["corpus"].eq("rbi")
    ].iloc[0]
    print(f"RBI corpus status: {row['corpus_status']}")
    print(f"Valid real RBI records: {int(row['valid_record_count'])}")
    print(f"Manual action required: {'yes' if row['manual_action_required'] else 'no'}")
    print("Guide: docs/nlp_real_data_acquisition_guide.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
