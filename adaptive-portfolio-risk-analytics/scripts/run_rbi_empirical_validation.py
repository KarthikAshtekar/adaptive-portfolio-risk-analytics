"""Run real-RBI empirical validation only when intake is ready."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import (  # noqa: E402
    run_rbi_empirical_validation,
    validate_nlp_corpus_intake,
)


OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "reports" / "phase_4a3_real_rbi_macro_validation"
)


def main() -> int:
    intake = validate_nlp_corpus_intake()
    rbi = intake["corpora"]["rbi"]
    if rbi["summary"]["valid_record_count"] == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "intake_status.md").write_text(
            "# RBI Empirical Validation\n\n"
            "Manual action required: no valid real RBI corpus is available.\n"
            "See `docs/nlp_real_data_acquisition_guide.md`.\n",
            encoding="utf-8",
        )
        print("RBI empirical validation not run: manual action required.")
        return 0
    index = pd.bdate_range("2020-01-01", pd.Timestamp.today().normalize())
    rng = np.random.default_rng(47)
    returns = pd.DataFrame(
        {"SYNTH_A": rng.normal(0, 0.01, len(index)),
         "SYNTH_B": rng.normal(0, 0.01, len(index))},
        index=index,
    )
    regimes = pd.Series("Normal", index=index)
    run_rbi_empirical_validation(
        rbi["manifest_path"],
        returns,
        regimes,
        pd.Series("Unknown", index=index),
        OUTPUT_DIR,
    )
    print("RBI empirical validation completed with real intake records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
