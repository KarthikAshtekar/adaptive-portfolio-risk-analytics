"""Validate RBI, earnings-call, and news real-corpus intake manifests."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sentiment import validate_nlp_corpus_intake  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "reports" / "nlp_corpus_intake_validation"
)


def run_intake_validation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rbi_manifest: str | Path | None = None,
    earnings_manifest: str | Path | None = None,
    news_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Persist corpus-level and row-level diagnostics without failing on gaps."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = validate_nlp_corpus_intake(
        rbi_manifest=rbi_manifest,
        earnings_manifest=earnings_manifest,
        news_manifest=news_manifest,
    )
    result["intake_status"].to_csv(output / "intake_status.csv", index=False)
    for corpus in ("rbi", "earnings", "news"):
        result["row_diagnostics"][corpus].to_csv(
            output / f"{corpus}_status.csv", index=False
        )
    status_lines = "\n".join(
        f"- {row.corpus}: {row.status_message}"
        for row in result["intake_status"].itertuples(index=False)
    )
    valid_counts = result["valid_real_records_by_corpus"]
    summary = f"""# NLP Corpus Intake Validation

Generated: {date.today().isoformat()}

## Status

**Manual action required: {'Yes' if result['manual_action_required'] else 'No'}**

{status_lines}

## Valid real records

- RBI: {valid_counts['rbi']}
- Earnings calls: {valid_counts['earnings']}
- News/geopolitical: {valid_counts['news']}

Placeholder and synthetic fixture rows are excluded. Missing corpora do not
cause the validator to fail; they remain explicit manual-action items. Intake
readiness does not establish predictive value, and NLP remains monitoring-only.

See `docs/nlp_real_data_acquisition_guide.md` for collection instructions.
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    result["output_dir"] = output
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate governed real NLP corpus intake manifests."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rbi-manifest", default=None)
    parser.add_argument("--earnings-manifest", default=None)
    parser.add_argument("--news-manifest", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_intake_validation(
        output_dir=args.output_dir,
        rbi_manifest=args.rbi_manifest,
        earnings_manifest=args.earnings_manifest,
        news_manifest=args.news_manifest,
    )
    counts = result["valid_real_records_by_corpus"]
    print(
        "Intake validation completed: "
        f"RBI={counts['rbi']}, earnings={counts['earnings']}, "
        f"news={counts['news']} valid real record(s)."
    )
    print(
        "Manual action required: "
        f"{'yes' if result['manual_action_required'] else 'no'}"
    )
    print(f"Outputs: {Path(result['output_dir']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
