# RBI Real-Corpus Intake Notes

Target a bounded, reviewed corpus:

- 5–8 Monetary Policy Statements.
- 5–8 MPC Minutes.
- 3–5 Governor speeches.
- 3–5 Financial Stability or macro reports.

Cover the 2020 COVID period, the 2022 inflation/rate-hike period, 2023–2024
normalization, and the latest available period.

Store private/raw documents under `raw/`, fill `manifest.csv` from
`manifest_template.csv`, preserve official source URLs and retrieval dates, and
run `python scripts/validate_nlp_corpus_intake.py`. The example manifest is not
real data and must never be copied into empirical results unchanged.
