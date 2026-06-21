# Real NLP Data Acquisition Guide

This guide defines the governed intake workflow for real, publication-timestamped
text used by the Phase 4A.6 validation harness. NLP remains monitoring-only.
Do not add market reactions, future returns, or price-derived labels to these
corpora.

## Common intake rules

1. Record the original publication time/date, retrieval time/date, source URL,
   language, and enough entity/topic metadata to audit every row.
2. Keep synthetic fixtures and `DO_NOT_USE_PLACEHOLDER` examples separate from
   real records.
3. Store private, licensed, large, or copyrighted source files only under the
   ignored `raw/` directories. Commit metadata only when licensing permits.
4. Prefer UTF-8 `.txt` or `.md` extracts. Keep the original URL and note any
   manual conversion from PDF or HTML.
5. Avoid documents whose primary content is only a post-event market-price move.
6. Run `python scripts/validate_nlp_corpus_intake.py` before collection.

## 1. RBI and monetary-policy communications

### Recommended sources

- RBI Monetary Policy Committee resolutions and minutes.
- RBI monetary-policy statements and press releases.
- Governor or Deputy Governor speeches hosted on the official RBI website.
- RBI Financial Stability Reports and annual reports.

Use official `rbi.org.in` publication pages wherever possible. Do not perform
high-frequency scraping; download a bounded reviewed corpus manually.

### What to record

- The reviewed text or a local PDF-to-text extract.
- Official publication date, title, document type, source URL, retrieval date,
  language, and preparation notes.

### Files and location

- Save private/raw source material under `data/sentiment/rbi_real/raw/`.
- Prefer UTF-8 `.txt` or `.md`.
- Fill `data/sentiment/rbi_real/manifest.csv` using
  `manifest_template.csv`.

Required columns:

```text
document_id,publication_date,document_type,title,local_path,source_url,retrieval_date,language,notes
```

Allowed document types are `mpc_minutes`, `monetary_policy_statement`,
`governor_speech`, `press_release`, `financial_stability_report`,
`annual_report`, and `unknown`.

### Quality checks

- Confirm the local file exists and is non-empty.
- Confirm publication and retrieval dates are valid.
- Use unique document IDs.
- Verify the source URL resolves to the stated RBI communication.
- Check duplicate title/date/source combinations.
- Review text extraction for broken pages, repeated headers, and OCR errors.

### Legal caution and validation

RBI public communications should still retain attribution and source URLs. Do
not remove copyright or use restrictions from downloaded material.

```bash
python scripts/validate_nlp_corpus_intake.py
python scripts/check_rbi_corpus_status.py
python scripts/run_rbi_empirical_validation.py
```

## 2. Earnings-call and company transcripts

### Recommended sources

- Company investor-relations pages.
- Exchange filings containing legally redistributable call transcripts.
- Licensed transcript services only when local storage and analysis are
  permitted by the applicable agreement.

Do not commit paid or copyrighted transcripts unless redistribution is
explicitly permitted.

### What to record

- Company, ticker, sector, quarter, publication date, title, source URL,
  retrieval date, language, and licensing/provenance notes.
- A local reviewed transcript only when storage is permitted.

### Files and location

- Store private transcripts under `data/sentiment/earnings_calls/raw/`.
- Fill `data/sentiment/earnings_calls/manifest.csv` using
  `manifest_template.csv`.

Required columns:

```text
document_id,company,ticker,sector,quarter,publication_date,title,local_path,source_url,retrieval_date,language,notes
```

### Quality checks

- Confirm the file exists, is non-empty, and matches the stated company/quarter.
- Require sector, publication date, retrieval date, language, and source URL.
- Use unique document IDs and flag duplicate title/date/source rows.
- Note edited, summarized, machine-generated, or partial transcripts.

### Legal caution and validation

Keep licensed or private text under the ignored `raw/` directory. Metadata can
be committed only when it does not disclose restricted content.

```bash
python scripts/validate_nlp_corpus_intake.py
python scripts/collect_real_nlp_data.py --config config/nlp_providers.example.yaml --start-date 2020-01-01 --end-date 2026-06-21 --no-live
```

## 3. Geopolitical and financial news

### Recommended sources

- Public or licensed news APIs with explicit retention terms.
- Official government, central-bank, exchange, or company releases.
- Manually reviewed metadata exports where full article storage is not allowed.

Prefer storing a short permitted summary or source-provided snippet rather than
copyrighted full article text.

### What to record

- Publication time, retrieval time, source, provider, document type,
  entity/ticker/sector/country, title, permitted text or summary, URL,
  language, and provenance notes.

### Files and location

- Fill `data/sentiment/news_real/manifest.csv` using
  `manifest_template.csv`.
- Store any permitted private exports under `data/sentiment/news_real/raw/`.

Required columns:

```text
record_id,publication_time,source,provider,document_type,entity,ticker,sector,country,title,text,url,language,retrieval_time,notes
```

Recommended themes include India inflation, RBI rate hikes, liquidity stress,
oil-price shocks, geopolitical risk, banking stress, currency pressure, global
recession risk, and supply-chain disruption.

### Quality checks

- Require publication time no later than retrieval time.
- Require source, provider, URL, language, title, and non-empty permitted text.
- Use unique record IDs and flag duplicate title/time/source rows.
- Verify entity/topic relevance and avoid syndicated duplicates.
- Avoid pure market reaction articles where the primary content is only that
  prices moved.

### Legal caution and validation

Do not commit copyrighted third-party articles or paid-feed payloads. Retain
only permitted fields and keep private exports ignored.

```bash
python scripts/validate_nlp_corpus_intake.py
python scripts/collect_real_nlp_data.py --config config/nlp_providers.example.yaml --start-date 2020-01-01 --end-date 2026-06-21 --no-live
python scripts/validate_real_nlp_signal.py --input-records outputs/reports/phase_4a6_real_nlp_validation/deduped_sentiment_records.csv
```

## Intake readiness

The intake validator writes corpus-level and row-level diagnostics under
`outputs/reports/nlp_corpus_intake_validation/`. Missing manifests, zero valid
real records, placeholders, and invalid rows remain manual-action items. Passing
intake validation only establishes provenance and format readiness; it does not
establish predictive value or authorize allocation testing.
