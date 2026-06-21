# Manual RBI Corpus Download Guide

Phase 4A.3 uses locally stored, publicly available Reserve Bank of India
communications. The dashboard and validation runner do not scrape RBI during
tests or normal analysis.

## Recommended documents

- Monetary Policy Statements
- Monetary Policy Committee Minutes
- Governor speeches
- Financial Stability Reports
- Annual Reports
- Press releases related to monetary policy, liquidity, or inflation

## Download workflow

1. Open the relevant document on the official RBI website.
2. Record the publication date, document type, title, RBI source URL, local
   file name, and retrieval date.
3. Save the document as UTF-8 `.txt` or `.md` under
   `data/sentiment/rbi_real/raw/`.
4. Add one row to `data/sentiment/rbi_real/manifest.csv`.
5. Run the manifest validator before empirical analysis.

The exact manifest header is:

```text
document_id,publication_date,document_type,title,local_path,source_url,retrieval_date,language,notes
```

Use one of these document types:

```text
mpc_minutes
monetary_policy_statement
governor_speech
press_release
financial_stability_report
annual_report
unknown
```

`local_path` is resolved relative to the manifest directory unless a separate
base directory is supplied. A typical value is `raw/rbi_mpc_2024_02_22.txt`.

## PDF workflow

PDF extraction is optional. If `pypdf` is available, the corpus validator can
read a local PDF. The preferred reproducible workflow is:

1. Keep large raw PDFs outside Git.
2. Extract and review their text manually.
3. Save the reviewed text as UTF-8 `.txt` or `.md`.
4. Record the extracted text path and original RBI URL in the manifest.

## Source and copyright rules

- Use only publicly available RBI communications.
- Do not copy copyrighted third-party news articles into the repository.
- Preserve the official RBI source URL and retrieval date.
- Save long documents as `.txt` or `.md`.
- Review extracted text for page headers, footers, and OCR errors.

## Validation

From the repository root:

```python
from src.sentiment.rbi_corpus_builder import validate_rbi_manifest

result = validate_rbi_manifest("data/sentiment/rbi_real/manifest.csv")
print(result["diagnostics"])
print(result["invalid_documents"])
```

Only valid rows are used by the empirical validation runner. Invalid rows are
reported without aborting the entire corpus.

## Optional transformer scoring

Transformer scoring is optional and must not download model weights during
tests. For future local evaluation, the adapter recognizes the gtfintechlab
RBI stance, certainty, and time-orientation models:

```text
gtfintechlab/model_reserve_bank_of_india_stance_label
gtfintechlab/model_reserve_bank_of_india_certain_label
gtfintechlab/model_reserve_bank_of_india_time_label
```

Install and cache any chosen model separately, then use
`score_rbi_sentences(..., method="transformer")`. If the dependency or local
weights are unavailable, scoring falls back sentence-by-sentence to the
deterministic lexicon and records fallback metadata.
