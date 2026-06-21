# Real RBI Corpus

This directory is the local, reproducible input surface for Phase 4A.3.

## Structure

- `raw/`: manually downloaded `.txt` or `.md` RBI communications. Large raw PDFs should remain outside Git.
- `processed/`: optional normalized text produced during manual preparation.
- `manifest.csv`: exact source and timestamp metadata for each document.

The manifest columns are:

```text
document_id,publication_date,document_type,title,local_path,source_url,retrieval_date,language,notes
```

No real RBI documents are bundled by default. Follow
[`docs/rbi_corpus_manual_download.md`](../../../docs/rbi_corpus_manual_download.md)
to populate the corpus. Synthetic fixtures remain under
`data/sentiment/rbi_documents/` for tests and pipeline validation.
