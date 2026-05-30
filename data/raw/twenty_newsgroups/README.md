# 20 Newsgroups (text-clustering benchmark)

Public, well-vetted text-clustering benchmark used in thousands of ML papers
since 1995. The corpus is **~20,000 newsgroup posts** spread across **20
topical categories** (atheism, autos, baseball, cryptography, electronics,
graphics, hardware, hockey, medicine, motorcycles, ms-windows, politics,
religion, sci-space, sport, etc.).

## Why this dataset?

- **Multiple natural ground-truth topics** → lets us validate that the
  pipeline's unsupervised clustering recovers semantically coherent groups.
- **Long-form prose** → exercises the transformer branch of `text_vectorizer`
  (when sentence-transformers is installed) AND the TF-IDF + TruncatedSVD
  fallback path.
- **Distributed by scikit-learn** so there's no Kaggle login, no scraping, no
  unverified third-party download.

## Source / safety

- Distributed inside `sklearn.datasets.fetch_20newsgroups`. scikit-learn is
  already a project dependency, and the dataset has been audited by the
  scientific Python community for ~25 years.
- Downloaded from `https://ndownloader.figshare.com/files/5975967` (the
  sklearn-hosted mirror) on first call and cached in `~/scikit_learn_data/`.
- Public domain news posts — **no PII, malware, phishing payloads, or
  proprietary content.**
- We strip post headers/quotes/signatures with sklearn's `remove=('headers',
  'footers', 'quotes')` arg so the cluster signal comes from the article
  body, not boilerplate.

## How to download

```bash
python data/raw/twenty_newsgroups/download.py
```

The script writes:

- `data/raw/twenty_newsgroups/twenty_newsgroups.csv` — columns:
  - `id`        — original integer index from sklearn
  - `category`  — the ground-truth newsgroup label (for validation only;
                  the pipeline runs UNsupervised — labels are never fed in)
  - `text`      — the cleaned post body

By default the script downloads the **training subset** (~11k posts) so
runs are fast. Pass `--subset all` for the full 20k-post corpus.

## How to run the pipeline on it

```bash
python run_pipeline.py \\
  --data data/raw/twenty_newsgroups/twenty_newsgroups.csv \\
  --modality text \\
  --text-column text
```

Or run the offline benchmark (no LLM calls, asserts the pipeline produces
meaningful topical clusters):

```bash
python experiments/benchmark_text_clustering.py
```
