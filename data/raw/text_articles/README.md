# Sample text dataset — `text_articles`

A tiny, synthetic corpus for exercising the **text-clustering** modality of the
pipeline. 36 short articles spanning three clear themes:

- **Sports** (rows 1–10, 31, 34) — matches, records, tournaments
- **Technology** (rows 11–20, 32, 35) — devices, AI, software, security
- **Cooking** (rows 21–30, 33, 36) — recipes and dishes

## Columns
- `id` — row identifier
- `title` — a short headline
- `body` — the free-text article (this is the column the TextPreparer embeds)

## Usage
```bash
python run_pipeline.py --data data/raw/text_articles/text_articles.csv \
                       --modality text --text-column body
```
Or set `modality: text` (and optionally `text_column: body`) in `config.yaml`.

With auto-detection the pipeline should also pick `body` as the text column and
discover roughly three themed clusters. It is intentionally small so a full run
is fast and cheap.
