# Changelog — 2026-05-16

Branch: `claude/interactive-cluster-interface-Pu2Ci`
Commits: `8c0bc6a` (`feat: live UI + adaptive escalation + multi-domain datasets`) + `df9bbeb` (`chore: remove .png screenshots`)

A single long working session that took the project from "pipeline + standalone view-only UI" to a live, interactive, demo-recordable system.

## 1. Pipeline ↔ UI fused into one process

- `run_pipeline.py` now boots the Flask UI in a background thread + auto-opens the browser (use `--no-ui` for headless).
- `OrchestratorBus` writes an incremental JSONL event stream (`outputs/pipeline_events.jsonl`) consumed by the UI via Server-Sent Events (`/api/events/stream`).
- On every `OrchestratorBus.__init__`, the bus truncates the event log AND wipes per-run files (`personas.json`, `cluster_profiles.json`, `classifier_metrics.json`, `silhouette_curve.json`, `cluster_lineage.json`, `last_upload_preview.json`, `pca_iterations.json`, `pending_*.json`) — guaranteed clean slate on every restart.
- Live "agentic" UI structure:
  - **Architecture graph** at the top (orchestrator at center, 7 agents in a wide row, edges that pulse + light up as agents talk to the Decision Maker).
  - **Tokens & cost panel** directly below the graph.
  - **Agent log** (full streaming events).
  - **Agent ↔ Decision Maker conversations** (chat bubbles typed live).
  - **Pipeline timeline** with per-step status.
  - **Right column** — per-agent computed-result cards with iteration history pills.
  - **3 tabs**: Live pipeline | Data & evidence | Named clusters.

## 2. Adaptive Decision Maker escalation

After every Clusterer iteration:

- `silhouette_target` (default **0.5** in `config.yaml`) — if Clusterer's silhouette is below target → loop back to FeatureSelector.
- `max_reselect_failures` (default **3**) — after 3 consecutive misses → **re-engineer features from raw data** + clear algorithm pick so the LLM chooses fresh next round.
- `max_relax_failures` (default **3**) — after 3 consecutive misses → in **bypass** mode auto-lowers `silhouette_target` by −0.1; in **interactive** mode opens a relax modal so the user picks the new bar.

Both rules fire together at 3 misses. Target trajectory: 0.5 → 0.4 → 0.3 → … (floor 0.05). All escalations appear as **Orchestrator** entries in the right-column agent outputs panel.

## 3. Mode toggle + interactive warnings

- Topbar **Bypass / Interactive** toggle persisted to `outputs/pipeline_mode.json`.
- **Bypass**: when a warning fires, the frontend auto-POSTs to `/api/explain` (category=`evidence`) and renders the LLM's recommended action in active voice ("The pipeline chose to apply log-transform…") inside the agent's output card. Prior decision text stays visible during update — no flash of "loading…" wipes.
- **Interactive**: the bus pauses on warnings, the decision modal lets the user write guidance, which is saved as a high-priority memory rule (`global_rule` in `outputs/user_feedback_log.jsonl`) and picked up by the very next iteration's prompts.

## 4. Three cost ledgers

`OrchestratorBus.ask(..., category=...)` now accepts a `category` parameter:

| Ledger | Source | Purpose |
|---|---|---|
| **`pipeline`** | Every Orchestrator-mediated LLM call during the pipeline | Bills the cost of the agents' actual work |
| **`evidence`** | `/api/explain` (bypass auto-decisions + manual "explain this warning" button) | Transparency-only spend, kept separate |
| **`naming`** | `/api/cluster-chat` (per-cluster multi-turn discussion with the LLM) | User chat with the agent to refine a cluster's name |

The Tokens & cost panel shows three independent stat boxes + three collapsible tables.

## 5. Data & Evidence tab

- **Dataset profile** card (populated from upload preview, enriched by DatasetExaminer).
- **Per-column skewness** bar chart with inline mini-histograms + a separate "distribution evidence" card with full-size histograms for the top-3 worst-skew columns (e.g. `amt` skew=25.6 on fraudTrain).
- **Suggested feature groups** card (from DatasetExaminer's `context.group_details`).
- **Per-iteration history cards** for FeatureEngineer / FeatureSelector / Clusterer — every iter stacked vertically. Each Clusterer row shows an inline 2-D **PCA scatter** (saved per iter by `Orchestrator._save_pca_projection`, up to 1500 sampled points coloured by cluster id).
- **PersonaNamer + Classifier**: only the **best** iteration shown (winner = Clarity-Gate-passed with max avg_confidence / max F1).
- **Silhouette curve / cluster sizes / per-class F1 / lineage** — populated on `pipeline_complete`.
- **Final pipeline summary card** pinned to the top of Evidence when complete: 5 big numbers (iterations · tokens · cost · time · winning iter) + per-iter table with Algo / k / Silhouette / F1 / Naming gate / Tokens / Cost / Time. Winning row highlighted green with ★.
- **All-pipeline-outputs file list** filtered by `mtime` so only files from this run show.

## 6. Per-cluster chat (new naming workflow)

- New `POST /api/cluster-chat` endpoint — stateless multi-turn chat about one cluster. The system prompt is grounded in the cluster's actual top above-/below-average features + values, so it cites real evidence (no hallucinations).
- UI: Each cluster card opens a detail panel with a **chat box** in addition to the existing inline-edit fields and one-shot regenerate-with-hint textarea.
- **Conclude → propose action** button asks the LLM for a structured JSON proposal `{summary, action, new_name?, merge_with?, reason}`, then offers clickable Apply buttons that map to existing endpoints (rename → `PUT /api/personas/<cid>`, merge → `POST /api/clusters/merge`, keep/recluster → save as `global_rule` memory).
- Cost lands in the **naming** ledger.

## 7. Memory drawer redesign

- Inline **"+ Add a new memory rule"** form right in the drawer (no need to find the topbar button).
- Filter chips: All / Global rules / Naming hints / Manual edits / Merges / Inactive.
- Each row labeled `user_change : <priority> : <YYYY-MM-DD>`.
- New `DELETE /api/feedback/<id>` endpoint + Delete button per row.

## 8. Data + recording

- `download_datasets.py` extended to **12 UCI / Kaggle datasets** across domains:
  - **Humans**: adult_census (48k × 15), credit_default (30k × 24), bank_marketing (45k × 17), ibm_hr (1.5k × 35)
  - **Products**: wholesale_customers (440 × 8), wine_quality (6.5k × 12), news_popularity (40k × 60)
  - **Signals**: magic_gamma (19k × 10), breast_cancer_wdbc (569 × 30), pendigits (11k × 16), air_quality (9k × 13), occupancy (20k × 6)
  - **Transactions**: online_retail (542k × 8), mall_customers (200 × 5)
  - **Images-as-tabular** (Kaggle): fashion_mnist (70k × 784) — Kaggle key required
  - All landed under `data/raw/` (force-added past `*.csv` gitignore).
  - Smallest for quick demos: **wholesale_customers** (~5 s / iter).
- `record_demo.py` (Playwright-based): launches **7 Chromium windows**, one per `?demo=<area>` URL (intent / graph / log / convos / outputs / evidence / tokens), records each to its own `.webm` while the pipeline runs. Auto-submits intent or waits for the user (`--no-auto-submit`). Auto-renames files at end to `recordings/<area>.webm`. Uses `--disable-background-timer-throttling` flags so non-focused windows don't lag.

## 9. New demo URL params

Add `?demo=<area>` to any URL to isolate one part of the UI for a clean screen recording:

| Area | What's visible |
|---|---|
| `intent` | The intent form, full-screen |
| `graph` | Architecture graph only |
| `log` | Streaming agent log only |
| `convos` | Agent ↔ Decision Maker chat bubbles only |
| `outputs` | Right-column agent outputs only |
| `evidence` | Data & evidence tab only |
| `tokens` | Tokens & cost panel only |

A small red "× Exit demo mode" button appears at top-right to return to the normal UI.

## 10. Critical lessons we hit (so we don't repeat them)

- `OrchestratorBus()` instantiated by anything (e.g. `ui/llm_bridge.py` for the regenerate / merge / explain / cluster-chat endpoints) used to truncate the live event log + delete the upload preview because `__init__` always touches `DEFAULT_EVENT_LOG`. **Fix:** `ui/llm_bridge.py` now passes `event_log_path=None`. Any new code that creates a transient bus for a UI endpoint MUST do the same — otherwise it'll silently kill the running pipeline's state.
- Re-applying the same CSS class to the SVG graph nodes on every event was restarting the CSS animation from frame 0 — animations looked frozen. **Fix:** `data-statusClass` dedupe in `setArchNodeStatus`.
- Playwright's `wait_until="networkidle"` never fires for a page with a long-lived SSE connection. **Fix:** use `"domcontentloaded"` + a short `wait_for_timeout`.
- Chromium throttles non-focused windows. **Fix:** pass `--disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows` when recording multiple windows at once.
- The auto-decision text in bypass mode used to disappear briefly while the next LLM call was in flight (overwrite-on-trigger). **Fix:** preserve prior decision text until the new one arrives.
- **Closing Cursor kills the pipeline + recording.** Parent chain runs Cursor → Cursor Helper → zsh → claude → python. **Fix:** always launch with `nohup setsid` (see `docs/HOWTO_run_and_record.md`).
