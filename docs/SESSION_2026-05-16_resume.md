# Resume notes — 2026-05-16 evening session

A snapshot of what we did and what's running, so the next session can pick up safely.

## Conversation summary

1. **User asked if I remembered prior work.** I recalled the 2026-05-16 build (branch `claude/interactive-cluster-interface-Pu2Ci`, commits `8c0bc6a` + `df9bbeb`): live UI fused with pipeline, adaptive escalation, Bypass/Interactive modes, 3 cost ledgers, Data & Evidence tab, memory drawer redesign, 12 datasets, Playwright 7-window recorder.

2. **User asked to restart the pipeline + launch the 7-window recorder + run both under nohup so closing Cursor wouldn't kill them.**

3. **First launch failed silently.** The old memory said `nohup setsid python ... &`, but **macOS does not ship `setsid`** — `nohup` reported `setsid: No such file or directory` and nothing started. The pipeline PID I quoted was a ghost.

4. **Fix:** relaunched using Python's `subprocess.Popen(..., start_new_session=True)`, which calls `os.setsid()` in the child before exec. Both processes ended up with `PPID=1` (launchd) — fully detached.

5. **User typed intent in the `?demo=intent` Chromium window.** Pipeline began working (lots of `POST /api/explain` traffic — Bypass-mode auto-explanations). All 7 `.webm` files were growing.

6. **User confirmed they want to close Cursor.** Confirmed it's safe: the pipeline + recorder are children of launchd, not of Cursor, so SIGHUP won't reach them. Closing Cursor kills this Claude session and the Cursor terminal, but the run survives.

## Running state at time of writing

| Component | PID | PPID | What it's doing |
|---|---|---|---|
| Pipeline (`run_pipeline.py --data data/raw/wholesale_customers/wholesale_customers.csv`) | 72633 | 1 | Flask UI on `:5057`, orchestrator executing iterations |
| Recorder (`record_demo.py --no-auto-submit --timeout 1500`) | 72660 | 1 | 7 Chromium windows recording to `recordings/<area>/page@*.webm` |

Logs:
- `/tmp/pipeline_run.log`
- `/tmp/record_demo.log`

Per-window webm files were already at: graph 13 MB, log 3.3 MB, convos 2.5 MB, intent/outputs/tokens/evidence 1–1.5 MB and growing.

## The user's last instruction (still in flight)

> "I already inputed my intent" — leave the pipeline + recorder running, close Cursor, the recording must complete on its own.

When the pipeline emits `pipeline_complete` (or the 1500 s recorder timeout fires), `record_demo.py` will:
1. Close each Chromium context cleanly → flushes each `.webm`.
2. Rename `recordings/<area>/page@<hash>.webm` → `recordings/<area>.webm`.
3. Clean up empty subdirectories.

## What to verify when resuming

From any new terminal (Cursor or Terminal.app):

```bash
# Are they still alive?
ps -p 72633,72660 -o pid,ppid,stat,etime,command

# Did the pipeline finish?
grep -E "pipeline_complete|FATAL|Traceback" /tmp/pipeline_run.log | tail

# Are the final webm files in place at the TOP level of recordings/?
ls -lh recordings/*.webm

# If the per-window files are still in subdirs (recorder was killed before
# cleanup ran), rename manually using the snippet in docs/HOWTO_run_and_record.md
# under "Stopping cleanly".
```

If the pipeline crashed mid-run, the per-window files in `recordings/<area>/page@*.webm` are still recoverable — they just need the manual rename step.

## Important macOS gotcha (now fixed in memory + HOWTO)

- `setsid` is **not** a binary on macOS by default.
- The correct detach recipe on macOS is the `subprocess.Popen(..., start_new_session=True)` wrapper shown at the top of `docs/HOWTO_run_and_record.md`.
- Always verify with `ps -p <pid> -o pid,ppid,stat` — **PPID must be 1**, otherwise the process is still tied to Cursor and will die when Cursor closes.

## Loose ends / things to consider next session

- Decide whether the `setsid`-via-`nohup` line should be removed from the HOWTO entirely, or kept as a Linux fallback (currently kept as fallback).
- The uncommitted changes from before this session are still uncommitted: `record_demo.py`, `ui/static/style.css`, plus the new docs in this folder. Decide whether to commit after the recording finishes and the videos look good.
- If recordings look right, also consider committing the updated `docs/HOWTO_run_and_record.md` macOS fix.
