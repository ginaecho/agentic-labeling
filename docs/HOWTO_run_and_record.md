# How to run the pipeline + record a demo (safely)

This guide covers the practical commands to start the pipeline, launch the multi-window recording, and — critically — how to **keep them running even after you close Cursor**.

## TL;DR — fully detach so closing Cursor doesn't kill the run

**macOS does NOT ship `setsid` as a binary.** `nohup setsid python ... &` fails instantly with `setsid: No such file or directory` and nothing launches. Use the Python wrapper below (it calls `os.setsid()` internally via `start_new_session=True`):

```bash
python -c "
import subprocess, sys
subprocess.Popen(
    [sys.executable, 'run_pipeline.py',
     '--data', 'data/raw/wholesale_customers/wholesale_customers.csv'],
    stdout=open('/tmp/pipeline_run.log', 'wb'),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True, close_fds=True,
)
"

python -c "
import subprocess, sys
subprocess.Popen(
    [sys.executable, 'record_demo.py', '--no-auto-submit', '--timeout', '1500'],
    stdout=open('/tmp/record_demo.log', 'wb'),
    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    start_new_session=True, close_fds=True,
)
"
```

After both spawn, **verify PPID is 1**:
```bash
ps -p <pid> -o pid,ppid,stat,command   # PPID must be 1, STAT must contain 's'
```

If `setsid` IS available (Linux, or `brew install util-linux` on macOS providing `gsetsid`), the older recipe also works:

```bash
nohup setsid python run_pipeline.py \
    > /tmp/pipeline_run.log 2>&1 < /dev/null &
```

- `nohup` ignores SIGHUP (the signal sent to children when a parent terminal closes).
- `setsid` puts the python process into its own **session**, detached from the controlling terminal.
- `< /dev/null` closes stdin so the process can't block waiting on it.
- `> /tmp/...log 2>&1` keeps stdout + stderr reachable from any new terminal.

After this you can close Cursor freely — the python processes get re-parented to PID 1 (launchd) and keep running. Check progress from any new terminal:

```bash
tail -f /tmp/pipeline_run.log
tail -f /tmp/record_demo.log
```

## Why this matters

When launched from inside Cursor, the parent chain is:

```
Cursor → Cursor Helper (terminal) → zsh → claude → python (pipeline | record_demo) → chromium...
```

Plain `python ...&` or `exec python ...&` keeps the python process attached to that chain. **Closing Cursor → all children die mid-recording**, leaving `.webm` files truncated or corrupted. `nohup setsid` breaks the chain.

## Common scenarios

### Quick smoke test on the tiniest dataset

```bash
nohup setsid python run_pipeline.py \
    --data data/raw/wholesale_customers/wholesale_customers.csv \
    > /tmp/pipeline_run.log 2>&1 < /dev/null &
# ~5 seconds per iteration → entire run finishes in well under a minute
```

### Full demo run with recording

```bash
# 1. Start the pipeline
nohup setsid python run_pipeline.py \
    > /tmp/pipeline_run.log 2>&1 < /dev/null &

# 2. Wait until you see 'awaiting_intent' in the log
sleep 4 && grep awaiting_intent /tmp/pipeline_run.log

# 3. Launch the 7-window recording (user fills intent manually)
nohup setsid python record_demo.py --no-auto-submit --timeout 1500 \
    > /tmp/record_demo.log 2>&1 < /dev/null &

# 4. Fill the intent form in the ?demo=intent window.
# 5. Walk away — recording auto-stops on pipeline_complete and saves
#    seven .webm files to recordings/.
```

### Headless run (no UI window)

```bash
nohup setsid python run_pipeline.py --no-ui \
    > /tmp/pipeline_run.log 2>&1 < /dev/null &
```

### Different port

```bash
nohup setsid python run_pipeline.py --ui-port 5058 \
    > /tmp/pipeline_run.log 2>&1 < /dev/null &
```

## Stopping cleanly

```bash
# Stop pipeline (also stops the embedded Flask UI)
pkill -f "python.*run_pipeline"

# Stop recording — SIGINT first so .webm files get properly flushed + renamed
pkill -INT -f "record_demo"
sleep 5      # let the cleanup code run
pkill -f "record_demo"      # if anything's still alive

# Force-kill stragglers (last resort)
pkill -9 -f "chromium-headless-shell"
```

If the recording is interrupted before its cleanup runs, `.webm` files end up in subdirectories like `recordings/graph/page@<hash>.webm` rather than at `recordings/graph.webm`. Rename manually:

```bash
cd recordings
for region in intent graph log convos outputs evidence tokens; do
  [ -d "$region" ] || continue
  f=$(ls -S "$region"/*.webm 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    mv "$f" "${region}.webm"
    rm -rf "$region"
  fi
done
```

## Viewing the .webm files

macOS QuickTime can't play `.webm` natively. Three options:

1. **Browser**: `open recordings/graph.webm` opens it in your default browser (Chrome/Safari/Firefox all play webm).
2. **VLC**: `brew install --cask vlc` then `open -a VLC recordings/graph.webm`.
3. **Convert to MP4**: `brew install ffmpeg` then
   ```bash
   cd recordings && for f in *.webm; do
     ffmpeg -i "$f" -c:v libx264 -crf 18 "${f%.webm}.mp4"
   done
   ```

## Demo URL reference

Add `?demo=<area>` to `http://127.0.0.1:5057/` to isolate one UI region for a clean recording:

| URL | Captures |
|---|---|
| `?demo=intent` | The intent form, full-screen — record yourself filling + submitting it |
| `?demo=graph` | Architecture graph lighting up agent-by-agent |
| `?demo=log` | Streaming text log of every pipeline event (largest text-content view) |
| `?demo=convos` | Agent ↔ Decision Maker chat bubbles typing live |
| `?demo=outputs` | Right-column agent outputs with iteration history pills |
| `?demo=evidence` | Data & evidence tab updating per iteration (PCA scatters appear here) |
| `?demo=tokens` | Three-ledger Tokens & cost panel ticking up |

A red **× Exit demo mode** button appears top-right of every demo page.
