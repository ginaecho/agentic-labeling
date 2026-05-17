"""
record_demo.py — Capture one .webm per UI region during a live pipeline run.

What it does
------------
1. Launches a headed Chromium (you can see it on screen)
2. Opens FIVE browser contexts (one per `?demo=...` URL), each at 1280×720
3. Each context records its own video to `recordings/<area>.webm`
4. Submits the intent automatically via the UI form so the pipeline starts
5. Watches for `pipeline_complete` (or the configured timeout) then stops cleanly

Usage
-----
    # Make sure run_pipeline.py is already running (pipeline waits for intent):
    python run_pipeline.py
    # In another terminal:
    python record_demo.py

Outputs land in `recordings/`:
    recordings/intent.webm     ← the user filling + submitting the intent form
    recordings/graph.webm      ← architecture graph lighting up agents
    recordings/log.webm        ← streaming text log of every pipeline event
    recordings/convos.webm     ← agent ↔ Decision Maker chat bubbles typing live
    recordings/outputs.webm    ← right-column per-agent computed results
    recordings/evidence.webm   ← Data & evidence tab updating per iteration
    recordings/tokens.webm     ← 3-ledger Tokens & cost panel ticking up
    recordings/named.webm      ← Named Clusters tab (populates after pipeline_complete)

Notes
-----
- Each .webm is the FULL viewport of one context (cleanly cropped to that region
  because the `?demo=X` URL hides everything else).
- Convert with ffmpeg if you need mp4:
    for f in recordings/*.webm; do ffmpeg -i "$f" -c:v libx264 -crf 18 "${f%.webm}.mp4"; done
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import requests

BASE_URL = "http://127.0.0.1:5057"   # overridden by --base-url / --port at runtime
# Order matters for tiling: intent first (user fills it), then live views, then
# 'named' which only has content after pipeline_complete (the cluster grid +
# detail panel populate once personas land in outputs/personas.json).
REGIONS = ["intent", "graph", "log", "convos", "outputs", "evidence", "tokens", "named"]
VIEWPORT = {"width": 1280, "height": 720}
# Bare "full" walkthrough — keep the window small enough to FIT inside the
# user's visible Mac screen (minus the menu bar + dock). 1280×800 fits on every
# MacBook 13"/14"/16" without parts of the UI being clipped offscreen, which is
# what made scrolling unrecordable on the previous attempt.
VIEWPORT_FULL = {"width": 1280, "height": 800}
OUT_DIR = pathlib.Path("recordings")


def _status() -> dict:
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=5)
        return r.json()
    except Exception:
        return {}


def _submit_intent(target: str, purpose: str, dataset_path: str, k: int | None) -> None:
    """POST /api/intent — the pipeline picks this up from outputs/pending_intent.json."""
    payload = {
        "target_entity": target,
        "business_purpose": purpose,
        "dataset_path": dataset_path,
        "constraints": "",
        "n_clusters_requested": k,
        "must_have_clusters": [],
    }
    r = requests.post(f"{BASE_URL}/api/intent", json=payload, timeout=5)
    r.raise_for_status()
    print(f"  [demo] intent submitted: target={target!r} k={k} dataset={dataset_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="customers",
                    help="target_entity for the pipeline")
    ap.add_argument("--purpose",
                    default="cluster customers based on spending preferences and behaviour",
                    help="business_purpose for the pipeline")
    ap.add_argument("--dataset", default="data/uploads/fraudTrain_4.csv",
                    help="dataset path used by the pipeline")
    ap.add_argument("--k", type=int, default=8,
                    help="n_clusters_requested hint")
    ap.add_argument("--timeout", type=int, default=600,
                    help="max seconds to wait for pipeline_complete (default 600 = 10min)")
    ap.add_argument("--auto-submit", action="store_true", default=True,
                    help="auto-submit the intent (default true)")
    ap.add_argument("--no-auto-submit", dest="auto_submit", action="store_false",
                    help="don't submit intent automatically — wait for user")
    ap.add_argument("--regions", nargs="*", default=REGIONS,
                    help=f"which regions to record (default: all {len(REGIONS)})")
    ap.add_argument("--port", type=int, default=None,
                    help="UI port (default 5057 — must match run_pipeline.py --ui-port)")
    ap.add_argument("--base-url", default=None,
                    help="full UI URL like http://127.0.0.1:5090 (overrides --port)")
    ap.add_argument("--skip-pipeline-check", action="store_true",
                    help="don't require pipeline_running=true (useful when recording "
                         "post-completion views like 'named' against already-saved personas)")
    ap.add_argument("--stop-on-key", action="store_true",
                    help="record until you press Enter in this terminal instead of waiting "
                         "for pipeline_complete (use when capturing manual interactions like "
                         "renaming clusters)")
    args = ap.parse_args()

    # Resolve the UI base URL: --base-url wins, then --port, then BASE_URL default.
    global BASE_URL
    if args.base_url:
        BASE_URL = args.base_url.rstrip("/")
    elif args.port:
        BASE_URL = f"http://127.0.0.1:{args.port}"
    print(f"  [demo] using UI at {BASE_URL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check: pipeline UI is reachable
    status = _status()
    if not status:
        sys.exit(f"ERROR: cannot reach {BASE_URL}/api/status — is the pipeline running?")
    if not status.get("pipeline_running") and not args.skip_pipeline_check:
        sys.exit("ERROR: pipeline is not in 'running' state. Start `python run_pipeline.py` "
                 "first, or pass --skip-pipeline-check to record against the previous run's "
                 "saved personas (useful for the 'named' region).")
    if "awaiting_intent" not in {e for e in []} and not args.auto_submit:
        print("  [demo] note: pipeline state =", json.dumps(status, indent=2))

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        sys.exit(f"ERROR: playwright not importable ({exc}). Run: pip install playwright && playwright install chromium")

    with sync_playwright() as p:
        # Launch one shared browser, separate context per region (each gets its own video)
        # Disable Chromium's background-tab throttling so all 5 windows render
        # in real time even when they're not the focused window. Without these
        # flags, non-focused windows slow their JS / fetches / animations and
        # the Data & Evidence tab visibly lags the others.
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--window-position=0,0",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=CalculateNativeWinOcclusion",
            ],
        )
        contexts = {}
        pages = {}
        for region in args.regions:
            ctx_dir = OUT_DIR / region
            ctx_dir.mkdir(exist_ok=True)
            _vp = VIEWPORT_FULL if region == "full" else VIEWPORT
            ctx = browser.new_context(
                viewport=_vp,
                record_video_dir=str(ctx_dir),
                record_video_size=_vp,
            )
            page = ctx.new_page()
            # 'full' is a special pseudo-region: load the bare UI (no ?demo= param)
            # so topbar, tabs, detail panel — everything — is interactive and
            # visible. Use this when you want to capture a free-form walkthrough.
            url = BASE_URL + "/" if region == "full" else f"{BASE_URL}/?demo={region}"
            # 'networkidle' never fires because the UI keeps an SSE connection
            # open forever — use 'domcontentloaded' + a small explicit sleep.
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(800)
            contexts[region] = ctx
            pages[region] = page
            print(f"  [demo] opened {url}  → recording to {ctx_dir}/")

        # Submit intent so the pipeline kicks off (recordings already running).
        # Skip when --skip-pipeline-check (no live pipeline) or --stop-on-key
        # (recording a manual interaction, not a fresh run).
        if args.auto_submit and not args.skip_pipeline_check and not args.stop_on_key:
            time.sleep(2)   # let pages settle
            try:
                _submit_intent(args.target, args.purpose, args.dataset, args.k)
            except Exception as exc:
                print(f"  [demo] WARNING: intent submit failed: {exc}")
                print("  [demo] submit it manually in the browser — recording continues")

        if args.stop_on_key:
            print("\n  [demo] recording … interact with the browser, then press Enter here to stop.")
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                pass
            print("  [demo] Enter received — closing pages and writing video files")
        elif args.skip_pipeline_check:
            # No live pipeline to wait for — just record for the full --timeout,
            # giving the user time to interact (rename clusters, etc.) in the
            # already-open browser window.
            print(f"  [demo] recording for {args.timeout}s (no live pipeline). "
                  f"Interact in the browser; the recording stops when timeout hits "
                  f"or you Ctrl+C this process.")
            try:
                time.sleep(args.timeout)
            except KeyboardInterrupt:
                print("\n  [demo] interrupted — saving recordings up to this point")
        else:
            # Wait for pipeline_complete OR timeout
            print(f"  [demo] recording … waiting for pipeline_complete (timeout {args.timeout}s)")
            deadline = time.time() + args.timeout
            try:
                while time.time() < deadline:
                    s = _status()
                    last_complete = s.get("last_complete")
                    if last_complete:
                        elapsed = int(args.timeout - (deadline - time.time()))
                        # 20 s tail so the Named Clusters tab has time to (a) detect
                        # pipeline_complete via SSE, (b) auto-switch view, and (c) render
                        # all cluster cards from the freshly-saved personas.json before
                        # the recording stops. 8 s was too short for slower machines.
                        print(f"  [demo] pipeline_complete fired (status={last_complete.get('status')}) "
                              f"after {elapsed}s — recording 20 more seconds for Named Clusters tab to render")
                        time.sleep(20)
                        break
                    time.sleep(2)
            except KeyboardInterrupt:
                print("\n  [demo] interrupted — saving recordings up to this point")

        # Close pages, then contexts → flushes video files to disk
        for region, page in pages.items():
            try:
                page.close()
            except Exception:
                pass
        for region, ctx in contexts.items():
            try:
                ctx.close()
                # Rename the auto-generated video file to <region>.webm
                files = list((OUT_DIR / region).glob("*.webm"))
                if files:
                    src = files[0]
                    dst = OUT_DIR / f"{region}.webm"
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
                    (OUT_DIR / region).rmdir()
                    size_mb = dst.stat().st_size / (1024 * 1024)
                    print(f"  ✓ saved {dst}  ({size_mb:.1f} MB)")
                else:
                    print(f"  ✗ no video file produced for {region}")
            except Exception as exc:
                print(f"  ✗ {region}: {exc}")
        browser.close()

    print(f"\n  [demo] All recordings in: {OUT_DIR.resolve()}")
    print("  [demo] Convert .webm → .mp4 with:")
    print("       for f in recordings/*.webm; do ffmpeg -i \"$f\" -c:v libx264 -crf 18 \"${f%.webm}.mp4\"; done")


if __name__ == "__main__":
    main()
