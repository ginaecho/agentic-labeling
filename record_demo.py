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
    recordings/graph.webm
    recordings/convos.webm
    recordings/outputs.webm
    recordings/evidence.webm
    recordings/tokens.webm

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

BASE_URL = "http://127.0.0.1:5057"
REGIONS = ["graph", "convos", "outputs", "evidence", "tokens"]
VIEWPORT = {"width": 1280, "height": 720}
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
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check: pipeline UI is reachable
    status = _status()
    if not status:
        sys.exit(f"ERROR: cannot reach {BASE_URL}/api/status — is the pipeline running?")
    if not status.get("pipeline_running"):
        sys.exit("ERROR: pipeline is not in 'running' state. Start `python run_pipeline.py` first.")
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
            ctx = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(ctx_dir),
                record_video_size=VIEWPORT,
            )
            page = ctx.new_page()
            url = f"{BASE_URL}/?demo={region}"
            # 'networkidle' never fires because the UI keeps an SSE connection
            # open forever — use 'domcontentloaded' + a small explicit sleep.
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(800)
            contexts[region] = ctx
            pages[region] = page
            print(f"  [demo] opened {url}  → recording to {ctx_dir}/")

        # Submit intent so the pipeline kicks off (recordings already running)
        if args.auto_submit:
            time.sleep(2)   # let pages settle
            try:
                _submit_intent(args.target, args.purpose, args.dataset, args.k)
            except Exception as exc:
                print(f"  [demo] WARNING: intent submit failed: {exc}")
                print("  [demo] submit it manually in the browser — recording continues")

        # Wait for pipeline_complete OR timeout
        print(f"  [demo] recording … waiting for pipeline_complete (timeout {args.timeout}s)")
        deadline = time.time() + args.timeout
        try:
            while time.time() < deadline:
                s = _status()
                last_complete = s.get("last_complete")
                if last_complete:
                    elapsed = int(args.timeout - (deadline - time.time()))
                    print(f"  [demo] pipeline_complete fired (status={last_complete.get('status')}) "
                          f"after {elapsed}s — recording 8 more seconds for final UI render")
                    time.sleep(8)
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
