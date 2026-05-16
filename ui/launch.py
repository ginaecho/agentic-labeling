"""Launch the interactive cluster fine-tuning UI in the default browser.

Usage:  python -m ui.launch  [--port 5057] [--host 127.0.0.1] [--no-open]
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

from ui.app import main as serve, PERSONAS_PATH


def _open_browser(url: str, delay: float = 1.2) -> None:
    def _go():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5057)
    parser.add_argument('--no-open', action='store_true',
                        help='Do not auto-open the browser')
    args = parser.parse_args()

    if not PERSONAS_PATH.exists():
        print(f'[ui] {PERSONAS_PATH} not found — UI will show the live pipeline view.')
        print('     Once `python run_pipeline.py` finishes, the cluster grid appears here.')

    if not args.no_open:
        _open_browser(f'http://{args.host}:{args.port}/')
    serve(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
