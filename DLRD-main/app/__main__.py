"""Entry point: ``python -m app``.

Starts uvicorn bound to 127.0.0.1 ONLY (never 0.0.0.0) and opens the browser
on the local config/analysis screen. The bind host is hard-coded to loopback by
design — there is no flag to expose this server on the network.
"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from .config import Settings

# Hard-coded: the server is reachable only from this machine.
HOST = "127.0.0.1"


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    settings = Settings.load()
    ap = argparse.ArgumentParser(prog="python -m app", description="Data-Lineage and Retro-Documentation (local)")
    ap.add_argument("--port", type=int, default=settings.port, help=f"port (default {settings.port})")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--debug", action="store_true", help="verbose logging (never logs file contents)")
    args = ap.parse_args()

    debug = args.debug or settings.debug
    logging.basicConfig(
        level=logging.INFO if debug else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    url = f"http://{HOST}:{args.port}/"
    print(f"\n  Data-Lineage and Retro-Documentation (local) -> {url}")
    print(f"  Bound to {HOST} only. Single network egress = your configured LLMAAS apiBase.\n")
    if not args.no_browser:
        threading.Timer(1.0, _open_browser, args=(url,)).start()

    uvicorn.run(
        "app.server:app",
        host=HOST,
        port=args.port,
        log_level="info" if debug else "warning",
        access_log=debug,
    )


if __name__ == "__main__":
    main()
