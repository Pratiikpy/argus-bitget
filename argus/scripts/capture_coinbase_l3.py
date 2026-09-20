"""Capture Coinbase Exchange's real, free, public `full` WebSocket channel — genuine L3
market-by-order data (received/open/done/change events, one per real order lifecycle event) —
to a local JSONL file, for `eval/queue_l3_comparison.py` to reconstruct real queue-position-to-fill
ground truth from.

No API key, no cost. `wss://ws-feed.exchange.coinbase.com`, channel `full`. Confirmed real and
public via Coinbase's own documentation (not vendored code — this is our own capture client
against their own public WebSocket API, the same way any market-data consumer would connect).

Run standalone, long-lived, in the background: `python scripts/capture_coinbase_l3.py --minutes 60`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import websocket

WS_URL = "wss://ws-feed.exchange.coinbase.com"

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Coinbase's real L3 full channel")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument(
        "--out", default=None, help="output JSONL path (default: data/coinbase_l3_<product>.jsonl)"
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else DATA / f"coinbase_l3_{args.product}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + args.minutes * 60
    count = 0
    started_at = datetime.now(UTC).isoformat()

    ws = websocket.create_connection(WS_URL, timeout=30)
    try:
        subscribe = {
            "type": "subscribe",
            "product_ids": [args.product],
            "channels": ["full"],
        }
        ws.send(json.dumps(subscribe))
        print(f"subscribed to full channel for {args.product}, capturing to {out_path}")

        with out_path.open("a", encoding="utf-8") as f:
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv()
                except Exception as exc:  # real network hiccup — reconnect, don't crash the capture
                    print(f"recv error: {exc!r}; reconnecting", file=sys.stderr)
                    try:
                        ws.close()
                    except Exception:
                        pass
                    time.sleep(2)
                    ws = websocket.create_connection(WS_URL, timeout=30)
                    ws.send(json.dumps(subscribe))
                    continue
                received_at = datetime.now(UTC).isoformat()
                f.write(json.dumps({"received_at": received_at, "raw": json.loads(raw)}) + "\n")
                f.flush()
                count += 1
                if count % 500 == 0:
                    remaining = deadline - time.monotonic()
                    print(f"{count} messages captured, {remaining:.0f}s remaining")
    finally:
        try:
            ws.close()
        except Exception:
            pass

    print(f"done: {count} messages, started {started_at}, saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
