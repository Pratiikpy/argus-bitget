"""A forward tape of Bitget's public order books and trades, recorded to disk as it happens.

The execution arena planned in the rival review of 2026-09-24 scores every order-splitting method
on a tape none of them was fitted on. REST snapshots (`data/book_tape.jsonl`, 49 per symbol over
nine days) cannot show what happens between them — how fast a level refills after a sweep, which
resting order a trade consumed — so this records Bitget's own push stream: ``books5`` (the top five
levels, a full snapshot on every push, about nine a second per symbol) and every ``publicTrade``
print, for the stock perpetuals and two crypto controls, each message stamped with the instant it
arrived.

**Why five levels and not the full book.** The full-depth incremental ``books`` topic was measured
on 2026-09-24 at about 100 KB/s for these ten symbols — 8 GB a day before compression, more than
this drive can hold for the week the arena needs. ``books5`` measured about 1.1 GB a day gzipped
on the first hour of recording, still too much, so each symbol's five-level snapshot is kept at
most every :data:`BOOK_SPACING` seconds (a snapshot is complete in itself, so dropping the ones
between loses no state, only time resolution), and the two crypto controls keep their trades but
not their books: about 0.4 GB a day. What is lost is replenishment below the fifth level; the
arena's full-depth test day comes from Tardis's ``incremental_book_L2`` instead, and the 200-level
REST sweeps stay in `data/book_tape.jsonl`. (``books15`` does not exist on the v3 public stream;
the server refuses it.)

**A WebSocket client in the standard library, deliberately.** The desk package carries no network
dependency beyond ``urllib``; RFC 6455 needs a TLS socket, one HTTP upgrade, and a frame parser for
text, ping, pong and close — about a hundred lines, written here against the RFC rather than adding
a package for them. Client frames are masked (RFC 6455 §5.3); server frames are not.

**Disk is budgeted, not assumed.** Messages are appended to one gzip file per UTC hour under
``data/tape/<date>/``; the recorder stops writing and says so when the drive's free space falls
below :data:`MIN_FREE_GB`, because a full drive stops every other process on the machine too (it
happened on 2026-09-24).

    python -m argus.market.ws_tape --probe 20      # measure the stream for 20 s, write nothing
    python -m argus.market.ws_tape                 # record until stopped
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import json
import os
import shutil
import socket
import ssl
import struct
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOST = "ws.bitget.com"
PATH = "/v3/ws/public"
STOCKS = ("NVDA", "TSLA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "COIN", "MSTR", "QQQ")
CONTROLS = ("BTC", "ETH")
TOPICS = ("books5", "publicTrade")
BOOK_SPACING = 0.25
"""Seconds between kept five-level snapshots of one symbol."""
PING_EVERY = 20.0
MIN_FREE_GB = 3.0
TAPE_DIR = Path(__file__).resolve().parents[3] / "data" / "tape"


class Closed(ConnectionError):
    """The server closed the connection, or it dropped."""


class Socket:
    """One RFC 6455 client connection over TLS."""

    def __init__(self, host: str = HOST, path: str = PATH, timeout: float = 30.0) -> None:
        raw = socket.create_connection((host, 443), timeout=timeout)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   f"Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(request.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise Closed("the handshake was cut off")
            head += chunk
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise Closed(f"handshake refused: {status.decode(errors='replace')}")
        self.buffer = head.split(b"\r\n\r\n", 1)[1]

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise Closed("the connection dropped")
            self.buffer += chunk
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def send(self, text: str, opcode: int = 0x1) -> None:
        payload = text.encode()
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def receive(self) -> str | None:
        """The next text message; ``None`` for a control frame that carried none."""
        first, second = self._read(2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read(8))[0]
        payload = self._read(length)
        if opcode == 0x8:
            raise Closed("the server closed the connection")
        if opcode == 0x9:
            self.send(payload.decode(errors="replace"), opcode=0xA)
            return None
        if opcode in (0x1, 0x0):
            return payload.decode()
        return None

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


def subscription(symbols: Sequence[str], topics: Sequence[str] = TOPICS,
                 books: Sequence[str] | None = None) -> str:
    """Every topic for every symbol, except that only ``books`` (default: all) get the book."""
    with_book = set(symbols if books is None else books)
    return json.dumps({"op": "subscribe", "args": [
        {"instType": "usdt-futures", "topic": topic, "symbol": f"{s}USDT"}
        for s in symbols for topic in topics
        if not (topic.startswith("books") and s not in with_book)]})


def keep(message: str, arrived: float, last: dict[str, float],
         spacing: float = BOOK_SPACING) -> bool:
    """Whether to write ``message``: every trade and event, and a symbol's book snapshot only when
    ``spacing`` seconds have passed since the last one kept."""
    if '"topic":"books' not in message:
        return True
    try:
        symbol = str(json.loads(message)["arg"]["symbol"])
    except (ValueError, KeyError, TypeError):
        return True
    if arrived - last.get(symbol, 0.0) < spacing:
        return False
    last[symbol] = arrived
    return True


def stream(symbols: Sequence[str], topics: Sequence[str] = TOPICS,
           books: Sequence[str] | None = None) -> Iterator[tuple[float, str]]:
    """(arrival time, raw message) forever, reconnecting with a growing pause after a drop."""
    pause = 1.0
    while True:
        conn: Socket | None = None
        try:
            conn = Socket()
            conn.send(subscription(symbols, topics, books))
            last_ping = time.monotonic()
            pause = 1.0
            while True:
                if time.monotonic() - last_ping > PING_EVERY:
                    conn.send("ping")
                    last_ping = time.monotonic()
                message = conn.receive()
                if message is not None and message != "pong":
                    yield time.time(), message
        except (Closed, OSError, ssl.SSLError) as exc:
            yield time.time(), json.dumps({"argus_event": "disconnected", "reason": str(exc)})
            time.sleep(pause)
            pause = min(pause * 2, 60.0)
        finally:
            if conn is not None:
                conn.close()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def record(symbols: Sequence[str], directory: Path = TAPE_DIR,
           books: Sequence[str] | None = None) -> None:  # pragma: no cover - live
    directory.mkdir(parents=True, exist_ok=True)
    handle: Any = None
    hour = ""
    checked = 0.0
    last_book: dict[str, float] = {}
    for arrived, message in stream(symbols, books=books):
        if not keep(message, arrived, last_book):
            continue
        now = datetime.fromtimestamp(arrived, UTC)
        stamp = now.strftime("%Y-%m-%d/%H")
        if stamp != hour:
            if handle is not None:
                handle.close()
            target = directory / f"{stamp}.jsonl.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = gzip.open(target, "at", encoding="utf-8")  # noqa: SIM115 - one per hour
            hour = stamp
        if arrived - checked > 60:
            checked = arrived
            if free_gb(directory) < MIN_FREE_GB:
                handle.write(json.dumps({"t": arrived, "argus_event": "stopped_low_disk"}) + "\n")
                handle.close()
                print(f"stopped: under {MIN_FREE_GB} GB free on the tape's drive")
                return
        handle.write(json.dumps({"t": round(arrived, 3), "m": message}) + "\n")


def probe(seconds: float, symbols: Sequence[str]) -> dict[str, Any]:  # pragma: no cover - live
    counts: dict[str, int] = {}
    size = 0
    started = time.time()
    first: dict[str, str] = {}
    for arrived, message in stream(symbols):
        if arrived - started > seconds:
            break
        size += len(message)
        try:
            body = json.loads(message)
        except ValueError:
            body = {}
        arg = body.get("arg") or {}
        key = f"{arg.get('topic', body.get('event', '?'))}:{arg.get('symbol', '')}"
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, message[:400])
    elapsed = time.time() - started
    return {"seconds": round(elapsed, 1), "messages": sum(counts.values()),
            "bytes_per_second": round(size / elapsed), "counts": counts, "first": first}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(description="Record Bitget's public book and trade stream.")
    parser.add_argument("--probe", type=float, default=0.0, help="measure for N seconds, no write")
    args = parser.parse_args(argv)
    symbols = (*STOCKS, *CONTROLS)
    if args.probe:
        print(json.dumps(probe(args.probe, symbols), indent=1)[:6000])
        return 0
    record(symbols, books=STOCKS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
