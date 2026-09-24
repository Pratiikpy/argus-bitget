"""The forward tape's WebSocket client: framing per RFC 6455, and the subscription it sends.

The live stream is exercised by `python -m argus.market.ws_tape --probe 20`; these pin the parts
that can be wrong without a network — masked client frames, unmasked server frames of every length
form, ping answered with pong, and close raised rather than read as data.
"""

from __future__ import annotations

import json
import struct

import pytest

from argus.market.ws_tape import TOPICS, Closed, Socket, keep, subscription


class FakeSock:
    def __init__(self, incoming: bytes = b"") -> None:
        self.incoming = incoming
        self.sent = b""

    def recv(self, n: int) -> bytes:
        chunk, self.incoming = self.incoming[:n], self.incoming[n:]
        return chunk

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def close(self) -> None:
        pass


def _client(incoming: bytes = b"") -> tuple[Socket, FakeSock]:
    conn = object.__new__(Socket)
    fake = FakeSock(incoming)
    conn.sock = fake  # type: ignore[assignment]
    conn.buffer = b""
    return conn, fake


def _server_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    n = len(payload)
    if n < 126:
        return struct.pack("!BB", 0x80 | opcode, n) + payload
    if n < 65536:
        return struct.pack("!BBH", 0x80 | opcode, 126, n) + payload
    return struct.pack("!BBQ", 0x80 | opcode, 127, n) + payload


@pytest.mark.parametrize("size", [5, 300, 70_000])
def test_server_frames_of_every_length_form_are_read(size: int) -> None:
    text = "x" * size
    conn, _ = _client(_server_frame(text.encode()))
    assert conn.receive() == text


def test_client_frames_are_masked_and_unmask_to_the_text() -> None:
    conn, fake = _client()
    conn.send("ping")
    first, second = fake.sent[0], fake.sent[1]
    assert first == 0x81 and second & 0x80
    mask, body = fake.sent[2:6], fake.sent[6:]
    assert bytes(b ^ mask[i % 4] for i, b in enumerate(body)) == b"ping"


def test_a_ping_is_answered_and_a_close_is_raised() -> None:
    conn, fake = _client(_server_frame(b"hi", opcode=0x9) + _server_frame(b"", opcode=0x8))
    assert conn.receive() is None
    assert fake.sent[0] == 0x8A  # pong
    with pytest.raises(Closed):
        conn.receive()


def test_the_subscription_names_every_symbol_and_topic() -> None:
    body = json.loads(subscription(["NVDA", "BTC"]))
    assert body["op"] == "subscribe"
    assert len(body["args"]) == 2 * len(TOPICS)
    assert {"instType": "usdt-futures", "topic": "books5", "symbol": "NVDAUSDT"} in body["args"]


def test_the_controls_can_keep_trades_without_their_book() -> None:
    body = json.loads(subscription(["NVDA", "BTC"], books=["NVDA"]))
    topics = {(a["symbol"], a["topic"]) for a in body["args"]}
    assert ("BTCUSDT", "publicTrade") in topics and ("BTCUSDT", "books5") not in topics
    assert ("NVDAUSDT", "books5") in topics


def test_book_snapshots_are_thinned_per_symbol_and_trades_never() -> None:
    last: dict[str, float] = {}
    book = json.dumps({"arg": {"topic": "books5", "symbol": "NVDAUSDT"}, "data": []},
                      separators=(",", ":"))
    trade = json.dumps({"arg": {"topic": "publicTrade", "symbol": "NVDAUSDT"}},
                       separators=(",", ":"))
    assert keep(book, 10.0, last)
    assert not keep(book, 10.1, last)
    assert keep(trade, 10.1, last)
    assert keep(book, 10.3, last)
