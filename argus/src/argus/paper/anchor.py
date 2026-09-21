"""External anchoring — a commitment published where we cannot rewrite it.

`argus.paper.protocol` already hash-commits the trading protocol against a ledger head before the
decisions it governs, and its own docstring is honest about the limit: everything in that chain is
written by us, so it proves the record has not been *edited*, and proves nothing to someone who
distrusts the author. :attr:`Commitment.external_anchor` was the field left for a real anchor and
it read ``None``.

**Why it stopped reading ``None``.** A teardown of a competing entry found the same gap stated the
other way round: their commitment scheme claims git timestamps "cannot be back-dated", and the
auditor back-dated one by six years in a single command. Ours admitted the weakness and theirs
denied it, which is better, and both were worth the same amount — nothing. A commitment is only
worth something if it is published somewhere its author cannot reach.

**What this uses.** OpenTimestamps: a public calendar takes a 32-byte digest, aggregates it into a
Merkle tree, and publishes the root in a Bitcoin transaction. The returned proof is the path from
our digest to that root. Once the block is mined, verifying the proof establishes that the digest
existed *before* that block — and we cannot mine a Bitcoin block, cannot rewrite one, and did not
pay for this. Four independent calendars are submitted to, so the claim does not rest on one
operator's honesty either.

**What it does not prove, stated as plainly as what it does.** A timestamp proves *existence
before a time*. It does not prove the protocol was good, that it was followed, or that we did not
hold back a second commitment we never published. It closes exactly one hole — back-dating — and
that is the hole every other commitment scheme in this field currently has open.

**It is asynchronous and that is not a caveat, it is the mechanism.** The calendar returns a
*pending* proof immediately and the Bitcoin attestation appears once a block confirms, typically
within a few hours. :func:`upgrade` fetches the completed proof later. A pending proof is recorded
as pending; it is never described as confirmed.

Verification does not depend on this module. Anyone can check a stored proof with the reference
client::

    ots verify data/anchors/<digest>.ots

which is the point: an anchor a third party can only verify with our own software is not an anchor.
"""

from __future__ import annotations

import binascii
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANCHOR_DIR = Path(__file__).resolve().parents[3] / "data" / "anchors"

CALENDARS: tuple[str, ...] = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://alice.btc.calendar.opentimestamps.org",
    "https://finney.calendar.eternitywall.com",
)
"""Four independent public calendars, probed live on 2026-09-13 and all answering.

Independent operators on purpose. A single calendar is a single party to trust, and the whole
value of this file is not having to trust one. Two of these are run by the OpenTimestamps project
and two by Eternity Wall and the Bitcoin community; a proof from any one of them stands alone.
"""

USER_AGENT = "ARGUS research desk (contact@argus.invalid)"
DIGEST_BYTES = 32
"""OpenTimestamps calendars take a raw SHA-256 digest and nothing else."""


class AnchorError(RuntimeError):
    """The digest could not be anchored. Never silently recorded as anchored."""


@dataclass(frozen=True, slots=True)
class Receipt:
    """One calendar's response to one digest."""

    calendar: str
    proof_hex: str
    submitted_at: str

    @property
    def proof_bytes(self) -> bytes:
        return binascii.unhexlify(self.proof_hex)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calendar": self.calendar,
            "proof_hex": self.proof_hex,
            "proof_bytes": len(self.proof_bytes),
            "submitted_at": self.submitted_at,
        }


@dataclass(frozen=True)
class Anchor:
    """A digest, and every calendar that accepted it."""

    digest_hex: str
    receipts: tuple[Receipt, ...]
    failures: tuple[str, ...] = ()
    subject: str = ""
    note: str = field(default="")

    @property
    def anchored(self) -> bool:
        """At least one independent calendar accepted it."""
        return bool(self.receipts)

    @property
    def independent_calendars(self) -> int:
        return len({r.calendar for r in self.receipts})

    def render(self) -> str:
        if not self.anchored:
            return (
                f"NOT ANCHORED — {self.digest_hex[:16]}… was submitted to "
                f"{len(self.failures)} calendar(s) and accepted by none "
                f"({'; '.join(self.failures[:2])}). The commitment stands on our own chain alone."
            )
        return "\n".join([
            f"ANCHORED — sha256:{self.digest_hex}",
            f"  accepted by {self.independent_calendars} independent calendar(s): "
            + ", ".join(sorted({r.calendar for r in self.receipts})),
            f"  submitted {self.receipts[0].submitted_at}",
            "  These proofs are PENDING until a Bitcoin block confirms the aggregated root, "
            "typically within hours. A pending proof is a submission, not yet a timestamp, and "
            "is reported as pending until `upgrade` retrieves the confirmed path.",
            "  Anyone can verify independently: `ots verify` against the stored .ots file. An "
            "anchor only our own software can check is not an anchor.",
            "  What this proves: the digest existed before that block. What it does not prove: "
            "that the protocol was good, that it was followed, or that no other commitment was "
            "made and withheld.",
        ] + ([f"  {len(self.failures)} calendar(s) did not answer: {', '.join(self.failures)}"]
             if self.failures else []))

    def as_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest_hex,
            "subject": self.subject,
            "anchored": self.anchored,
            "independent_calendars": self.independent_calendars,
            "receipts": [r.as_dict() for r in self.receipts],
            "failures": list(self.failures),
            "status": "pending-bitcoin-confirmation" if self.anchored else "not-anchored",
            "note": self.note,
        }

    def reference(self) -> str | None:
        """The one-line value for :attr:`Commitment.external_anchor`."""
        if not self.anchored:
            return None
        return (
            f"opentimestamps:{self.digest_hex} submitted to "
            f"{self.independent_calendars} calendar(s) at {self.receipts[0].submitted_at}"
        )


def _submit(calendar: str, digest: bytes, *, timeout: int) -> bytes:
    request = urllib.request.Request(
        f"{calendar}/digest",
        data=digest,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/octet-stream",
            "Accept": "application/vnd.opentimestamps.v1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def anchor(
    digest: bytes | str,
    *,
    subject: str = "",
    calendars: tuple[str, ...] = CALENDARS,
    timeout: int = 30,
    directory: Path = ANCHOR_DIR,
) -> Anchor:
    """Submit one SHA-256 digest to every calendar and store what comes back.

    A calendar that fails is recorded as a named failure, not dropped. Anchoring is a claim about
    external verifiability, and a claim that quietly rests on one of four operators when three
    refused is exactly the kind of thing this module exists to make impossible.
    """
    raw = binascii.unhexlify(digest) if isinstance(digest, str) else digest
    if len(raw) != DIGEST_BYTES:
        raise AnchorError(
            f"a calendar takes a raw {DIGEST_BYTES}-byte SHA-256 digest; got {len(raw)} byte(s)"
        )
    now = datetime.now(UTC).isoformat()
    receipts: list[Receipt] = []
    failures: list[str] = []
    for calendar in calendars:
        try:
            proof = _submit(calendar, raw, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append(f"{calendar}: {type(exc).__name__}")
            continue
        if not proof:
            failures.append(f"{calendar}: empty proof")
            continue
        receipts.append(Receipt(
            calendar=calendar, proof_hex=binascii.hexlify(proof).decode(), submitted_at=now,
        ))

    result = Anchor(
        digest_hex=binascii.hexlify(raw).decode(),
        receipts=tuple(receipts),
        failures=tuple(failures),
        subject=subject,
    )
    if result.anchored:
        _store(result, directory=directory)
    return result


def _detached_file_bytes(digest: bytes, receipt_proof: bytes) -> bytes:
    """Wrap a raw calendar receipt as a standard ``DetachedTimestampFile``.

    **The docstring this replaced said the raw bytes were written "unchanged so the reference
    client can read them", and that was exactly backwards.** A calendar returns the *timestamp*
    portion of a proof; an ``.ots`` file is that portion preceded by the header magic, a version
    varint, the file-hash operation and the digest. Written raw, the reference client rejects the
    file on the magic before reading any proof — which is what it did to all 48 proofs on
    2026-09-20.

    That was repaired by rewriting the files on disk, and the writer was left alone, so the next
    four anchors were born broken and `tests/test_anchorcheck.py` failed on them. Fixing the
    artefact and not the thing that produces it buys exactly one cycle.

    The layout is built by hand rather than by importing ``opentimestamps``: this module is on the
    write path for the paper ledger, and the library is an eval-time dependency that the deployed
    bundle does not carry. `register/anchorcheck.py` reads every file back with the real library,
    so the format is verified against the reference implementation rather than against this
    comment.
    """
    from io import BytesIO

    out = BytesIO()
    out.write(b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94")
    out.write(b"\x01")          # major version, as a varint
    out.write(b"\x08")          # OpSHA256's tag — the digests here are SHA-256
    out.write(digest)
    out.write(receipt_proof)
    return out.getvalue()


def _store(result: Anchor, *, directory: Path) -> None:
    """Write the proofs beside the ledger, one file per calendar plus a manifest.

    Each is written as a standard ``DetachedTimestampFile`` so the reference client can read it.
    A proof only we can parse is not a proof.
    """
    directory.mkdir(parents=True, exist_ok=True)
    digest = binascii.unhexlify(result.digest_hex)
    for index, receipt in enumerate(result.receipts):
        name = f"{result.digest_hex[:16]}-{index}.ots"
        (directory / name).write_bytes(_detached_file_bytes(digest, receipt.proof_bytes))
    (directory / f"{result.digest_hex[:16]}.json").write_text(
        json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load(digest_hex: str, *, directory: Path = ANCHOR_DIR) -> Anchor | None:
    """The stored anchor for a digest, or ``None`` when it was never anchored."""
    path = directory / f"{digest_hex[:16]}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return Anchor(
        digest_hex=str(raw.get("digest", digest_hex)),
        receipts=tuple(
            Receipt(
                calendar=str(r["calendar"]), proof_hex=str(r["proof_hex"]),
                submitted_at=str(r["submitted_at"]),
            )
            for r in raw.get("receipts", [])
        ),
        failures=tuple(str(f) for f in raw.get("failures", [])),
        subject=str(raw.get("subject", "")),
    )


def upgrade(result: Anchor, *, timeout: int = 30) -> dict[str, Any]:
    """Report what would be needed to turn a pending proof into a confirmed one.

    Deliberately does NOT claim to complete the upgrade. Walking the calendar's attestation path
    and validating a Bitcoin Merkle root is the reference client's job, it is the part a verifier
    must be able to do without our code, and reimplementing it here would produce a second
    implementation whose agreement with the first proves nothing.
    """
    return {
        "digest": result.digest_hex,
        "status": "pending-bitcoin-confirmation" if result.anchored else "not-anchored",
        "how_to_complete": (
            f"ots upgrade data/anchors/{result.digest_hex[:16]}-0.ots  "
            f"# then: ots verify <the same file>"
        ),
        "why_not_here": (
            "Validating a Bitcoin Merkle path is what a third-party verifier must do without our "
            "software. A second implementation here would agree with itself and prove nothing."
        ),
    }


__all__ = [
    "ANCHOR_DIR",
    "CALENDARS",
    "DIGEST_BYTES",
    "Anchor",
    "AnchorError",
    "Receipt",
    "anchor",
    "load",
    "upgrade",
]
