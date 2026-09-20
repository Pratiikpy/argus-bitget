"""The vendored third-party baselines load, run their real code, and cannot silently drift."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from argus.eval.baselines.alphalens_ic_loader import AlphalensIcLoadError, load_ic_module
from argus.eval.baselines.crypto_sor_loader import CryptoSorSubprocessError, run_new_order
from argus.eval.baselines.lean_pairs_ranking_loader import (
    LeanPairsRankingLoadError,
    load_pairs_ranking_module,
)
from argus.eval.baselines.loader import BaselineLoadError, load_baseline
from argus.eval.baselines.maxme_arbitrer_loader import (
    MaxmeArbitrerLoadError,
    load_arbitrer_module,
)
from argus.eval.baselines.pytaa_signal_loader import (
    PytaaSignalLoadError,
    load_signal_module,
)
from argus.eval.baselines.pytaa_vigilant_allocation_loader import (
    PytaaVigilantAllocationLoadError,
    load_vigilant_allocation_module,
)
from argus.eval.baselines.qlib_loader import QlibBaselineLoadError, load_qlib_baseline
from argus.eval.baselines.quantconnect_sue_loader import (
    QuantConnectSueLoadError,
    load_sue_module,
)
from argus.eval.baselines.rdagent_factor_loader import load_factor_module
from argus.eval.baselines.serenity_loader import (
    SerenityBaselineLoadError,
    load_chained_journal_class,
)
from argus.eval.baselines.stumpy_squared_distance_loader import (
    StumpySquaredDistanceLoadError,
    load_squared_distance_module,
)
from argus.eval.baselines.tradingagents_feedlist_loader import (
    TradingAgentsFeedlistLoadError,
    load_interface_module,
)
from argus.eval.baselines.tradingagents_loader import (
    TradingAgentsBaselineLoadError,
    load_trading_memory_log_class,
)
from argus.eval.baselines.vectorbt_loader import (
    VectorbtBaselineLoadError,
    load_vectorbt_baseline,
)
from argus.eval.baselines.whale_signals_event_study_loader import (
    WhaleSignalsEventStudyLoadError,
    load_event_study_module,
)

_BASELINES_DIR = Path(__file__).resolve().parents[1] / "src" / "argus" / "eval" / "baselines"

# Computed once via `tail -n <original-file-line-count> <vendored-file> | sha256sum` and
# cross-checked directly against github.com/HKUDS/Vibe-Trading at commit
# 8452a8448f947dfa1d5fe55b582f1944d4d9b696 with `diff` against a local clone before being pinned
# here — see `eval/baselines/vibe_trading_*.py`'s own header comments for the full provenance. A
# hash mismatch means the vendored body no longer matches what was verified byte-identical to
# upstream, which this test alone can catch WITHOUT needing a clone of that repo on disk.
_ENFORCEMENT_BODY_SHA256 = "b302b0021c8910e2acdc1a53749a71939e9a7e248527e6a1798b72ea75dbbe52"
_ENFORCEMENT_ORIGINAL_LINE_COUNT = 797
_MODEL_BODY_SHA256 = "c8e34168449ca512273b1dac9d7648eb31e86b198ffa95e1978d26dd84e78be7"
_MODEL_ORIGINAL_LINE_COUNT = 148

# Same reasoning, for `qlib_cs_processor.py` — cross-checked against github.com/microsoft/qlib at
# commit 79633dd9506ea689e5400dea0197717b5b3d74b7.
_QLIB_PROCESSOR_BODY_SHA256 = "fa02f4232e755b0a3a34960dc5eeca287b02da0b0280bfbdfe2aba764d8b751d"
_QLIB_PROCESSOR_ORIGINAL_LINE_COUNT = 419

# Same reasoning, for `vectorbt_dsr_metrics.py` — cross-checked against
# github.com/polakowo/vectorbt at commit 34b6d5935e3ea3eccd549e2592bc0f455b8045f5.
_VECTORBT_METRICS_BODY_SHA256 = "08be40b895d2d753ea6dc1027c412eb211da8e51b28070529839bc00b456a656"
_VECTORBT_METRICS_ORIGINAL_LINE_COUNT = 36

# Same reasoning, for `tradingagents_memory.py` and `tradingagents_rating.py` — cross-checked
# against github.com/TauricResearch/TradingAgents at commit
# be952b8eccb49720509af544c6675233bc1f10d0.
_TA_MEMORY_BODY_SHA256 = "66644c9a6960ad1b6f81d49f30bac4003aadf68913ffff1fac5580254a89a770"
_TA_MEMORY_ORIGINAL_LINE_COUNT = 334
_TA_RATING_BODY_SHA256 = "09fd9c75b0260cb56ecc0dadbb3003895974d7baa4ecb042e0f6c53af03f0457"
_TA_RATING_ORIGINAL_LINE_COUNT = 83

# Same reasoning, for `serenity_journal.py` and `serenity_guards.py` — cross-checked against
# github.com/SerenityTn/serenity-guardrails at commit 13d46dc1c6204f27fe321bd66023691918017cce.
_SERENITY_JOURNAL_BODY_SHA256 = "570819e6322cd96f717c6b6875b051be67ad2c7ffb4742b81f22cdf0f80f8d85"
_SERENITY_JOURNAL_ORIGINAL_LINE_COUNT = 96
_SERENITY_GUARDS_BODY_SHA256 = "5821c12de43cc425c214926273004495b8b2a87f847f27664a4b2e2558749f74"
_SERENITY_GUARDS_ORIGINAL_LINE_COUNT = 194

# Same reasoning, for the four `tradingagents_feedlist_*.py` files — cross-checked against
# github.com/TauricResearch/TradingAgents at commit be952b8eccb49720509af544c6675233bc1f10d0
# (same repo/commit already used for the memory/rating baselines above).
_TA_FEEDLIST_INTERFACE_BODY_SHA256 = (
    "deb9c97edb5ac6bc6720cc71655aab1c11fe610be2f60537ea4de97b234c9a4f"
)
_TA_FEEDLIST_INTERFACE_ORIGINAL_LINE_COUNT = 262
_TA_FEEDLIST_ERRORS_BODY_SHA256 = "456570572c89446cc814f1eceaf9a7fd1c740f35dce2e3dda755dbf89e99e280"
_TA_FEEDLIST_ERRORS_ORIGINAL_LINE_COUNT = 55
_TA_FEEDLIST_CONFIG_BODY_SHA256 = "c498661b6ab17a087ca93c9148e80e62e5f9bdcfee017f5424ce826e6ded855d"
_TA_FEEDLIST_CONFIG_ORIGINAL_LINE_COUNT = 41
_TA_FEEDLIST_DEFAULT_CONFIG_BODY_SHA256 = (
    "e1e4261ca519781f6974bd8c8ea428da71ec3226ba1a91561243275fd562c4f2"
)
_TA_FEEDLIST_DEFAULT_CONFIG_ORIGINAL_LINE_COUNT = 170

# Same reasoning, for `qlib_expression_base.py` and `qlib_expression_ops.py` — cross-checked
# against github.com/microsoft/qlib at the same commit as `qlib_cs_processor.py` above
# (79633dd9506ea689e5400dea0197717b5b3d74b7). See `qlib_expression_base.py`'s own header for why
# this file was wrongly marked unvendorable by an earlier session, and corrected.
_QLIB_EXPR_BASE_BODY_SHA256 = "cc29103b4fcb5a7c87f29194fc3f0fd0a12b621250a6fdeadbce5172db230479"
_QLIB_EXPR_BASE_ORIGINAL_LINE_COUNT = 281
_QLIB_EXPR_OPS_BODY_SHA256 = "6f78562448aaeca33375274e2406368a619744d4ca25733837e7463369858691"
_QLIB_EXPR_OPS_ORIGINAL_LINE_COUNT = 1681

# `qlib_eval_surface.py` is two DISJOINT excerpts (qlib/utils/__init__.py:277-302 and
# qlib/data/data.py:383-407, same commit as above) concatenated with a divider comment this
# vendoring added — not a single contiguous tail of one upstream file, so it cannot use
# `_body_sha256`'s "last N lines" slicing. Each excerpt is instead located by its own marker
# comment (also added by this vendoring, see `qlib_eval_surface.py`'s header) and hashed alone.
_QLIB_EVAL_SURFACE_PARSE_FIELD_MARKER = (
    b"# qlib/utils/__init__.py:277-302 \xe2\x80\x94 Copyright (c) Microsoft Corporation, "
    b"MIT License.\n"
)
_QLIB_EVAL_SURFACE_PARSE_FIELD_SHA256 = (
    "442ffe83341d3478543fa731b0dd553b3061c560cdb653cbbae6dc1789529e42"
)
_QLIB_EVAL_SURFACE_DIVIDER = b"\n\n# ---\n"
_QLIB_EVAL_SURFACE_EXPR_PROVIDER_MARKER = (
    b"# qlib/data/data.py:383-407 -- Copyright (c) Microsoft Corporation, MIT License.\n"
)
_QLIB_EVAL_SURFACE_EXPR_PROVIDER_SHA256 = (
    "2458c485a32be33f2405b6ce3ffa499a4e0ac968b29a94340959a10bc45b9448"
)

# Same reasoning, for the six `rdagent_*.py` files — cross-checked against
# github.com/microsoft/RD-Agent at commit 32b3d395e73d9db5eee3fe9063d69aec0fdc83bd.
_RDAGENT_EXPERIMENT_BODY_SHA256 = "46c137dd3c74560ad4915ee8207cc76d9ff312683f047d498a124cda21d138f3"
_RDAGENT_EXPERIMENT_ORIGINAL_LINE_COUNT = 410
_RDAGENT_FACTOR_BODY_SHA256 = "f92bca5c9a60a33c57a0aa3c9005b2841476ab79861d03e765b1fddf4f426c42"
_RDAGENT_FACTOR_ORIGINAL_LINE_COUNT = 231
_RDAGENT_COSTEER_TASK_BODY_SHA256 = (
    "17bb7e1f95aa5ea3b2377b68211d201d23ea59fea7d93fd2e2a27c3bb5219d90"
)
_RDAGENT_COSTEER_TASK_ORIGINAL_LINE_COUNT = 9
_RDAGENT_EXCEPTION_BODY_SHA256 = "e0812ccd02d8a416652190b795958c7a56150132291632d1ef6b5be983a926e2"
_RDAGENT_EXCEPTION_ORIGINAL_LINE_COUNT = 82
_RDAGENT_CACHE_WITH_PICKLE_MARKER = (
    b"# rdagent/core/utils.py:159-219 -- Copyright (c) Microsoft Corporation, MIT License.\n"
)
_RDAGENT_CACHE_WITH_PICKLE_SHA256 = (
    "502210464020dfc4351caaae2da72854ad70465c38811c33049f5c6f7936e027"
)
_RDAGENT_CACHE_UTILS_DIVIDER = b"\n\n# ---\n"
_RDAGENT_MD5_HASH_MARKER = (
    b"# rdagent/utils/__init__.py:202-206 -- Copyright (c) Microsoft Corporation, MIT License.\n"
)
_RDAGENT_MD5_HASH_SHA256 = "1daadfe499c31cd0e2506646a79d66eee34b1fc3bf7bbe6c4777e357f3e5440e"

# Same reasoning, for `stumpy_squared_distance.py` — cross-checked against
# github.com/TDAmeritrade/stumpy at commit e4caf8a7ba519d1ba04796cd06aa78d91e9ca6ee. The repo's
# LICENSE.txt is textbook 3-Clause BSD despite GitHub's API classifying it "Other" (see that
# vendored file's own header for why — an extra trademark sentence ahead of the standard BSD text
# trips GitHub's auto-detector; the actual terms were read directly, not inferred from the badge).
_STUMPY_SQUARED_DISTANCE_BODY_SHA256 = (
    "b15518425cef96e944011c67551ccdc04da11c39c34751f15aa19db2d352f56c"
)
_STUMPY_SQUARED_DISTANCE_ORIGINAL_LINE_COUNT = 66

# Same reasoning, for `lean_pairs_ranking.py` — cross-checked against
# github.com/QuantConnect/Lean at commit 23b735d99a357807dc0df9f4c51d30f05fe0d277, Apache-2.0.
# Marker-extracted (like the qlib/rdagent excerpts) rather than tail-sliced: the vendored body
# sits between a start marker comment and a `return corr` line this vendoring's own wrapper adds.
_LEAN_PAIRS_RANKING_MARKER = (
    b"VENDORED FROM HERE ==============================\n"
)
_LEAN_PAIRS_RANKING_END_MARKER = b"            return corr\n"
_LEAN_PAIRS_RANKING_SHA256 = "6b73baf904d689244e7723d6851d97f7f6826be7dab947e585bc9e353f6b4e67"

# Same reasoning, for `maxme_arbitrer.py` — cross-checked against
# github.com/maxme/bitcoin-arbitrage at commit f41684a3226710853096a4e93c3b823f92079abf, MIT.
# Marker-extracted: the vendored body starts immediately after this vendoring's own wrapper
# `__init__`'s last real line (the wrapper class/`__init__` sit between the license marker and
# the real vendored body, which starts at its own `def get_profit_for`).
_MAXME_ARBITRER_START = b"        self.max_tx_volume = max_tx_volume\n\n"
_MAXME_ARBITRER_SHA256 = "34caf305eaa43bb03efdb306fd41479683d3f65e48edc5b13f82c2a0bf0b000b"

# Same reasoning, for `pytaa_vigilant_allocation.py` and `pytaa_signal.py` — cross-checked against
# github.com/oronimbus/tactical-asset-allocation at commit 317ad1c6618def4e1dc0fb9879050c6f9f2f026c,
# MIT. Both are the standard "VENDORED FROM HERE" full-file marker, hashed to end of file.
_VENDORED_FROM_HERE_MARKER = b"VENDORED FROM HERE ==============================\n"
_PYTAA_VIGILANT_ALLOCATION_SHA256 = (
    "17278e507a8e51eee75ccf7aec871da1b42102cee9394a9daf0bbcc17df2665b"
)
_PYTAA_SIGNAL_SHA256 = "450e8717021f951e6442f09968b467013b658a78225105206ad32b28460b0ff7"

# Same reasoning, for `whale_signals_event_study.py` — cross-checked against
# github.com/zty05070242/whale-signals at commit 6be10a598c9319aa6b64bf517618eac2dd2ec2f5, MIT.
# Two real functions concatenated with a `\n\n# ---\n\n` divider (like qlib_eval_surface.py);
# hashed together as the full post-marker body, matching how the file is actually structured.
_WHALE_SIGNALS_EVENT_STUDY_SHA256 = (
    "c254d5047ff83c2a453a026708289f0c42047fababbb58310a162df69b9327e9"
)

# Same reasoning, for the two vendored TypeScript files in `crypto_sor_shim/src/lib/` — cross-
# checked against github.com/Is0tope/crypto_sor at commit
# e1bc5c85a160135889a250c21c7c20a056f8a245, MIT. Language-agnostic marker extraction: the same
# `_marker_extract_sha256` helper works on `//`-style comments exactly like `#`-style ones.
_TS_VENDORED_FROM_HERE_MARKER = (
    b"// ============================== VENDORED FROM HERE ==============================\n"
)
_CRYPTO_SOR_COMPOSITE_ORDER_BOOK_SHA256 = (
    "3bfbe4c972aa60cebdeb45b55e7b5f8e92d357219fe8471f9addee75527e5929"
)
_CRYPTO_SOR_COMMON_SHA256 = "2b6f6d723d3b6ca6762924d404317b0312481bd653bcd8983e607a1df1462331"

# Same reasoning, for `quantconnect_sue.py` — cross-checked against
# github.com/QuantConnect/Tutorials at commit 4a341890296f7e79e095508f06170c72ccaa629c,
# Apache-2.0. Standard "VENDORED FROM HERE" marker, hashed to end of file.
_QUANTCONNECT_SUE_SHA256 = "5d11845fc20fa18f595d5f886b2755111d609dd430985afcaeba5bb867fb2b2e"

# Same reasoning, for `quantconnect_preholiday_decision.py` — cross-checked against
# github.com/QuantConnect/Tutorials at the same commit as `quantconnect_sue.py` above
# (4a341890296f7e79e095508f06170c72ccaa629c). Standard "VENDORED FROM HERE" marker, hashed to
# end of file.
_QUANTCONNECT_PREHOLIDAY_DECISION_SHA256 = (
    "796d4999c27c99101c9809ea8b08f37b29ecd8717c1136000027b27e44d098b3"
)

# Vendored DATA, not code — the real US equity holiday dates from Lean's own real
# `market-hours-database.json` — cross-checked against github.com/QuantConnect/Lean at the same
# commit already used for `lean_pairs_ranking.py` (23b735d99a357807dc0df9f4c51d30f05fe0d277),
# Apache-2.0. Whole-file hash: this JSON is the vendored artifact in full, not an excerpt, so no
# marker extraction is needed.
_LEAN_MARKET_HOLIDAYS_USA_SHA256 = (
    "5661185bfc2a48a553141d060346ce70525f17b7a109ac3fe8d7c5cfa1134b90"
)

# Same reasoning, for `alphalens_ic.py` — cross-checked against
# github.com/stefan-jansen/alphalens-reloaded (the maintained fork of the original, archived
# quantopian/alphalens) at commit f0a07c22d554e4b4036983cc80320b432714fe7e, Apache-2.0. Three real
# functions from two different files in the same repo, concatenated with a `\n\n# ---\n\n`
# divider (like qlib_eval_surface.py) — marker-extracted, each excerpt but the last ends at the
# divider.
_ALPHALENS_IC_MARKER = (
    b"# alphalens/performance.py:28-77 -- Copyright 2017 Quantopian, Inc., "
    b"Apache-2.0 License.\n"
)
_ALPHALENS_IC_DIVIDER = b"\n\n# ---\n\n"
_ALPHALENS_IC_SHA256 = "b8040c3af43cbd08df6f0f786b19aa6567eedbad196a0aefedb7decc87980c90"
_ALPHALENS_COLS_MARKER = (
    b"# alphalens/utils.py:916-935 -- Copyright 2018 Quantopian, Inc., Apache-2.0 License.\n"
)
_ALPHALENS_COLS_SHA256 = "7c7572d0ab72175dcc663fae601ea9048884afad127594619740910d3c5e6606"
_ALPHALENS_TABLE_MARKER = (
    b"# alphalens/plotting.py:180-195 -- Copyright 2017 Quantopian, Inc., Apache-2.0 License.\n"
)
_ALPHALENS_TABLE_SHA256 = "1b1ddd62c603fa497404a7ed9e5174d99421f8e5533d8dc3af5f70122e8218ff"


def _marker_extract_sha256(path: Path, start_marker: bytes, end: bytes | None) -> str:
    """Locate one excerpt by its own marker comment and hash everything from just after it up
    to `end` (another marker's bytes) or, when `end` is None, to the end of the file."""
    raw = path.read_bytes()
    start = raw.index(start_marker) + len(start_marker)
    body = raw[start : raw.index(end, start)] if end is not None else raw[start:]
    return hashlib.sha256(body).hexdigest()


def _body_sha256(vendored_path: Path, original_line_count: int) -> str:
    lines = vendored_path.read_bytes().splitlines(keepends=True)
    body = b"".join(lines[-original_line_count:])
    return hashlib.sha256(body).hexdigest()


class TestVendoredFilesHaveNotDrifted:
    """Independent of the loader: byte-level proof the vendored files are what they claim to be."""

    def test_enforcement_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "vibe_trading_enforcement.py", _ENFORCEMENT_ORIGINAL_LINE_COUNT
        )
        assert got == _ENFORCEMENT_BODY_SHA256

    def test_mandate_model_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "vibe_trading_mandate_model.py", _MODEL_ORIGINAL_LINE_COUNT
        )
        assert got == _MODEL_BODY_SHA256

    def test_qlib_processor_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "qlib_cs_processor.py", _QLIB_PROCESSOR_ORIGINAL_LINE_COUNT
        )
        assert got == _QLIB_PROCESSOR_BODY_SHA256

    def test_vectorbt_metrics_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "vectorbt_dsr_metrics.py", _VECTORBT_METRICS_ORIGINAL_LINE_COUNT
        )
        assert got == _VECTORBT_METRICS_BODY_SHA256

    def test_tradingagents_memory_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_memory.py", _TA_MEMORY_ORIGINAL_LINE_COUNT
        )
        assert got == _TA_MEMORY_BODY_SHA256

    def test_tradingagents_rating_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_rating.py", _TA_RATING_ORIGINAL_LINE_COUNT
        )
        assert got == _TA_RATING_BODY_SHA256

    def test_serenity_journal_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "serenity_journal.py", _SERENITY_JOURNAL_ORIGINAL_LINE_COUNT
        )
        assert got == _SERENITY_JOURNAL_BODY_SHA256

    def test_serenity_guards_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "serenity_guards.py", _SERENITY_GUARDS_ORIGINAL_LINE_COUNT
        )
        assert got == _SERENITY_GUARDS_BODY_SHA256

    def test_feedlist_interface_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_feedlist_interface.py",
            _TA_FEEDLIST_INTERFACE_ORIGINAL_LINE_COUNT,
        )
        assert got == _TA_FEEDLIST_INTERFACE_BODY_SHA256

    def test_feedlist_errors_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_feedlist_errors.py",
            _TA_FEEDLIST_ERRORS_ORIGINAL_LINE_COUNT,
        )
        assert got == _TA_FEEDLIST_ERRORS_BODY_SHA256

    def test_feedlist_config_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_feedlist_config.py",
            _TA_FEEDLIST_CONFIG_ORIGINAL_LINE_COUNT,
        )
        assert got == _TA_FEEDLIST_CONFIG_BODY_SHA256

    def test_feedlist_default_config_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "tradingagents_feedlist_default_config.py",
            _TA_FEEDLIST_DEFAULT_CONFIG_ORIGINAL_LINE_COUNT,
        )
        assert got == _TA_FEEDLIST_DEFAULT_CONFIG_BODY_SHA256

    def test_qlib_expression_base_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "qlib_expression_base.py", _QLIB_EXPR_BASE_ORIGINAL_LINE_COUNT
        )
        assert got == _QLIB_EXPR_BASE_BODY_SHA256

    def test_qlib_expression_ops_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "qlib_expression_ops.py", _QLIB_EXPR_OPS_ORIGINAL_LINE_COUNT
        )
        assert got == _QLIB_EXPR_OPS_BODY_SHA256

    def test_qlib_eval_surface_parse_field_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "qlib_eval_surface.py",
            _QLIB_EVAL_SURFACE_PARSE_FIELD_MARKER,
            _QLIB_EVAL_SURFACE_DIVIDER,
        )
        assert got == _QLIB_EVAL_SURFACE_PARSE_FIELD_SHA256

    def test_qlib_eval_surface_expression_provider_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "qlib_eval_surface.py",
            _QLIB_EVAL_SURFACE_EXPR_PROVIDER_MARKER,
            None,
        )
        assert got == _QLIB_EVAL_SURFACE_EXPR_PROVIDER_SHA256

    def test_rdagent_experiment_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "rdagent_experiment.py", _RDAGENT_EXPERIMENT_ORIGINAL_LINE_COUNT
        )
        assert got == _RDAGENT_EXPERIMENT_BODY_SHA256

    def test_rdagent_factor_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "rdagent_factor.py", _RDAGENT_FACTOR_ORIGINAL_LINE_COUNT
        )
        assert got == _RDAGENT_FACTOR_BODY_SHA256

    def test_rdagent_costeer_task_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "rdagent_costeer_task.py", _RDAGENT_COSTEER_TASK_ORIGINAL_LINE_COUNT
        )
        assert got == _RDAGENT_COSTEER_TASK_BODY_SHA256

    def test_rdagent_exception_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "rdagent_exception.py", _RDAGENT_EXCEPTION_ORIGINAL_LINE_COUNT
        )
        assert got == _RDAGENT_EXCEPTION_BODY_SHA256

    def test_rdagent_cache_with_pickle_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "rdagent_cache_utils.py",
            _RDAGENT_CACHE_WITH_PICKLE_MARKER,
            _RDAGENT_CACHE_UTILS_DIVIDER,
        )
        assert got == _RDAGENT_CACHE_WITH_PICKLE_SHA256

    def test_rdagent_md5_hash_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "rdagent_cache_utils.py", _RDAGENT_MD5_HASH_MARKER, None,
        )
        assert got == _RDAGENT_MD5_HASH_SHA256

    def test_stumpy_squared_distance_body_matches_the_pinned_hash(self) -> None:
        got = _body_sha256(
            _BASELINES_DIR / "stumpy_squared_distance.py",
            _STUMPY_SQUARED_DISTANCE_ORIGINAL_LINE_COUNT,
        )
        assert got == _STUMPY_SQUARED_DISTANCE_BODY_SHA256

    def test_lean_pairs_ranking_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "lean_pairs_ranking.py",
            _LEAN_PAIRS_RANKING_MARKER,
            _LEAN_PAIRS_RANKING_END_MARKER,
        )
        assert got == _LEAN_PAIRS_RANKING_SHA256

    def test_maxme_arbitrer_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "maxme_arbitrer.py", _MAXME_ARBITRER_START, None,
        )
        assert got == _MAXME_ARBITRER_SHA256

    def test_pytaa_vigilant_allocation_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "pytaa_vigilant_allocation.py", _VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _PYTAA_VIGILANT_ALLOCATION_SHA256

    def test_pytaa_signal_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "pytaa_signal.py", _VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _PYTAA_SIGNAL_SHA256

    def test_whale_signals_event_study_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "whale_signals_event_study.py", _VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _WHALE_SIGNALS_EVENT_STUDY_SHA256

    def test_crypto_sor_composite_order_book_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "crypto_sor_shim" / "src" / "lib" / "CompositeOrderBook.ts",
            _TS_VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _CRYPTO_SOR_COMPOSITE_ORDER_BOOK_SHA256

    def test_crypto_sor_common_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "crypto_sor_shim" / "src" / "lib" / "common.ts",
            _TS_VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _CRYPTO_SOR_COMMON_SHA256

    def test_quantconnect_sue_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "quantconnect_sue.py", _VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _QUANTCONNECT_SUE_SHA256

    def test_quantconnect_preholiday_decision_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "quantconnect_preholiday_decision.py",
            _VENDORED_FROM_HERE_MARKER, None,
        )
        assert got == _QUANTCONNECT_PREHOLIDAY_DECISION_SHA256

    def test_lean_market_holidays_usa_body_matches_the_pinned_hash(self) -> None:
        got = hashlib.sha256(
            (_BASELINES_DIR / "lean_market_holidays_usa.json").read_bytes()
        ).hexdigest()
        assert got == _LEAN_MARKET_HOLIDAYS_USA_SHA256

    def test_alphalens_ic_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "alphalens_ic.py", _ALPHALENS_IC_MARKER, _ALPHALENS_IC_DIVIDER,
        )
        assert got == _ALPHALENS_IC_SHA256

    def test_alphalens_cols_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "alphalens_ic.py", _ALPHALENS_COLS_MARKER, _ALPHALENS_IC_DIVIDER,
        )
        assert got == _ALPHALENS_COLS_SHA256

    def test_alphalens_table_body_matches_the_pinned_hash(self) -> None:
        got = _marker_extract_sha256(
            _BASELINES_DIR / "alphalens_ic.py", _ALPHALENS_TABLE_MARKER, None,
        )
        assert got == _ALPHALENS_TABLE_SHA256


class TestAlphalensIcLoaderMakesTheRealCodeRunnable:
    def test_load_ic_module_succeeds(self) -> None:
        module = load_ic_module()
        assert module.factor_information_coefficient.__name__ == "factor_information_coefficient"

    def test_a_loader_conflict_raises_rather_than_silently_reusing_a_foreign_module(
        self,
    ) -> None:
        import sys

        dotted = "argus_eval_alphalens_ic"
        real = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(AlphalensIcLoadError):
                load_ic_module()
        finally:
            sys.modules.pop(dotted, None)
            if real is not None:
                sys.modules[dotted] = real

    def test_it_computes_a_real_information_coefficient_on_constructed_panel_data(self) -> None:
        import pandas as pd

        module = load_ic_module()
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2026-01-01", periods=4, freq="D"), ["A", "B", "C"]],
            names=["date", "asset"],
        )
        factor_data = pd.DataFrame(
            {"factor": [1.0, 2.0, 3.0] * 4, "1D": [1.0, 2.0, 3.0] * 4}, index=idx
        )
        ic = module.factor_information_coefficient(factor_data)
        assert ic["1D"].iloc[0] == pytest.approx(1.0)

    def test_it_computes_the_real_naive_significance_table_on_real_ic_output(self) -> None:
        import pandas as pd

        module = load_ic_module()
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2026-01-01", periods=10, freq="D"), ["A", "B", "C", "D"]],
            names=["date", "asset"],
        )
        factor_data = pd.DataFrame(
            {"factor": list(range(40)), "1D": [v % 7 - 3 for v in range(40)]}, index=idx
        )
        ic = module.factor_information_coefficient(factor_data)
        table = module.plot_information_table(ic, return_df=True)
        assert "t-stat(IC)" in table.columns
        assert "p-value(IC)" in table.columns


class TestMaxmeArbitrerLoaderMakesTheRealCodeRunnable:
    def test_load_arbitrer_module_succeeds(self) -> None:
        module = load_arbitrer_module()
        assert module.ArbitrerProfitDetector.__name__ == "ArbitrerProfitDetector"

    def test_it_computes_a_real_profit_figure_on_real_order_book_depth(self) -> None:
        module = load_arbitrer_module()
        depths = {
            "ex_a": {"asks": [{"price": 100.0, "amount": 1.0}], "bids": []},
            "ex_b": {"asks": [], "bids": [{"price": 100.5, "amount": 1.0}]},
        }
        detector = module.ArbitrerProfitDetector(depths, max_tx_volume=10.0)
        profit, volume, _ask, _bid, _wbuy, _wsell = detector.arbitrage_depth_opportunity(
            "ex_a", "ex_b"
        )
        assert profit == pytest.approx(0.5)
        assert volume == pytest.approx(1.0)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_arbitrer_module()
        second = load_arbitrer_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import maxme_arbitrer_loader as loader_module

        dotted = "argus_eval_maxme_arbitrer"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(MaxmeArbitrerLoadError):
                loader_module.load_arbitrer_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestLeanPairsRankingLoaderMakesTheRealCodeRunnable:
    def test_load_pairs_ranking_module_succeeds(self) -> None:
        module = load_pairs_ranking_module()
        assert module.rank_pairs_by_correlation.__name__ == "rank_pairs_by_correlation"

    def test_it_runs_real_scipy_pearsonr_on_real_data(self) -> None:
        import pandas as pd
        from scipy.stats import pearsonr

        module = load_pairs_ranking_module()
        df = pd.DataFrame({0: [1.0, 2, 3, 4, 5], 1: [2.0, 3, 4, 5, 6], 2: [5.0, 4, 3, 2, 1]})
        result = module.rank_pairs_by_correlation(df, 3, pearsonr)
        assert result[-1][1] == pytest.approx(1.0)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_pairs_ranking_module()
        second = load_pairs_ranking_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import lean_pairs_ranking_loader as loader_module

        dotted = "argus_eval_lean_pairs_ranking"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(LeanPairsRankingLoadError):
                loader_module.load_pairs_ranking_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestPytaaVigilantAllocationLoaderMakesTheRealCodeRunnable:
    def test_load_vigilant_allocation_module_succeeds(self) -> None:
        module = load_vigilant_allocation_module()
        assert module.vigilant_allocation.__name__ == "vigilant_allocation"

    def test_it_runs_the_real_breadth_rule_on_real_pandas_input(self) -> None:
        import pandas as pd

        module = load_vigilant_allocation_module()
        data = pd.Series({"A": 0.05, "B": -0.02, "C": 0.01, "SAFE": 0.03})
        out = module.vigilant_allocation(
            data, risk_assets=["A", "B", "C"], safe_assets=["SAFE"], top_k=2, step=0.25,
        )
        assert out["SAFE"].iloc[0] == pytest.approx(0.25)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_vigilant_allocation_module()
        second = load_vigilant_allocation_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import pytaa_vigilant_allocation_loader as loader_module

        dotted = "argus_eval_pytaa_vigilant_allocation"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(PytaaVigilantAllocationLoadError):
                loader_module.load_vigilant_allocation_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestPytaaSignalLoaderMakesTheRealCodeRunnable:
    def test_load_signal_module_succeeds(self) -> None:
        module = load_signal_module()
        assert module.Signal.__name__ == "Signal"

    def test_it_computes_a_real_momentum_score_on_real_pandas_input(self) -> None:
        import pandas as pd

        module = load_signal_module()
        dates = pd.date_range("2025-01-31", periods=13, freq="BME")
        df = pd.DataFrame({"X": [100 * (1.01**i) for i in range(13)]}, index=dates)
        score = module.Signal(df).momentum_score()
        assert score["X"].iloc[-1] == pytest.approx(0.49106933133397135)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_signal_module()
        second = load_signal_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import pytaa_signal_loader as loader_module

        dotted = "argus_eval_pytaa_signal"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(PytaaSignalLoadError):
                loader_module.load_signal_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestWhaleSignalsEventStudyLoaderMakesTheRealCodeRunnable:
    def test_load_event_study_module_succeeds(self) -> None:
        module = load_event_study_module()
        assert module.compute_hit_rates.__name__ == "compute_hit_rates"
        assert module.compute_base_rate.__name__ == "compute_base_rate"

    def test_it_runs_the_real_binomial_test_on_real_pandas_input(self) -> None:
        import pandas as pd

        module = load_event_study_module()
        rows = [{
            "tx_category": "exchange_withdrawal",
            "fwd_return_1h": 0.01, "fwd_return_6h": 0.01, "fwd_return_24h": 0.01,
        } for _ in range(40)]
        result = module.compute_hit_rates(pd.DataFrame(rows))
        assert result["exchange_withdrawal"][24]["hit_rate"] == pytest.approx(1.0)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_event_study_module()
        second = load_event_study_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import whale_signals_event_study_loader as loader_module

        dotted = "argus_eval_whale_signals_event_study"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(WhaleSignalsEventStudyLoadError):
                loader_module.load_event_study_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestCryptoSorLoaderRunsTheRealTypeScriptSubprocess:
    """Unlike every other loader in this file, there is no Python module to import — the real
    code is TypeScript, run through a real `node`/`ts-node` subprocess each call."""

    def test_the_real_router_fills_from_the_best_quoted_price(self) -> None:
        executions = run_new_order(
            "T", "BUY", 10.0,
            [
                {"exchange": "A", "side": "SELL", "price": 100.0, "size": 5.0},
                {"exchange": "B", "side": "SELL", "price": 99.0, "size": 5.0},
                {"exchange": "C", "side": "SELL", "price": 101.0, "size": 20.0},
            ],
        )
        assert executions[0]["exchange"] == "B"
        assert executions[0]["lastPrice"] == pytest.approx(99.0)
        assert executions[1]["exchange"] == "A"

    def test_a_missing_shim_directory_is_reported_clearly(self) -> None:
        from argus.eval.baselines import crypto_sor_loader as loader_module

        original = loader_module._SHIM_DIR
        loader_module._SHIM_DIR = original / "does-not-exist"
        try:
            with pytest.raises(CryptoSorSubprocessError, match="shim directory not found"):
                run_new_order("T", "BUY", 1.0, [])
        finally:
            loader_module._SHIM_DIR = original


class TestQuantConnectSueLoaderMakesTheRealCodeRunnable:
    def test_load_sue_module_succeeds(self) -> None:
        module = load_sue_module()
        assert module.SueSorter.__name__ == "SueSorter"

    def test_it_computes_a_real_sue_value_on_real_input_shape(self) -> None:
        module = load_sue_module()
        monthly = []
        for q in (2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
            monthly.extend([q, q, q])
        sorter = module.SueSorter(
            eps_by_symbol={"X": monthly}, months_count=36, months_eps_change=12,
        )
        out: dict = {}

        class _Stock:
            Symbol = "X"

        sorter.compute(_Stock(), out)
        assert out["X"] == pytest.approx(4.319593977248311)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_sue_module()
        second = load_sue_module()
        assert first is second

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import quantconnect_sue_loader as loader_module

        dotted = "argus_eval_quantconnect_sue"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(QuantConnectSueLoadError):
                loader_module.load_sue_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestStumpySquaredDistanceLoaderMakesTheRealCodeRunnable:
    def test_load_squared_distance_module_succeeds(self) -> None:
        module = load_squared_distance_module()
        assert module._calculate_squared_distance.__name__ == "_calculate_squared_distance"

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_squared_distance_module()
        second = load_squared_distance_module()
        assert first is second

    def test_it_genuinely_jit_compiles_not_a_stub(self) -> None:
        """A numba `CPUDispatcher` object, not a plain Python function — confirms the real
        `@njit(fastmath=...)` decorator actually ran against a real `config`/`np`, not a shim
        that silently dropped the decorator."""
        module = load_squared_distance_module()
        assert type(module._calculate_squared_distance).__name__ == "CPUDispatcher"

    def test_a_duplicate_sys_modules_entry_is_refused(self) -> None:
        import sys

        from argus.eval.baselines import stumpy_squared_distance_loader as loader_module

        dotted = "argus_eval_stumpy_squared_distance"
        existing = sys.modules.pop(dotted, None)
        try:
            sys.modules[dotted] = object()  # type: ignore[assignment]
            with pytest.raises(StumpySquaredDistanceLoadError):
                loader_module.load_squared_distance_module()
        finally:
            del sys.modules[dotted]
            if existing is not None:
                sys.modules[dotted] = existing


class TestRdAgentFactorLoaderMakesTheRealCodeRunnable:
    @pytest.fixture(autouse=True)
    def _clean_rdagent_workspace(self):
        yield
        import shutil

        root = Path(__file__).resolve().parents[1] / "data"
        for name in ("_rdagent_workspace_tmp", "_rdagent_pickle_cache_tmp"):
            shutil.rmtree(root / name, ignore_errors=True)

    def test_load_factor_module_succeeds_and_exposes_the_real_symbols(self) -> None:
        experiment_module, factor_module = load_factor_module()
        assert experiment_module.FBWorkspace.__name__ == "FBWorkspace"
        assert factor_module.FactorTask.__name__ == "FactorTask"
        assert factor_module.FactorFBWorkspace.__name__ == "FactorFBWorkspace"

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_factor_module()
        second = load_factor_module()
        assert first[0] is second[0]
        assert first[1] is second[1]

    def test_a_shimmed_dependency_this_comparison_never_uses_raises_if_ever_called(self) -> None:
        _experiment_module, factor_module = load_factor_module()
        task = factor_module.FactorTask(
            factor_name="never_run", factor_description="d", factor_formulation="N/A",
        )
        task.version = 2
        ws = factor_module.FactorFBWorkspace(target_task=task, raise_exception=True)
        ws.file_dict = {"factor.py": "pass\n"}
        with pytest.raises(NotImplementedError, match="KAGGLE_IMPLEMENT_SETTING"):
            ws.execute()


class TestTradingAgentsLoaderMakesTheRealCodeRunnable:
    def test_load_trading_memory_log_class_succeeds(self) -> None:
        cls = load_trading_memory_log_class()
        assert cls.__name__ == "TradingMemoryLog"

    def test_a_real_write_and_read_round_trips(self) -> None:
        import tempfile
        from pathlib import Path

        cls = load_trading_memory_log_class()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "mem.md"
            log = cls({"memory_log_path": str(log_path)})
            log.store_decision("NVDA", "2026-01-01", "buy")
            log.update_with_outcome(
                "NVDA", "2026-01-01", raw_return=0.05, alpha_return=0.02,
                holding_days=1, reflection="noted", resolution_date="2026-01-02",
            )
            context = log.get_past_context("NVDA")
            assert "NVDA" in context

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_trading_memory_log_class()
        second = load_trading_memory_log_class()
        assert first is second

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        import types

        sentinel = types.ModuleType("tradingagents")
        sys.modules["tradingagents"] = sentinel
        try:
            with pytest.raises(TradingAgentsBaselineLoadError, match="did not create"):
                load_trading_memory_log_class()
        finally:
            del sys.modules["tradingagents"]


class TestTradingAgentsFeedlistLoaderMakesTheRealCodeRunnable:
    def test_load_interface_module_succeeds_and_exposes_the_real_symbols(self) -> None:
        module = load_interface_module()
        assert callable(module.route_to_vendor)
        assert "core_stock_apis" in module.TOOLS_CATEGORIES
        assert "macro_data" in module.OPTIONAL_CATEGORIES

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_interface_module()
        second = load_interface_module()
        assert first is second

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        import types

        sentinel = types.ModuleType("tradingagents")
        sys.modules["tradingagents"] = sentinel
        try:
            with pytest.raises(TradingAgentsFeedlistLoadError, match="did not create"):
                load_interface_module()
        finally:
            del sys.modules["tradingagents"]


class TestVectorbtLoaderMakesTheRealCodeRunnable:
    def test_load_vectorbt_baseline_succeeds_and_exposes_the_real_symbols(self) -> None:
        baseline = load_vectorbt_baseline()
        assert callable(baseline.deflated_sharpe_ratio)
        assert callable(baseline.approx_exp_max_sharpe)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_vectorbt_baseline()
        second = load_vectorbt_baseline()
        assert first.deflated_sharpe_ratio is second.deflated_sharpe_ratio

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        import types

        sentinel = types.ModuleType("vectorbt")
        sys.modules["vectorbt"] = sentinel
        try:
            with pytest.raises(VectorbtBaselineLoadError, match="did not create"):
                load_vectorbt_baseline()
        finally:
            del sys.modules["vectorbt"]


class TestQlibLoaderMakesTheRealCodeRunnable:
    def test_load_qlib_baseline_succeeds_and_exposes_the_real_symbols(self) -> None:
        baseline = load_qlib_baseline()
        assert callable(baseline.cs_rank_norm)
        assert callable(baseline.get_group_columns)

    def test_a_shimmed_dependency_this_comparison_never_uses_raises_if_ever_called(self) -> None:
        """The shims fail loud, not silent. `CSZScoreNorm` (not `CSRankNorm` — this comparison
        never calls it) is the one real class in the vendored file that reaches a shimmed
        `qlib.utils.data.zscore` at call time, so it is the honest way to prove the shim raises
        rather than silently returning a plausible wrong answer."""
        import pandas as pd

        baseline = load_qlib_baseline()
        df = pd.DataFrame({"datetime": ["d", "d"], "value": [1.0, 2.0]}).set_index("datetime")
        processor = baseline.cs_zscore_norm(fields_group=None, method="zscore")
        with pytest.raises(NotImplementedError):
            processor(df.copy())

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_qlib_baseline()
        second = load_qlib_baseline()
        assert first.cs_rank_norm is second.cs_rank_norm

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        import types

        sentinel = types.ModuleType("qlib")
        sys.modules["qlib"] = sentinel
        try:
            with pytest.raises(QlibBaselineLoadError, match="did not create"):
                load_qlib_baseline()
        finally:
            del sys.modules["qlib"]


class TestSerenityLoaderMakesTheRealCodeRunnable:
    def test_load_chained_journal_class_succeeds(self) -> None:
        cls = load_chained_journal_class()
        assert cls.__name__ == "ChainedJournal"

    def test_a_real_append_runs_without_raising(self) -> None:
        """`verify()`'s own real behaviour on this platform — including its Windows CRLF bug
        that fires on even a single entry — is exercised and documented in
        `test_journal_comparison.py`, not duplicated here; this test only confirms the loader
        makes `append()` itself callable."""
        import tempfile
        from pathlib import Path

        cls = load_chained_journal_class()
        with tempfile.TemporaryDirectory() as tmp:
            journal = cls(Path(tmp) / "j.jsonl")
            journal.append({"seq": 1})
            assert (Path(tmp) / "j.jsonl").exists()

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        first = load_chained_journal_class()
        second = load_chained_journal_class()
        assert first is second

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        import types

        sentinel = types.ModuleType("etoro_trading")
        sys.modules["etoro_trading"] = sentinel
        try:
            with pytest.raises(SerenityBaselineLoadError, match="did not create"):
                load_chained_journal_class()
        finally:
            del sys.modules["etoro_trading"]


class TestLoaderMakesTheRealCodeRunnable:
    def test_load_baseline_succeeds_and_exposes_the_real_symbols(self) -> None:
        baseline = load_baseline()
        assert callable(baseline.check_mandate)
        assert {m.value for m in baseline.instrument_type} == {
            "equity", "etf", "option", "crypto", "forex", "cfd",
        }
        assert {m.value for m in baseline.asset_class} == {
            "us_equity", "us_etf", "hk_equity", "cn_equity", "in_equity", "crypto", "forex",
        }

    def test_a_clean_order_is_allowed_by_the_real_function(self) -> None:
        b = load_baseline()
        caps = b.hard_caps(
            account_funding_usd=100_000.0, max_order_notional_usd=20_000.0,
            max_total_exposure_usd=80_000.0, max_leverage=1.0,
            allowed_instruments=(b.instrument_type.EQUITY,), max_trades_per_day=10,
        )
        universe = b.universe_constraint(
            asset_classes=(b.asset_class.US_EQUITY,), min_market_cap_usd=None,
            min_avg_daily_volume_usd=None, exclude_symbols=(),
        )
        consent = b.consent_meta(
            created_at="2026-01-01T00:00:00+00:00", consent_token_sha256="x",
            broker="robinhood", account_ref="acc1", expires_at="2099-01-01T00:00:00+00:00",
        )
        mandate = b.mandate(schema_version=1, hard_caps=caps, universe=universe, consent=consent)
        intent = b.order_intent(
            symbol="AAPL", side="buy", notional_usd=5000.0, quantity=None,
            instrument_type=b.instrument_type.EQUITY, asset_class=b.asset_class.US_EQUITY,
        )
        verdict = b.check_mandate(
            mandate, intent, positions=[], balance={"equity": 100_000.0},
            broker="robinhood", remote_tool="place_order", daily_count=0,
        )
        assert verdict is None

    def test_a_leverage_breach_is_caught_by_the_real_function(self) -> None:
        b = load_baseline()
        caps = b.hard_caps(
            account_funding_usd=1000.0, max_order_notional_usd=20_000.0,
            max_total_exposure_usd=80_000.0, max_leverage=1.0,
            allowed_instruments=(b.instrument_type.EQUITY,), max_trades_per_day=10,
        )
        universe = b.universe_constraint(
            asset_classes=(b.asset_class.US_EQUITY,), min_market_cap_usd=None,
            min_avg_daily_volume_usd=None, exclude_symbols=(),
        )
        consent = b.consent_meta(
            created_at="2026-01-01T00:00:00+00:00", consent_token_sha256="x",
            broker="robinhood", account_ref="acc1", expires_at="2099-01-01T00:00:00+00:00",
        )
        mandate = b.mandate(schema_version=1, hard_caps=caps, universe=universe, consent=consent)
        intent = b.order_intent(
            symbol="AAPL", side="buy", notional_usd=5000.0, quantity=None,
            instrument_type=b.instrument_type.EQUITY, asset_class=b.asset_class.US_EQUITY,
        )
        verdict = b.check_mandate(
            mandate, intent, positions=[], balance={"equity": 1000.0},
            broker="robinhood", remote_tool="place_order", daily_count=0,
        )
        assert verdict is not None
        assert verdict.kind == "quantitative"
        assert verdict.limit == "max_leverage"
        assert verdict.attempted_value == pytest.approx(5.0)

    def test_loading_twice_in_one_process_is_idempotent(self) -> None:
        """The shim must not choke on being asked to load a second time in the same interpreter."""
        first = load_baseline()
        second = load_baseline()
        assert first.check_mandate is second.check_mandate

    def test_a_pre_existing_unrelated_module_under_the_shim_name_is_refused(self) -> None:
        """Fail-closed on a name collision, matching the vendored code's own posture."""
        import types

        sentinel = types.ModuleType("src")
        sys.modules["src"] = sentinel
        try:
            with pytest.raises(BaselineLoadError, match="did not create"):
                load_baseline()
        finally:
            del sys.modules["src"]
            # Any partial registrations this attempt made under "src.live"/"src.live.mandate"
            # would only exist if our own shim had created "src" first, which it did not reach
            # here — nothing further to clean up.
