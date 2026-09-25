"""Answers in the reader's language, with every figure locked to the English it came from."""
# ruff: noqa: RUF001 — the fixtures are real Chinese text, full-width punctuation included

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import translate as tr


@pytest.mark.parametrize(("text", "lang"), [
    ("英伟达的资金费率贵吗", "zh"), ("輝達的資金費率貴嗎", "zh-Hant"),
    ("エヌビディアは高い？", "ja"),
    ("엔비디아 펀딩비가 비싼가요", "ko"),
    ("¿Qué pasaría con mi cuenta si el mercado cae un 20%?", "es"),
    ("Was passiert mit meinem Depot, wenn der Markt fällt?", "de"),
    ("is NVDA overbought?", None),
])
def test_the_language_is_read_from_the_question(text: str, lang: str | None) -> None:
    assert tr.target_language(text) == lang


def test_a_changed_or_dropped_figure_fails_the_check() -> None:
    src = "NVDA last 225.55 USDT (+0.43% over 24h); spread 0.4bps, quoted 2026-09-25 06:37 UTC."
    good = "NVDA 最新价 225.55 USDT（24h +0.43%）；点差 0.4bps，报价时间 2026-09-25 06:37 UTC。"
    assert tr.figures_match(src, good)
    assert not tr.figures_match(src, good.replace("225.55", "225.5"))
    assert not tr.figures_match(src, good.replace("0.43%", "零点四三%"))
    assert not tr.figures_match(src, good + " 另加 5%")
    link = "Filing: https://www.sec.gov/a/b-index.htm"
    assert not tr.figures_match(link, "文件：https://www.sec.gov/a/c-index.htm")


def test_only_answers_this_console_signed_are_translated() -> None:
    lines = ["NVDA 225.55", "Data: Bitget."]
    token = tr.sign("zh", lines)
    assert tr.verify("zh", lines, token)
    assert not tr.verify("zh", [*lines, "anything else"], token)
    assert not tr.verify("es", lines, token)


class FakeModel:
    def __init__(self, lines: list[Any]) -> None:
        self.lines = lines

    def complete_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        return {"lines": self.lines}


def test_a_line_whose_figures_move_keeps_its_english() -> None:
    src = ["Actionable: a round trip costs 12.4bps.", "Funding: flat."]
    model = FakeModel(["可操作提示：往返成本约 12.5bps。", "资金费率：持平。"])
    out = tr.translate(src, "zh", model)
    assert out["lines"] == ["Actionable: a round trip costs 12.4bps.", "资金费率：持平。"]
    assert out["kept_english"] == [0] and out["note"].startswith("以下由 Qwen 翻译成中文")


def test_a_model_that_merges_lines_keeps_all_the_english() -> None:
    src = ["one 1", "two 2"]
    out = tr.translate(src, "de", FakeModel(["eins 1 zwei 2"]))
    assert out["lines"] == src and out["kept_english"] == [0, 1] and out["note"] is None


def test_no_model_means_english(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tr.translate(["x 1"], "zh", None)["lines"] == ["x 1"]


def test_the_server_offers_and_guards_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import server

    monkeypatch.setattr(server, "_router", lambda: object())
    payload: dict[str, Any] = {"lines": [server._language_note("英伟达的资金费率贵吗") or "",
                                         "NVDA 225.55", "Data: Bitget."]}
    server.offer_translation(payload, "英伟达的资金费率贵吗")
    offer = payload["translate"]
    assert offer["lang"] == "zh" and offer["skip"] == 1
    assert tr.verify("zh", ["NVDA 225.55", "Data: Bitget."], offer["token"])
    english: dict[str, Any] = {"lines": ["NVDA 225.55"]}
    server.offer_translation(english, "is NVDA overbought?")
    assert "translate" not in english
