"""Every sentence the console can say, in both languages it claims to speak.

**This exists because the claim was wider than the code.** `lui/question.py` carries seventeen
Chinese patterns and routes a Chinese question to the right intent — that part was real and is
tested. But every *answer* was assembled from English f-strings, so a judge asking 为什么没有交易
got the correct analysis in a language they had not used. The documents said "in English and
Chinese", which a live check on the deployed console refuted, and the claim was narrowed to
"understands English and Chinese" before this module was written. This is the part that makes the
original claim true.

**The English templates are transcribed exactly from the strings they replace.** That is the whole
safety property: rendering in English must be byte-identical to what the console said before, so
the existing phrasing tests remain a check on behaviour rather than being rewritten to match a new
implementation. A translation layer that also quietly reworded the English would make those tests
meaningless.

**Nothing here is translated by a model.** The console's defining constraint is that it answers
with no model key — `build_router` returns None without one and the deterministic layer takes over —
and a translation step that called out to a model would break exactly that property for any user
who asked in Chinese. Every rendering below is a fixed template.

**Numbers, symbols, hashes and verdicts are never translated.** `NVDAUSDT` is `NVDAUSDT`, a hash
prefix is a hash prefix, and `no_trade` stays `no_trade` because it is the literal value in the
ledger row a reader is being invited to check. Translating an identifier would break the one thing
this console promises: that every figure can be traced back to the record it came from.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Language(StrEnum):
    """Which language an answer is rendered in. Chosen from the question, never configured."""

    EN = "en"
    ZH = "zh"


class PhraseError(KeyError):
    """Raised for an unknown phrase key rather than falling back to the key itself.

    A missing key that renders as ``"perf.headline"`` would ship a debug string to a user. Raising
    turns it into a test failure instead, which is where it belongs.
    """


# Keys are ``area.name``. The English side is transcribed character for character from the string
# it replaced; changing one is a change to the console's output and should be visible as such.
PHRASES: dict[str, dict[Language, str]] = {
    # --- performance -------------------------------------------------------------------------
    "perf.undefined_stat": {
        Language.EN: "{stat}: not available — {why}",
        Language.ZH: "{stat}：无法给出 — {why}",
    },
    "perf.no_trades": {
        Language.EN: (
            "The log holds {abstentions} abstention(s) and {trades} settled trade(s) over "
            "{days} day(s). Sharpe, win rate and drawdown need trades; abstentions are scored "
            "separately, by abstention value."
        ),
        Language.ZH: (
            "记录中有 {abstentions} 次放弃交易、{trades} 笔已结算交易，跨度 {days} 天。"
            "夏普比率、胜率和回撤都需要成交才能计算；放弃交易另行评分，按放弃价值衡量。"
        ),
    },
    "perf.headline": {
        Language.EN: (
            "Sharpe {sharpe:.3f} · max drawdown {drawdown:.2f}% · win rate {win_rate:.1f}% over "
            "{trades} settled trade(s) in a {days}-day window."
        ),
        Language.ZH: (
            "夏普比率 {sharpe:.3f} · 最大回撤 {drawdown:.2f}% · 胜率 {win_rate:.1f}%，"
            "基于 {days} 天窗口内的 {trades} 笔已结算交易。"
        ),
    },
    "perf.largest": {
        Language.EN: "Largest contributor: {symbol} at {share}% of gross PnL ({trades} trade(s)).",
        Language.ZH: "最大贡献者：{symbol}，占毛盈亏的 {share}%（{trades} 笔交易）。",
    },
    "perf.net_pnl": {
        Language.EN: "Net PnL {net} on stated capital {capital}.",
        Language.ZH: "净盈亏 {net}，声明本金 {capital}。",
    },
    # --- one decision ------------------------------------------------------------------------
    "why.header": {
        Language.EN: "seq {seq} · {symbol} · {verdict} · decided {at} ({phase} session)",
        Language.ZH: "序号 {seq} · {symbol} · {verdict} · 决策时间 {at}（{phase} 时段）",
    },
    "why.confidence": {
        Language.EN: (
            "Stated confidence {confidence:.2f}. {hours:.0f}h to genuine price discovery."
        ),
        Language.ZH: "声明置信度 {confidence:.2f}。距离真实价格发现还有 {hours:.0f} 小时。",
    },
    "why.thesis": {
        Language.EN: "Thesis: {thesis}",
        Language.ZH: "论点：{thesis}",
    },
    "why.invalidation": {
        Language.EN: "Invalidation: {conditions}",
        Language.ZH: "证伪条件：{conditions}",
    },
    "why.settled": {
        Language.EN: "Settled {at}: net {net}, direction {direction}.",
        Language.ZH: "已于 {at} 结算：净额 {net}，方向{direction}。",
    },
    "why.direction_correct": {Language.EN: "correct", Language.ZH: "正确"},
    "why.direction_wrong": {Language.EN: "wrong", Language.ZH: "错误"},
    "why.unsettled": {
        Language.EN: "Not yet settled.",
        Language.ZH: "尚未结算。",
    },
    "why.hash": {
        Language.EN: "Entry hash {hash} (chain position after {prev}).",
        Language.ZH: "条目哈希 {hash}（链上位置紧接 {prev} 之后）。",
    },
    "why.multiple": {
        Language.EN: "({count} decisions matched; this is the most recent.)",
        Language.ZH: "（共匹配 {count} 条决策，此处显示最近的一条。）",
    },
    # --- abstentions -------------------------------------------------------------------------
    "abst.header": {
        Language.EN: (
            "{count} abstention(s){window}, across {symbols} symbol(s), all in {phases} session(s)."
        ),
        Language.ZH: (
            "{count} 次放弃交易{window}，涉及 {symbols} 个标的，"
            "全部发生在 {phases} 时段。"
        ),
    },
    "abst.explain": {
        Language.EN: (
            "An abstention here is a recorded decision, not an absence of one: it is hash-chained, "
            "settled against the move that actually happened, and scored by abstention value."
        ),
        Language.ZH: (
            "此处的放弃交易是一条被记录下来的决策，而不是决策的缺席：它进入哈希链，"
            "对照实际发生的行情结算，并按放弃价值评分。"
        ),
    },
    "abst.row": {
        Language.EN: "  seq {seq} {symbol}: {thesis}",
        Language.ZH: "  序号 {seq} {symbol}：{thesis}",
    },
    "abst.settled": {
        Language.EN: (
            "{count} of them have a counterfactual recorded — the move that would have happened — "
            "so standing aside can be graded rather than assumed correct."
        ),
        Language.ZH: (
            "其中 {count} 条已记录反事实结果，也就是当时若入场会发生的行情，"
            "因此选择观望可以被评分，而不是被默认为正确。"
        ),
    },
    "abst.unsettled": {
        Language.EN: (
            "None has settled yet, so none is graded. The counterfactual is attached at "
            "settlement, not asserted now."
        ),
        Language.ZH: (
            "目前没有任何一条已结算，因此都尚未评分。反事实结果在结算时附加，而不是现在断言。"
        ),
    },
    # --- windows -----------------------------------------------------------------------------
    "window.moved": {
        Language.EN: (
            "Assumed: {used}, the latest {unit} with a record; nothing is recorded for {asked}."
        ),
        Language.ZH: "假设：按 {used} 回答，这是最近一个有记录的{unit}；{asked}没有任何记录。",
    },
    "window.unit.day": {Language.EN: "day", Language.ZH: "日子"},
    "window.unit.weekend": {Language.EN: "weekend", Language.ZH: "周末"},
    "window.unit.week": {Language.EN: "week", Language.ZH: "一周"},
    "window.asked.today": {Language.EN: "today ({date})", Language.ZH: "今天（{date}）"},
    "window.asked.yesterday": {
        Language.EN: "yesterday ({date})", Language.ZH: "昨天（{date}）",
    },
    "window.asked.the_weekend": {
        Language.EN: "the weekend that began {date}", Language.ZH: "{date} 开始的周末",
    },
    "window.asked.last_weekend": {
        Language.EN: "last weekend ({span})", Language.ZH: "上周末（{span}）",
    },
    "window.asked.this_week": {
        Language.EN: "this week (from {date})", Language.ZH: "本周（{date} 起）",
    },
    "window.asked.last_week": {
        Language.EN: "last week ({span})", Language.ZH: "上周（{span}）",
    },
    "window.week_of": {
        Language.EN: "the week of {span}",
        Language.ZH: "{span} 这一周",
    },
    "window.span": {
        Language.EN: "{start} to {end}",
        Language.ZH: "{start} 至 {end}",
    },
    "window.weekend_of": {
        Language.EN: "the weekend of {span}",
        Language.ZH: "{span} 周末",
    },
    # --- decision list -----------------------------------------------------------------------
    "list.header": {
        Language.EN: "{count} decision(s){window}: {breakdown}",
        Language.ZH: "{count} 条决策{window}：{breakdown}",
    },
    "list.row": {
        Language.EN: "  seq {seq} {symbol} {verdict} (confidence {confidence:.2f}) {at}",
        Language.ZH: "  序号 {seq} {symbol} {verdict}（置信度 {confidence:.2f}）{at}",
    },
    # Voided rows are excluded from the count and said to be, never silently dropped. A total
    # that quietly shrinks is the same defect as one that quietly includes a refused position.
    "list.voided": {
        Language.EN: (
            "  {voided} row(s) excluded from these counts: they record positions the risk layer "
            "refused and the ledger booked anyway (see paper/corrections.py). They remain in the "
            "chain, listed but not counted."
        ),
        Language.ZH: (
            "  {voided} 条记录未计入统计：风险层已拒绝的仓位被错误写入账本"
            "（见 paper/corrections.py）。记录仍保留在链上，仅列出、不计数。"
        ),
    },
    # --- integrity ---------------------------------------------------------------------------
    "integ.header": {
        Language.EN: "Hash chain {state} across {count} entry(ies).",
        Language.ZH: "哈希链在 {count} 条记录上{state}。",
    },
    "integ.state_intact": {Language.EN: "intact", Language.ZH: "完整"},
    "integ.state_broken": {Language.EN: "BROKEN", Language.ZH: "已断裂"},
    "integ.break_at": {
        Language.EN: "First break at {at}.",
        Language.ZH: "首个断裂点位于 {at}。",
    },
    "integ.explain": {
        Language.EN: (
            "Each row carries the hash of the previous one, so editing any historical row breaks "
            "every hash after it. Settlement fields are deliberately excluded from that hash, so "
            "attaching an outcome does not invalidate the chain — but they are not unprotected: "
            "a separate settlement-seal row commits to each outcome the moment it is attached, "
            "and editing a settled trade's P&L after the fact breaks that seal's own chain link "
            "the same way tampering with a decision does."
        ),
        Language.ZH: (
            "每一行都带有上一行的哈希，因此修改任何一条历史记录都会破坏其后的全部哈希。"
            "结算字段被有意排除在该哈希之外，所以补录结果不会使链失效——但它们并非不受保护："
            "每次补录结果时都会写入一条独立的结算封存记录，事后修改已结算交易的盈亏"
            "同样会破坏该封存记录自身的链接，与篡改决策记录时的效果相同。"
        ),
    },
    "integ.head": {
        Language.EN: "Head hash {hash}.",
        Language.ZH: "链头哈希 {hash}。",
    },
    "integ.settlement_tampered": {
        Language.EN: (
            "Settlement seal mismatch on decision(s) {seqs}: the outcome recorded now does not "
            "match what was sealed at settlement time. That is what real tampering with a "
            "settled trade's P&L looks like from here."
        ),
        Language.ZH: (
            "决策 {seqs} 的结算封存不匹配：当前记录的结算结果与结算时封存的内容不一致。"
            "这正是已结算交易盈亏遭到真实篡改时的表现。"
        ),
    },
    "integ.settlement_unsealed": {
        Language.EN: (
            "Decision(s) {seqs} are marked settled with no settlement seal at all — either "
            "written before this record started sealing outcomes, or settled by a path that "
            "bypassed the ledger's own settle() entirely."
        ),
        Language.ZH: (
            "决策 {seqs} 被标记为已结算，但完全没有结算封存记录——"
            "要么是在本记录开始封存结算结果之前写入的，要么是绕过账本自身 settle() 的方式结算的。"
        ),
    },
    "integ.truncated": {
        Language.EN: "The anchor disagrees with the log: {note}",
        Language.ZH: "锚点记录与日志不一致：{note}",
    },
    # --- positions ---------------------------------------------------------------------------
    "pos.none": {Language.EN: "No open positions.", Language.ZH: "当前没有持仓。"},
    "pos.none_detail": {
        Language.EN: (
            "The record holds {total} decision(s), of which {abstentions} are abstentions."
        ),
        Language.ZH: (
            "记录中共有 {total} 条决策，其中 {abstentions} 条为放弃交易。"
        ),
    },
    "pos.header": {
        Language.EN: "{count} open position(s):",
        Language.ZH: "{count} 个持仓：",
    },
    "pos.row": {
        Language.EN: "  seq {seq} {symbol} {side} {quantity} @ {price}",
        Language.ZH: "  序号 {seq} {symbol} {side} {quantity} @ {price}",
    },
    # --- calibration -------------------------------------------------------------------------
    "calib.insufficient": {
        Language.EN: (
            "Not enough graded outcomes to state calibration: {graded} against a floor of {floor}."
        ),
        Language.ZH: "已评分结果不足以给出校准度：{graded} 条，下限为 {floor} 条。",
    },
    "calib.insufficient_why": {
        Language.EN: (
            "Calibration on fewer is noise wearing a decimal point, so none is reported. This is a "
            "refusal to compute, not a missing feature."
        ),
        Language.ZH: (
            "样本更少时算出的校准度只是披着小数点的噪声，因此不予报告。"
            "这是拒绝计算，而不是功能缺失。"
        ),
    },
    "calib.headline": {
        Language.EN: (
            "Expected calibration error {ece} · Brier {brier} · accuracy {accuracy}% over "
            "{graded} graded prediction(s)."
        ),
        Language.ZH: (
            "期望校准误差 {ece} · Brier 分数 {brier} · 准确率 {accuracy}%，"
            "基于 {graded} 条已评分预测。"
        ),
    },
    "calib.deterministic": {
        Language.EN: (
            "Every figure is deterministic arithmetic over recorded decisions; no model grades "
            "another."
        ),
        Language.ZH: (
            "每个数字都是对已记录决策的确定性算术运算；不存在由一个模型给另一个模型打分。"
        ),
    },
    # --- session -----------------------------------------------------------------------------
    "sess.header": {
        Language.EN: (
            "As of the last decision ({at}): {phase} session, {hours:.0f}h until genuine price "
            "discovery."
        ),
        Language.ZH: (
            "截至最近一次决策（{at}）：{phase} 时段，"
            "距离真实价格发现还有 {hours:.0f} 小时。"
        ),
    },
    "sess.explain": {
        Language.EN: (
            "Bitget's US stock perpetuals trade 7x24; the anchor US equity market does not. "
            "The gap between those two clocks is why the desk prices deliberation differently "
            "by session phase."
        ),
        Language.ZH: (
            "Bitget 的美股永续合约全天候 7x24 交易，而作为锚定标的的美股市场并非如此。"
            "正是这两个时钟之间的落差，使得交易台按时段对思考成本给出不同的定价。"
        ),
    },
    # --- evidence ----------------------------------------------------------------------------
    "ev.header": {
        Language.EN: "Evidence behind seq {seq} ({symbol}, {at}):",
        Language.ZH: "序号 {seq} 决策背后的证据（{symbol}，{at}）：",
    },
    "ev.thesis": {
        Language.EN: "Thesis as written: {thesis}",
        Language.ZH: "当时写下的论点：{thesis}",
    },
    "ev.hash": {
        Language.EN: (
            "Market state hash {hash} — the evidence set is committed to the chain, so what the "
            "desk saw cannot be revised after the fact."
        ),
        Language.ZH: (
            "市场状态哈希 {hash} — 证据集已写入哈希链，因此交易台当时看到的内容事后无法更改。"
        ),
    },
    "ev.feed": {
        Language.EN: (
            "The per-item feed (SEC EDGAR filings, verified RSS, Bitget skill server) is gathered "
            "by argus.market.evidence:gather under an as-of gate that raises on any item stamped "
            "after the decision instant."
        ),
        Language.ZH: (
            "逐条证据来源（SEC EDGAR 文件、经校验的 RSS、Bitget 技能服务）由 "
            "argus.market.evidence:gather 采集，并受时点闸门约束："
            "任何时间戳晚于决策时刻的条目都会直接抛错。"
        ),
    },
    # --- risk control ------------------------------------------------------------------------
    "risk.header": {
        Language.EN: (
            "The risk layer saw {decisions} decision(s), of which {offered} offered a position it "
            "could reduce."
        ),
        Language.ZH: (
            "风险层共处理 {decisions} 条决策，其中 {offered} 条提供了可供缩减的仓位。"
        ),
    },
    "risk.untested": {
        Language.EN: (
            "It intervened zero times because it was never invoked: every decision in this record "
            "was an abstention, and there was no exposure to narrow. That is an UNTESTED risk "
            "layer, not a restrained one, and reporting it as a low intervention rate over all "
            "decisions would be the flattering version of the same number."
        ),
        Language.ZH: (
            "它零次介入，是因为根本没有被调用：本记录中的每一条决策都是放弃交易，"
            "没有任何敞口可以收窄。这是一个**未经检验**的风险层，而不是一个克制的风险层；"
            "若按全部决策来报告一个很低的介入率，那只是同一个数字的粉饰版本。"
        ),
    },
}


WINDOW_LABELS_ZH: dict[str, str] = {
    "today": "今天",
    "yesterday": "昨天",
    "overnight": "隔夜",
    "this week": "本周",
    "last week": "上周",
    "the weekend": "周末",
    "last weekend": "上周末",
    "all time": "全部记录",
}
"""The windows `lui/question.py` resolves, named in Chinese for a Chinese answer's header."""

WINDOW_IN_ZH = "（{label}）"
"""How a Chinese header names its window: in full-width brackets after the count. Punctuation,
not a phrase, so it is kept out of :data:`PHRASES`, whose rule is that every Chinese entry is
written in Chinese."""


def t(key: str, language: Language = Language.EN, /, **kwargs: Any) -> str:
    """Render one phrase. Raises on an unknown key or a missing field rather than degrading.

    A phrase that silently rendered with a blank where a number should be would put an
    ungrounded-looking sentence in front of a reader, which is the one thing this console is built
    not to do.
    """
    entry = PHRASES.get(key)
    if entry is None:
        raise PhraseError(f"no phrase registered for {key!r}")
    template = entry.get(language) or entry[Language.EN]
    try:
        return template.format(**kwargs)
    except KeyError as exc:  # pragma: no cover - a template/caller mismatch is a build error
        raise PhraseError(f"phrase {key!r} needs field {exc.args[0]!r}") from exc


def language_of(text: str) -> Language:
    """Which language to answer in, decided from the question and nothing else.

    No setting, no header, no account preference: a reader who asks in Chinese is answered in
    Chinese, and the next question decides itself independently. The detection is the same Han
    check `lui/question.py` already uses to route intent, so the language of the answer can never
    disagree with the language the intent was matched in.
    """
    from argus.lui.question import has_chinese

    return Language.ZH if has_chinese(text) else Language.EN


__all__ = [
    "PHRASES",
    "WINDOW_IN_ZH",
    "WINDOW_LABELS_ZH",
    "Language",
    "PhraseError",
    "language_of",
    "t",
]
