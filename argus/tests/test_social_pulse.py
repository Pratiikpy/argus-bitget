"""The X and Reddit crowd read: stories counted by distinct people, and always dated."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from argus.market import social_pulse as sp

NOW = datetime(2026, 9, 25, 12, tzinfo=UTC)
LINE = "NVDA to 300 this week, loading calls, the squeeze is coming for everyone short"


def _post(i: int, author: str, text: str, minutes: int, channel: str = "x") -> sp.Post:
    return sp.Post(id=f"p{i}", claim=text, source=author,
                   available_at=NOW - timedelta(minutes=minutes), channel=channel)


def test_one_line_from_several_accounts_is_one_coordinated_story() -> None:
    posts = [_post(i, f"@acct{i}", LINE, 10 + i) for i in range(4)]
    posts.append(_post(9, "u/independent", "Nvidia guidance looks conservative given "
                       "hyperscaler capex plans for next year", 30, "reddit"))
    row = sp.pulse("NVDAUSDT", posts, ["twitter: 4 tweets"], NOW)
    assert row["posts"] == 5 and row["accounts"] == 5
    assert row["distinct_stories"] == 2
    assert row["coordinated_stories"] == 1 and row["coordinated_posts"] == 4
    assert row["by_channel"] == {"x": 4, "reddit": 1}


def test_one_account_repeating_itself_is_not_coordination() -> None:
    posts = [_post(i, "@samebot", LINE, 5 + i) for i in range(4)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW)
    assert row["distinct_stories"] == 1 and row["coordinated_stories"] == 0


def test_the_console_lines_state_the_snapshot_age_and_name_coordination() -> None:
    posts = [_post(i, f"@acct{i}", LINE, 10 + i) for i in range(4)]
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (posts, ["twitter: 4 tweets"]),
                        now=NOW)
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW + timedelta(hours=3))
    assert "3h ago" in lines[0] and "4 posts from 4 accounts" in lines[0]
    assert "look coordinated" in lines[1] and "repetition, not new information" in lines[1]


def test_a_stale_snapshot_says_so() -> None:
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: ([_post(1, "@a", LINE, 5)], []),
                        now=NOW)
    assert "stale" in sp.lines_for("NVDAUSDT", snapshot, now=NOW + timedelta(hours=20))[0]


def test_a_single_voice_is_not_reported_as_the_most_shared_story() -> None:
    posts = [_post(i, "@recapbot", f"Market recap {i} for Thursday", 5 + i) for i in range(3)]
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (posts, []), now=NOW)
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW)
    assert not any(line.startswith("Most-shared story") for line in lines)


def test_an_unreachable_platform_is_reported_not_silenced() -> None:
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: ([], [
        "twitter: unavailable (cli not installed)", "reddit: unavailable (TimeoutExpired)"]),
        now=NOW)
    line = sp.lines_for("NVDAUSDT", snapshot, now=NOW)[0]
    assert "not collected" in line and "cli not installed" in line


def test_a_name_that_was_not_swept_adds_nothing() -> None:
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: ([], []), now=NOW)
    assert sp.lines_for("TSLAUSDT", snapshot) == []
    assert sp.lines_for("NVDAUSDT", None) == []


# Abusive posts (2026-09-25): withheld before counting, never stored, never quoted. The lines
# below are written for these tests; no benchmark text is used here.
ABUSE = "NVDA bagholders are fucking idiots, kys"
BRIGADE = "NVDA holders are retarded morons lmao"


def test_an_abusive_post_is_withheld_not_counted() -> None:
    posts = [_post(1, "@a", LINE, 5), _post(2, "@b", ABUSE, 6)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW)
    assert row["posts"] == 1 and row["accounts"] == 1
    assert row["withheld_abusive"] == 1 and row["withheld_accounts"] == 1
    assert row["withheld_by_reason"] == {"threat_or_harassment": 1}
    assert ABUSE not in str(row)


def test_a_brigade_of_abuse_is_not_a_coordinated_story_and_is_not_quoted() -> None:
    posts = [_post(i, f"@acct{i}", BRIGADE, 10 + i) for i in range(4)]
    posts.append(_post(9, "u/independent", "Nvidia guidance looks conservative", 30, "reddit"))
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (posts, []), now=NOW)
    row = snapshot["symbols"][0]
    assert row["coordinated_stories"] == 0 and row["withheld_abusive"] == 4
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW)
    text = " ".join(lines)
    assert "retarded" not in text and "4 further posts were withheld as abusive" in text
    assert "abuse" in snapshot["abuse_screen"] or "abusive" in snapshot["abuse_screen"]


def test_without_the_screen_the_old_behaviour_is_reproduced() -> None:
    posts = [_post(i, f"@acct{i}", BRIGADE, 10 + i) for i in range(4)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW, screen=None)
    assert row["posts"] == 4 and row["coordinated_stories"] == 1
    assert row["withheld_abusive"] == 0 and row["top_stories"][0]["text"] == BRIGADE


def test_a_snapshot_written_before_the_screen_is_still_not_quoted() -> None:
    posts = [_post(i, f"@acct{i}", BRIGADE, 10 + i) for i in range(4)]
    old_row = sp.pulse("NVDAUSDT", posts, [], NOW, screen=None)
    for key in ("withheld_abusive", "withheld_by_reason", "withheld_accounts"):
        old_row.pop(key)
    snapshot = {"generated_at": NOW.isoformat(), "coordination_rule": "3+ accounts",
                "symbols": [old_row]}
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW)
    assert "look coordinated" in lines[1] and "retarded" not in " ".join(lines)


def test_an_abusive_lead_story_in_an_old_snapshot_is_named_but_not_quoted() -> None:
    posts = [_post(1, "@a", ABUSE, 5), _post(2, "@b", ABUSE, 200)]
    old_row = sp.pulse("NVDAUSDT", posts, [], NOW, screen=None)
    assert old_row["coordinated_stories"] == 0
    snapshot = {"generated_at": NOW.isoformat(), "symbols": [old_row]}
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW)
    assert any("is not quoted: it carries abusive language" in line for line in lines)
    assert "kys" not in " ".join(lines)


def test_when_every_post_is_abusive_the_console_says_so() -> None:
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: ([_post(1, "@a", ABUSE, 5)], []),
                        now=NOW)
    line = sp.lines_for("NVDAUSDT", snapshot, now=NOW)[0]
    assert "the one post naming NVDA" in line and "was withheld as abusive" in line
    two = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (
        [_post(1, "@a", ABUSE, 5), _post(2, "@b", BRIGADE, 6)], []), now=NOW)
    assert "all 2 posts naming NVDA" in sp.lines_for("NVDAUSDT", two, now=NOW)[0]


def test_an_abusive_handle_is_not_republished_but_still_counts_as_a_source() -> None:
    posts = [_post(1, "@FuckTheFed", LINE, 5), _post(2, "@calm_trader", LINE, 6),
             _post(3, "@other", LINE, 7)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW)
    story = row["top_stories"][0]
    assert story["accounts"] == 3 and story["coordinated"]
    assert sp.WITHHELD_HANDLE in story["authors"] and "@FuckTheFed" not in story["authors"]


def test_a_coordinated_story_is_described_never_quoted() -> None:
    # A line the screen does not flag, pushed by four accounts: sized, not republished.
    posts = [_post(i, f"@acct{i}", LINE, 10 + i) for i in range(4)]
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (posts, []), now=NOW)
    lines = sp.lines_for("NVDAUSDT", snapshot, now=NOW)
    assert "the largest is 4 posts from 4 accounts, not quoted" in lines[1]
    assert LINE[:40] not in " ".join(lines)
    old = sp.lines_for("NVDAUSDT", snapshot, now=NOW, screen=None)
    assert LINE[:40] in old[1]


def test_market_jargon_is_counted_not_withheld() -> None:
    posts = [_post(1, "@a", "Top ETF gainers & losers today: SQQQ leads", 5),
             _post(2, "@b", "dumb money is chasing NVDA at the highs", 6)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW)
    assert row["posts"] == 2 and row["withheld_abusive"] == 0


# Tone (2026-09-26): VADER per kept post, weighted so a coordinated story counts once. A keyword
# scorer stands in for VADER where the arithmetic is the point, so these do not need the clone.
BULL = "NVDA guidance is great, strong quarter ahead"
BEAR = "NVDA earnings were awful, weak margins"


def _scorer(text: str) -> float:
    return 0.6 if "great" in text else -0.6 if "awful" in text else 0.0


def test_a_coordinated_story_counts_once_in_the_tone() -> None:
    posts = [_post(i, f"@bull{i}", BULL, 10 + i) for i in range(6)]
    posts += [_post(20, "@bear1", BEAR, 30), _post(21, "u/calm", "NVDA reports on Wednesday",
                                                   40, "reddit")]
    row = sp.pulse("NVDAUSDT", posts, [], NOW, tone=_scorer)
    tone = row["tone"]
    assert tone["status"] == "ok" and row["coordinated_stories"] == 1
    assert tone["counts"] == {"positive": 6, "neutral": 1, "negative": 1}
    assert tone["share"] == pytest.approx({"positive": 1 / 3, "neutral": 1 / 3,
                                           "negative": 1 / 3})
    assert tone["share_unweighted"]["positive"] == pytest.approx(6 / 8)
    assert tone["voices"] == pytest.approx(3.0)
    assert tone["coordinated_stories_counted_once"] == 1
    assert tone["posts_in_coordinated_stories"] == 6
    assert len(row["post_tone"]) == 8 and BULL not in str(row["post_tone"])


def test_one_account_repeating_itself_is_one_voice_in_the_tone() -> None:
    posts = [_post(i, "@samebot", BULL, 5 + i) for i in range(4)]
    posts.append(_post(9, "@other", BEAR, 30))
    tone = sp.pulse("NVDAUSDT", posts, [], NOW, tone=_scorer)["tone"]
    assert tone["share"] == pytest.approx({"positive": 0.5, "neutral": 0.0, "negative": 0.5})


def test_independent_accounts_each_count() -> None:
    # two accounts, one line, far apart: not coordinated, so two voices
    posts = [_post(1, "@a", BULL, 5), _post(2, "@b", BULL, 300), _post(3, "@c", BEAR, 10)]
    tone = sp.pulse("NVDAUSDT", posts, [], NOW, tone=_scorer)["tone"]
    assert tone["share"]["positive"] == pytest.approx(2 / 3)


def test_the_tone_line_is_plain_and_sums_to_100() -> None:
    posts = [_post(i, f"@bull{i}", BULL, 10 + i) for i in range(6)]
    posts += [_post(20, "@bear1", BEAR, 30), _post(21, "u/calm", "NVDA reports on Wednesday",
                                                   40, "reddit")]
    snapshot = sp.sweep(["NVDAUSDT"], fetch=lambda s, n: (posts, []), now=NOW)
    snapshot["symbols"][0] = sp.pulse("NVDAUSDT", posts, [], NOW, tone=_scorer)
    line = sp.lines_for("NVDAUSDT", snapshot, now=NOW)[-1]
    assert line.startswith("Crowd tone on NVDA: 34% positive, 33% neutral, 33% negative across 8 "
                           "posts from 8 accounts (VADER; the coordinated story counts once).")
    # VADER's measured lean is said on the same line (TweetEval: about 20 points more positive)
    assert "20 points more positive than human labellers" in line
    assert sum(sp.percentages({"positive": 1 / 3, "neutral": 1 / 3,
                               "negative": 1 / 3}).values()) == 100
    assert sp.percentages({"positive": 0.586, "neutral": 0.3, "negative": 0.114}) == {
        "positive": 59, "neutral": 30, "negative": 11}


def test_withheld_posts_are_not_scored_and_the_line_says_so() -> None:
    posts = [_post(1, "@a", BULL, 5), _post(2, "@b", ABUSE, 6)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW, tone=_scorer)
    assert row["tone"]["posts_scored"] == 1 and row["tone"]["withheld_not_scored"] == 1
    assert [entry["id"] for entry in row["post_tone"]] == ["p1"]
    snapshot = {"generated_at": NOW.isoformat(), "symbols": [row]}
    line = sp.lines_for("NVDAUSDT", snapshot, now=NOW)[-1]
    assert "100% positive" in line and "The 1 withheld abusive post is not scored" in line


def test_an_old_snapshot_without_tone_adds_no_tone_line() -> None:
    row = sp.pulse("NVDAUSDT", [_post(1, "@a", LINE, 5)], [], NOW, tone=_scorer)
    row.pop("tone")
    row.pop("post_tone")
    lines = sp.lines_for("NVDAUSDT", {"generated_at": NOW.isoformat(), "symbols": [row]},
                         now=NOW)
    assert lines and not any("Crowd tone" in line for line in lines)


def test_without_vader_the_tone_is_reported_unscored_not_invented(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.eval.baselines import vader_loader

    def missing() -> sp.Tone:
        raise vader_loader.VaderLoadError("no VADER clone at /nowhere")

    monkeypatch.setattr(sp, "vader_tone", missing)
    row = sp.pulse("NVDAUSDT", [_post(1, "@a", BULL, 5)], [], NOW)
    assert row["tone"]["status"].startswith("not scored") and row["post_tone"] == []
    assert row["posts"] == 1  # the volume read is unaffected
    line = sp.lines_for("NVDAUSDT", {"generated_at": NOW.isoformat(), "symbols": [row]},
                        now=NOW)[-1]
    assert line.startswith("Crowd tone on NVDA: not scored in this snapshot")


def test_the_existing_fields_are_unchanged() -> None:
    row = sp.pulse("NVDAUSDT", [_post(1, "@a", LINE, 5)], [], NOW, tone=_scorer)
    for key in ("symbol", "posts", "accounts", "by_channel", "distinct_stories",
                "coordinated_stories", "coordinated_posts", "withheld_abusive",
                "withheld_by_reason", "withheld_accounts", "top_stories", "status",
                "checked_at"):
        assert key in row


def test_the_real_vader_scores_the_crowd() -> None:
    from argus.eval.baselines import vader_loader

    try:
        sp.vader_tone()
    except vader_loader.VaderLoadError:
        pytest.skip("VADER clone not on this machine")
    posts = [_post(1, "@a", "Great quarter, strong guidance, love this stock", 5),
             _post(2, "@b", "Terrible quarter, awful guidance, hate this stock", 6)]
    row = sp.pulse("NVDAUSDT", posts, [], NOW)
    assert row["tone"]["counts"] == {"positive": 1, "neutral": 0, "negative": 1}
    assert "VADER" in row["tone"]["scorer"] and "0.05" in row["tone"]["scorer"]
