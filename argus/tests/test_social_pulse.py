"""The X and Reddit crowd read: stories counted by distinct people, and always dated."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
