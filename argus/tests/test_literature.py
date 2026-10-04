"""The literature reader: the question's subject, the ranking, and the answer's honesty lines."""

from __future__ import annotations

from typing import Any

from argus.lui.research import literature


def _work(title: str, cited: int, venue: str, year: int = 2021) -> dict[str, Any]:
    return {"display_name": title, "publication_year": year, "cited_by_count": cited,
            "doi": f"https://doi.org/10.1/{cited}",
            "primary_location": {"source": {"display_name": venue}},
            "abstract_inverted_index": {"We": [0], "study": [1], "momentum.": [2], "Then": [3]},
            "authorships": [{"author": {"display_name": "A. Author"}}]}


PAYLOAD = {"meta": {"count": 6305}, "results": [
    _work("Conditional heteroskedasticity in asset returns", 900, "Journal of Statistics"),
    _work("Cryptocurrencies and momentum", 143, "Economics Letters"),
    _work("Cryptocurrencies and momentum", 20, "SSRN Electronic Journal"),
    _work("Momentum trading in cryptocurrencies", 60, "SSRN Electronic Journal"),
]}


def test_the_subject_is_the_question_without_the_asking_words() -> None:
    assert literature.topic("is there any research on funding rates predicting crypto returns?"
                            ) == "funding rates predicting crypto returns"
    assert literature.topic("what does the literature say about momentum in cryptocurrencies"
                            ) == "momentum in cryptocurrencies"


def test_the_title_match_ranks_before_citations_and_preprints_are_said() -> None:
    said = literature.lines("any papers on momentum in cryptocurrencies",
                            fetch=lambda url, timeout: PAYLOAD)
    assert said is not None and "6,305 published works" in said[0]
    assert said[1].startswith("Cryptocurrencies and momentum (2021; A. Author; Economics Letters")
    assert "Abstract opens: \"We study momentum.\"" in said[1]
    assert "preprint, not peer reviewed" in said[2]
    assert sum("Cryptocurrencies and momentum (" in x for x in said) == 1
    assert "does not vouch for their findings" in said[0]


def test_an_outage_is_said_and_nothing_stands_in() -> None:
    def down(url: str, timeout: float) -> Any:
        raise OSError("no route")

    said = literature.lines("is there research on bitcoin and inflation", fetch=down)
    assert said is not None and "did not answer just now" in said[0]


def test_an_ordinary_question_is_not_taken() -> None:
    assert literature.lines("what is BTC doing today") is None
