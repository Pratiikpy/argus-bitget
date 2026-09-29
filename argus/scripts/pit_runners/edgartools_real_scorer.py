"""Score ``edgartools_real_runner.py``'s dump of the real edgartools package against the
artefact's ``edgartools_fiscal`` rebuild and against ``argus_dated_after``, with ARGUS's own
``TRUTH_YOY_DAYS``, ``is_true_yoy`` and ``mcnemar_exact``. Run under ARGUS's venv from ``argus/``:

    python scripts/pit_runners/edgartools_real_scorer.py <runner output .json>

Writes ``data/edgartools_real_check.json``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from argus.eval.general_sue_comparison import is_true_yoy, mcnemar_exact

REAL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("edgartools_real_output.json")
ARTEFACT_PATH = Path("data/general_sue_comparison.json")
OUT_PATH = Path("data/edgartools_real_check.json")


def is_true_yoy_iso(end_iso: str, prior_end_iso: str) -> bool:
    from datetime import date
    return is_true_yoy(date.fromisoformat(end_iso), date.fromisoformat(prior_end_iso))


def real_correct(entry: dict) -> bool:
    """Same definition as CORRECTNESS_DEFINITION in general_sue_comparison.py: finite SUE and
    every pair genuinely year-over-year (320-410 days)."""
    if entry.get("status") != "ok":
        return False
    sue = entry.get("sue")
    if sue is None or not math.isfinite(sue):
        return False
    pairs = entry.get("pairs") or []
    if not pairs:
        return False
    return all(is_true_yoy_iso(a, b) for a, b in pairs)


def main() -> int:
    real = json.loads(REAL_PATH.read_text(encoding="utf-8"))
    artefact = json.loads(ARTEFACT_PATH.read_text(encoding="utf-8"))
    edgartools_block = artefact["edgartools_fiscal"]
    rows_by_cik = {r["cik"]: r for r in edgartools_block["filer_rows"]}

    assert set(real["filers"]) == set(rows_by_cik), (
        f"filer set mismatch: real has {len(real['filers'])}, artefact has {len(rows_by_cik)}; "
        f"only in real: {sorted(set(real['filers']) - set(rows_by_cik))[:10]}; "
        f"only in artefact: {sorted(set(rows_by_cik) - set(real['filers']))[:10]}"
    )

    both_correct = only_real = only_rebuild = neither = 0
    real_vs_argus_a = real_vs_argus_only_real = real_vs_argus_only_argus = real_vs_argus_neither = 0
    disagreements: list[dict] = []
    sue_value_mismatches: list[dict] = []

    for cik, entry in real["filers"].items():
        row = rows_by_cik[cik]
        rebuild_arm = row["arms"]["edgartools_fiscal"]
        argus_arm = row["arms"]["argus_dated_after"]

        r_correct = real_correct(entry)
        rebuild_correct = bool(rebuild_arm["correct"])
        argus_correct = bool(argus_arm["correct"])

        if r_correct and rebuild_correct:
            both_correct += 1
        elif r_correct:
            only_real += 1
        elif rebuild_correct:
            only_rebuild += 1
        else:
            neither += 1

        if r_correct and argus_correct:
            real_vs_argus_a += 1
        elif r_correct:
            real_vs_argus_only_real += 1
        elif argus_correct:
            real_vs_argus_only_argus += 1
        else:
            real_vs_argus_neither += 1

        if r_correct != rebuild_correct:
            reason = "unknown"
            real_sue = entry.get("sue")
            rebuild_sue = rebuild_arm.get("sue")
            if entry.get("status") != "ok":
                reason = f"real status={entry.get('status')}"
            elif real_sue is None:
                reason = f"real refused: {entry.get('refusal')}"
            elif not math.isfinite(real_sue):
                reason = "real non_finite SUE"
            elif not (entry.get("pairs") or []):
                reason = "real produced no pairs"
            elif not all(is_true_yoy_iso(a, b) for a, b in entry["pairs"]):
                gaps = [
                    (__import__("datetime").date.fromisoformat(a)
                     - __import__("datetime").date.fromisoformat(b)).days
                    for a, b in entry["pairs"]
                ]
                reason = f"real has a non-year-over-year pair, gaps={gaps}"
            elif rebuild_arm["status"] != "value":
                reason = f"rebuild status={rebuild_arm['status']}"
            elif not rebuild_arm["all_pairs_yoy"]:
                reason = ("rebuild has a non-year-over-year pair, "
                          f"gaps={rebuild_arm['pair_gaps_days']}")
            disagreements.append({
                "cik": cik, "real_correct": r_correct, "rebuild_correct": rebuild_correct,
                "real_sue": real_sue, "rebuild_sue": rebuild_sue,
                "real_quarters": entry.get("quarters"),
                "rebuild_quarters": row.get("edgartools_quarters"),
                "real_fiscal_year_end_month": entry.get("fiscal_year_end_month"),
                "rebuild_fiscal_year_end_month": row.get("edgartools_fiscal_year_end_month"),
                "real_pairs": entry.get("pairs"),
                "rebuild_pair_gaps_days": rebuild_arm.get("pair_gaps_days"),
                "reason": reason,
            })
        elif (entry.get("sue") is not None and rebuild_arm.get("sue") is not None
              and math.isfinite(entry["sue"]) and math.isfinite(rebuild_arm["sue"])
              and abs(entry["sue"] - rebuild_arm["sue"])
              > 1e-9 * max(1.0, abs(rebuild_arm["sue"]))):
            sue_value_mismatches.append({
                "cik": cik, "real_sue": entry["sue"], "rebuild_sue": rebuild_arm["sue"],
            })

    n = len(real["filers"])
    mcnemar_vs_rebuild = mcnemar_exact(only_real, only_rebuild)
    mcnemar_vs_argus = mcnemar_exact(real_vs_argus_only_real, real_vs_argus_only_argus)

    real_correct_total = both_correct + only_real
    rebuild_correct_total = artefact["edgartools_fiscal"]["correct"]["edgartools_fiscal"]["correct"]
    argus_correct_total = artefact["edgartools_fiscal"]["correct"]["argus_dated_after"]["correct"]

    out = {
        "generated_by": "scripts/pit_runners/edgartools_real_runner.py + ad hoc scorer under "
                        "argus's own venv (scripts/pit_runners/edgartools_real_scorer.py)",
        "edgartools_version": real["edgartools_version"],
        "method": real["method"],
        "offline_or_live": "offline: EntityFactsParser.parse_company_facts fed the frozen "
                            "companyfacts snapshot directly, no network access",
        "companyfacts_source": real["companyfacts_source"],
        "filers_scored": n,
        "correct_counts": {
            "real_edgartools_package": real_correct_total,
            "argus_rebuild_edgartools_fiscal": rebuild_correct_total,
            "argus_dated_after": argus_correct_total,
        },
        "real_vs_rebuild": {
            "filers": n,
            "both_correct": both_correct,
            "only_real_correct": only_real,
            "only_rebuild_correct": only_rebuild,
            "neither_correct": neither,
            "per_filer_verdicts_agree": both_correct + neither,
            "per_filer_verdicts_disagree": only_real + only_rebuild,
            "mcnemar_exact": mcnemar_vs_rebuild,
        },
        "real_vs_argus_dated_after": {
            "filers": n,
            "both_correct": real_vs_argus_a,
            "only_real_correct": real_vs_argus_only_real,
            "only_argus_correct": real_vs_argus_only_argus,
            "neither_correct": real_vs_argus_neither,
            "difference_in_correct_share": (real_vs_argus_only_real - real_vs_argus_only_argus) / n,
            "mcnemar_exact": mcnemar_vs_argus,
        },
        "disagreements_with_rebuild": disagreements,
        "sue_value_mismatches_on_agreeing_verdicts": sue_value_mismatches,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({
        "correct_counts": out["correct_counts"],
        "real_vs_rebuild": out["real_vs_rebuild"],
        "real_vs_argus_dated_after": out["real_vs_argus_dated_after"],
        "n_disagreements": len(disagreements),
        "n_sue_value_mismatches": len(sue_value_mismatches),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
