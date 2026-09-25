"""ARGUS's abstention scoring vs. the general-purpose selective-prediction evaluators — run on the
same frozen record of 637 settled refusals, and lost before it was won.

**Why a non-trading rival.** "Abstention scored as a decision" was OWNED against the one trading
repository that scores refusals (`eval/abstention_comparison.py`, AutonomousTradeAgents' Ghost P&L).
The function itself is not a trading idea. A system that may decline to act, graded on what it
declined, is *selective prediction*, and the field that evaluates it has spent a decade on exactly
the failure `abstention_quality` has: scoring the refusals without asking whether the confidence
that chose them carries information. The strongest open implementations were picked on method, not
stars:

* **fd-shifts** (`IML-DKFZ/fd-shifts`, Apache-2.0, commit c4467aec) — the reference code of Traub
  et al., "Overcoming Common Flaws in the Evaluation of Selective Classification Systems" (NeurIPS
  2024 spotlight), which audited how AURC is computed across the field and introduced AUGRC. Its
  `analysis/rc_stats.py` + `rc_stats_utils.py` are tie-aware, accept continuous residuals, carry an
  optimal reference and working-point selection. Read in full.
* **torch-uncertainty** (`torch-uncertainty/torch-uncertainty`, Apache-2.0, commit 3f82fe5d) — a
  maintained PyTorch library shipping AURC, AUGRC, RiskAtxCov and CovAtxRisk as metrics
  (`metrics/classification/risk_coverage.py`, read in full).
* Considered and not run: MAPIE (BSD-3) — conformal risk control *chooses* a threshold with a
  guarantee rather than *scoring* a record of refusals, a different function; AbstentionBench
  (Meta, non-permissive licence) — grades LLM abstention with a model judge against labelled
  "should abstain" prompts, which a trading record does not have; Open Bandit Pipeline — off-policy
  evaluation estimates an unobserved counterfactual, and here the counterfactual move is observed.

**The same input.** The live paper ledger frozen at ``seq <= 684`` and ``settled_at <=
2026-09-25T13:47:40Z`` (ledger head ``5d144cbef0af9faf`` at freeze): 684 decisions, every one a
refusal; 637 settled with a counterfactual move; 470 of those stated a lean. Each lean is a withheld
prediction: acting on it would have netted ``direction * move - 12bps``. The rivals ran on the 470
(confidence = ``lean_confidence``, residual = 1 when that net is <= 0) through
`eval/baselines/selective_rivals_runner.py`, in their own environment because fd-shifts needs
NumPy < 2 — the command, the file hashes and both runs are in ``RIVAL_PROVENANCE``, the numbers in
``RIVAL_RECORDED``. A second, independent run reproduced every number exactly.

**What was measured, and who won each** — see :func:`criteria` for the table the artefact carries.

1. *Does the score depend on which calls were refused?* `abstention_quality` does not read a
   confidence, so a perfect gate, a random one and an inverted one score identically on the same
   refusals — run here, not argued. fd-shifts separates them (AURC 0.177 / 0.489 / 0.868 against a
   random reference of 0.532). **fd-shifts won; ARGUS's existing capability lost.**
2. *Is the score a property of the record or of its row order?* torch-uncertainty ranks by plain
   ``argsort`` (`risk_coverage.py:190-203`), so tied confidences are split by row order. On a gate
   with no information at all (six calls at one confidence) its AURC runs from 0.297 to 0.653 on
   the same rows reversed — "skilled" to "anti-skilled". On the live record, 200 row orders move it
   by 0.031, more than the entire gap between the desk's gate and a random one (0.024). fd-shifts
   and ARGUS do not move. **torch-uncertainty lost.**
3. *Is the loss priced?* Selective-classification residuals live in ``[0, 1]``. fd-shifts' working
   point at a 50% target risk on the live record — act on leans at confidence >= 0.68 — meets its
   target exactly, and would have lost **3,344bps** over its 42 calls. Only a curve in basis points,
   fee included, sees that. **ARGUS won.**
4. *Degenerate cases.* fd-shifts' ``get_working_point`` raises ``ValueError: attempt to get argmax
   of an empty sequence`` when no point meets the target (`rc_stats.py:610-612`) — on this record,
   under the abstention-doubt ranking. ARGUS returns ``None``. **ARGUS won.**
5. *Honest uncertainty.* fd-shifts' interval resamples calls independently (`rc_stats.py:538-574`).
   Refusals decided on one day share one market move: the intraclass correlation of the loss
   indicator within a day is 0.121, a design effect of 5.6, so 470 calls carry roughly the
   information of 84. Its 95% AURC interval is 0.132 wide; resampling whole days gives 0.249.
   **ARGUS won** on the property, not on a tighter number — the wider interval is the honest one.
6. *Graded against the right trade.* Running ARGUS's own production path on the same rows found
   that `eval/scorecard.py` graded refusals against ``side`` — a field `eval/shadow.py` had already
   measured as uninformative — rather than the lean. 84 of 470 leans were graded against the
   opposite direction and 167 with no lean were graded as the side the schema was filled with
   (164 BUY, 3 SELL), and the headline flipped:
   **-2,824bps** ("standing aside cost money") where the withheld trades say **+2,381bps**. Fixed
   in `eval/scorecard.py:lean_side`. Neither sign is distinguishable from zero on twelve days
   (day-clustered interval roughly -30,800 to +28,600bps), which is stated rather than hidden.

**Then adapted and re-run.** fd-shifts is Apache-2.0, so its RC traversal, AUC, optimal references
and working-point rule were adapted into `eval/abstention_coverage.py` with ``file:line``
attribution. Re-run on the identical 470 calls it reproduces every fd-shifts quantity to within
1e-14 (1.1e-16 on the live record; 3.4e-15 at worst, on the planted inverted arm — summation
order), and adds the value curve, the day-clustered interval, a chronological out-of-sample check on
the chosen threshold and the None-not-crash working point. The result on the desk: no detectable
ranking skill in either confidence (the day-clustered interval on ``AURC_random - AURC`` contains
zero under both rankings), and abstaining on every call was the best threshold in hindsight.

**Scope**: see ``SCOPE_STATEMENT``.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.eval.abstention_coverage import (
    Extraction,
    LeanCall,
    augrc,
    augrc_optimal,
    aurc,
    aurc_optimal,
    calls_from_entries,
    rc_points,
    score,
    working_point,
)
from argus.eval.observatory import AbstentionOutcome, abstention_quality

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "general_abstention_comparison.json"
LEDGER_PATH = DATA / "paper_ledger.jsonl"

FROZEN_UPTO_SEQ = 684
FROZEN_SETTLED_BY = "2026-09-25T13:47:40.945561+00:00"
FROZEN_LEDGER_HEAD = "5d144cbef0af9faf"
"""The ledger head at freeze. The ledger only grows and a settlement is written once and sealed,
so ``(FROZEN_UPTO_SEQ, FROZEN_SETTLED_BY)`` selects the same rows for as long as the chain is
intact."""

FROZEN_COUNTS = {"lean_calls": 470, "no_lean": 167, "settled": 637}

RIVAL_PROVENANCE: dict[str, Any] = {
    "fd_shifts": {
        "repo": "https://github.com/IML-DKFZ/fd-shifts",
        "commit": "c4467aec134e99691359da209f811d91283fc1e3",
        "licence": "Apache-2.0",
        "files_run_unmodified": {
            "fd_shifts/analysis/rc_stats.py": "sha256[:16]=4633dae55f54cb61",
            "fd_shifts/analysis/rc_stats_utils.py": "sha256[:16]=de9cf04030f6bb23",
        },
        "how": "the two files loaded as a package straight from the clone (their package __init__ "
               "imports loguru/omegaconf/faiss for unrelated experiment code); "
               "RiskCoverageStats(confids, residuals) with its own aurc/augrc/aurc_optimal/"
               "augrc_optimal/eaurc/get_working_point/evaluate_ci (np.random.seed(0), n_bs=2000)",
    },
    "torch_uncertainty": {
        "repo": "https://github.com/torch-uncertainty/torch-uncertainty",
        "commit": "3f82fe5d15a7bf877a821731baadef1e4731c31d",
        "licence": "Apache-2.0",
        "files_run_unmodified": {
            "src/torch_uncertainty/metrics/classification/risk_coverage.py":
                "sha256[:16]=cf4ea12256328d8f",
        },
        "how": "the metric file loaded straight from the clone; AURC, AUGRC, RiskAtxCov(0.5), "
               "CovAtxRisk(0.5) fed probs [1-p, p] with p = 0.5 + confidence/2 (monotone, so the "
               "ranking is the confidence's) and target = predicted class when the lean netted > 0",
    },
    "runner": "src/argus/eval/baselines/selective_rivals_runner.py (imports nothing from argus; "
              "fd-shifts needs numpy < 2 for np.infty and np.trapz, so the rivals run in their "
              "own environment)",
    "command": "python -m argus.eval.general_abstention_comparison --write-rival-input "
               "frozen.json; python src/argus/eval/baselines/selective_rivals_runner.py "
               "--fd-shifts <clone> --torch-uncertainty <clone> --input frozen.json "
               "--out rival_results.json",
    "runs": [
        {"date": "2026-09-25", "environment": "python 3.11, numpy 1.26.4, torch 2.14.0+cpu, "
                                              "torchmetrics 1.9.0, scikit-learn 1.6.1",
         "note": "first run; its planted 'random' arm used numpy's shuffle, a different "
                 "permutation from ARGUS's"},
        {"date": "2026-09-26", "environment": "python 3.11.9, numpy 1.26.4, torch 2.2.2, "
                                              "torchmetrics 1.9.0",
         "note": "independent re-run through the runner above, 43s wall: every recorded number "
                 "reproduced exactly (difference 0.0) except the planted 'random' arm, which "
                 "now shares ARGUS's random.Random(7) permutation and is recorded from this run"},
    ],
}

RIVAL_RECORDED: dict[str, Any] = {
    "live": {
        "lean_confidence": {
            "fd_shifts": {
                "aurc": 0.5559536683764247,
                "augrc": 0.27413535536441835,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": 0.3793645888814956,
                "n_rc_points": 19,
                "working_point": {
                    "coverage": 0.08936170212765958,
                    "risk": 0.5,
                    "threshold": 0.68
                },
                "aurc_ci95_iid": [0.4924421785712619, 0.6247347764892018]
            },
            "torch_uncertainty": {
                "aurc": 0.5644876472913577,
                "augrc": 0.2775620377942818,
                "risk_at_50cov": 0.5617021276595745,
                "cov_at_50risk": 0.08936170488595963
            }
        },
        "abstention_doubt": {
            "fd_shifts": {
                "aurc": 0.542880484665719,
                "augrc": 0.2617587143503848,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": 0.36629140517078984,
                "n_rc_points": 24,
                "working_point": "error: attempt to get argmax of an empty sequence",
                "aurc_ci95_iid": [0.48106765029844417, 0.602007990671065]
            },
            "torch_uncertainty": {
                "aurc": 0.5390435615466155,
                "augrc": 0.2630200061968887,
                "risk_at_50cov": 0.5276595744680851,
                "cov_at_50risk": 0.6042553186416626
            }
        }
    },
    "row_order_sensitivity": {
        "permutations": 200,
        "fd_shifts_aurc": {
            "min": 0.5559536683764247,
            "max": 0.5559536683764247,
            "spread": 0.0
        },
        "torch_uncertainty_aurc": {
            "min": 0.5424596606719646,
            "max": 0.5733687736479951,
            "spread": 0.030909112976030495
        },
        "torch_uncertainty_augrc": {
            "min": 0.2699677902428817,
            "max": 0.2801252100869095,
            "spread": 0.010157419844027793
        }
    },
    "planted_skill": {
        "oracle": {
            "fd_shifts": {
                "aurc": 0.1765884501474384,
                "augrc": 0.14146672702580354,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": -6.293474907295149e-07,
                "n_rc_points": 471,
                "working_point": {
                    "coverage": 0.9361702127659575,
                    "risk": 0.5,
                    "threshold": 0.10756929637526652
                }
            },
            "torch_uncertainty": {
                "aurc": 0.1769649713498713,
                "augrc": 0.14176836181511995,
                "risk_at_50cov": 0.06382978723404255,
                "cov_at_50risk": 0.936170220375061
            }
        },
        "random": {
            "fd_shifts": {
                "aurc": 0.4888880176382817,
                "augrc": 0.25208239022181983,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": 0.3122989381433526,
                "n_rc_points": 471,
                "working_point": {
                    "coverage": 0.6212765957446809,
                    "risk": 0.5,
                    "threshold": 0.39157782515991474
                }
            },
            "torch_uncertainty": {
                "aurc": 0.4877982269030237,
                "augrc": 0.2526176112300706,
                "risk_at_50cov": 0.4808510638297872,
                "cov_at_50risk": 0.6212766170501709
            }
        },
        "inverted": {
            "fd_shifts": {
                "aurc": 0.8676982621960763,
                "augrc": 0.3904481665912177,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": 0.6911091827011471,
                "n_rc_points": 471,
                "working_point": "error: attempt to get argmax of an empty sequence"
            },
            "torch_uncertainty": {
                "aurc": 0.8674161690090418,
                "augrc": 0.39127841040312183,
                "risk_at_50cov": 1.0,
                "cov_at_50risk": None
            }
        },
        "constant": {
            "fd_shifts": {
                "aurc": 0.5319148936170213,
                "augrc": 0.26595744680851063,
                "aurc_optimal": 0.17658907949492914,
                "augrc_optimal": 0.1414667270258035,
                "eaurc": 0.3553258141220921,
                "n_rc_points": 2,
                "working_point": "error: attempt to get argmax of an empty sequence"
            },
            "torch_uncertainty": {
                "aurc": 0.5334052848485897,
                "augrc": 0.2660890075993253,
                "risk_at_50cov": 0.5148936170212766,
                "cov_at_50risk": 0.47659575939178467
            }
        }
    },
    "fixtures": {
        "ties": {
            "confids": [0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5],
            "residuals": [0, 1, 0, 0, 1, 1, 0, 1],
            "fd_shifts": {
                "aurc": 0.26339285714285715,
                "augrc": 0.1796875,
                "aurc_optimal": 0.15342640972002758,
                "augrc_optimal": 0.125,
                "eaurc": 0.10996644742282959,
                "n_rc_points": 6,
                "working_point": {
                    "coverage": 1.0,
                    "risk": 0.5,
                    "threshold": 0.5
                }
            },
            "torch_uncertainty": {
                "aurc": 0.3802721088435374,
                "augrc": 0.23214285714285715,
                "risk_at_50cov": 0.25,
                "cov_at_50risk": 1.0
            },
            "torch_uncertainty_reversed_rows": {
                "aurc": 0.25646258503401365,
                "augrc": 0.17857142857142858,
                "risk_at_50cov": 0.25,
                "cov_at_50risk": 1.0
            }
        },
        "all_tied": {
            "confids": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            "residuals": [1, 0, 1, 1, 0, 0],
            "fd_shifts": {
                "aurc": 0.5,
                "augrc": 0.25,
                "aurc_optimal": 0.15342640972002758,
                "augrc_optimal": 0.125,
                "eaurc": 0.3465735902799725,
                "n_rc_points": 2,
                "working_point": {
                    "coverage": 1.0,
                    "risk": 0.5,
                    "threshold": 0.6
                }
            },
            "torch_uncertainty": {
                "aurc": 0.6533333286643027,
                "augrc": 0.3666666647791862,
                "risk_at_50cov": 0.6666666666666666,
                "cov_at_50risk": 1.0
            },
            "torch_uncertainty_reversed_rows": {
                "aurc": 0.2966666638851166,
                "augrc": 0.2166666660706202,
                "risk_at_50cov": 0.3333333333333333,
                "cov_at_50risk": 1.0
            }
        },
        "perfect": {
            "confids": [0.9, 0.8, 0.7, 0.6],
            "residuals": [0, 0, 1, 1],
            "fd_shifts": {
                "aurc": 0.14583333333333331,
                "augrc": 0.125,
                "aurc_optimal": 0.15342640972002758,
                "augrc_optimal": 0.125,
                "eaurc": -0.007593076386694264,
                "n_rc_points": 5,
                "working_point": {
                    "coverage": 1.0,
                    "risk": 0.5,
                    "threshold": 0.6
                }
            },
            "torch_uncertainty": {
                "aurc": 0.19444444444444442,
                "augrc": 0.16666666666666666,
                "risk_at_50cov": 0.0,
                "cov_at_50risk": 1.0
            },
            "torch_uncertainty_reversed_rows": {
                "aurc": 0.19444444444444442,
                "augrc": 0.16666666666666666,
                "risk_at_50cov": 0.0,
                "cov_at_50risk": 1.0
            }
        },
        "inverted": {
            "confids": [0.9, 0.8, 0.7, 0.6],
            "residuals": [1, 1, 0, 0],
            "fd_shifts": {
                "aurc": 0.8541666666666666,
                "augrc": 0.375,
                "aurc_optimal": 0.15342640972002758,
                "augrc_optimal": 0.125,
                "eaurc": 0.7007402569466391,
                "n_rc_points": 5,
                "working_point": {
                    "coverage": 1.0,
                    "risk": 0.5,
                    "threshold": 0.6
                }
            },
            "torch_uncertainty": {
                "aurc": 0.8055555555555555,
                "augrc": 0.4583333333333333,
                "risk_at_50cov": 1.0,
                "cov_at_50risk": 1.0
            },
            "torch_uncertainty_reversed_rows": {
                "aurc": 0.8055555555555555,
                "augrc": 0.4583333333333333,
                "risk_at_50cov": 1.0,
                "cov_at_50risk": 1.0
            }
        }
    }
}
"""Every rival number, exactly as the run printed it (NaN recorded as None)."""


# =============================================================================================
# The frozen input.
# =============================================================================================


def frozen_extraction(ledger_path: Path = LEDGER_PATH) -> Extraction:
    from argus.paper.ledger import PaperLedger

    ledger = PaperLedger(path=ledger_path)
    return calls_from_entries(
        ledger.entries, upto_seq=FROZEN_UPTO_SEQ, settled_by=FROZEN_SETTLED_BY,
    )


def rival_input(extraction: Extraction) -> dict[str, Any]:
    """The frozen calls as `eval/baselines/selective_rivals_runner.py` reads them.

    Only what a rival needs: the two confidences, the loss indicator and the net. Written to a file
    by ``--write-rival-input`` so the rivals run in their own environment on exactly these rows.
    """
    return {
        "upto_seq": FROZEN_UPTO_SEQ, "settled_by": FROZEN_SETTLED_BY,
        "ledger_head_at_freeze": FROZEN_LEDGER_HEAD,
        "calls": [{"seq": c.seq, "confidence": c.confidence,
                   "stated_confidence": c.stated_confidence, "error": c.error,
                   "net_bps": c.net_bps} for c in extraction.calls],
    }


def frozen_settled_entries(ledger_path: Path = LEDGER_PATH) -> list[Any]:
    """The 637 settled refusals, in ledger order — the production scorecard's view of the input."""
    from datetime import datetime

    from argus.paper.ledger import PaperLedger

    cutoff = datetime.fromisoformat(FROZEN_SETTLED_BY)
    return [
        e for e in PaperLedger(path=ledger_path).entries
        if e.is_abstention and e.seq <= FROZEN_UPTO_SEQ and e.counterfactual_move_bps is not None
        and e.settled_at is not None and datetime.fromisoformat(e.settled_at) <= cutoff
    ]


# =============================================================================================
# 1 — Does the score depend on which calls were refused?
# =============================================================================================


def planted_confidences(calls: tuple[LeanCall, ...]) -> dict[str, list[float]]:
    """The real outcomes, confidences re-assigned — the rival runner's own construction.

    ``oracle`` ranks the best trades most confident; ``random`` is the same values shuffled by
    ``random.Random(7)``; ``inverted`` the reverse of oracle; ``constant`` is a gate with no
    information. `eval/baselines/selective_rivals_runner.py:planted` builds the identical four
    arms, so every arm is the same input on both sides (pinned by a test).
    """
    n = len(calls)
    rank = sorted(range(n), key=lambda i: calls[i].net_bps)
    oracle = [0.0] * n
    for r, i in enumerate(rank):
        oracle[i] = 0.05 + 0.9 * r / (n - 1)
    shuffled = list(oracle)
    random.Random(7).shuffle(shuffled)
    return {"oracle": oracle, "random": shuffled, "inverted": [1.0 - x for x in oracle],
            "constant": [0.6] * n}


def _quality_of(calls: tuple[LeanCall, ...]) -> dict[str, Any]:
    """`abstention_quality` on the withheld trades, exactly as the fixed scorecard feeds it."""
    return abstention_quality([
        AbstentionOutcome(
            decision_id=str(c.seq), counterfactual_move_bps=_dec(c.move_bps),
            intended_side="BUY" if c.lean == "up" else "SELL",
        )
        for c in calls
    ])


def _dec(x: float) -> Any:
    from decimal import Decimal

    return Decimal(repr(x))


def planted_skill(calls: tuple[LeanCall, ...]) -> dict[str, Any]:
    errors = [c.error for c in calls]
    nets = [c.net_bps for c in calls]
    arms: dict[str, dict[str, Any]] = {}
    quality = _quality_of(calls)
    for label, conf in planted_confidences(calls).items():
        pts = rc_points(conf, errors, nets)
        arms[label] = {
            "argus_abstention_quality": quality,
            "argus_coverage_aurc": aurc(pts),
            "fd_shifts_aurc": RIVAL_RECORDED["planted_skill"][label]["fd_shifts"]["aurc"],
            "torch_uncertainty_aurc":
                RIVAL_RECORDED["planted_skill"][label]["torch_uncertainty"]["aurc"],
        }
    err_rate = sum(errors) / len(errors)

    def orders(key: str) -> bool:
        return bool(arms["oracle"][key] < arms["random"][key] < arms["inverted"][key])

    return {
        "arms": arms,
        "aurc_random_reference": err_rate,
        "abstention_quality_is_identical_across_arms":
            len({repr(a["argus_abstention_quality"]) for a in arms.values()}) == 1,
        "coverage_orders_oracle_random_inverted": orders("argus_coverage_aurc"),
        "fd_shifts_orders_oracle_random_inverted": orders("fd_shifts_aurc"),
        "torch_uncertainty_orders_oracle_random_inverted": orders("torch_uncertainty_aurc"),
        "max_abs_diff_argus_vs_fd_shifts": max(
            abs(a["argus_coverage_aurc"] - a["fd_shifts_aurc"]) for a in arms.values()),
    }


# =============================================================================================
# 2 — Row order.
# =============================================================================================


def row_order_sensitivity(
    calls: tuple[LeanCall, ...], *, permutations: int = 200,
) -> dict[str, Any]:
    conf = [c.confidence for c in calls]
    err = [c.error for c in calls]
    rng = random.Random(20260925)
    idx = list(range(len(calls)))
    values = []
    for _ in range(permutations):
        rng.shuffle(idx)
        values.append(aurc(rc_points([conf[i] for i in idx], [err[i] for i in idx])))
    rival = RIVAL_RECORDED["row_order_sensitivity"]
    gap = abs(RIVAL_RECORDED["live"]["lean_confidence"]["fd_shifts"]["aurc"] - sum(err) / len(err))
    return {
        "permutations": permutations,
        "argus_aurc_spread": max(values) - min(values),
        "fd_shifts_aurc_spread": rival["fd_shifts_aurc"]["spread"],
        "torch_uncertainty_aurc_spread": rival["torch_uncertainty_aurc"]["spread"],
        "torch_uncertainty_augrc_spread": rival["torch_uncertainty_augrc"]["spread"],
        "gate_vs_random_gap": gap,
        "torch_uncertainty_spread_exceeds_the_effect":
            rival["torch_uncertainty_aurc"]["spread"] > gap,
        "all_tied_fixture": {
            "torch_uncertainty_aurc_rows_as_given":
                RIVAL_RECORDED["fixtures"]["all_tied"]["torch_uncertainty"]["aurc"],
            "torch_uncertainty_aurc_rows_reversed":
                RIVAL_RECORDED["fixtures"]["all_tied"]["torch_uncertainty_reversed_rows"]["aurc"],
            "fd_shifts_aurc": RIVAL_RECORDED["fixtures"]["all_tied"]["fd_shifts"]["aurc"],
            "random_reference": 0.5,
        },
    }


# =============================================================================================
# Parity with fd-shifts, fixtures and live.
# =============================================================================================


def parity() -> dict[str, Any]:
    """ARGUS's adapted implementation against fd-shifts' real output, quantity by quantity."""
    worst = 0.0
    rows = []
    for label, fx in RIVAL_RECORDED["fixtures"].items():
        conf, res = fx["confids"], fx["residuals"]
        pts = rc_points(conf, res)
        err = sum(res) / len(res)
        ours = {"aurc": aurc(pts), "augrc": augrc(pts), "aurc_optimal": aurc_optimal(err),
                "augrc_optimal": augrc_optimal(err), "n_rc_points": len(pts)}
        theirs = fx["fd_shifts"]
        diffs = {k: abs(float(ours[k]) - float(theirs[k])) for k in ours}
        worst = max(worst, *diffs.values())
        wp = working_point(pts, target_risk=0.5)
        rows.append({"fixture": label, "ours": ours, "fd_shifts": theirs, "abs_diff": diffs,
                     "working_point_matches": wp is not None
                     and abs(wp.coverage - theirs["working_point"]["coverage"]) < 1e-12
                     and abs(wp.threshold - theirs["working_point"]["threshold"]) < 1e-12})
    return {"fixtures": rows, "max_abs_diff": worst}


def live_parity(report: dict[str, Any], ranking: str) -> dict[str, Any]:
    theirs = RIVAL_RECORDED["live"][ranking]["fd_shifts"]
    keys = ("aurc", "augrc", "aurc_optimal", "augrc_optimal")
    diffs = {k: abs(float(report[k]) - float(theirs[k])) for k in keys}
    diffs["e_aurc"] = abs(float(report["e_aurc"]) - float(theirs["eaurc"]))
    return {"ranking": ranking, "abs_diff": diffs, "max_abs_diff": max(diffs.values()),
            "fd_shifts": theirs, "torch_uncertainty": RIVAL_RECORDED["live"][ranking]
            ["torch_uncertainty"]}


# =============================================================================================
# 3 — Is the loss priced?  4 — degenerate cases.  5 — honest uncertainty.
# =============================================================================================


def priced_working_point(calls: tuple[LeanCall, ...]) -> dict[str, Any]:
    """What fd-shifts' own working point at 50% target risk would actually have earned."""
    wp = RIVAL_RECORDED["live"]["lean_confidence"]["fd_shifts"]["working_point"]
    acted = [c for c in calls if c.confidence >= wp["threshold"]]
    total = sum(c.net_bps for c in acted)
    return {
        "fd_shifts_working_point": wp,
        "acted": len(acted),
        "loss_rate": sum(c.error for c in acted) / len(acted),
        "total_net_bps": total,
        "mean_net_bps": total / len(acted),
        "meets_its_risk_target_and_loses_money": wp["risk"] <= 0.5 and total < 0,
    }


def degenerate_cases(calls: tuple[LeanCall, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    doubt = [1.0 - c.stated_confidence for c in calls]
    errors = [c.error for c in calls]
    cases = {
        "abstention_doubt_live": (doubt, errors,
                                  RIVAL_RECORDED["live"]["abstention_doubt"]["fd_shifts"]),
        "inverted_planted": (planted_confidences(calls)["inverted"], errors,
                             RIVAL_RECORDED["planted_skill"]["inverted"]["fd_shifts"]),
        "constant_planted": ([0.6] * len(calls), errors,
                             RIVAL_RECORDED["planted_skill"]["constant"]["fd_shifts"]),
    }
    for label, (conf, err, fd) in cases.items():
        wp = working_point(rc_points(conf, err), target_risk=0.5)
        out[label] = {
            "fd_shifts_get_working_point": fd["working_point"],
            "argus_working_point": None if wp is None else wp.as_dict(),
            "fd_shifts_raised": isinstance(fd["working_point"], str),
        }
    return out


def uncertainty(calls: tuple[LeanCall, ...], report: dict[str, Any]) -> dict[str, Any]:
    from collections import defaultdict

    from argus.eval.forecasts import design_effect, intraclass_correlation

    by_day: dict[str, list[float]] = defaultdict(list)
    for c in calls:
        by_day[c.cluster].append(float(c.error))
    icc = intraclass_correlation(list(by_day.values()))
    deff = design_effect(icc, len(calls) / len(by_day))
    iid = RIVAL_RECORDED["live"]["lean_confidence"]["fd_shifts"]["aurc_ci95_iid"]
    ours = report["aurc_ci95_by_day"]
    return {
        "days": len(by_day),
        "icc_of_loss_within_day": icc,
        "design_effect": deff,
        "effective_calls": len(calls) / deff,
        "fd_shifts_aurc_ci95_iid": iid,
        "fd_shifts_iid_width": iid[1] - iid[0],
        "argus_aurc_ci95_by_day": ours,
        "argus_by_day_width": ours[1] - ours[0],
    }


# =============================================================================================
# 6 — Graded against the right trade.
# =============================================================================================


def direction_fidelity(entries: list[Any]) -> dict[str, Any]:
    """ARGUS's production path before and after the `eval/scorecard.py:lean_side` fix."""
    from decimal import Decimal

    from argus.eval.scorecard import lean_side

    by_side = [
        AbstentionOutcome(decision_id=str(e.seq),
                          counterfactual_move_bps=Decimal(e.counterfactual_move_bps),
                          intended_side=e.side)
        for e in entries
    ]
    by_lean = [
        AbstentionOutcome(decision_id=str(e.seq),
                          counterfactual_move_bps=Decimal(e.counterfactual_move_bps),
                          intended_side=side)
        for e in entries if (side := lean_side(e)) is not None
    ]
    opposite = sum(1 for e in entries
                   if (side := lean_side(e)) is not None and side != e.side.upper())
    before = abstention_quality(by_side)
    after = abstention_quality(by_lean)
    return {
        "graded_before": len(by_side),
        "graded_after": len(by_lean),
        "graded_against_the_opposite_of_the_lean_before": opposite,
        "no_lean_graded_as_a_side_before": len(by_side) - len(by_lean),
        "before_side_based": before,
        "after_lean_based": after,
        "sign_flipped":
            (float(before["total_value_bps"]) < 0) != (float(after["total_value_bps"]) < 0),
    }


# =============================================================================================
# Costs, criteria, scope.
# =============================================================================================


def measure_costs(extraction: Extraction) -> dict[str, Any]:
    calls = extraction.calls
    start = time.perf_counter()
    for _ in range(20):
        _quality_of(calls)
    quality_ms = (time.perf_counter() - start) / 20 * 1000
    start = time.perf_counter()
    for _ in range(20):
        aurc(rc_points([c.confidence for c in calls], [c.error for c in calls],
                       [c.net_bps for c in calls]))
    curve_ms = (time.perf_counter() - start) / 20 * 1000
    start = time.perf_counter()
    score(extraction)
    full_ms = (time.perf_counter() - start) * 1000
    return {"abstention_quality_ms": quality_ms, "argus_rc_curve_ms": curve_ms,
            "argus_full_report_with_day_bootstrap_ms": full_ms,
            "note_rival_costs": "the rival run took 2m10s wall for everything in RIVAL_RECORDED, "
                                "dominated by fd-shifts' 2,000-draw iid bootstrap and 200 "
                                "permutations x 2 libraries; not comparable per call"}


def criteria(result: dict[str, Any]) -> list[dict[str, str]]:
    ps = result["planted_skill"]
    ro = result["row_order"]
    pw = result["priced_working_point"]
    un = result["uncertainty"]
    df = result["direction_fidelity"]
    dg = result["degenerate_cases"]
    return [
        {"property": "the score depends on which calls were refused (planted skill)",
         "argus_before": "no: abstention_quality identical across oracle/random/inverted"
         if ps["abstention_quality_is_identical_across_arms"] else "yes",
         "fd_shifts": "yes" if ps["fd_shifts_orders_oracle_random_inverted"] else "no",
         "torch_uncertainty":
             "yes" if ps["torch_uncertainty_orders_oracle_random_inverted"] else "no",
         "argus_after": "yes" if ps["coverage_orders_oracle_random_inverted"] else "no",
         "winner": "fd-shifts over ARGUS-before; absorbed"},
        {"property": "invariant to row order under tied confidences",
         "argus_before": "yes (a sum)", "fd_shifts": f"yes (spread {ro['fd_shifts_aurc_spread']})",
         "torch_uncertainty": f"no (AURC spread {ro['torch_uncertainty_aurc_spread']:.4f} > "
                              f"gate-vs-random gap {ro['gate_vs_random_gap']:.4f})",
         "argus_after": f"yes (spread {ro['argus_aurc_spread']})",
         "winner": "fd-shifts = ARGUS; torch-uncertainty loses"},
        {"property": "the loss is priced in bps, fee included",
         "argus_before": "yes (value only)", "fd_shifts": "no: its 50%-risk working point "
         f"would have lost {-pw['total_net_bps']:,.0f}bps over {pw['acted']} calls",
         "torch_uncertainty": "no", "argus_after": "yes (value curve beside the risk curve)",
         "winner": "ARGUS"},
        {"property": "no crash when no threshold meets the target risk",
         "argus_before": "n/a", "fd_shifts": "raises ValueError on "
         f"{sum(1 for v in dg.values() if v['fd_shifts_raised'])} of {len(dg)} cases",
         "torch_uncertainty": "returns NaN (documented)", "argus_after": "returns None",
         "winner": "ARGUS"},
        {"property": "interval respects same-day dependence",
         "argus_before": "no interval", "fd_shifts": f"iid, width {un['fd_shifts_iid_width']:.3f}",
         "torch_uncertainty": "no interval",
         "argus_after": f"by day, width {un['argus_by_day_width']:.3f} "
                        f"(ICC {un['icc_of_loss_within_day']:.3f}, design effect "
                        f"{un['design_effect']:.1f})",
         "winner": "ARGUS"},
        {"property": "graded against the trade actually withheld (the lean)",
         "argus_before": f"no: {df['graded_against_the_opposite_of_the_lean_before']} opposite, "
                         f"{df['no_lean_graded_as_a_side_before']} no-lean graded as a side; "
                         f"total {df['before_side_based']['total_value_bps']}bps",
         "fd_shifts": "n/a (the encoding is the caller's)", "torch_uncertainty": "n/a",
         "argus_after": f"yes: total {df['after_lean_based']['total_value_bps']}bps",
         "winner": "found by this comparison, fixed in eval/scorecard.py"},
    ]


SCOPE_STATEMENT = """\
Claimed: on the same frozen record (470 leans from 637 settled refusals), ARGUS's pre-existing \
abstention score (`observatory.abstention_quality`) is blind to which calls were refused — run on \
a perfect, a random and an inverted gate over identical outcomes it returns identical output — \
while the general-purpose selective-prediction evaluator fd-shifts ranks them correctly. That is \
a loss, and it is published. fd-shifts' method (Apache-2.0) was adapted into \
`eval/abstention_coverage.py`, which reproduces fd-shifts' AURC, AUGRC, optimal references, \
e-AURC and working point on the identical input to within 1e-14, and then differs where this \
record needs it to: a value curve in bps (fd-shifts' own 50%-risk working point would have lost \
3,344bps), a day-clustered interval (ICC 0.121, design effect 5.6), no crash where fd-shifts \
raises, and an out-of-sample check on the chosen threshold. torch-uncertainty's AURC depends on \
row order under ties by more than the effect being measured. Running ARGUS's own production path \
on the same rows exposed that the scorecard graded refusals against `side` instead of the lean, \
flipping the sign of the headline; that is fixed.

NOT claimed: that the desk's refusals were right. Under both confidence rankings the day-clustered \
interval on AURC_random - AURC contains zero — no detectable ranking skill — and the abstention \
value's interval spans roughly -30,800 to +28,600bps on twelve days: the sign of what abstaining \
was worth is not established. NOT claimed that fd-shifts is a weak tool: it is the stronger of the \
two rivals and the one ARGUS copied; its iid interval and [0,1] residuals are the right defaults \
for image classification, the setting it was built for, and the wrong ones for a record of \
same-day refusals priced in basis points. NOT claimed that ARGUS beats fd-shifts on selective \
classification in general — only on this input, on the three properties listed (priced loss, no \
crash, day-clustered interval); on ranking and row-order invariance the two tie.
"""


def run(ledger_path: Path = LEDGER_PATH) -> dict[str, Any]:
    extraction = frozen_extraction(ledger_path)
    calls = extraction.calls
    lean = score(extraction, ranking="lean_confidence")
    doubt = score(extraction, ranking="abstention_doubt")
    result: dict[str, Any] = {
        "input": {
            "ledger": "data/paper_ledger.jsonl", "upto_seq": FROZEN_UPTO_SEQ,
            "settled_by": FROZEN_SETTLED_BY, "ledger_head_at_freeze": FROZEN_LEDGER_HEAD,
            "lean_calls": len(calls), "no_lean": extraction.no_lean,
            "matches_frozen_counts": len(calls) == FROZEN_COUNTS["lean_calls"]
            and extraction.no_lean == FROZEN_COUNTS["no_lean"],
        },
        "rival_provenance": RIVAL_PROVENANCE,
        "argus_adapted": {"lean_confidence": lean, "abstention_doubt": doubt},
        "parity_fixtures": parity(),
        "parity_live": [live_parity(lean, "lean_confidence"),
                        live_parity(doubt, "abstention_doubt")],
        "planted_skill": planted_skill(calls),
        "row_order": row_order_sensitivity(calls),
        "priced_working_point": priced_working_point(calls),
        "degenerate_cases": degenerate_cases(calls),
        "uncertainty": uncertainty(calls, lean),
        "direction_fidelity": direction_fidelity(frozen_settled_entries(ledger_path)),
        "costs": measure_costs(extraction),
        "rival_recorded": RIVAL_RECORDED,
    }
    result["criteria"] = criteria(result)
    result["verdict"] = {
        "before_adaptation": "rival_wins: fd-shifts scores whether the gate ranks; "
                             "abstention_quality cannot",
        "after_adaptation": "ARGUS reproduces fd-shifts on every shared quantity (a tie on "
                            "ranking and row-order invariance) and adds three properties this "
                            "record needs that fd-shifts lacks: the loss priced in bps, no crash "
                            "where no threshold meets the target, and a day-clustered interval; "
                            "separately, the comparison exposed and fixed a grading defect in "
                            "ARGUS's own scorecard (side instead of lean)",
        "desk_finding": f"{lean['gate_skill_verdict']}; {lean['abstention_verdict']}",
    }
    result["scope_statement"] = SCOPE_STATEMENT
    return result


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ARGUS abstention scoring vs fd-shifts and "
                                                 "torch-uncertainty on the frozen record")
    parser.add_argument("--write-rival-input", type=Path, default=None, metavar="PATH",
                        help="write the frozen calls for selective_rivals_runner.py and stop")
    args = parser.parse_args(argv)
    if args.write_rival_input is not None:
        blob = rival_input(frozen_extraction())
        args.write_rival_input.write_text(json.dumps(blob, allow_nan=False), encoding="utf-8")
        print(f"{len(blob['calls'])} calls -> {args.write_rival_input}")
        return 0
    result = run()
    undefined = artefact.write(REPORT_PATH, result)
    for row in result["criteria"]:
        print(f"- {row['property']}: {row['winner']}")
    print(f"\nparity with fd-shifts: fixtures max |diff| "
          f"{result['parity_fixtures']['max_abs_diff']:.2e}, live "
          f"{max(p['max_abs_diff'] for p in result['parity_live']):.2e}")
    print(f"desk: {result['verdict']['desk_finding']}")
    print(f"\nsaved -> {REPORT_PATH}" + (f" (null: {undefined})" if undefined else ""))
    return 0


__all__ = [
    "FROZEN_COUNTS",
    "FROZEN_LEDGER_HEAD",
    "FROZEN_SETTLED_BY",
    "FROZEN_UPTO_SEQ",
    "RIVAL_PROVENANCE",
    "RIVAL_RECORDED",
    "SCOPE_STATEMENT",
    "criteria",
    "degenerate_cases",
    "direction_fidelity",
    "frozen_extraction",
    "frozen_settled_entries",
    "live_parity",
    "main",
    "measure_costs",
    "parity",
    "planted_confidences",
    "planted_skill",
    "priced_working_point",
    "rival_input",
    "row_order_sensitivity",
    "run",
    "uncertainty",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
