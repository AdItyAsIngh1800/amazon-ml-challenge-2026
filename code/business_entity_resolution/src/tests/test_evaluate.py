"""Tests for src.evaluate: official per-S1 F0.5 rules, breakdowns, recall report."""

from __future__ import annotations

import pandas as pd
import pytest

from src import evaluate


# --- f05_single: one test per official rule -------------------------------

def test_readme_example_is_0714() -> None:
    """README: pred 3 IDs, truth 2 of them -> P=2/3, R=1 -> 0.714."""
    score = evaluate.f05_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert round(score, 3) == 0.714


def test_empty_truth_empty_pred_is_one() -> None:
    """Correct singleton."""
    assert evaluate.f05_single(set(), set()) == 1.0


def test_empty_truth_nonempty_pred_is_zero() -> None:
    """False merge on a singleton."""
    assert evaluate.f05_single({"S2-1"}, set()) == 0.0


def test_nonempty_truth_empty_pred_is_zero() -> None:
    """Missed a matched S1 entirely."""
    assert evaluate.f05_single(set(), {"S2-1"}) == 0.0


def test_no_overlap_is_zero() -> None:
    """P = R = 0 -> 0, not a division error."""
    assert evaluate.f05_single({"S2-2"}, {"S2-1"}) == 0.0


def test_formula_general_case() -> None:
    """P = 1/2, R = 1/3 -> 1.25*P*R / (0.25*P + R)."""
    p, r = 1 / 2, 1 / 3
    expected = 1.25 * p * r / (0.25 * p + r)
    got = evaluate.f05_single({"a", "x"}, {"a", "b", "c"})
    assert got == pytest.approx(expected)


def test_perfect_match_is_one() -> None:
    """Exact set -> 1.0."""
    assert evaluate.f05_single({"a", "b"}, {"a", "b"}) == 1.0


# --- macro_f05 ---------------------------------------------------------------

def test_macro_f05_averages_over_all_s1_and_defaults_missing_pred_to_empty() -> None:
    """S1-3 is absent from pred -> treated as empty (correct singleton)."""
    truth = {"S1-1": {"S2-47", "S3-812"}, "S1-2": {"S2-1"}, "S1-3": set()}
    pred = {"S1-1": {"S2-47", "S2-193", "S3-812"}, "S1-2": set()}
    score = evaluate.macro_f05(pred, truth, ["S1-1", "S1-2", "S1-3"])
    assert score == pytest.approx((0.7142857 + 0.0 + 1.0) / 3, abs=1e-6)


def test_macro_f05_only_uses_given_s1_ids() -> None:
    """Scoring a subset ignores other S1s."""
    truth = {"S1-1": set(), "S1-2": {"S2-1"}}
    assert evaluate.macro_f05({}, truth, ["S1-1"]) == 1.0


def test_macro_f05_unknown_s1_raises() -> None:
    """An S1 without ground truth is a caller bug."""
    with pytest.raises(ValueError, match="S1-9"):
        evaluate.macro_f05({}, {"S1-1": set()}, ["S1-1", "S1-9"])


# --- breakdown ----------------------------------------------------------------

def test_f05_breakdown_segments() -> None:
    """Overall, singleton, non-singleton and one row per country."""
    s1 = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
                       "country": ["US", "US", "India", "France"]})
    truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1"}, "S1-4": set()}
    pred = {"S1-1": {"S2-1"}, "S1-2": {"S2-9"}, "S1-3": set(), "S1-4": set()}
    got = evaluate.f05_breakdown(pred, truth, s1).set_index("segment")
    assert got.loc["overall", "f05"] == pytest.approx(0.5)
    assert got.loc["singleton", "f05"] == pytest.approx(0.5)
    assert got.loc["non_singleton", "f05"] == pytest.approx(0.5)
    assert got.loc["country=US", "f05"] == pytest.approx(0.5)
    assert got.loc["country=India", "f05"] == 0.0
    assert got.loc["country=France", "f05"] == 1.0
    assert got.loc["overall", "n_s1"] == 4 and got.loc["singleton", "n_s1"] == 2


def test_f05_breakdown_missing_column_raises() -> None:
    """S1 frame without 'country' is rejected at the boundary."""
    with pytest.raises(ValueError, match="country"):
        evaluate.f05_breakdown({}, {"S1-1": set()}, pd.DataFrame({"entity_id": ["S1-1"]}))


# --- blocking recall ------------------------------------------------------------

def test_blocking_recall_report() -> None:
    """3 true pairs, 2 found; S1-2 fully covered, S1-3 singleton counts as covered."""
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": set()}
    cands = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-9"],
        "cand_id": ["S2-1", "S2-7", "S2-2", "S3-5", "S2-8", "S2-1"],
    })
    rep = evaluate.blocking_recall_report(cands, truth, ["S1-1", "S1-2", "S1-3"], chunk_rows=2)
    assert rep.n_true_pairs == 3 and rep.n_found_pairs == 2
    assert rep.pair_recall == pytest.approx(2 / 3)
    assert rep.pct_s1_fully_covered == pytest.approx(100 * 2 / 3)
    assert rep.avg_candidates_per_s1 == pytest.approx(5 / 3)  # S1-9 row ignored


def test_blocking_recall_missing_column_raises() -> None:
    """Candidates without cand_id are rejected at the boundary."""
    with pytest.raises(ValueError, match="cand_id"):
        evaluate.blocking_recall_report(pd.DataFrame({"s1_id": []}), {}, [])
