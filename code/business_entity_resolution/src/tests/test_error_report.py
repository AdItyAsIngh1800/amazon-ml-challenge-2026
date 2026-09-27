"""Tests for error_report.py on decide's synthetic train split (hand-computed F0.5 points)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import error_report, io_utils
from src.config import Paths
from src.tests.test_decide import OOF, train_paths  # noqa: F401  (fixture)


def _setup(paths: Paths, t: float, one_owner: bool, extra: list[tuple[str, str, float, int]]) -> None:
    """Write decision_config.json and oof_train.parquet (OOF + ``extra`` rows)."""
    cfg = {"t": t, "t_empty": 0.0, "one_owner": one_owner, "score_column": "prob", "train_f05": 0.0}
    (paths.artifacts_dir / "decision_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    oof = pd.DataFrame(OOF + extra, columns=["s1_id", "cand_id", "prob", "label"]).assign(fold=0)
    io_utils.save_parquet(oof, paths.artifacts_dir / "oof_train.parquet")


def _points(paths: Paths) -> dict[str, float]:
    """F0.5 points lost per bucket (overall table)."""
    errors, _ = error_report.error_pairs(*error_report._decision(paths))
    return {str(k): float(v) for k, v in errors.groupby("bucket", observed=True)["points"].sum().items()}


def test_b2_and_b4_threshold(train_paths: Paths) -> None:  # noqa: F811
    """t=0.75: S1-1 predicts S2-47 + wrong S2-193 (P .5 R .5 -> F .5), misses S3-812 (p .7)."""
    _setup(train_paths, 0.75, False, [])
    pts = _points(train_paths)
    assert set(pts) == {"B2", "B4_threshold"}
    assert pts["B2"] == pytest.approx((1.25 / 1.5 - 0.5) / 3)  # drop S2-193 -> F .8333
    assert pts["B4_threshold"] == pytest.approx((2.5 / 3.5 - 0.5) / 3)  # add S3-812 -> F .7143
    report = error_report.run(train_paths, 5).read_text(encoding="utf-8")
    assert "macro F0.5 0.5000" in report and "== Examples B2 (1 random" in report
    assert "true  S3-812" in report and "rank=2/3" in report


def test_b1_and_b4_one_owner(train_paths: Paths) -> None:  # noqa: F811
    """One-owner hands S3-812 to singleton S1-2 (p .95): B1 on S1-2, B4_one_owner on S1-1."""
    _setup(train_paths, 0.55, True, [("S1-2", "S3-812", 0.95, 0)])
    pts = _points(train_paths)
    assert pts["B1"] == pytest.approx(1 / 3)  # S1-2 empty truth: any prediction scores 0
    assert pts["B4_one_owner"] == pytest.approx((2.5 / 3.5 - 0.5) / 3)
    assert "B4_threshold" not in pts


def test_ignores_unblocked_s1(train_paths: Paths, caplog: pytest.LogCaptureFixture) -> None:  # noqa: F811
    """With a blocking S1 subset, only blocked S1 are scored and the exclusion is logged."""
    _setup(train_paths, 0.55, False, [])
    oof = pd.read_parquet(train_paths.artifacts_dir / "oof_train.parquet")
    io_utils.save_parquet(oof[oof["s1_id"] == "S1-1"], train_paths.artifacts_dir / "oof_train.parquet")
    pd.DataFrame({"s1_id": ["S1-1"]}).to_parquet(io_utils.blocked_s1_path(train_paths.artifacts_dir, "train"))
    with caplog.at_level("INFO"):
        errors, f05 = error_report.error_pairs(*error_report._decision(train_paths))
    assert any("analysing 1 of 3 train S1; 2 unblocked S1 excluded" in r.getMessage() for r in caplog.records)
    assert set(errors["bucket"]) == {"B2"} and set(errors["country"]) == {"US"}
    assert f05 == pytest.approx(2.5 / 3.5)  # S1-1 predicts all three; S1-2 / S1-3 not counted
