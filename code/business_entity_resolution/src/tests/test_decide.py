"""Tests for decide.py on synthetic splits (decide on train, write on test)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, decide, evaluate, io_utils, normalize, run_pipeline
from src.config import Paths

Row = tuple[str, str, str, str]

# S1-1: README example (truth S2-47 + S3-812; S2-193 is wrong).
# S1-2: singleton with one weak candidate. S1-3: its true match S2-5 was never a candidate.
TRAIN = {
    1: [("S1-1", "Acme Corp", "1 Main St", "US"), ("S1-2", "Raj Traders", "MG Road", "India"),
        ("S1-3", "Zen Labs", "2 Elm St", "US")],
    2: [("S2-47", "Acme Corp", "1 Main St", "US"), ("S2-193", "Acme Co", "9 Oak St", "US"),
        ("S2-5", "Zen Labs", "2 Elm St", "US"), ("S2-9", "Raj Trading", "MG Rd", "India")],
    3: [("S3-812", "ACME CORP", "1 Main Street", "US")],
}
TRUTH = {"S1-1": {"S2-47", "S3-812"}, "S1-2": set(), "S1-3": {"S2-5"}}
OOF = [("S1-1", "S2-47", 0.9, 1), ("S1-1", "S2-193", 0.8, 0), ("S1-1", "S3-812", 0.7, 1),
       ("S1-2", "S2-9", 0.6, 0), ("S1-3", "S2-193", 0.3, 0)]


def _make_split(paths: Paths, split: str, rows: dict[int, list[Row]]) -> None:
    """Write source TSVs and build records_{split}.parquet with prep v0."""
    cols = list(io_utils.SOURCE_COLUMNS)
    for s, r in rows.items():
        io_utils.write_source_tsv(pd.DataFrame(r, columns=cols), paths.data_dir / split / f"{split}_source{s}.tsv")
    normalize.run_stage(paths, split)


@pytest.fixture
def train_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Paths:
    """Synthetic train split with records, ground truth and oof_train.parquet."""
    monkeypatch.setattr(config, "SHARED_ARTIFACTS_DIR", tmp_path / "shared")
    paths = Paths(tmp_path / "data", tmp_path / "art", tmp_path / "out")
    _make_split(paths, "train", TRAIN)
    io_utils.write_id_list_tsv(TRUTH, list(TRUTH), paths.data_dir / "train" / "train_ground_truth.tsv",
                               io_utils.GROUND_TRUTH_COLUMNS)
    oof = pd.DataFrame(OOF, columns=["s1_id", "cand_id", "prob", "label"]).assign(fold=0)
    io_utils.save_parquet(oof, paths.artifacts_dir / "oof_train.parquet")
    return paths


def _brute_force_best(one_owner: bool) -> float:
    """Best macro F0.5 over every observed score as t / t_empty (exhaustive), via evaluate.macro_f05."""
    rows = OOF
    if one_owner:
        best_owner: dict[str, tuple[float, str]] = {}
        for s1, c, p, _ in rows:
            if c not in best_owner or p > best_owner[c][0]:
                best_owner[c] = (p, s1)
        rows = [r for r in rows if best_owner[r[1]][1] == r[0]]
    best = -1.0
    grid = sorted({float(np.float32(p)) for _, _, p, _ in OOF})
    for t in grid:
        for te in (0.0, *grid):
            mx = {s1: max(p for s, _, p, _ in rows if s == s1) for s1, *_ in rows}
            pred = {s1: {c for s, c, p, _ in rows if s == s1 and p >= t} if mx[s1] >= te else set() for s1 in mx}
            best = max(best, evaluate.macro_f05(pred, TRUTH, list(TRUTH)))
    return best


def test_readme_example_through_decide(train_paths: Paths) -> None:
    """t=0.5, no gate, no one-owner: S1-1 predicts 3 IDs, 2 correct -> 0.714."""
    ids = decide.load_split_ids(train_paths, "train")
    pairs = decide.load_pairs(train_paths.artifacts_dir / "oof_train.parquet", "prob", ids, with_label=True)
    keep = np.ones(len(pairs.s1), dtype=bool)
    sel = decide.select(pairs, keep, decide.max_score_per_s1(pairs, keep, len(ids.s1)), 0.5, 0.0)
    scores = decide.per_s1_scores(pairs, sel, decide.truth_counts(train_paths.data_dir, ids))
    assert round(float(scores[ids.s1.get_loc("S1-1")]), 3) == 0.714
    assert scores[ids.s1.get_loc("S1-3")] == 0.0  # true match never a candidate


def test_one_owner_keeps_best_s1() -> None:
    """S2-193 is kept only for S1-1 (0.8 beats 0.3); ties go to the lower S1 code."""
    pairs = decide.Pairs(s1=np.array([0, 2, 1, 0], np.int32), cand=np.array([5, 5, 7, 7], np.int32),
                         score=np.array([0.8, 0.3, 0.6, 0.6], np.float32), label=None)
    assert decide.one_owner_mask(pairs, 8).tolist() == [True, False, False, True]


def test_threshold_grid_adapts_to_score_range() -> None:
    """Grid values are observed scores, deduplicated and ascending, for RRF-range and probability scores."""
    rrf = np.array([0.017, 0.05, 0.05, 0.18, 0.033], np.float32)
    grid = decide.threshold_grid(rrf, 200)
    assert grid == tuple(sorted(float(v) for v in set(rrf.tolist())))
    probs = np.linspace(0, 1, 10_001, dtype=np.float32)
    grid = decide.threshold_grid(probs, 200)
    assert len(grid) == 200 and grid[0] == 0.0 and grid[-1] == 1.0 and set(grid) <= set(probs.tolist())
    with pytest.raises(ValueError):
        decide.threshold_grid(np.array([], np.float32), 200)


@pytest.mark.parametrize("one_owner", [True, False])
def test_grid_matches_brute_force(train_paths: Paths, monkeypatch: pytest.MonkeyPatch, one_owner: bool) -> None:
    """Vectorised grid search finds the same best macro F0.5 as scoring every grid point with evaluate.py."""
    monkeypatch.setattr(config, "DECIDE_ONE_OWNER", one_owner)
    cfg = decide.run_decide(train_paths)
    assert cfg.train_f05 == pytest.approx(_brute_force_best(one_owner), abs=1e-6)
    saved = json.loads((train_paths.artifacts_dir / "decision_config.json").read_text(encoding="utf-8"))
    assert saved["one_owner"] is one_owner and set(saved) >= {"t", "t_empty", "one_owner", "score_column"}


def test_decide_via_runner_with_baseline_score(train_paths: Paths) -> None:
    """--decide-score-column score reads a rule-baseline 'score' column; experiment row is logged."""
    base = pd.DataFrame(OOF, columns=["s1_id", "cand_id", "score", "label"])
    io_utils.save_parquet(base, train_paths.artifacts_dir / "oof_train.parquet")
    try:
        run_pipeline.main(["--stage", "decide", "--split", "train", "--data-dir", str(train_paths.data_dir),
                           "--artifacts-dir", str(train_paths.artifacts_dir), "--out-dir", str(train_paths.output_dir),
                           "--decide-score-column", "score"])
    finally:
        config.DECIDE_SCORE_COLUMN = "prob"
    saved = json.loads((train_paths.artifacts_dir / "decision_config.json").read_text(encoding="utf-8"))
    assert saved["score_column"] == "score"
    assert saved["train_f05"] == pytest.approx(_brute_force_best(True), abs=1e-6)
    log = (config.SHARED_ARTIFACTS_DIR / "experiments.tsv").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2 and "\tA\tdecide v1 quantile grid (score)\t" in log[1]


def test_bad_inputs_raise(train_paths: Paths) -> None:
    """Missing score column or unknown IDs are rejected at the boundary."""
    ids = decide.load_split_ids(train_paths, "train")
    with pytest.raises(ValueError, match="score"):
        decide.load_pairs(train_paths.artifacts_dir / "oof_train.parquet", "score", ids, with_label=True)
    bad = pd.DataFrame([("S1-99", "S2-47", 0.5, 1)], columns=["s1_id", "cand_id", "prob", "label"])
    io_utils.save_parquet(bad, train_paths.artifacts_dir / "bad.parquet")
    with pytest.raises(ValueError, match="missing from records"):
        decide.load_pairs(train_paths.artifacts_dir / "bad.parquet", "prob", ids, with_label=True)


TEST = {
    1: [("S1-10", "Acme Corp", "1 Main St", "US"), ("S1-11", "Ecole Lune", "3 Rue X, Lille", "France"),
        ("S1-12", "Raj Traders", "MG Road", "India")],
    2: [("S2-1", "Acme Corp", "1 Main St", "US"), ("S2-2", "Acme Co", "9 Oak", "US")],
    3: [("S3-1", "ACME", "1 Main", "US"), ("S3-2", "Raj", "MG Rd", "India")],
}
PRED = [("S1-10", "S2-1", 0.95), ("S1-10", "S3-1", 0.7), ("S1-10", "S2-2", 0.2),
        ("S1-12", "S3-2", 0.4), ("S1-12", "S2-2", 0.1)]  # S1-11 (France) has no candidates


@pytest.fixture
def test_paths(tmp_path: Path) -> Paths:
    """Synthetic test split with records, pred_test.parquet and a decision config."""
    paths = Paths(tmp_path / "data", tmp_path / "art", tmp_path / "out")
    _make_split(paths, "test", TEST)
    io_utils.save_parquet(pd.DataFrame(PRED, columns=["s1_id", "cand_id", "prob"]),
                          paths.artifacts_dir / "pred_test.parquet")
    cfg = {"t": 0.5, "t_empty": 0.6, "one_owner": True, "score_column": "prob", "train_f05": 0.0}
    (paths.artifacts_dir / "decision_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return paths


def test_write_outputs_and_validator_pass(test_paths: Paths) -> None:
    """One row per test S1 (France incl.), matches subset of candidates, best-first order, validator PASS."""
    decide.run_stage(test_paths, "test")
    m = io_utils.read_ground_truth_pairs(test_paths.output_dir / "matching_results.tsv")
    raw = (test_paths.output_dir / "candidate_pairs.tsv").read_bytes()
    assert raw.startswith(b"source1_entity_id\tcandidate_entity_ids\n") and b"\r" not in raw
    assert raw.decode("utf-8").splitlines()[1:] == ["S1-10\tS2-1,S3-1,S2-2", "S1-11\t", "S1-12\tS3-2,S2-2"]
    # t=0.5 keeps S2-1 and S3-1; S1-12's max 0.4 < t_empty -> empty; S2-2 goes to S1-10 (one-owner)
    assert m.values.tolist() == [["S1-10", "S2-1"], ["S1-10", "S3-1"], ["S1-11", ""], ["S1-12", ""]]
    assert not list(test_paths.output_dir.glob("*.tmp"))


def test_write_fails_loudly_when_validator_fails(test_paths: Paths, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-PASS validator raises and leaves no final output files behind."""
    fake_root = tmp_path / "fake_repo"
    (fake_root / "utils").mkdir(parents=True)
    (fake_root / "utils" / "validate_submission.py").write_text(
        "import sys\nprint('FAIL - 1 issue(s)')\nsys.exit(1)\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", fake_root)
    with pytest.raises(RuntimeError, match="did NOT pass"):
        decide.run_stage(test_paths, "test")
    assert not (test_paths.output_dir / "matching_results.tsv").exists()
    assert not (test_paths.output_dir / "candidate_pairs.tsv").exists()


def test_id_keys_rejects_malformed_ids() -> None:
    """IDs must look like S<1-3>-<digits>; keys keep sources apart."""
    import pyarrow as pa

    keys = decide.id_keys(pa.array(["S2-5", "S3-5", "S1-123"]))
    assert len(set(keys.tolist())) == 3
    with pytest.raises(ValueError, match="must match"):
        decide.id_keys(pa.array(["S2-5", "X-1"]))


def _main(paths: Paths, stage: str, split: str, *extra: str) -> None:
    """Run one stage through the CLI entry point."""
    run_pipeline.main(["--stage", stage, "--split", split, "--data-dir", str(paths.data_dir),
                       "--artifacts-dir", str(paths.artifacts_dir), "--out-dir", str(paths.output_dir), *extra])


def _save_candidates(paths: Paths, split: str, rows: list[tuple[str, str, float]]) -> None:
    """candidates_{split}.parquet with a synthetic rrf_score (Lane B's column)."""
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "rrf_score"]).astype({"rrf_score": np.float32})
    io_utils.save_parquet(df, paths.artifacts_dir / f"candidates_{split}.parquet")


def test_baseline_end_to_end(train_paths: Paths) -> None:
    """baseline(train) -> decide(score) -> baseline(test) -> write --decision-config -> validator PASS."""
    paths = train_paths
    _make_split(paths, "test", TEST)
    _save_candidates(paths, "train", [(s1, c, p) for s1, c, p, _ in OOF])
    _save_candidates(paths, "test", PRED)
    try:
        _main(paths, "baseline", "train", "--force")  # fixture already wrote a prob-based oof_train
        oof = pd.read_parquet(paths.artifacts_dir / "oof_train.parquet")
        assert oof["label"].tolist() == [lbl for *_, lbl in OOF]  # labels from the ground truth
        assert (oof["score"].dtype, oof["label"].dtype, oof["fold"].dtype) == (np.float32, np.int8, np.int8)
        _main(paths, "decide", "train", "--decide-score-column", "score")
        _main(paths, "baseline", "test")
        assert pd.read_parquet(paths.artifacts_dir / "pred_test.parquet").columns.tolist() == [
            "s1_id", "cand_id", "score"]
        # Config tuned "elsewhere": moved out of the artifacts dir and passed explicitly.
        tuned = paths.data_dir.parent / "tuned" / "decision_config.json"
        tuned.parent.mkdir()
        (paths.artifacts_dir / "decision_config.json").rename(tuned)
        with pytest.raises(FileNotFoundError, match="Decision config not found"):
            _main(paths, "write", "test")
        _main(paths, "write", "test", "--decision-config", str(tuned))  # raises unless validator prints PASS
    finally:
        config.DECIDE_SCORE_COLUMN = "prob"
        config.DECISION_CONFIG_PATH = None
    cfg = json.loads(tuned.read_text(encoding="utf-8"))
    assert cfg["score_column"] == "score" and cfg["train_f05"] == pytest.approx(_brute_force_best(True), abs=1e-6)
    rows = (paths.output_dir / "candidate_pairs.tsv").read_text(encoding="utf-8").splitlines()
    assert [r.split("\t")[0] for r in rows[1:]] == ["S1-10", "S1-11", "S1-12"]


def test_baseline_without_rrf_score_names_lane_b(train_paths: Paths) -> None:
    """A candidates file from blocking v0 (no rrf_score) fails with a clear message."""
    df = pd.DataFrame([("S1-1", "S2-47", 0.9)], columns=["s1_id", "cand_id", "best_block_score"])
    io_utils.save_parquet(df, train_paths.artifacts_dir / "candidates_train.parquet")
    with pytest.raises(ValueError, match="rrf_score.*Lane B"):
        decide.run_baseline(train_paths, "train")
    assert not (train_paths.artifacts_dir / "oof_train.parquet.tmp").exists()
