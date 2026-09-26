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
    grid = decide.threshold_grid(rrf, 200, step=0)
    assert grid == tuple(sorted(float(v) for v in set(rrf.tolist())))
    assert set(grid) < set(decide.threshold_grid(rrf, 200))  # default adds the fixed step grid
    probs = np.linspace(0, 1, 10_001, dtype=np.float32)
    grid = decide.threshold_grid(probs, 200, step=0)
    assert len(grid) == 200 and grid[0] == 0.0 and grid[-1] == 1.0 and set(grid) <= set(probs.tolist())
    with pytest.raises(ValueError):
        decide.threshold_grid(np.array([], np.float32), 200)


def test_threshold_grid_covers_bimodal_probabilities() -> None:
    """Near-0 bulk + few high probs: quantiles skip 0.02-0.99, the fixed step grid restores the optimum.

    Positives score 0.3-0.5, top negatives 0.2-0.25: the best t lies in (0.25, 0.3],
    which no quantile reaches. The grid search must equal brute force over every
    distinct score (all distinct thresholds for ``score >= t``).
    """
    rng = np.random.default_rng(0)
    n_s1 = 200
    s1 = np.repeat(np.arange(n_s1, dtype=np.int32), 400)
    score = rng.uniform(0.0, 0.01, len(s1)).astype(np.float32)  # bulk near 0
    label = np.zeros(len(s1), np.int8)
    first = np.arange(n_s1) * 400
    score[first[:150]] = rng.uniform(0.3, 0.5, 150).astype(np.float32)  # true matches
    label[first[:150]] = 1
    score[first + 1] = rng.uniform(0.2, 0.25, n_s1).astype(np.float32)  # hard negatives
    pairs = decide.Pairs(s1, np.arange(len(s1), dtype=np.int32), score, label)
    n_truth = np.bincount(s1[label == 1], minlength=n_s1).astype(np.int64)
    keep = np.ones(len(s1), dtype=bool)
    old = decide.threshold_grid(score, 200, step=0)
    grid = decide.threshold_grid(score, 200)
    assert set(old) < set(grid) and float(np.float32(0.005)) in grid and float(np.float32(0.995)) in grid
    assert not any(0.25 < t < 0.3 for t in old)
    mx = decide.max_score_per_s1(pairs, keep, n_s1)
    brute = max(float(decide.per_s1_scores(pairs, decide.select(pairs, keep, mx, float(t), te), n_truth).mean())
                for t in np.unique(score) for te in (0.0, 0.3))
    assert brute == 1.0
    assert decide.grid_search(pairs, keep, n_truth, old)[2] < brute - 0.05
    assert decide.grid_search(pairs, keep, n_truth, grid)[2] == pytest.approx(brute, abs=1e-12)


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


# --------------------------------------------------------------------------- decide v1


def _random_pairs(seed: int, n_s1: int = 40, n_cand: int = 60, n_pairs: int = 300) -> tuple[
        decide.Pairs, np.ndarray, np.ndarray]:
    """Random labelled pairs with coarse (tie-heavy) scores; returns pairs, n_truth, is_s3 per pair."""
    rng = np.random.default_rng(seed)
    key = np.unique(rng.integers(0, n_s1 * n_cand, n_pairs))
    s1, cand = (key // n_cand).astype(np.int32), (key % n_cand).astype(np.int32)
    score = (rng.integers(0, 12, len(key)) / 11).astype(np.float32)
    label = (rng.random(len(key)) < score * 0.8).astype(np.int8)
    missed = rng.integers(0, 2, n_s1) * (rng.random(n_s1) < 0.2)  # truths blocking missed
    n_truth = np.bincount(s1[label == 1], minlength=n_s1).astype(np.int64) + missed
    return decide.Pairs(s1, cand, score, label), n_truth, cand % 3 == 0


def _macro(pairs: decide.Pairs, sel: np.ndarray, n_truth: np.ndarray) -> float:
    """Macro F0.5 of a selection, scored per S1 with evaluate.f05_single (independent of decide)."""
    assert pairs.label is not None
    total = 0.0
    for i in range(len(n_truth)):
        mine = pairs.s1 == i
        pred = {int(c) for c in pairs.cand[mine & sel]}
        found = {int(c) for c in pairs.cand[mine & (pairs.label == 1)]}
        truth = found | {-k - 1 for k in range(int(n_truth[i]) - len(found))}  # missed truths: unmatched IDs
        total += evaluate.f05_single({str(c) for c in pred}, {str(c) for c in truth})
    return total / len(n_truth)


@pytest.mark.parametrize("seed", range(4))
def test_sorted_sweep_matches_exhaustive_grid(seed: int) -> None:
    """Incremental sweep = scoring every (t, t_empty) on the grid, same winner incl. tie-breaking."""
    pairs, n_truth, _ = _random_pairs(seed)
    n = len(n_truth)
    keep = decide.one_owner_mask(pairs, 60) if seed % 2 else np.ones(len(pairs.s1), dtype=bool)
    grid = decide.threshold_grid(pairs.score, 200, step=0)  # brute force is O(grid^2-3)
    mx = decide.max_score_per_s1(pairs, keep, n)
    best = (grid[0], 0.0, -1.0)
    for t in grid:  # v0 algorithm: ascending, strict improvement
        for te in (0.0, *grid):
            f = float(decide.per_s1_scores(pairs, decide.select(pairs, keep, mx, t, te), n_truth).mean())
            if f > best[2] + 1e-12:
                best = (t, te, f)
    t, te, f = decide.grid_search(pairs, keep, n_truth, grid)
    assert (t, te) == best[:2] and f == pytest.approx(best[2], abs=1e-12)
    assert f == pytest.approx(_macro(pairs, decide.select(pairs, keep, mx, t, te), n_truth), abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_per_source_between_global_and_brute_force(seed: int) -> None:
    """Coordinate descent: >= global threshold, <= exhaustive (t_s2, t_s3, t_empty), and its F is real."""
    pairs, n_truth, is_s3 = _random_pairs(seed)
    n = len(n_truth)
    keep = np.ones(len(pairs.s1), dtype=bool)
    grid = decide.threshold_grid(pairs.score, 200, step=0)  # brute force is O(grid^2-3)
    start = decide.grid_search(pairs, keep, n_truth, grid)
    t2, t3, te, f = decide.per_source_search(pairs, keep, is_s3, n_truth, grid, start)
    mx = decide.max_score_per_s1(pairs, keep, n)
    brute = max(float(decide.per_s1_scores(pairs, decide.select(pairs, keep, mx, np.where(is_s3, b, a), e),
                                           n_truth).mean())
                for a in grid for b in grid for e in (0.0, *grid))
    assert start[2] - 1e-12 <= f <= brute + 1e-12
    sel = decide.select(pairs, keep, mx, np.where(is_s3, np.float32(t3), np.float32(t2)), te)
    assert f == pytest.approx(_macro(pairs, sel, n_truth), abs=1e-12)


def test_per_source_finds_separate_thresholds() -> None:
    """S2 matches score 0.9 vs S2 non-matches 0.6; S3 matches 0.5 vs non-matches 0.2: needs t_s2 > t_s3."""
    s1 = np.array([0, 0, 1, 1, 2, 3], np.int32)
    cand = np.array([0, 1, 2, 3, 4, 5], np.int32)
    score = np.array([0.9, 0.6, 0.5, 0.2, 0.6, 0.2], np.float32)
    label = np.array([1, 0, 1, 0, 0, 0], np.int8)
    is_s3 = np.array([False, False, True, True, False, True])
    pairs, n_truth = decide.Pairs(s1, cand, score, label), np.array([1, 1, 0, 0], np.int64)
    keep = np.ones(6, dtype=bool)
    grid = decide.threshold_grid(score, 200)
    start = decide.grid_search(pairs, keep, n_truth, grid)
    t2, t3, _, f = decide.per_source_search(pairs, keep, is_s3, n_truth, grid, start)
    assert start[2] < 1.0 and f == 1.0 and t2 > 0.6 and 0.2 < t3 <= 0.5


def _expected_naive(pairs: decide.Pairs, keep: np.ndarray) -> set[int]:
    """Per-S1 expected-F0.5 prefix choice written as a plain loop."""
    chosen: set[int] = set()
    for i in np.unique(pairs.s1[keep]):
        idx = [int(j) for j in np.flatnonzero(keep & (pairs.s1 == i))]
        idx.sort(key=lambda j: -float(pairs.score[j]))  # stable: file order among ties
        p = [float(pairs.score[j]) for j in idx]
        best_k, best = 0, float(np.prod([1 - q for q in p]))
        for k in range(1, len(p) + 1):
            e = 1.25 * sum(p[:k]) / (0.25 * sum(p) + k)
            if e > best + 1e-12:
                best_k, best = k, e
        chosen |= set(idx[:best_k])
    return chosen


@pytest.mark.parametrize("chunk", [1, 3, 10_000])
def test_expected_f05_matches_naive(monkeypatch: pytest.MonkeyPatch, chunk: int) -> None:
    """Vectorised, chunked selection = plain per-S1 loop, for chunks smaller than one S1 and larger than all."""
    monkeypatch.setattr(config, "DECIDE_EXPECTED_CHUNK_PAIRS", chunk)
    for seed in range(3):
        pairs, n_truth, _ = _random_pairs(seed)
        pairs = decide.Pairs(pairs.s1, pairs.cand, np.random.default_rng(seed).random(len(pairs.s1)).astype(np.float32)
                             ** 3, pairs.label)
        keep = decide.one_owner_mask(pairs, 60)
        sel = decide.expected_f05_select(pairs, keep, len(n_truth))
        assert set(np.flatnonzero(sel).tolist()) == _expected_naive(pairs, keep)


def test_expected_f05_simple_cases() -> None:
    """Lone weak candidate -> empty; lone strong -> taken; one strong + one weak -> strong only."""
    pairs = decide.Pairs(s1=np.array([0, 1, 2, 2], np.int32), cand=np.arange(4, dtype=np.int32),
                         score=np.array([0.3, 0.8, 0.95, 0.1], np.float32), label=None)
    assert decide.expected_f05_select(pairs, np.ones(4, dtype=bool), 3).tolist() == [False, True, True, False]


def test_calibration_table() -> None:
    """Equal-width bins: counts, mean probability and positive rate per bin; empty bins dropped."""
    prob = np.array([0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 1.0], np.float32)
    label = np.array([0, 1, 1, 1, 1, 0, 1], np.int8)
    t = decide.calibration_table(prob, label)
    assert t["bin"].tolist() == ["[0.0,0.1)", "[0.9,1.0)"] and t["n"].tolist() == [2, 5]
    assert t["pos_rate"].tolist() == pytest.approx([0.5, 0.8]) and t["mean_prob"].iloc[1] == pytest.approx(0.96)


def test_auto_one_owner_and_per_source_end_to_end(train_paths: Paths) -> None:
    """--decide-one-owner-auto + --decide-method per_source: choice recorded, write applies t_s2 / t_s3."""
    paths = train_paths
    _make_split(paths, "test", TEST)
    io_utils.save_parquet(pd.DataFrame(PRED, columns=["s1_id", "cand_id", "prob"]),
                          paths.artifacts_dir / "pred_test.parquet")
    try:
        _main(paths, "decide", "train", "--decide-method", "per_source", "--decide-one-owner-auto")
        cfg = json.loads((paths.artifacts_dir / "decision_config.json").read_text(encoding="utf-8"))
        assert cfg["method"] == "per_source" and cfg["one_owner_auto"] is True
        by = cfg["f05_by_one_owner"]
        assert cfg["one_owner"] is (by["true"] >= by["false"]) and cfg["train_f05"] == max(by.values())
        assert cfg["train_f05"] >= _brute_force_best(cfg["one_owner"]) - 1e-6
        cfg.update(t_s2=0.9, t_s3=0.5, t_empty=0.0, one_owner=False)  # S2 needs 0.9, S3 needs 0.5
        (paths.artifacts_dir / "decision_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        _main(paths, "write", "test")
    finally:
        config.DECIDE_METHOD, config.DECIDE_ONE_OWNER_AUTO = "threshold", False
    m = (paths.output_dir / "matching_results.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assert m == ["S1-10\tS2-1,S3-1", "S1-11\t", "S1-12\t"]


def test_default_decision_config_unchanged(train_paths: Paths) -> None:
    """Flags off: decision_config.json has exactly the v0 keys."""
    decide.run_decide(train_paths, log_row=False)
    cfg = json.loads((train_paths.artifacts_dir / "decision_config.json").read_text(encoding="utf-8"))
    assert list(cfg) == ["t", "t_empty", "one_owner", "score_column", "train_f05"]


def test_expected_f05_needs_probabilities(train_paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    """expected_f05 on a rule-baseline score is refused before any work."""
    monkeypatch.setattr(config, "DECIDE_METHOD", "expected_f05")
    monkeypatch.setattr(config, "DECIDE_SCORE_COLUMN", "score")
    with pytest.raises(ValueError, match="needs probabilities"):
        decide.run_decide(train_paths)


def test_compare_stage_table(train_paths: Paths) -> None:
    """compare writes one row per method x one-owner with overall / singleton / country columns."""
    _main(train_paths, "compare", "train")
    t = pd.read_csv(train_paths.artifacts_dir / "decide_compare.tsv", sep="\t")
    assert t["variant"].tolist() == ["threshold+one_owner", "per_source+one_owner", "expected_f05+one_owner",
                                     "threshold", "per_source", "expected_f05"]
    assert {"overall", "singleton", "non_singleton", "country=US", "country=India"} <= set(t.columns)
    assert t.loc[0, "overall"] == pytest.approx(_brute_force_best(True), abs=1e-4)
    assert t.loc[3, "overall"] == pytest.approx(_brute_force_best(False), abs=1e-4)
    assert (t["overall"][[1, 4]].to_numpy() >= t["overall"][[0, 3]].to_numpy() - 1e-9).all()
    assert not (train_paths.artifacts_dir / "decision_config.json").exists()


def test_write_with_expected_f05_config(test_paths: Paths) -> None:
    """write applies method=expected_f05 per S1: S1-10 keeps its two strong candidates, S1-12 (0.4, 0.1) goes empty."""
    cfg_file = test_paths.artifacts_dir / "decision_config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8")) | {"method": "expected_f05"}
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    decide.run_stage(test_paths, "test")
    m = (test_paths.output_dir / "matching_results.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assert m == ["S1-10\tS2-1,S3-1", "S1-11\t", "S1-12\t"]


def test_blocking_s1_subset_restricts_train_s1(tmp_path: Path) -> None:
    """With blocking_s1_subset_train.parquet, decide sees only those S1 and their truth."""
    paths = Paths(tmp_path / "d", tmp_path / "a", tmp_path / "o")
    paths.artifacts_dir.mkdir(parents=True)
    (paths.data_dir / "train").mkdir(parents=True)
    pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S2-1", "S2-2"], "source": ["S1", "S1", "S2", "S2"],
                  "country": ["US"] * 4}).to_parquet(paths.artifacts_dir / "records_train.parquet", index=False)
    (paths.data_dir / "train" / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\n", encoding="utf-8", newline="\n")
    pd.DataFrame({"s1_id": ["S1-2"]}).to_parquet(paths.artifacts_dir / "blocking_s1_subset_train.parquet")
    ids = decide.load_split_ids(paths, "train")
    assert list(ids.s1) == ["S1-2"] and list(ids.cand) == ["S2-1", "S2-2"]
    assert list(decide.truth_counts(paths.data_dir, ids)) == [1]
