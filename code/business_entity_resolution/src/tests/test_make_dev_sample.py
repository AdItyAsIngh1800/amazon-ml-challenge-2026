"""Tests for src.make_dev_sample on a small synthetic train split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import io_utils
from src.make_dev_sample import build_dev_sample


def _fake_train(root: Path) -> None:
    """40 S1 (2 countries x matched/singleton), 3 matches each, 80 pure distractors."""
    Row = tuple[str, str, str, str]
    s1: list[Row] = []
    s23: dict[int, list[Row]] = {2: [], 3: []}
    gt: dict[str, set[str]] = {}
    for i in range(40):
        country = "US" if i % 2 else "India"
        s1_id = f"S1-{i}"
        s1.append((s1_id, f"Biz {i}", f'{i} "Main" St', country))
        matches = [] if i % 4 < 2 else [f"S2-{i}a", f"S2-{i}b", f"S3-{i}"]
        gt[s1_id] = set(matches)
        for m in matches:
            s23[int(m[1])].append((m, f"Biz {i}", "", country))
    for j in range(40):
        s23[2].append((f"S2-x{j}", "NA", "Somewhere", "US"))
        s23[3].append((f"S3-x{j}", "Other", "", "India"))
    cols = list(io_utils.SOURCE_COLUMNS)
    io_utils.write_source_tsv(pd.DataFrame(s1, columns=cols), root / "train" / "train_source1.tsv")
    for s in (2, 3):
        io_utils.write_source_tsv(pd.DataFrame(s23[s], columns=cols), root / "train" / f"train_source{s}.tsv")
    io_utils.write_id_list_tsv(gt, list(gt), root / "train" / "train_ground_truth.tsv", io_utils.GROUND_TRUTH_COLUMNS)


def test_dev_sample_contents(tmp_path: Path) -> None:
    """Stratified S1, all their matches, matching ratio, readable format, deterministic."""
    _fake_train(tmp_path / "full")
    build_dev_sample(tmp_path / "full", tmp_path / "dev", frac=0.5, seed=1)

    d = tmp_path / "dev" / "train"
    s1 = io_utils.read_source(d / "train_source1.tsv")
    s23 = pd.concat([io_utils.read_source(d / f"train_source{s}.tsv") for s in (2, 3)])
    gt = io_utils.read_ground_truth(d / "train_ground_truth.tsv")

    assert len(s1) == 20  # 10 per stratum x 4 strata x 0.5
    assert set(gt) == set(s1["entity_id"])
    singles = sum(not v for v in gt.values())
    assert singles == 10
    assert s1["country"].value_counts().to_dict() == {"US": 10, "India": 10}
    ids23 = set(s23["entity_id"])
    assert set().union(*gt.values()) <= ids23  # every true match kept
    full_ratio = (60 + 80) / 40
    assert len(s23) == round(full_ratio * 20)
    assert '"Main"' in s1["business_address"].iloc[0]

    build_dev_sample(tmp_path / "full", tmp_path / "dev2", frac=0.5, seed=1)
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"):
        assert (d / name).read_bytes() == (tmp_path / "dev2" / "train" / name).read_bytes()
