from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from genui_quality_gate.ml.data import (
    DATASET_REVISION,
    RATINGS_PATH,
    aggregate_ratings,
    content_digest,
    html_path,
    load_dataset,
    mirror_training_pairs,
    split_dataset,
)


def rating(annotator: str, winner: str, **overrides: object) -> dict[str, object]:
    return {
        "annotatorId": annotator,
        "userQuery": "Build a form",
        "linkAUrl": "first/001.html",
        "linkBUrl": "second/001.html",
        "overallWinner": winner,
        **overrides,
    }


def pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": f"pair-{index}-{target}",
                "prompt_id": str(index),
                "prompt": f"Prompt {index}",
                "html_a": "first",
                "html_b": "second",
                "target": target,
            }
            for index in range(30)
            for target in (0, 1)
        ]
    )


def test_aggregation_removes_duplicates_flags_and_nonmajority() -> None:
    ratings = [
        rating("one", "A"),
        rating("two", "A"),
        rating("three", "tie"),
        rating("one", "A"),
        rating("bad", "B", isFlagged=True),
        rating("bad2", "invalid"),
        rating("four", "A", userQuery="Another prompt"),
        rating("five", "B", userQuery="Another prompt"),
        rating("six", "tie", userQuery="Another prompt"),
        rating("seven", "tie", userQuery="Tied prompt"),
    ]
    frame, summary = aggregate_ratings(ratings)
    assert len(frame) == 1
    assert frame.iloc[0]["agreement"] == pytest.approx(2 / 3)
    assert frame.iloc[0]["rating_count"] == 3
    assert summary["excluded_duplicate_ratings"] == 1
    assert summary["excluded_flagged_ratings"] == 1
    assert summary["excluded_invalid_ratings"] == 1
    assert summary["excluded_no_majority_pairs"] == 2
    assert summary["retained_prompts"] == 1


def test_aggregation_canonicalizes_reversed_pair_and_whitespace() -> None:
    frame, _ = aggregate_ratings(
        [
            rating("one", "A"),
            rating("two", "B", linkAUrl="second/001.html", linkBUrl="first/001.html"),
            rating("three", "A", userQuery=" Build   a form "),
        ]
    )
    row = frame.iloc[0]
    assert row["agreement"] == 1
    winner = row["path_a"] if row["target"] else row["path_b"]
    assert winner == "generated_uis/first/001.html"


def test_aggregation_requires_usable_pairs() -> None:
    with pytest.raises(ValueError, match="No pairs"):
        aggregate_ratings([rating("one", "tie")])


@pytest.mark.parametrize(
    "path",
    ["../secret", "https://example.com/1.html", "dir/../../1.html", "../001.html", "./001.html"],
)
def test_html_path_rejects_untrusted_locations(path: str) -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        html_path(path)


def test_group_splits_are_reproducible_and_disjoint() -> None:
    frame = pairs()
    splits = split_dataset(frame)
    assert [len(part) for part in splits.values()] == [36, 12, 12]
    assert sum(len(part) for part in splits.values()) == len(frame)
    prompts = [set(part["prompt_id"]) for part in splits.values()]
    assert not prompts[0] & prompts[1]
    assert not prompts[0] & prompts[2]
    assert not prompts[1] & prompts[2]
    pd.testing.assert_frame_equal(splits["train"], split_dataset(frame)["train"])


def test_splits_reject_too_few_prompts_and_missing_classes() -> None:
    with pytest.raises(ValueError, match="ten"):
        split_dataset(pairs().iloc[:4])
    frame = pairs()
    frame["target"] = 1
    with pytest.raises(ValueError, match="both target"):
        split_dataset(frame)


def test_mirroring_balances_training_and_preserves_source() -> None:
    frame = pairs().query("target == 1")
    frame["path_a"] = "first"
    frame["path_b"] = "second"
    original = frame.copy(deep=True)
    augmented = mirror_training_pairs(frame)
    assert len(augmented) == 2 * len(frame)
    assert augmented["target"].mean() == 0.5
    assert augmented.iloc[-1]["html_a"] == "second"
    assert augmented.iloc[-1]["html_b"] == "first"
    assert augmented.iloc[-1]["path_a"] == "second"
    assert augmented.iloc[-1]["path_b"] == "first"
    assert augmented.iloc[-1]["pair_id"].endswith(":reversed")
    pd.testing.assert_frame_equal(frame, original)


def test_digest_tracks_content_but_not_row_order() -> None:
    frame = pairs()
    assert content_digest(frame) == content_digest(frame.iloc[::-1])
    changed = frame.copy()
    changed.loc[0, "html_a"] = "changed"
    assert content_digest(frame) != content_digest(changed)


def test_loading_pins_download_revision_and_records_checksums(tmp_path: Path) -> None:
    ratings_file = tmp_path / RATINGS_PATH
    ratings_file.parent.mkdir(parents=True)
    ratings_file.write_text(json.dumps({"submissions": [rating("one", "A")]}))
    for folder in ("first", "second"):
        file = tmp_path / "generated_uis" / folder / "001.html"
        file.parent.mkdir(parents=True)
        file.write_text(f"<main>{folder}</main>")
    with patch("genui_quality_gate.ml.data.snapshot_download") as download:
        frame, manifest = load_dataset(tmp_path)
    assert download.call_args.kwargs["revision"] == DATASET_REVISION
    assert len(frame) == 1
    assert "<main>" in frame.iloc[0]["html_a"]
    assert len(manifest["sha256"]) == 3
    assert manifest["license"] == "CC-BY-NC-4.0"
