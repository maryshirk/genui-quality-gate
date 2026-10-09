from copy import deepcopy

import pandas as pd
import pytest

from genui_quality_gate.ml.data import annotation_statistics, split_dataset
from genui_quality_gate.ml.eda import (
    build_analysis,
    criterion_table,
    document_table,
    feature_preferences,
    split_tables,
)
from tests.unit.test_ml_training import training_frame


def test_annotation_statistics_retains_ties_and_uses_comparable_votes() -> None:
    ratings = [
        {
            "annotatorId": str(index),
            "userQuery": "Build a form",
            "linkAUrl": "one/001.html",
            "linkBUrl": "two/001.html",
            "overallWinner": overall,
            "dimensionEvaluations": [{"dimensionId": "usability", "winner": winner}],
        }
        for index, (overall, winner) in enumerate([("A", "A"), ("A", "B"), ("tie", "tie")])
    ]
    statistics = annotation_statistics(ratings)
    assert statistics["ratings_per_pair"] == {3: 1}
    assert statistics["annotators"] == 3
    assert not any(statistics["missing_required_values"].values())
    row = criterion_table(statistics).iloc[0]
    assert row["tie_share"] == pytest.approx(1 / 3)
    assert row["comparable_votes"] == 2
    assert row["overall_agreement"] == 0.5
    assert row["rated_pairs"] == 1
    assert row["mean_top_vote_share"] == pytest.approx(1 / 3)
    assert row["no_preference_majority_share"] == 1


def test_criterion_agreement_accounts_for_reversed_presentation() -> None:
    original = {
        "userQuery": "Prompt",
        "linkAUrl": "one/001.html",
        "linkBUrl": "two/001.html",
        "overallWinner": "A",
        "dimensionEvaluations": [{"dimensionId": "usability", "winner": "A"}],
    }
    reversed_rating = {
        **original,
        "linkAUrl": "two/001.html",
        "linkBUrl": "one/001.html",
        "overallWinner": "B",
        "dimensionEvaluations": [{"dimensionId": "usability", "winner": "B"}],
    }
    row = criterion_table(annotation_statistics([original, reversed_rating])).iloc[0]
    assert row["unanimous_pair_share"] == 1
    assert row["mean_top_vote_share"] == 1
    assert row["no_preference_majority_share"] == 0


def test_documents_are_counted_once_and_repeated_content_is_identified() -> None:
    frame = training_frame()
    documents = document_table(frame)
    assert len(documents) == 2
    frame["path_a"] = [f"generated_uis/one/{index}.html" for index in range(len(frame))]
    frame["path_b"] = [f"generated_uis/two/{index}.html" for index in range(len(frame))]
    documents = document_table(frame)
    assert len(documents) == 2 * len(frame)
    assert documents["sha256"].nunique() == 2


def test_split_checks_detect_shared_content_despite_distinct_prompts() -> None:
    sizes, overlap = split_tables(split_dataset(training_frame()))
    assert sizes["prompts"].sum() == 20
    assert not overlap["prompt_overlap"].any()
    assert (overlap["html_digest_overlap"] == 2).all()


def test_feature_preferences_are_oriented_by_winner() -> None:
    preferences = feature_preferences(training_frame())
    assert preferences.loc["html_length", "winner_more_share"] == 1
    assert preferences.loc["interactive_count", "winner_less_share"] == 0
    assert preferences.loc["image_count", "equal_share"] == 1
    assert (
        preferences[["winner_more_share", "equal_share", "winner_less_share"]].sum(axis=1) == 1
    ).all()


def test_feature_analysis_does_not_use_validation_or_test_labels() -> None:
    frame = training_frame()
    splits = split_dataset(frame)
    first = build_analysis(frame, splits, {})
    changed = deepcopy(splits)
    for name in ("validation", "test"):
        changed[name]["target"] = 1 - changed[name]["target"]
    second = build_analysis(frame, changed, {})
    pd.testing.assert_frame_equal(
        first.tables["feature_preferences"], second.tables["feature_preferences"]
    )
    assert first.summary["feature_analysis_scope"] == "original training split only"
    assert "## Ограничения" in first.report
    assert "feature_correlations" in first.figures


def test_all_tie_criterion_has_no_overall_agreement() -> None:
    manifest = {
        "annotations": {
            "criteria": {"usability": {"tie": 3}},
            "overall_votes": {"tie": 3},
        }
    }
    frame = training_frame()
    result = build_analysis(frame, split_dataset(frame), manifest)
    assert pd.isna(result.tables["criteria"].iloc[0]["overall_agreement"])
    assert "annotations" in result.figures
