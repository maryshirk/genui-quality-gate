from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.model_selection import GroupShuffleSplit

DATASET_REPOSITORY = "SALT-NLP/GenUI"
DATASET_REVISION = "11c9182f5f1452f28a8714bf3dea6fbd218963c7"
DATASET_SOURCE = f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/tree/{DATASET_REVISION}"
RATINGS_PATH = "raw_evaluation_results/raw_human_evaluation_results.json"
INPUT_COLUMNS = ["prompt", "html_a", "html_b"]


def content_digest(frame: pd.DataFrame) -> str:
    records = frame.sort_values("pair_id").to_json(orient="records", force_ascii=False)
    return hashlib.sha256(records.encode()).hexdigest()


def html_path(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[0-9]+\.html", value) or value.split("/")[0] in {
        ".",
        "..",
    }:
        raise ValueError("Unexpected dataset HTML path")
    return f"generated_uis/{value}"


def aggregate_ratings(submissions: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    summary = Counter({"raw_ratings": len(submissions)})
    for item in submissions:
        if any(item.get(flag, False) for flag in ("isTrap", "isFlagged", "isDisqualified")):
            summary["excluded_flagged_ratings"] += 1
            continue
        prompt = " ".join(str(item["userQuery"]).split())
        first, second = html_path(item["linkAUrl"]), html_path(item["linkBUrl"])
        winner = item["overallWinner"]
        if not prompt or first == second or winner not in {"A", "B", "tie"}:
            summary["excluded_invalid_ratings"] += 1
            continue
        low, high = sorted((first, second))
        key = (prompt, low, high)
        annotation_key = (*key, str(item["annotatorId"]))
        if annotation_key in seen:
            summary["excluded_duplicate_ratings"] += 1
            continue
        seen.add(annotation_key)
        if first != low and winner != "tie":
            winner = "B" if winner == "A" else "A"
        groups[key].append(winner)
    records = []
    for (prompt, low, high), votes in sorted(groups.items()):
        counts = Counter(votes)
        winner, count = counts.most_common(1)[0]
        summary["valid_ratings"] += len(votes)
        summary["tie_ratings"] += counts["tie"]
        if winner == "tie" or count <= len(votes) / 2:
            summary["excluded_no_majority_pairs"] += 1
            continue
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
        pair_id = hashlib.sha256(f"{prompt_id}|{low}|{high}".encode()).hexdigest()
        flipped = int(pair_id[-1], 16) % 2 == 1
        records.append(
            {
                "pair_id": pair_id,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "path_a": high if flipped else low,
                "path_b": low if flipped else high,
                "target": int((winner == "A") != flipped),
                "agreement": count / len(votes),
                "rating_count": len(votes),
            }
        )
    if not records:
        raise ValueError("No pairs with a strict majority remain")
    frame = pd.DataFrame.from_records(records).sort_values("pair_id").reset_index(drop=True)
    summary["total_pairs"] = len(groups)
    summary["retained_pairs"] = len(frame)
    summary["retained_prompts"] = frame["prompt_id"].nunique()
    return frame, dict(summary)


def annotation_statistics(submissions: list[dict[str, Any]]) -> dict[str, Any]:
    criteria: dict[str, Counter[str]] = defaultdict(Counter)
    criterion_votes: dict[tuple[str, tuple[str, str, str]], list[str]] = defaultdict(list)
    pairs: Counter[tuple[str, str, str]] = Counter()
    for item in submissions:
        prompt = " ".join(str(item.get("userQuery", "")).split())
        first, second = sorted((str(item.get("linkAUrl", "")), str(item.get("linkBUrl", ""))))
        pair = (prompt, first, second)
        pairs[pair] += 1
        overall = item.get("overallWinner")
        for evaluation in item.get("dimensionEvaluations", []):
            counts = criteria[str(evaluation["dimensionId"])]
            winner = evaluation.get("winner")
            counts[str(winner)] += 1
            if winner in {"A", "B", "tie"}:
                canonical = winner
                if item.get("linkAUrl") != first and winner != "tie":
                    canonical = "B" if winner == "A" else "A"
                criterion_votes[(str(evaluation["dimensionId"]), pair)].append(canonical)
            if winner in {"A", "B"} and overall in {"A", "B"}:
                counts["comparable_to_overall"] += 1
                counts["matches_overall"] += int(winner == overall)
    required = ("userQuery", "linkAUrl", "linkBUrl", "overallWinner", "annotatorId")
    criterion_results = {}
    for name, counts in criteria.items():
        votes = [
            Counter(values)
            for (criterion, _), values in criterion_votes.items()
            if criterion == name
        ]
        criterion_results[name] = {
            **dict(counts),
            "rated_pairs": len(votes),
            "unanimous_pairs": sum(len(value) == 1 for value in votes),
            "all_tie_pairs": sum(set(value) == {"tie"} for value in votes),
            "strict_preference_majority_pairs": sum(
                max(value["A"], value["B"]) > value.total() / 2 for value in votes
            ),
            "mean_top_vote_share": (
                sum(max(value.values()) / value.total() for value in votes) / len(votes)
                if votes
                else None
            ),
        }
    return {
        "overall_votes": dict(Counter(str(item.get("overallWinner")) for item in submissions)),
        "criteria": criterion_results,
        "ratings_per_pair": dict(Counter(pairs.values())),
        "annotators": len({str(item.get("annotatorId")) for item in submissions}),
        "missing_required_values": {
            name: sum(item.get(name) in (None, "") for item in submissions) for name in required
        },
        "flags": {
            name: sum(bool(item.get(name, False)) for item in submissions)
            for name in ("isTrap", "isFlagged", "isDisqualified")
        },
    }


def load_dataset(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = directory.resolve()
    snapshot_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=[RATINGS_PATH, "generated_uis/*/*.html"],
        local_dir=directory,
        max_workers=4,
    )
    ratings_file = directory / RATINGS_PATH
    submissions = json.loads(ratings_file.read_text())["submissions"]
    frame, counts = aggregate_ratings(submissions)
    paths = sorted(set(frame["path_a"]) | set(frame["path_b"]))
    html = {path: (directory / path).read_text(encoding="utf-8") for path in paths}
    frame["html_a"] = frame["path_a"].map(html)
    frame["html_b"] = frame["path_b"].map(html)
    manifest = {
        "repository": DATASET_REPOSITORY,
        "revision": DATASET_REVISION,
        "source": DATASET_SOURCE,
        "license": "CC-BY-NC-4.0",
        "counts": counts,
        "annotations": annotation_statistics(submissions),
        "source_html_files": sum(1 for _ in (directory / "generated_uis").glob("*/*.html")),
        "sha256": {
            path: hashlib.sha256((directory / path).read_bytes()).hexdigest()
            for path in [RATINGS_PATH, *paths]
        },
        "transformations": [
            "exclude trap, flagged, disqualified, invalid and duplicate annotator ratings",
            "canonicalize unordered HTML pairs within normalized prompts",
            "retain only strict A/B majority across all votes, including ties",
            "choose reproducible A/B orientation using pair SHA256 parity",
            "split by prompt before augmenting training pairs with reversed pairs",
            "parse HTML only; no browser, external resources or JavaScript execution",
        ],
    }
    return frame, manifest


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    if frame["prompt_id"].nunique() < 10:
        raise ValueError("At least ten distinct prompts are required")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development_index, test_index = next(splitter.split(frame, groups=frame["prompt_id"]))
    development = frame.iloc[development_index]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_index, validation_index = next(
        splitter.split(development, groups=development["prompt_id"])
    )
    splits = {
        "train": development.iloc[train_index].reset_index(drop=True),
        "validation": development.iloc[validation_index].reset_index(drop=True),
        "test": frame.iloc[test_index].reset_index(drop=True),
    }
    if any(set(split["target"].unique()) != {0, 1} for split in splits.values()):
        raise ValueError("Each split must contain both target classes")
    return splits


def mirror_training_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    mirrored = frame.copy()
    mirrored["html_a"] = frame["html_b"]
    mirrored["html_b"] = frame["html_a"]
    if {"path_a", "path_b"}.issubset(frame.columns):
        mirrored["path_a"] = frame["path_b"]
        mirrored["path_b"] = frame["path_a"]
    mirrored["target"] = 1 - frame["target"]
    mirrored["pair_id"] = frame["pair_id"] + ":reversed"
    return pd.concat([frame, mirrored], ignore_index=True)
