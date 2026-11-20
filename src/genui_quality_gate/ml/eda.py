from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from genui_quality_gate.ml.data import content_digest
from genui_quality_gate.ml.features import (
    CONTENT_FEATURES,
    STRUCTURAL_FEATURES,
    PairFeatures,
    document_features,
)

FEATURES = STRUCTURAL_FEATURES + CONTENT_FEATURES
SELECTED_FEATURES = [
    "html_length",
    "tag_count",
    "interactive_count",
    "semantic_count",
    "text_length",
    "prompt_overlap",
    "image_alt_ratio",
    "accessible_input_ratio",
]
CRITERIA = {
    "query_interface_consistency": "Query match",
    "task_efficiency": "Task efficiency",
    "usability": "Usability",
    "learnability": "Learnability",
    "information_clarity": "Information clarity",
    "aesthetic_appeal": "Visual appeal",
    "interaction_satisfaction": "Interaction satisfaction",
}
SOURCE_LABELS = {
    "generative_ui_claude_3.7": "GenUI (full)",
    "conversational_ui_claude_3.7": "Conversational (Claude 3.7)",
    "conversational_ui_gpt_4o": "Conversational (GPT-4o)",
    "instructed_ui_claude3.7": "Instructed (Claude 3.7)",
    "generative_ui_without_adaptive_reward_claude_3.7": "GenUI without reward",
    "generative_ui_without_adaptive_reward_and_structured_representation_claude_3.7": (
        "GenUI without reward/structure"
    ),
    (
        "generative_ui_without_adaptive_reward_and_structured_representation"
        "_and_iterative_refinement_claude_3.7"
    ): "GenUI without reward/structure/refinement",
}


@dataclass
class Analysis:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Figure]
    report: str


def document_table(frame: pd.DataFrame) -> pd.DataFrame:
    documents = {}
    for row in frame.to_dict(orient="records"):
        for side in ("a", "b"):
            html = str(row[f"html_{side}"])
            digest = hashlib.sha256(html.encode()).hexdigest()
            path = str(row.get(f"path_{side}", digest))
            if path not in documents:
                family = path.split("/")[-2] if "/" in path else "synthetic"
                documents[path] = {
                    "document": path,
                    "source_family": family,
                    "sha256": digest,
                    "characters": len(html),
                    "blank_html": not html.strip(),
                    **dict(
                        zip(
                            FEATURES, document_features(str(row["prompt"]), html, True), strict=True
                        )
                    ),
                }
    return pd.DataFrame(documents.values())


def split_tables(splits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    documents = {}
    for name, part in splits.items():
        documents[name] = document_table(part)
        rows.append(
            {
                "split": name,
                "pairs": len(part),
                "prompts": int(part["prompt_id"].nunique()),
                "documents": len(documents[name]),
                "a_win_share": float(part["target"].mean()),
                "unanimous_share": float((part["agreement"] == 1).mean()),
            }
        )
    overlap = []
    for first, second in combinations(splits, 2):
        overlap.append(
            {
                "splits": f"{first} / {second}",
                "prompt_overlap": len(
                    set(splits[first]["prompt_id"]) & set(splits[second]["prompt_id"])
                ),
                "document_overlap": len(
                    set(documents[first]["document"]) & set(documents[second]["document"])
                ),
                "html_digest_overlap": len(
                    set(documents[first]["sha256"]) & set(documents[second]["sha256"])
                ),
            }
        )
    return pd.DataFrame(rows).set_index("split"), pd.DataFrame(overlap)


def criterion_table(annotations: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, counts in annotations.get("criteria", {}).items():
        total = sum(counts.get(label, 0) for label in ("A", "B", "tie"))
        comparable = counts.get("comparable_to_overall", 0)
        rated_pairs = counts.get("rated_pairs", 0)
        rows.append(
            {
                "criterion": name,
                "votes": total,
                "a_share": counts.get("A", 0) / max(1, total),
                "b_share": counts.get("B", 0) / max(1, total),
                "tie_share": counts.get("tie", 0) / max(1, total),
                "comparable_votes": comparable,
                "overall_agreement": (
                    counts.get("matches_overall", 0) / comparable if comparable else np.nan
                ),
                "rated_pairs": rated_pairs,
                "mean_top_vote_share": counts.get("mean_top_vote_share", np.nan),
                "unanimous_pair_share": counts.get("unanimous_pairs", 0) / max(1, rated_pairs),
                "all_tie_pair_share": counts.get("all_tie_pairs", 0) / max(1, rated_pairs),
                "no_preference_majority_share": (
                    1 - counts.get("strict_preference_majority_pairs", 0) / rated_pairs
                    if rated_pairs
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_preferences(train: pd.DataFrame) -> pd.DataFrame:
    differences = PairFeatures(extended=True).transform(train)
    oriented = differences * (2 * train["target"].to_numpy() - 1)[:, None]
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "winner_more_share": np.mean(oriented > 0, axis=0),
            "equal_share": np.mean(oriented == 0, axis=0),
            "winner_less_share": np.mean(oriented < 0, axis=0),
            "median_winner_minus_loser": np.median(oriented, axis=0),
        }
    ).set_index("feature")


def markdown_table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    rows.append("| " + " | ".join("---" for _ in frame.columns) + " |")
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def render_report(summary: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    quality = summary["quality"]
    annotations = summary.get("annotations", {})
    criteria = tables["criteria"]
    overlap = tables["split_overlap"]
    preferences = tables["feature_preferences"]
    lines = [
        "# EDA GenUI",
        "",
        "## Данные и единица наблюдения",
        "",
        f"Источник: {summary.get('source', 'synthetic smoke data')}. "
        f"Ревизия: `{summary.get('revision', 'synthetic-v1')}`.",
        "Одна строка после обработки — пара интерфейсов для одного запроса. "
        "Целевая переменная задаётся строгим большинством голосов оценщиков. "
        "target=1 означает победу A, target=0 — победу B.",
        f"Получено {summary['pairs']} пар для {summary['prompts']} запросов. "
        f"В этих парах используются {quality['unique_documents']} разных документов.",
        "",
        "## Качество данных",
        "",
        f"Пустых HTML: {quality['blank_html']}; повторных pair_id: "
        f"{quality['duplicate_pair_ids']}; пропусков в полях модели: "
        f"{quality['missing_model_values']}.",
        f"Разных SHA256 HTML: {quality['unique_html_digests']}. "
        "Повторное использование одного документа в нескольких парах учитывается "
        "отдельно от совпадения содержимого разных файлов.",
        f"Пар с полностью одинаковым HTML: {quality['identical_html_pairs']}. "
        "HTML-модель не может различить входы такой пары. Предпочтение оценщиков "
        "для неё не объясняется используемыми признаками; этот случай сохраняется "
        "в исходном benchmark и учитывается как ограничение разметки.",
        f"Отчёт о фильтрации: `{summary.get('counts', {})}`.",
        f"Число оценок на пару до фильтрации: `{annotations.get('ratings_per_pair', {})}`. "
        f"Пропуски исходных полей: `{annotations.get('missing_required_values', {})}`.",
        "",
        "## Победители и согласие оценщиков",
        "",
        f"Доля побед A после воспроизводимой перестановки — {summary['a_win_share']:.3f}. "
        f"Средняя доля голосов за победителя — {summary['mean_agreement']:.3f}. "
        "Ничьи и отсутствие большинства исключаются из обучения, но остаются "
        "в анализе исходной разметки.",
        "![Распределения](distributions.png)",
        "",
        "## Критерии оценки",
        "",
        "A/B в этом разделе обозначают исходные стороны анкеты. "
        "Согласие с overallWinner рассчитано только для оценок, где оба выбора "
        "не являются ничьёй; это описательная связь, а не причинный эффект.",
        "Средняя доля наиболее частого ответа в паре отдельно характеризует "
        "согласие между оценщиками. Здесь ничья считается отдельным ответом, "
        "поэтому единогласная ничья означает согласие, но не задаёт победителя.",
    ]
    if not criteria.empty:
        most_ties = criteria.loc[criteria["tie_share"].idxmax()]
        comparable = criteria.dropna(subset=["overall_agreement"])
        agreement_text = "Сопоставимых оценок без ничьих нет."
        if not comparable.empty:
            closest = comparable.loc[comparable["overall_agreement"].idxmax()]
            agreement_text = (
                f"Наибольшее согласие с общим выбором: `{closest['criterion']}` "
                f"({closest['overall_agreement']:.1%})."
            )
        lines.extend(
            [
                f"Наибольшая доля ничьих: `{most_ties['criterion']}` "
                f"({most_ties['tie_share']:.1%}). {agreement_text}",
                markdown_table(criteria),
                "![Критерии](annotations.png)",
            ]
        )
    lines.extend(
        [
            "",
            "## Покрытие и размеры документов",
            "",
            f"Число пар на запрос: от {summary['pairs_per_prompt_min']} "
            f"до {summary['pairs_per_prompt_max']}. "
            "Число документов по источникам показано после фильтрации пар. "
            "Взаимное сравнение генераторов неравномерно: набор пар ограничен "
            "дизайном исходного исследования, поэтому общий win rate генератора "
            "нельзя интерпретировать как универсальный рейтинг.",
            "![Покрытие](coverage.png)",
            "Размеры HTML рассчитаны по уникальным документам. "
            "Широкий диапазон размеров обосновывает преобразование счётчиков log1p. "
            "Длина HTML сама по себе не измеряет качество интерфейса.",
            "![Размеры](documents.png)",
            "",
            "## Разбиение и возможная утечка",
            "",
            markdown_table(tables["splits"].reset_index()),
            markdown_table(overlap),
            "Разбиение выполняется по запросам до зеркалирования train. "
            "Кроме query_id проверяются пересечения путей и SHA256 содержимого HTML.",
            "![Разбиение](splits.png)",
            "",
            "## Признаки на обучающей выборке",
            "",
            "Этот раздел использует только исходный train, до зеркалирования. "
            "Для каждой пары признаки победителя сравниваются с признаками проигравшего. "
            "Доли показывают направление различия, но не доказывают значимость "
            "или причинность: пары одного запроса зависимы.",
            markdown_table(preferences.loc[SELECTED_FEATURES].reset_index()),
            "![Признаки и предпочтения](feature_preferences.png)",
            "Корреляции Spearman рассчитаны по уникальным документам train. "
            "Связанные счётчики описывают близкие свойства HTML; регуляризация "
            "уменьшает чувствительность регрессии к избыточным признакам.",
            "![Корреляции признаков](feature_correlations.png)",
            "",
            "## Ограничения и решения",
            "",
            "Данные малы и не являются независимой проверкой продукта в эксплуатации. "
            "Большинство голосов скрывает часть разногласий людей. "
            "Используются HTML-признаки; JavaScript и внешние ресурсы не исполняются, "
            "визуальное качество по скриншотам не оценивается. "
            "Лицензия исходных данных — CC BY-NC 4.0.",
            "Для моделей фиксируются групповой split, порог 0.5 и seed 42. "
            "Основная метрика — balanced accuracy. Сравниваются baseline, "
            "структурные признаки и расширенные признаки. Выбор выполняется "
            "по validation, test используется после выбора. "
            "Результаты EDA не используются для повторной настройки по test.",
            "",
        ]
    )
    return "\n\n".join(lines)


def build_analysis(
    frame: pd.DataFrame, splits: dict[str, pd.DataFrame], manifest: dict[str, Any]
) -> Analysis:
    documents = document_table(frame)
    train_documents = document_table(splits["train"])
    split_sizes, overlap = split_tables(splits)
    criteria = criterion_table(manifest.get("annotations", {}))
    preferences = feature_preferences(splits["train"])
    correlations = train_documents[SELECTED_FEATURES].corr(method="spearman")
    coverage = frame.groupby("prompt_id").size().rename("pairs")
    families = documents.groupby("source_family").agg(
        documents=("document", "size"), median_characters=("characters", "median")
    )
    summary = {
        **manifest,
        "dataset_digest": content_digest(frame),
        "pairs": len(frame),
        "prompts": int(frame["prompt_id"].nunique()),
        "a_win_share": float(frame["target"].mean()),
        "mean_agreement": float(frame["agreement"].mean()),
        "pairs_per_prompt_min": int(coverage.min()),
        "pairs_per_prompt_max": int(coverage.max()),
        "feature_analysis_scope": "original training split only",
        "split": {
            name: {"pairs": len(part), "prompts": int(part["prompt_id"].nunique())}
            for name, part in splits.items()
        },
        "quality": {
            "unique_documents": len(documents),
            "unique_html_digests": int(documents["sha256"].nunique()),
            "blank_html": int(documents["blank_html"].sum()),
            "duplicate_pair_ids": int(frame["pair_id"].duplicated().sum()),
            "missing_model_values": int(
                frame[["prompt", "html_a", "html_b", "target"]].isna().sum().sum()
            ),
            "identical_html_pairs": int((frame["html_a"] == frame["html_b"]).sum()),
        },
        "limitations": [
            "small benchmark and dependent pairs within prompts",
            "majority labels exclude ties and compress annotator disagreement",
            "HTML features do not evaluate screenshots or JavaScript behavior",
            "source comparisons are constrained by the benchmark pairing design",
            "CC-BY-NC-4.0 non-commercial research data",
        ],
    }
    tables = {
        "documents": documents,
        "source_coverage": families,
        "prompt_coverage": coverage.to_frame(),
        "criteria": criteria,
        "splits": split_sizes,
        "split_overlap": overlap,
        "feature_preferences": preferences,
        "training_feature_statistics": train_documents[FEATURES].describe().T,
        "training_feature_correlations": correlations,
    }
    figures = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    frame["target"].value_counts().reindex([0, 1], fill_value=0).plot.bar(
        ax=axes[0], rot=0, color=["steelblue", "darkorange"]
    )
    axes[0].set(xticklabels=["B", "A"], title="Majority winner", ylabel="Pairs")
    axes[1].hist(frame["agreement"], bins=[0.5, 0.7, 0.9, 1.01])
    axes[1].set(title="Annotator agreement", xlabel="Winning vote share", ylabel="Pairs")
    votes = manifest.get("annotations", {}).get("overall_votes", {})
    if votes:
        axes[2].bar(["A", "B", "tie"], [votes.get(label, 0) for label in ("A", "B", "tie")])
        axes[2].set(title="Raw overall votes", ylabel="Ratings")
    else:
        axes[2].hist(np.log10(documents["characters"].clip(lower=1)), bins=20)
        axes[2].set(title="Unique HTML size", xlabel="log10(characters)", ylabel="Documents")
    figures["distributions"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    split_sizes[["pairs", "prompts"]].plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Prompt-disjoint split", ylabel="Count")
    split_sizes["a_win_share"].plot.bar(ax=axes[1], rot=0, ylim=(0, 1))
    axes[1].axhline(0.5, color="grey", linestyle="--")
    axes[1].set(title="Class balance by split", ylabel="A-win share")
    figures["splits"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    coverage.value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
    axes[0].set(title="Retained pairs per prompt", xlabel="Pairs", ylabel="Prompts")
    families["documents"].rename(index=SOURCE_LABELS).sort_values().plot.barh(ax=axes[1])
    axes[1].set(title="Unique documents by source", xlabel="Documents", ylabel="")
    figures["coverage"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(np.log10(documents["characters"].clip(lower=1)), bins=20)
    axes[0].set(title="Unique HTML document size", xlabel="log10(characters)", ylabel="Documents")
    prompts = frame.drop_duplicates("prompt_id")["prompt"].str.split().str.len()
    axes[1].hist(prompts, bins=15)
    axes[1].set(title="Unique prompt size", xlabel="Words", ylabel="Prompts")
    figures["documents"] = fig
    if not criteria.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = criteria["criterion"].map(lambda value: CRITERIA.get(value, value))
        plot = criteria.set_index(labels)
        plot[["a_share", "b_share", "tie_share"]].plot.barh(
            stacked=True, ax=axes[0], color=["darkorange", "steelblue", "lightgrey"]
        )
        axes[0].set(title="Raw criterion vote shares", xlabel="Share", ylabel="", xlim=(0, 1))
        axes[0].legend(["A", "B", "Tie"], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        plot[["overall_agreement", "mean_top_vote_share"]].plot.barh(
            ax=axes[1], color=["seagreen", "darkorange"]
        )
        axes[1].set(
            title="Agreement measures by criterion",
            xlabel="Agreement",
            ylabel="",
            xlim=(0, 1),
        )
        axes[1].legend(
            ["With overall (non-tie)", "Between annotators (incl. ties)"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
        )
        figures["annotations"] = fig
    fig, ax = plt.subplots(figsize=(9, 4.5))
    preferences.loc[
        SELECTED_FEATURES, ["winner_more_share", "equal_share", "winner_less_share"]
    ].plot.barh(stacked=True, ax=ax, color=["seagreen", "lightgrey", "salmon"])
    ax.set(
        title="Winner vs loser features: original train pairs",
        xlabel="Pair share",
        ylabel="",
        xlim=(0, 1),
    )
    ax.legend(
        ["Winner greater", "Equal", "Winner smaller"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
    )
    figures["feature_preferences"] = fig
    fig, ax = plt.subplots(figsize=(9, 7))
    heatmap = ax.imshow(correlations.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(SELECTED_FEATURES)), SELECTED_FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(SELECTED_FEATURES)), SELECTED_FEATURES)
    ax.set_title("Spearman correlations: unique training documents")
    for row in range(len(SELECTED_FEATURES)):
        for column in range(len(SELECTED_FEATURES)):
            value = correlations.iloc[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > 0.65 else "black",
                )
    fig.colorbar(heatmap, ax=ax, shrink=0.7)
    figures["feature_correlations"] = fig
    for figure in figures.values():
        figure.tight_layout()
        plt.close(figure)
    return Analysis(summary, tables, figures, render_report(summary, tables))


def log_analysis(analysis: Analysis) -> None:
    mlflow.log_dict(analysis.summary, "eda/summary.json")
    mlflow.log_text(analysis.report, "eda/report.md")
    for name, table in analysis.tables.items():
        mlflow.log_text(
            table.to_csv(index=name not in {"documents", "criteria", "split_overlap"}),
            f"eda/tables/{name}.csv",
        )
    for name, figure in analysis.figures.items():
        mlflow.log_figure(figure, f"eda/{name}.png")
