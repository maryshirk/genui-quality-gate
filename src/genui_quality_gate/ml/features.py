from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STRUCTURAL_FEATURES = [
    "html_length",
    "tag_count",
    "interactive_count",
    "heading_count",
    "image_count",
    "link_count",
    "form_count",
    "semantic_count",
]
CONTENT_FEATURES = [
    "text_length",
    "unique_words",
    "prompt_overlap",
    "image_alt_ratio",
    "accessible_input_ratio",
    "aria_count",
    "has_viewport",
    "media_query_count",
]


def document_features(prompt: str, html: str, extended: bool) -> NDArray[np.float64]:
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all(True)
    images = soup.find_all("img")
    values = [
        len(html),
        len(tags),
        len(soup.find_all(["button", "input", "select", "textarea"])),
        len(soup.find_all(re.compile(r"^h[1-6]$"))),
        len(images),
        len(soup.find_all("a")),
        len(soup.find_all("form")),
        len(soup.find_all(["main", "nav", "section", "article", "header", "footer"])),
    ]
    structural = np.log1p(values)
    if not extended:
        return structural
    inputs = soup.find_all(["input", "select", "textarea"])
    labels = {label.get("for") for label in soup.find_all("label", attrs={"for": True})}
    accessible = sum(
        bool(field.get("aria-label") or field.get("aria-labelledby") or field.get("id") in labels)
        for field in inputs
    )
    aria_count = sum(any(str(key).startswith("aria-") for key in tag.attrs) for tag in tags)
    viewport = float(soup.find("meta", attrs={"name": "viewport"}) is not None)
    media_queries = len(re.findall(r"@media\b", html))
    alt_ratio = sum(image.has_attr("alt") for image in images) / max(1, len(images))
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text_words = set(re.findall(r"[a-z]{3,}", text.lower()))
    prompt_words = set(re.findall(r"[a-z]{3,}", prompt.lower()))
    content = [
        np.log1p(len(text)),
        np.log1p(len(text_words)),
        len(text_words & prompt_words) / max(1, len(prompt_words)),
        alt_ratio,
        accessible / max(1, len(inputs)),
        np.log1p(aria_count),
        viewport,
        np.log1p(media_queries),
    ]
    return np.concatenate([structural, np.asarray(content, dtype=np.float64)])


class PairFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, extended: bool = False) -> None:
        self.extended = extended

    def fit(self, frame: pd.DataFrame, target: Any = None) -> PairFeatures:
        return self

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float64]:
        return np.asarray(
            [
                document_features(prompt, first, self.extended)
                - document_features(prompt, second, self.extended)
                for prompt, first, second in frame[["prompt", "html_a", "html_b"]].itertuples(
                    index=False, name=None
                )
            ],
            dtype=np.float64,
        )

    def get_feature_names_out(self, input_features: Any = None) -> NDArray[np.str_]:
        names = STRUCTURAL_FEATURES + (CONTENT_FEATURES if self.extended else [])
        return np.asarray(names, dtype=str)


def build_pipeline(variant: str) -> Pipeline:
    if variant not in {"baseline", "structure", "content"}:
        raise ValueError("Unknown model variant")
    classifier = (
        DummyClassifier(strategy="prior")
        if variant == "baseline"
        else LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000, random_state=42)
    )
    return Pipeline(
        [
            ("features", PairFeatures(extended=variant == "content")),
            ("scale", StandardScaler(with_mean=False)),
            ("classifier", classifier),
        ]
    )
