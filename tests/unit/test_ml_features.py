from __future__ import annotations

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from genui_quality_gate.ml.features import (
    CONTENT_FEATURES,
    STRUCTURAL_FEATURES,
    PairFeatures,
    build_pipeline,
    document_features,
)


def example() -> pd.DataFrame:
    rich = '<main><h1>Register</h1><label for="email">Email</label><input id="email"></main>'
    plain = "<div>Register</div>"
    return pd.DataFrame(
        {
            "prompt": ["Register email", "Register email"],
            "html_a": [rich, plain],
            "html_b": [plain, rich],
        }
    )


def test_feature_difference_is_antisymmetric() -> None:
    transformer = PairFeatures(extended=True)
    features = transformer.fit_transform(example())
    assert features.shape == (2, len(STRUCTURAL_FEATURES) + len(CONTENT_FEATURES))
    np.testing.assert_allclose(features[0], -features[1])
    assert len(transformer.get_feature_names_out()) == features.shape[1]


def test_feature_extraction_ignores_script_text_and_never_executes_it() -> None:
    html = '<script>throw new Error("register")</script><style>.register{}</style><p>Nothing</p>'
    features = document_features("Register", html, True)
    overlap = (STRUCTURAL_FEATURES + CONTENT_FEATURES).index("prompt_overlap")
    assert features[overlap] == 0
    assert np.isfinite(features).all()


def test_empty_and_malformed_html_are_supported() -> None:
    assert np.isfinite(document_features("", "", True)).all()
    assert np.isfinite(document_features("abc", "<main><input", True)).all()
    assert len(document_features("", "", False)) == len(STRUCTURAL_FEATURES)


def test_accessibility_features_and_embedded_images() -> None:
    html = (
        '<meta name="viewport"><style>@media (width>10px){}</style>'
        '<noscript><img alt="fallback"></noscript><img alt=""><input aria-label="email">'
    )
    values = dict(
        zip(
            STRUCTURAL_FEATURES + CONTENT_FEATURES,
            document_features("email", html, True),
            strict=True,
        )
    )
    assert values["image_alt_ratio"] == 1
    assert values["accessible_input_ratio"] == 1
    assert values["has_viewport"] == 1
    assert values["media_query_count"] > 0


@pytest.mark.parametrize("variant", ["structure", "content"])
def test_model_probabilities_are_symmetric_and_serializable(variant: str) -> None:
    frame = example()
    pipeline = build_pipeline(variant).fit(frame, [1, 0])
    probability = pipeline.predict_proba(frame)[:, 1]
    assert probability[0] > 0.5
    assert probability.sum() == pytest.approx(1)
    restored = cloudpickle.loads(cloudpickle.dumps(pipeline))
    np.testing.assert_allclose(restored.predict_proba(frame), pipeline.predict_proba(frame))
    identical = frame.iloc[:1].copy()
    identical["html_b"] = identical["html_a"]
    assert pipeline.predict_proba(identical)[0, 1] == pytest.approx(0.5)


def test_baseline_is_chance_and_unknown_variant_rejected() -> None:
    pipeline = build_pipeline("baseline").fit(example(), [1, 0])
    np.testing.assert_allclose(pipeline.predict_proba(example()), 0.5)
    with pytest.raises(ValueError, match="Unknown"):
        build_pipeline("bad")
