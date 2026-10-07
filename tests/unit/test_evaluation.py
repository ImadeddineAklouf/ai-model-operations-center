import numpy as np
import pandas as pd
import pytest

from app.core.model_schemas import (
    ModelEvaluationConfig,
)
from training.evaluation import (
    EXPECTED_LABELS,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    evaluate_classifier,
    validate_evaluation_data,
    validate_predictions,
)


class FixedPredictionModel:
    """
    Modèle factice utilisé pour les tests.
    """

    def __init__(
        self,
        predictions: list[str],
    ) -> None:
        self.predictions = np.asarray(
            predictions,
            dtype=str,
        )

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        if len(features) != len(
            self.predictions
        ):
            raise ValueError(
                "Nombre d'observations incorrect."
            )

        return self.predictions


def create_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "feature_b": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
        }
    )


def create_target() -> pd.Series:
    return pd.Series(
        [
            "GOOD",
            "GOOD",
            "ACCEPTABLE",
            "ACCEPTABLE",
            "POOR",
            "POOR",
        ],
        name="quality_label",
    )


def test_expected_labels() -> None:
    assert EXPECTED_LABELS == [
        "GOOD",
        "ACCEPTABLE",
        "POOR",
    ]


def test_valid_evaluation_data() -> None:
    validate_evaluation_data(
        X_evaluation=create_features(),
        y_evaluation=create_target(),
    )


def test_empty_features_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="X_evaluation",
    ):
        validate_evaluation_data(
            X_evaluation=pd.DataFrame(),
            y_evaluation=create_target(),
        )


def test_misaligned_data_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="même nombre",
    ):
        validate_evaluation_data(
            X_evaluation=create_features(),
            y_evaluation=create_target().iloc[:-1],
        )


def test_unknown_target_is_rejected() -> None:
    target = create_target()
    target.iloc[0] = "UNKNOWN"

    with pytest.raises(
        ValueError,
        match="labels inconnus",
    ):
        validate_evaluation_data(
            X_evaluation=create_features(),
            y_evaluation=target,
        )


def test_incorrect_prediction_count_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="nombre de prédictions",
    ):
        validate_predictions(
            predictions=np.asarray(
                ["GOOD"],
                dtype=str,
            ),
            expected_count=2,
        )


def test_unknown_prediction_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="labels inconnus",
    ):
        validate_predictions(
            predictions=np.asarray(
                [
                    "GOOD",
                    "UNKNOWN",
                ],
                dtype=str,
            ),
            expected_count=2,
        )


def test_perfect_metrics() -> None:
    target = create_target()

    predictions = target.to_numpy(
        dtype=str
    )

    metrics = calculate_classification_metrics(
        y_true=target,
        predictions=predictions,
        prediction_latency_seconds=0.01,
    )

    assert metrics.accuracy == pytest.approx(
        1.0
    )

    assert metrics.precision_macro == pytest.approx(
        1.0
    )

    assert metrics.recall_macro == pytest.approx(
        1.0
    )

    assert metrics.f1_macro == pytest.approx(
        1.0
    )

    assert (
        metrics.metrics_by_class["POOR"].recall
        == pytest.approx(1.0)
    )


def test_perfect_confusion_matrix() -> None:
    target = create_target()

    matrix = calculate_confusion_matrix(
        y_true=target,
        predictions=target.to_numpy(
            dtype=str
        ),
    )

    assert matrix == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]


def test_perfect_model_passes_evaluation() -> None:
    features = create_features()
    target = create_target()

    model = FixedPredictionModel(
        target.tolist()
    )

    result = evaluate_classifier(
        model=model,
        X_evaluation=features,
        y_evaluation=target,
        model_name="test-model",
        model_version="1.0.0",
        evaluation_config=(
            ModelEvaluationConfig(
                minimum_f1_macro=0.80,
                minimum_poor_recall=0.85,
            )
        ),
        evaluated_split="validation",
    )

    assert result.passed is True
    assert result.rejection_reasons == []

    assert result.metrics.accuracy == pytest.approx(
        1.0
    )

    assert result.metrics.f1_macro == pytest.approx(
        1.0
    )

    assert result.labels == EXPECTED_LABELS

    assert result.confusion_matrix == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]


def test_model_without_predict_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="méthode predict",
    ):
        evaluate_classifier(
            model=object(),
            X_evaluation=create_features(),
            y_evaluation=create_target(),
            model_name="invalid-model",
            model_version="1.0.0",
            evaluation_config=(
                ModelEvaluationConfig()
            ),
        )