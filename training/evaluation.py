from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.core.model_schemas import (
    ClassMetrics,
    ClassificationMetrics,
    ModelEvaluationConfig,
    ModelEvaluationResult,
)


EXPECTED_LABELS = [
    "GOOD",
    "ACCEPTABLE",
    "POOR",
]


def validate_evaluation_data(
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
) -> None:
    """
    Vérifie que les données d'évaluation
    peuvent être utilisées.
    """

    if X_evaluation.empty:
        raise ValueError(
            "X_evaluation ne peut pas être vide."
        )

    if y_evaluation.empty:
        raise ValueError(
            "y_evaluation ne peut pas être vide."
        )

    if len(X_evaluation) != len(y_evaluation):
        raise ValueError(
            "X_evaluation et y_evaluation doivent "
            "contenir le même nombre d'observations."
        )

    if X_evaluation.isna().any().any():
        raise ValueError(
            "X_evaluation contient des valeurs manquantes."
        )

    if y_evaluation.isna().any():
        raise ValueError(
            "y_evaluation contient des valeurs manquantes."
        )

    actual_labels = set(
        y_evaluation.astype(str).unique()
    )

    unknown_labels = (
        actual_labels
        - set(EXPECTED_LABELS)
    )

    if unknown_labels:
        raise ValueError(
            "y_evaluation contient des labels inconnus : "
            f"{sorted(unknown_labels)}"
        )


def validate_predictions(
    predictions: np.ndarray,
    expected_count: int,
) -> None:
    """
    Vérifie les prédictions produites par le modèle.
    """

    if len(predictions) != expected_count:
        raise ValueError(
            "Le nombre de prédictions ne correspond pas "
            "au nombre d'observations à évaluer."
        )

    prediction_labels = {
        str(prediction)
        for prediction in predictions
    }

    unknown_predictions = (
        prediction_labels
        - set(EXPECTED_LABELS)
    )

    if unknown_predictions:
        raise ValueError(
            "Le modèle a produit des labels inconnus : "
            f"{sorted(unknown_predictions)}"
        )


def calculate_metrics_by_class(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, ClassMetrics]:
    """
    Calcule la précision, le recall et le F1-score
    pour chaque classe.
    """

    report = classification_report(
        y_true,
        predictions,
        labels=EXPECTED_LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics_by_class: dict[
        str,
        ClassMetrics,
    ] = {}

    for label in EXPECTED_LABELS:
        label_metrics = report[label]

        metrics_by_class[label] = ClassMetrics(
            precision=float(
                label_metrics["precision"]
            ),
            recall=float(
                label_metrics["recall"]
            ),
            f1_score=float(
                label_metrics["f1-score"]
            ),
            support=int(
                label_metrics["support"]
            ),
        )

    return metrics_by_class


def calculate_classification_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    prediction_latency_seconds: float,
) -> ClassificationMetrics:
    """
    Calcule les métriques globales et par classe.
    """

    metrics_by_class = (
        calculate_metrics_by_class(
            y_true=y_true,
            predictions=predictions,
        )
    )

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        precision_macro=float(
            precision_score(
                y_true,
                predictions,
                labels=EXPECTED_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        recall_macro=float(
            recall_score(
                y_true,
                predictions,
                labels=EXPECTED_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        f1_macro=float(
            f1_score(
                y_true,
                predictions,
                labels=EXPECTED_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        prediction_count=len(
            predictions
        ),
        prediction_latency_seconds=float(
            prediction_latency_seconds
        ),
        metrics_by_class=metrics_by_class,
    )


def evaluate_acceptance_thresholds(
    metrics: ClassificationMetrics,
    evaluation_config: ModelEvaluationConfig,
) -> list[str]:
    """
    Vérifie si le modèle respecte les seuils
    définis dans la configuration.

    Une liste vide signifie que le modèle
    respecte tous les seuils.
    """

    rejection_reasons: list[str] = []

    if (
        metrics.f1_macro
        < evaluation_config.minimum_f1_macro
    ):
        rejection_reasons.append(
            "F1 macro inférieur au seuil minimum : "
            f"{metrics.f1_macro:.4f} < "
            f"{evaluation_config.minimum_f1_macro:.4f}"
        )

    poor_recall = (
        metrics
        .metrics_by_class["POOR"]
        .recall
    )

    if (
        poor_recall
        < evaluation_config.minimum_poor_recall
    ):
        rejection_reasons.append(
            "Recall POOR inférieur au seuil minimum : "
            f"{poor_recall:.4f} < "
            f"{evaluation_config.minimum_poor_recall:.4f}"
        )

    return rejection_reasons


def calculate_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> list[list[int]]:
    """
    Calcule la matrice de confusion.

    Les lignes correspondent aux vraies classes.
    Les colonnes correspondent aux prédictions.

    L'ordre est :
    GOOD, ACCEPTABLE, POOR.
    """

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=EXPECTED_LABELS,
    )

    return [
        [
            int(value)
            for value in row
        ]
        for row in matrix.tolist()
    ]


def evaluate_classifier(
    *,
    model: Any,
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    model_name: str,
    model_version: str,
    evaluation_config: ModelEvaluationConfig,
    evaluated_split: str = "validation",
) -> ModelEvaluationResult:
    """
    Évalue complètement un modèle de classification.
    """

    if not model_name.strip():
        raise ValueError(
            "Le nom du modèle ne peut pas être vide."
        )

    if not model_version.strip():
        raise ValueError(
            "La version du modèle ne peut pas être vide."
        )

    if not evaluated_split.strip():
        raise ValueError(
            "Le nom de l'ensemble évalué "
            "ne peut pas être vide."
        )

    if not hasattr(model, "predict"):
        raise TypeError(
            "Le modèle doit posséder une méthode predict."
        )

    validate_evaluation_data(
        X_evaluation=X_evaluation,
        y_evaluation=y_evaluation,
    )

    prediction_start = perf_counter()

    predictions = np.asarray(
        model.predict(
            X_evaluation
        ),
        dtype=str,
    )

    prediction_latency_seconds = (
        perf_counter()
        - prediction_start
    )

    validate_predictions(
        predictions=predictions,
        expected_count=len(
            y_evaluation
        ),
    )

    metrics = calculate_classification_metrics(
        y_true=y_evaluation,
        predictions=predictions,
        prediction_latency_seconds=(
            prediction_latency_seconds
        ),
    )

    rejection_reasons = (
        evaluate_acceptance_thresholds(
            metrics=metrics,
            evaluation_config=(
                evaluation_config
            ),
        )
    )

    matrix = calculate_confusion_matrix(
        y_true=y_evaluation,
        predictions=predictions,
    )

    return ModelEvaluationResult(
        model_name=model_name,
        model_version=model_version,
        evaluated_split=evaluated_split,
        passed=len(
            rejection_reasons
        ) == 0,
        rejection_reasons=rejection_reasons,
        metrics=metrics,
        labels=EXPECTED_LABELS.copy(),
        confusion_matrix=matrix,
    )