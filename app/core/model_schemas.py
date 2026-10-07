from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


PrimaryMetric = Literal[
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]


class BaselineTrainingConfig(BaseModel):
    """
    Paramètres d'entraînement de la régression logistique.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    maximum_iterations: int = Field(
        default=1000,
        ge=1,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
    )

    class_weight: str | None = None


class ModelEvaluationConfig(BaseModel):
    """
    Seuils d'acceptation du modèle.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    primary_metric: PrimaryMetric = "f1_macro"

    minimum_f1_macro: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    minimum_poor_recall: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )


class ModelInputConfig(BaseModel):
    """
    Emplacement des données préparées.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    processed_data_directory: Path


class ModelOutputConfig(BaseModel):
    """
    Emplacements des artefacts du modèle.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_directory: Path
    evaluation_directory: Path


class BaselineModelConfig(BaseModel):
    """
    Configuration complète du modèle baseline.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    training: BaselineTrainingConfig
    evaluation: ModelEvaluationConfig
    input: ModelInputConfig
    output: ModelOutputConfig


class ClassMetrics(BaseModel):
    """
    Métriques calculées pour une classe.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    precision: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    f1_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    support: int = Field(
        ge=0,
    )


class ClassificationMetrics(BaseModel):
    """
    Métriques globales et métriques par classe.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    precision_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    f1_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    prediction_count: int = Field(
        ge=0,
    )

    prediction_latency_seconds: float = Field(
        ge=0.0,
    )

    metrics_by_class: dict[
        str,
        ClassMetrics,
    ]


class ModelEvaluationResult(BaseModel):
    """
    Résultat complet de l'évaluation.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_name: str
    model_version: str
    evaluated_split: str

    passed: bool
    rejection_reasons: list[str]

    metrics: ClassificationMetrics

    labels: list[str]
    confusion_matrix: list[list[int]]