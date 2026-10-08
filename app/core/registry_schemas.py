from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


ExperimentStatus = Literal[
    "RUNNING",
    "COMPLETED",
    "FAILED",
]


class ExperimentArtifact(BaseModel):
    """
    Artefact produit pendant une expérience.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    artifact_name: str = Field(
        min_length=1,
    )

    artifact_type: str = Field(
        min_length=1,
    )

    artifact_path: str = Field(
        min_length=1,
    )


class ExperimentCandidateMetric(BaseModel):
    """
    Résumé des métriques d'un modèle candidat.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    passed: bool

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    f1_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    poor_recall: float = Field(
        ge=0.0,
        le=1.0,
    )

    prediction_latency_seconds: float = Field(
        ge=0.0,
    )

    training_duration_seconds: float = Field(
        ge=0.0,
    )


class ExperimentRecord(BaseModel):
    """
    Enregistrement complet d'une expérience ML.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    experiment_id: str = Field(
        min_length=1,
    )

    created_at: str = Field(
        min_length=1,
    )

    completed_at: str | None = None

    status: ExperimentStatus

    random_seed: int = Field(
        ge=0,
    )

    dataset_path: str = Field(
        min_length=1,
    )

    candidate_count: int = Field(
        ge=0,
    )

    champion_model_name: str | None = None
    champion_model_version: str | None = None

    candidates: list[
        ExperimentCandidateMetric
    ]

    artifacts: list[
        ExperimentArtifact
    ]

    error_message: str | None = None


class ExperimentIndexEntry(BaseModel):
    """
    Entrée légère stockée dans l'index du registre.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    experiment_id: str = Field(
        min_length=1,
    )

    created_at: str = Field(
        min_length=1,
    )

    completed_at: str | None = None

    status: ExperimentStatus

    champion_model_name: str | None = None
    champion_model_version: str | None = None

    experiment_path: str = Field(
        min_length=1,
    )


class ExperimentRegistryIndex(BaseModel):
    """
    Index global des expériences enregistrées.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    registry_version: str = Field(
        default="1.0.0",
        min_length=1,
    )

    experiment_count: int = Field(
        default=0,
        ge=0,
    )

    experiments: list[
        ExperimentIndexEntry
    ] = Field(
        default_factory=list,
    )