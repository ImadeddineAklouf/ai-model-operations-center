from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QualityLabel(StrEnum):
    """
    Classe de qualité attribuée à un dataset.
    """

    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"


class DatasetQualityFeatures(BaseModel):
    """
    Métriques techniques décrivant la qualité d'un dataset.

    Ces champs deviendront les variables d'entrée du modèle
    de machine learning.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    row_count: int = Field(
        ge=1,
        description="Nombre total de lignes du dataset.",
    )

    column_count: int = Field(
        ge=1,
        description="Nombre total de colonnes du dataset.",
    )

    missing_value_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion de valeurs manquantes.",
    )

    duplicate_row_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion de lignes dupliquées.",
    )

    invalid_type_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion de valeurs avec un type invalide.",
    )

    primary_key_uniqueness: float = Field(
        ge=0.0,
        le=1.0,
        description="Taux d'unicité de la clé primaire.",
    )

    foreign_key_match_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Taux de correspondance des clés étrangères.",
    )

    schema_change_count: int = Field(
        ge=0,
        description="Nombre de changements de schéma détectés.",
    )

    outlier_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion de valeurs aberrantes.",
    )

    freshness_delay_hours: float = Field(
        ge=0.0,
        description="Retard de fraîcheur en heures.",
    )


class DatasetQualityRecord(DatasetQualityFeatures):
    """
    Observation complète utilisée pour l'entraînement.

    Elle contient les features ainsi que la classe cible.
    """

    quality_label: QualityLabel


class DatasetGenerationConfig(BaseModel):
    """
    Configuration de la génération du dataset synthétique.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    number_of_records: int = Field(
        default=3000,
        ge=30,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
    )

    good_ratio: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
    )

    acceptable_ratio: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )

    poor_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_class_ratios(
        self,
    ) -> "DatasetGenerationConfig":
        """
        Vérifie que la somme des proportions vaut exactement 1.
        """

        ratio_sum = (
            self.good_ratio
            + self.acceptable_ratio
            + self.poor_ratio
        )

        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(
                "La somme des proportions de classes "
                "doit être égale à 1."
            )

        return self