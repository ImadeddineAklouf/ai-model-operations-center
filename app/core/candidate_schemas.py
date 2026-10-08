from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class LogisticRegressionCandidateConfig(BaseModel):
    """
    Configuration de la régression logistique candidate.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    enabled: bool = True

    version: str = Field(
        default="1.0.0",
        min_length=1,
    )

    maximum_iterations: int = Field(
        default=1000,
        ge=1,
    )

    class_weight: str | None = None


class RandomForestCandidateConfig(BaseModel):
    """
    Configuration du Random Forest.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    enabled: bool = True

    version: str = Field(
        default="1.0.0",
        min_length=1,
    )

    number_of_estimators: int = Field(
        default=200,
        ge=1,
    )

    maximum_depth: int | None = Field(
        default=None,
        ge=1,
    )

    minimum_samples_split: int = Field(
        default=2,
        ge=2,
    )

    minimum_samples_leaf: int = Field(
        default=1,
        ge=1,
    )

    class_weight: str | None = None


class GradientBoostingCandidateConfig(BaseModel):
    """
    Configuration du Gradient Boosting.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    enabled: bool = True

    version: str = Field(
        default="1.0.0",
        min_length=1,
    )

    number_of_estimators: int = Field(
        default=100,
        ge=1,
    )

    learning_rate: float = Field(
        default=0.10,
        gt=0.0,
    )

    maximum_depth: int = Field(
        default=3,
        ge=1,
    )


class CandidateModelsConfig(BaseModel):
    """
    Configuration des trois modèles candidats.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    logistic_regression: (
        LogisticRegressionCandidateConfig
    )

    random_forest: RandomForestCandidateConfig

    gradient_boosting: (
        GradientBoostingCandidateConfig
    )

    @model_validator(mode="after")
    def validate_enabled_models(
        self,
    ) -> "CandidateModelsConfig":
        """
        Vérifie qu'au moins un modèle est activé.
        """

        enabled_models = [
            self.logistic_regression.enabled,
            self.random_forest.enabled,
            self.gradient_boosting.enabled,
        ]

        if not any(enabled_models):
            raise ValueError(
                "Au moins un modèle candidat "
                "doit être activé."
            )

        return self


class CandidateEvaluationConfig(BaseModel):
    """
    Seuils communs appliqués à tous les modèles.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

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


class CandidateInputConfig(BaseModel):
    """
    Emplacement des données préparées.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    processed_data_directory: Path


class CandidateOutputConfig(BaseModel):
    """
    Emplacements des modèles et rapports.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    candidates_directory: Path
    comparison_directory: Path


class CandidateExperimentConfig(BaseModel):
    """
    Configuration complète de l'expérience comparative.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
    )

    models: CandidateModelsConfig
    evaluation: CandidateEvaluationConfig
    input: CandidateInputConfig
    output: CandidateOutputConfig

class CandidateModelResult(BaseModel):
    """
    Résultat synthétique d'un modèle candidat.
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

    rejection_reasons: list[str]


class CandidateComparisonReport(BaseModel):
    """
    Rapport final de comparaison des modèles candidats.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    champion_model_name: str | None

    champion_model_version: str | None

    candidate_count: int = Field(
        ge=1,
    )

    validated_candidate_count: int = Field(
        ge=0,
    )

    ranking: list[CandidateModelResult]