from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.candidate_schemas import (
    CandidateEvaluationConfig,
    CandidateExperimentConfig,
    CandidateInputConfig,
    CandidateModelsConfig,
    CandidateOutputConfig,
    GradientBoostingCandidateConfig,
    LogisticRegressionCandidateConfig,
    RandomForestCandidateConfig,
)


def create_models_config() -> CandidateModelsConfig:
    """
    Construit une configuration valide des modèles.
    """

    return CandidateModelsConfig(
        logistic_regression=(
            LogisticRegressionCandidateConfig()
        ),
        random_forest=(
            RandomForestCandidateConfig()
        ),
        gradient_boosting=(
            GradientBoostingCandidateConfig()
        ),
    )


def test_default_logistic_regression_config() -> None:
    config = LogisticRegressionCandidateConfig()

    assert config.enabled is True
    assert config.version == "1.0.0"
    assert config.maximum_iterations == 1000
    assert config.class_weight is None


def test_default_random_forest_config() -> None:
    config = RandomForestCandidateConfig()

    assert config.enabled is True
    assert config.number_of_estimators == 200
    assert config.maximum_depth is None
    assert config.minimum_samples_split == 2
    assert config.minimum_samples_leaf == 1


def test_invalid_number_of_estimators_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RandomForestCandidateConfig(
            number_of_estimators=0
        )


def test_default_gradient_boosting_config() -> None:
    config = GradientBoostingCandidateConfig()

    assert config.enabled is True
    assert config.number_of_estimators == 100

    assert config.learning_rate == pytest.approx(
        0.10
    )

    assert config.maximum_depth == 3


def test_invalid_learning_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GradientBoostingCandidateConfig(
            learning_rate=0.0
        )


def test_at_least_one_model_must_be_enabled() -> None:
    with pytest.raises(
        ValidationError,
        match="Au moins un modèle",
    ):
        CandidateModelsConfig(
            logistic_regression=(
                LogisticRegressionCandidateConfig(
                    enabled=False
                )
            ),
            random_forest=(
                RandomForestCandidateConfig(
                    enabled=False
                )
            ),
            gradient_boosting=(
                GradientBoostingCandidateConfig(
                    enabled=False
                )
            ),
        )


def test_valid_candidate_experiment_config() -> None:
    config = CandidateExperimentConfig(
        random_seed=42,
        models=create_models_config(),
        evaluation=CandidateEvaluationConfig(),
        input=CandidateInputConfig(
            processed_data_directory=Path(
                "data/processed"
            )
        ),
        output=CandidateOutputConfig(
            candidates_directory=Path(
                "models/candidates"
            ),
            comparison_directory=Path(
                "reports/evaluation/candidates"
            ),
        ),
    )

    assert config.random_seed == 42
    assert config.models.random_forest.enabled is True

    assert (
        config.evaluation.minimum_f1_macro
        == pytest.approx(0.80)
    )


def test_invalid_evaluation_threshold_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateEvaluationConfig(
            minimum_f1_macro=1.5
        )