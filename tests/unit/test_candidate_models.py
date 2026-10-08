from pathlib import Path

import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

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
from training.candidate_models import (
    build_enabled_candidates,
    build_gradient_boosting_candidate,
    build_logistic_regression_candidate,
    build_random_forest_candidate,
    get_candidate_version,
)


def create_candidate_config() -> CandidateExperimentConfig:
    return CandidateExperimentConfig(
        random_seed=42,
        models=CandidateModelsConfig(
            logistic_regression=(
                LogisticRegressionCandidateConfig()
            ),
            random_forest=(
                RandomForestCandidateConfig()
            ),
            gradient_boosting=(
                GradientBoostingCandidateConfig()
            ),
        ),
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


def test_build_logistic_regression_candidate() -> None:
    model = build_logistic_regression_candidate(
        LogisticRegressionCandidateConfig(),
        42,
    )

    assert isinstance(
        model,
        LogisticRegression,
    )

    assert model.max_iter == 1000
    assert model.random_state == 42


def test_build_random_forest_candidate() -> None:
    model = build_random_forest_candidate(
        RandomForestCandidateConfig(
            number_of_estimators=150,
            maximum_depth=8,
        ),
        42,
    )

    assert isinstance(
        model,
        RandomForestClassifier,
    )

    assert model.n_estimators == 150
    assert model.max_depth == 8
    assert model.random_state == 42


def test_build_gradient_boosting_candidate() -> None:
    model = build_gradient_boosting_candidate(
        GradientBoostingCandidateConfig(
            number_of_estimators=120,
            learning_rate=0.05,
            maximum_depth=4,
        ),
        42,
    )

    assert isinstance(
        model,
        GradientBoostingClassifier,
    )

    assert model.n_estimators == 120

    assert model.learning_rate == pytest.approx(
        0.05
    )

    assert model.max_depth == 4


def test_build_all_enabled_candidates() -> None:
    config = create_candidate_config()

    candidates = build_enabled_candidates(
        config
    )

    assert set(candidates) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


def test_disabled_model_is_not_built() -> None:
    config = create_candidate_config()

    models = CandidateModelsConfig(
        logistic_regression=(
            config.models.logistic_regression
        ),
        random_forest=(
            config.models.random_forest.model_copy(
                update={
                    "enabled": False,
                }
            )
        ),
        gradient_boosting=(
            config.models.gradient_boosting
        ),
    )

    updated_config = config.model_copy(
        update={
            "models": models,
        }
    )

    candidates = build_enabled_candidates(
        updated_config
    )

    assert "random_forest" not in candidates

    assert set(candidates) == {
        "logistic_regression",
        "gradient_boosting",
    }


def test_get_candidate_version() -> None:
    config = create_candidate_config()

    assert get_candidate_version(
        "logistic_regression",
        config,
    ) == "1.0.0"

    assert get_candidate_version(
        "random_forest",
        config,
    ) == "1.0.0"

    assert get_candidate_version(
        "gradient_boosting",
        config,
    ) == "1.0.0"


def test_unknown_candidate_version_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="inconnu",
    ):
        get_candidate_version(
            "unknown_model",
            create_candidate_config(),
        )