from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from app.core.candidate_schemas import (
    CandidateExperimentConfig,
    GradientBoostingCandidateConfig,
    LogisticRegressionCandidateConfig,
    RandomForestCandidateConfig,
)


def build_logistic_regression_candidate(
    config: LogisticRegressionCandidateConfig,
    random_seed: int,
) -> LogisticRegression:
    """
    Construit une régression logistique candidate.
    """

    return LogisticRegression(
        max_iter=config.maximum_iterations,
        random_state=random_seed,
        class_weight=config.class_weight,
        solver="lbfgs",
    )


def build_random_forest_candidate(
    config: RandomForestCandidateConfig,
    random_seed: int,
) -> RandomForestClassifier:
    """
    Construit un Random Forest candidat.
    """

    return RandomForestClassifier(
        n_estimators=config.number_of_estimators,
        max_depth=config.maximum_depth,
        min_samples_split=(
            config.minimum_samples_split
        ),
        min_samples_leaf=(
            config.minimum_samples_leaf
        ),
        class_weight=config.class_weight,
        random_state=random_seed,
        n_jobs=-1,
    )


def build_gradient_boosting_candidate(
    config: GradientBoostingCandidateConfig,
    random_seed: int,
) -> GradientBoostingClassifier:
    """
    Construit un Gradient Boosting candidat.
    """

    return GradientBoostingClassifier(
        n_estimators=config.number_of_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.maximum_depth,
        random_state=random_seed,
    )


def build_enabled_candidates(
    config: CandidateExperimentConfig,
) -> dict[str, Any]:
    """
    Construit uniquement les modèles activés.

    Returns:
        Dictionnaire associant le nom d'un modèle
        à son instance Scikit-learn non entraînée.
    """

    candidates: dict[str, Any] = {}

    if config.models.logistic_regression.enabled:
        candidates["logistic_regression"] = (
            build_logistic_regression_candidate(
                config.models.logistic_regression,
                config.random_seed,
            )
        )

    if config.models.random_forest.enabled:
        candidates["random_forest"] = (
            build_random_forest_candidate(
                config.models.random_forest,
                config.random_seed,
            )
        )

    if config.models.gradient_boosting.enabled:
        candidates["gradient_boosting"] = (
            build_gradient_boosting_candidate(
                config.models.gradient_boosting,
                config.random_seed,
            )
        )

    if not candidates:
        raise ValueError(
            "Aucun modèle candidat n'est activé."
        )

    return candidates


def get_candidate_version(
    model_name: str,
    config: CandidateExperimentConfig,
) -> str:
    """
    Retourne la version configurée d'un modèle.
    """

    if model_name == "logistic_regression":
        return (
            config
            .models
            .logistic_regression
            .version
        )

    if model_name == "random_forest":
        return (
            config
            .models
            .random_forest
            .version
        )

    if model_name == "gradient_boosting":
        return (
            config
            .models
            .gradient_boosting
            .version
        )

    raise ValueError(
        f"Modèle candidat inconnu : {model_name}"
    )