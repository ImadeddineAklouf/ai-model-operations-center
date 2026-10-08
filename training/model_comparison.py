from app.core.candidate_schemas import (
    CandidateComparisonReport,
    CandidateModelResult,
)
from app.core.model_schemas import (
    ModelEvaluationResult,
)


def create_candidate_result(
    model_name: str,
    model_version: str,
    training_duration_seconds: float,
    evaluation_result: ModelEvaluationResult,
) -> CandidateModelResult:
    """
    Transforme un résultat d'évaluation détaillé
    en résultat synthétique exploitable pour le classement.
    """

    poor_recall = (
        evaluation_result
        .metrics
        .metrics_by_class["POOR"]
        .recall
    )

    return CandidateModelResult(
        model_name=model_name,
        model_version=model_version,
        passed=evaluation_result.passed,
        accuracy=(
            evaluation_result
            .metrics
            .accuracy
        ),
        precision_macro=(
            evaluation_result
            .metrics
            .precision_macro
        ),
        recall_macro=(
            evaluation_result
            .metrics
            .recall_macro
        ),
        f1_macro=(
            evaluation_result
            .metrics
            .f1_macro
        ),
        poor_recall=poor_recall,
        prediction_latency_seconds=(
            evaluation_result
            .metrics
            .prediction_latency_seconds
        ),
        training_duration_seconds=float(
            training_duration_seconds
        ),
        rejection_reasons=list(
            evaluation_result.rejection_reasons
        ),
    )


def candidate_sort_key(
    candidate: CandidateModelResult,
) -> tuple[
    int,
    float,
    float,
    float,
    float,
]:
    """
    Construit la clé utilisée pour classer un candidat.

    Une valeur élevée est préférable pour :

    - le statut validé ;
    - le F1 macro ;
    - le recall POOR.

    Une valeur faible est préférable pour :

    - la latence de prédiction ;
    - la durée d'entraînement.
    """

    if candidate.passed:
        validation_score = 1
    else:
        validation_score = 0

    return (
        validation_score,
        candidate.f1_macro,
        candidate.poor_recall,
        -candidate.prediction_latency_seconds,
        -candidate.training_duration_seconds,
    )


def rank_candidates(
    candidates: list[CandidateModelResult],
) -> list[CandidateModelResult]:
    """
    Classe les candidats du meilleur au moins bon.
    """

    if not candidates:
        raise ValueError(
            "La liste des candidats ne peut pas être vide."
        )

    ranking = sorted(
        candidates,
        key=candidate_sort_key,
        reverse=True,
    )

    return ranking


def select_champion(
    ranking: list[CandidateModelResult],
) -> CandidateModelResult | None:
    """
    Retourne le meilleur modèle validé.

    Aucun champion n'est retourné si tous les modèles
    ont été rejetés.
    """

    if not ranking:
        raise ValueError(
            "Le classement ne peut pas être vide."
        )

    for candidate in ranking:
        if candidate.passed:
            return candidate

    return None


def build_comparison_report(
    candidates: list[CandidateModelResult],
) -> CandidateComparisonReport:
    """
    Classe les candidats et construit le rapport final.
    """

    ranking = rank_candidates(
        candidates
    )

    champion = select_champion(
        ranking
    )

    validated_candidate_count = 0

    for candidate in ranking:
        if candidate.passed:
            validated_candidate_count += 1

    if champion is None:
        champion_model_name = None
        champion_model_version = None
    else:
        champion_model_name = (
            champion.model_name
        )

        champion_model_version = (
            champion.model_version
        )

    return CandidateComparisonReport(
        champion_model_name=(
            champion_model_name
        ),
        champion_model_version=(
            champion_model_version
        ),
        candidate_count=len(ranking),
        validated_candidate_count=(
            validated_candidate_count
        ),
        ranking=ranking,
    )