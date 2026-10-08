import pytest

from app.core.candidate_schemas import (
    CandidateModelResult,
)
from training.model_comparison import (
    build_comparison_report,
    candidate_sort_key,
    rank_candidates,
    select_champion,
)


def create_candidate(
    model_name: str,
    passed: bool,
    f1_macro: float,
    poor_recall: float,
    prediction_latency: float,
    training_duration: float = 1.0,
) -> CandidateModelResult:
    """
    Construit un résultat candidat pour les tests.
    """

    if passed:
        rejection_reasons = []
    else:
        rejection_reasons = [
            "Seuil non respecté"
        ]

    return CandidateModelResult(
        model_name=model_name,
        model_version="1.0.0",
        passed=passed,
        accuracy=f1_macro,
        precision_macro=f1_macro,
        recall_macro=f1_macro,
        f1_macro=f1_macro,
        poor_recall=poor_recall,
        prediction_latency_seconds=(
            prediction_latency
        ),
        training_duration_seconds=(
            training_duration
        ),
        rejection_reasons=rejection_reasons,
    )


def test_candidate_sort_key() -> None:
    """
    Vérifie la construction de la clé de classement.
    """

    candidate = create_candidate(
        model_name="random_forest",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.97,
        prediction_latency=0.03,
        training_duration=2.0,
    )

    sort_key = candidate_sort_key(
        candidate
    )

    assert sort_key == (
        1,
        0.95,
        0.97,
        -0.03,
        -2.0,
    )


def test_validated_model_is_prioritized() -> None:
    """
    Un modèle validé doit être mieux classé
    qu'un modèle rejeté, même si le modèle rejeté
    possède un meilleur F1.
    """

    validated_model = create_candidate(
        model_name="validated_model",
        passed=True,
        f1_macro=0.85,
        poor_recall=0.90,
        prediction_latency=0.05,
    )

    rejected_model = create_candidate(
        model_name="rejected_model",
        passed=False,
        f1_macro=0.99,
        poor_recall=0.99,
        prediction_latency=0.01,
    )

    ranking = rank_candidates(
        [
            rejected_model,
            validated_model,
        ]
    )

    assert ranking[0].model_name == (
        "validated_model"
    )


def test_f1_macro_is_primary_ranking_metric() -> None:
    """
    Le meilleur F1 gagne lorsque les deux modèles
    sont validés.
    """

    better_f1 = create_candidate(
        model_name="better_f1",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.90,
        prediction_latency=0.10,
    )

    lower_f1 = create_candidate(
        model_name="lower_f1",
        passed=True,
        f1_macro=0.90,
        poor_recall=0.99,
        prediction_latency=0.01,
    )

    ranking = rank_candidates(
        [
            lower_f1,
            better_f1,
        ]
    )

    assert ranking[0].model_name == (
        "better_f1"
    )


def test_poor_recall_breaks_f1_tie() -> None:
    """
    Le recall POOR départage deux F1 identiques.
    """

    better_poor_recall = create_candidate(
        model_name="better_poor_recall",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.98,
        prediction_latency=0.10,
    )

    lower_poor_recall = create_candidate(
        model_name="lower_poor_recall",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.90,
        prediction_latency=0.01,
    )

    ranking = rank_candidates(
        [
            lower_poor_recall,
            better_poor_recall,
        ]
    )

    assert ranking[0].model_name == (
        "better_poor_recall"
    )


def test_latency_breaks_metric_tie() -> None:
    """
    La plus faible latence départage deux modèles
    ayant les mêmes métriques.
    """

    faster_model = create_candidate(
        model_name="faster_model",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.98,
        prediction_latency=0.01,
    )

    slower_model = create_candidate(
        model_name="slower_model",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.98,
        prediction_latency=0.10,
    )

    ranking = rank_candidates(
        [
            slower_model,
            faster_model,
        ]
    )

    assert ranking[0].model_name == (
        "faster_model"
    )


def test_training_duration_breaks_final_tie() -> None:
    """
    La durée d'entraînement départage les modèles
    encore à égalité.
    """

    faster_training = create_candidate(
        model_name="faster_training",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.98,
        prediction_latency=0.01,
        training_duration=1.0,
    )

    slower_training = create_candidate(
        model_name="slower_training",
        passed=True,
        f1_macro=0.95,
        poor_recall=0.98,
        prediction_latency=0.01,
        training_duration=5.0,
    )

    ranking = rank_candidates(
        [
            slower_training,
            faster_training,
        ]
    )

    assert ranking[0].model_name == (
        "faster_training"
    )


def test_empty_candidate_list_is_rejected() -> None:
    """
    Un classement vide doit être refusé.
    """

    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        rank_candidates(
            []
        )


def test_select_champion() -> None:
    """
    Vérifie la sélection du premier candidat validé.
    """

    candidates = [
        create_candidate(
            model_name="random_forest",
            passed=True,
            f1_macro=0.95,
            poor_recall=0.96,
            prediction_latency=0.03,
        ),
        create_candidate(
            model_name="logistic_regression",
            passed=True,
            f1_macro=0.88,
            poor_recall=0.90,
            prediction_latency=0.01,
        ),
    ]

    ranking = rank_candidates(
        candidates
    )

    champion = select_champion(
        ranking
    )

    assert champion is not None

    assert champion.model_name == (
        "random_forest"
    )


def test_no_champion_when_all_are_rejected() -> None:
    """
    Aucun champion ne doit être sélectionné
    si tous les candidats sont rejetés.
    """

    candidates = [
        create_candidate(
            model_name="first",
            passed=False,
            f1_macro=0.70,
            poor_recall=0.60,
            prediction_latency=0.01,
        ),
        create_candidate(
            model_name="second",
            passed=False,
            f1_macro=0.75,
            poor_recall=0.65,
            prediction_latency=0.02,
        ),
    ]

    ranking = rank_candidates(
        candidates
    )

    champion = select_champion(
        ranking
    )

    assert champion is None


def test_build_comparison_report() -> None:
    """
    Vérifie le rapport final de comparaison.
    """

    candidates = [
        create_candidate(
            model_name="logistic_regression",
            passed=True,
            f1_macro=0.88,
            poor_recall=0.90,
            prediction_latency=0.01,
        ),
        create_candidate(
            model_name="random_forest",
            passed=True,
            f1_macro=0.94,
            poor_recall=0.96,
            prediction_latency=0.03,
        ),
        create_candidate(
            model_name="gradient_boosting",
            passed=False,
            f1_macro=0.79,
            poor_recall=0.80,
            prediction_latency=0.02,
        ),
    ]

    report = build_comparison_report(
        candidates
    )

    assert report.champion_model_name == (
        "random_forest"
    )

    assert report.champion_model_version == (
        "1.0.0"
    )

    assert report.candidate_count == 3

    assert (
        report.validated_candidate_count
        == 2
    )

    assert len(report.ranking) == 3

    assert report.ranking[0].model_name == (
        "random_forest"
    )


def test_report_has_no_champion_when_all_rejected() -> None:
    """
    Vérifie le rapport lorsque tous les modèles
    sont rejetés.
    """

    candidates = [
        create_candidate(
            model_name="first",
            passed=False,
            f1_macro=0.70,
            poor_recall=0.60,
            prediction_latency=0.01,
        ),
        create_candidate(
            model_name="second",
            passed=False,
            f1_macro=0.72,
            poor_recall=0.62,
            prediction_latency=0.02,
        ),
    ]

    report = build_comparison_report(
        candidates
    )

    assert report.champion_model_name is None
    assert report.champion_model_version is None

    assert (
        report.validated_candidate_count
        == 0
    )