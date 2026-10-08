import sys
from pathlib import Path

from app.core.candidate_config import (
    DEFAULT_CANDIDATE_CONFIG_PATH,
    load_candidate_config,
)
from app.core.candidate_schemas import (
    CandidateComparisonReport,
    CandidateExperimentConfig,
)
from app.core.config import settings
from app.core.experiment_registry import (
    generate_experiment_id,
    register_experiment,
    utc_now_iso,
)
from app.core.registry_schemas import (
    ExperimentArtifact,
    ExperimentCandidateMetric,
    ExperimentRecord,
)
from pipelines.train_candidates import (
    CHAMPION_FILENAME,
    COMPARISON_CSV_FILENAME,
    COMPARISON_JSON_FILENAME,
    MODEL_FILENAME,
    train_candidates_pipeline,
)


DEFAULT_REGISTRY_DIRECTORY = (
    settings.project_root
    / "models"
    / "registry"
)


def create_running_experiment(
    experiment_id: str,
    config: CandidateExperimentConfig,
) -> ExperimentRecord:
    """
    Crée l'enregistrement initial d'une expérience.

    À ce stade, aucun modèle n'est encore entraîné.
    """

    dataset_path = (
        config.input.processed_data_directory
        / "X_train.csv"
    )

    return ExperimentRecord(
        experiment_id=experiment_id,
        created_at=utc_now_iso(),
        completed_at=None,
        status="RUNNING",
        random_seed=config.random_seed,
        dataset_path=str(
            dataset_path.resolve()
        ),
        candidate_count=0,
        champion_model_name=None,
        champion_model_version=None,
        candidates=[],
        artifacts=[],
        error_message=None,
    )


def create_candidate_metrics(
    comparison_report: CandidateComparisonReport,
) -> list:
    """
    Convertit les résultats du classement
    en métriques enregistrables dans le registre.
    """

    candidate_metrics = []

    for candidate in comparison_report.ranking:
        metric = ExperimentCandidateMetric(
            model_name=candidate.model_name,
            model_version=(
                candidate.model_version
            ),
            passed=candidate.passed,
            accuracy=candidate.accuracy,
            f1_macro=candidate.f1_macro,
            poor_recall=candidate.poor_recall,
            prediction_latency_seconds=(
                candidate
                .prediction_latency_seconds
            ),
            training_duration_seconds=(
                candidate
                .training_duration_seconds
            ),
        )

        candidate_metrics.append(
            metric
        )

    return candidate_metrics


def create_experiment_artifacts(
    config: CandidateExperimentConfig,
    comparison_report: CandidateComparisonReport,
) -> list:
    """
    Construit la liste des artefacts produits
    pendant l'expérience.
    """

    comparison_directory = (
        config.output.comparison_directory
    )

    candidates_directory = (
        config.output.candidates_directory
    )

    artifacts = [
        ExperimentArtifact(
            artifact_name="comparison_json",
            artifact_type="json",
            artifact_path=str(
                (
                    comparison_directory
                    / COMPARISON_JSON_FILENAME
                ).resolve()
            ),
        ),
        ExperimentArtifact(
            artifact_name="comparison_csv",
            artifact_type="csv",
            artifact_path=str(
                (
                    comparison_directory
                    / COMPARISON_CSV_FILENAME
                ).resolve()
            ),
        ),
        ExperimentArtifact(
            artifact_name="champion_report",
            artifact_type="json",
            artifact_path=str(
                (
                    comparison_directory
                    / CHAMPION_FILENAME
                ).resolve()
            ),
        ),
    ]

    for candidate in comparison_report.ranking:
        model_path = (
            candidates_directory
            / candidate.model_name
            / MODEL_FILENAME
        )

        artifact = ExperimentArtifact(
            artifact_name=(
                f"{candidate.model_name}_model"
            ),
            artifact_type="joblib",
            artifact_path=str(
                model_path.resolve()
            ),
        )

        artifacts.append(
            artifact
        )

    return artifacts


def create_completed_experiment(
    running_experiment: ExperimentRecord,
    config: CandidateExperimentConfig,
    comparison_report: CandidateComparisonReport,
) -> ExperimentRecord:
    """
    Transforme l'expérience RUNNING
    en expérience COMPLETED.
    """

    return ExperimentRecord(
        experiment_id=(
            running_experiment.experiment_id
        ),
        created_at=(
            running_experiment.created_at
        ),
        completed_at=utc_now_iso(),
        status="COMPLETED",
        random_seed=config.random_seed,
        dataset_path=(
            running_experiment.dataset_path
        ),
        candidate_count=(
            comparison_report.candidate_count
        ),
        champion_model_name=(
            comparison_report
            .champion_model_name
        ),
        champion_model_version=(
            comparison_report
            .champion_model_version
        ),
        candidates=create_candidate_metrics(
            comparison_report
        ),
        artifacts=create_experiment_artifacts(
            config,
            comparison_report,
        ),
        error_message=None,
    )


def create_failed_experiment(
    running_experiment: ExperimentRecord,
    error: Exception,
) -> ExperimentRecord:
    """
    Transforme l'expérience RUNNING
    en expérience FAILED.
    """

    return ExperimentRecord(
        experiment_id=(
            running_experiment.experiment_id
        ),
        created_at=(
            running_experiment.created_at
        ),
        completed_at=utc_now_iso(),
        status="FAILED",
        random_seed=(
            running_experiment.random_seed
        ),
        dataset_path=(
            running_experiment.dataset_path
        ),
        candidate_count=0,
        champion_model_name=None,
        champion_model_version=None,
        candidates=[],
        artifacts=[],
        error_message=str(error),
    )


def run_candidate_experiment(
    config: CandidateExperimentConfig,
    registry_directory: Path,
) -> ExperimentRecord:
    """
    Exécute et enregistre une expérience complète.

    Étapes :

    1. génération de l'identifiant ;
    2. enregistrement du statut RUNNING ;
    3. entraînement et comparaison des candidats ;
    4. enregistrement du statut COMPLETED ;
    5. en cas d'erreur, enregistrement du statut FAILED.
    """

    experiment_id = generate_experiment_id()

    running_experiment = (
        create_running_experiment(
            experiment_id,
            config,
        )
    )

    register_experiment(
        registry_directory,
        running_experiment,
    )

    try:
        comparison_report = (
            train_candidates_pipeline(
                config
            )
        )

        completed_experiment = (
            create_completed_experiment(
                running_experiment,
                config,
                comparison_report,
            )
        )

        register_experiment(
            registry_directory,
            completed_experiment,
        )

        return completed_experiment

    except Exception as error:
        failed_experiment = (
            create_failed_experiment(
                running_experiment,
                error,
            )
        )

        register_experiment(
            registry_directory,
            failed_experiment,
        )

        raise


def print_experiment_summary(
    experiment: ExperimentRecord,
) -> None:
    """
    Affiche un résumé de l'expérience enregistrée.
    """

    print()
    print("=" * 70)
    print("CANDIDATE MODEL EXPERIMENT")
    print("=" * 70)

    print(
        f"Experiment ID : "
        f"{experiment.experiment_id}"
    )

    print(
        f"Statut : {experiment.status}"
    )

    print(
        f"Création : {experiment.created_at}"
    )

    print(
        f"Fin : {experiment.completed_at}"
    )

    print(
        f"Nombre de candidats : "
        f"{experiment.candidate_count}"
    )

    if experiment.champion_model_name is None:
        print(
            "Champion : aucun modèle sélectionné"
        )
    else:
        print(
            "Champion : "
            f"{experiment.champion_model_name} "
            f"version "
            f"{experiment.champion_model_version}"
        )

    print()
    print("Résultats des candidats :")

    for candidate in experiment.candidates:
        print(
            f"  - {candidate.model_name}"
        )

        print(
            f"    Validé : "
            f"{candidate.passed}"
        )

        print(
            f"    F1 macro : "
            f"{candidate.f1_macro:.4f}"
        )

        print(
            f"    Recall POOR : "
            f"{candidate.poor_recall:.4f}"
        )

    print()
    print(
        f"Nombre d'artefacts : "
        f"{len(experiment.artifacts)}"
    )

    print("=" * 70)
    print()


def main() -> int:
    """
    Point d'entrée de l'expérience candidate.
    """

    try:
        config = load_candidate_config(
            DEFAULT_CANDIDATE_CONFIG_PATH
        )

        experiment = run_candidate_experiment(
            config,
            DEFAULT_REGISTRY_DIRECTORY,
        )

        print_experiment_summary(
            experiment
        )

        return 0

    except FileNotFoundError as error:
        print(
            f"Fichier introuvable : {error}"
        )

        return 1

    except ValueError as error:
        print(
            "Configuration ou données invalides : "
            f"{error}"
        )

        return 1

    except RuntimeError as error:
        print(
            f"Erreur d'exécution : {error}"
        )

        return 1

    except Exception as error:
        print(
            f"Expérience interrompue : {error}"
        )

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)