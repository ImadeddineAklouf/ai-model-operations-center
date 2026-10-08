import pytest
from pydantic import ValidationError

from app.core.registry_schemas import (
    ExperimentArtifact,
    ExperimentCandidateMetric,
    ExperimentIndexEntry,
    ExperimentRecord,
    ExperimentRegistryIndex,
)


def create_valid_artifact() -> ExperimentArtifact:
    """
    Crée un artefact valide pour les tests.
    """

    return ExperimentArtifact(
        artifact_name="comparison_report",
        artifact_type="json",
        artifact_path=(
            "reports/evaluation/"
            "candidates/comparison.json"
        ),
    )


def create_valid_candidate_metric(
) -> ExperimentCandidateMetric:
    """
    Crée les métriques valides d'un modèle candidat.
    """

    return ExperimentCandidateMetric(
        model_name="random_forest",
        model_version="1.0.0",
        passed=True,
        accuracy=0.95,
        f1_macro=0.94,
        poor_recall=0.96,
        prediction_latency_seconds=0.01,
        training_duration_seconds=1.50,
    )


def create_valid_experiment_record(
) -> ExperimentRecord:
    """
    Crée un enregistrement d'expérience valide.
    """

    return ExperimentRecord(
        experiment_id=(
            "run_20261008_120500_ab12cd34"
        ),
        created_at=(
            "2026-10-08T12:05:00+00:00"
        ),
        completed_at=(
            "2026-10-08T12:05:05+00:00"
        ),
        status="COMPLETED",
        random_seed=42,
        dataset_path=(
            "data/processed/X_train.csv"
        ),
        candidate_count=1,
        champion_model_name=(
            "random_forest"
        ),
        champion_model_version="1.0.0",
        candidates=[
            create_valid_candidate_metric()
        ],
        artifacts=[
            create_valid_artifact()
        ],
        error_message=None,
    )


def create_valid_index_entry(
) -> ExperimentIndexEntry:
    """
    Crée une entrée valide pour l'index.
    """

    return ExperimentIndexEntry(
        experiment_id=(
            "run_20261008_120500_ab12cd34"
        ),
        created_at=(
            "2026-10-08T12:05:00+00:00"
        ),
        completed_at=(
            "2026-10-08T12:05:05+00:00"
        ),
        status="COMPLETED",
        champion_model_name=(
            "random_forest"
        ),
        champion_model_version="1.0.0",
        experiment_path=(
            "models/registry/runs/"
            "run_20261008_120500_ab12cd34/"
            "experiment.json"
        ),
    )


def test_valid_experiment_artifact() -> None:
    """
    Vérifie la création d'un artefact valide.
    """

    artifact = create_valid_artifact()

    assert artifact.artifact_name == (
        "comparison_report"
    )

    assert artifact.artifact_type == "json"

    assert artifact.artifact_path == (
        "reports/evaluation/"
        "candidates/comparison.json"
    )


def test_empty_artifact_name_is_rejected() -> None:
    """
    Vérifie que le nom de l'artefact
    ne peut pas être vide.
    """

    with pytest.raises(ValidationError):
        ExperimentArtifact(
            artifact_name="",
            artifact_type="json",
            artifact_path="reports/report.json",
        )


def test_empty_artifact_type_is_rejected() -> None:
    """
    Vérifie que le type de l'artefact
    ne peut pas être vide.
    """

    with pytest.raises(ValidationError):
        ExperimentArtifact(
            artifact_name="report",
            artifact_type="",
            artifact_path="reports/report.json",
        )


def test_empty_artifact_path_is_rejected() -> None:
    """
    Vérifie que le chemin de l'artefact
    ne peut pas être vide.
    """

    with pytest.raises(ValidationError):
        ExperimentArtifact(
            artifact_name="report",
            artifact_type="json",
            artifact_path="",
        )


def test_unknown_artifact_field_is_rejected() -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    artifact_data = {
        "artifact_name": "report",
        "artifact_type": "json",
        "artifact_path": "reports/report.json",
        "unknown_field": "unexpected",
    }

    with pytest.raises(ValidationError):
        ExperimentArtifact.model_validate(
            artifact_data
        )


def test_valid_candidate_metric() -> None:
    """
    Vérifie les métriques valides d'un candidat.
    """

    metric = create_valid_candidate_metric()

    assert metric.model_name == (
        "random_forest"
    )

    assert metric.model_version == "1.0.0"
    assert metric.passed is True

    assert metric.accuracy == pytest.approx(
        0.95
    )

    assert metric.f1_macro == pytest.approx(
        0.94
    )

    assert metric.poor_recall == pytest.approx(
        0.96
    )


def test_candidate_accuracy_above_one_is_rejected(
) -> None:
    """
    Vérifie que l'accuracy ne peut pas
    être supérieure à 1.
    """

    with pytest.raises(ValidationError):
        ExperimentCandidateMetric(
            model_name="random_forest",
            model_version="1.0.0",
            passed=True,
            accuracy=1.10,
            f1_macro=0.94,
            poor_recall=0.96,
            prediction_latency_seconds=0.01,
            training_duration_seconds=1.50,
        )


def test_negative_candidate_f1_is_rejected(
) -> None:
    """
    Vérifie que le F1 macro ne peut pas
    être négatif.
    """

    with pytest.raises(ValidationError):
        ExperimentCandidateMetric(
            model_name="random_forest",
            model_version="1.0.0",
            passed=True,
            accuracy=0.95,
            f1_macro=-0.10,
            poor_recall=0.96,
            prediction_latency_seconds=0.01,
            training_duration_seconds=1.50,
        )


def test_poor_recall_above_one_is_rejected(
) -> None:
    """
    Vérifie que le recall POOR est limité à 1.
    """

    with pytest.raises(ValidationError):
        ExperimentCandidateMetric(
            model_name="random_forest",
            model_version="1.0.0",
            passed=True,
            accuracy=0.95,
            f1_macro=0.94,
            poor_recall=1.20,
            prediction_latency_seconds=0.01,
            training_duration_seconds=1.50,
        )


def test_negative_prediction_latency_is_rejected(
) -> None:
    """
    Vérifie que la latence ne peut pas
    être négative.
    """

    with pytest.raises(ValidationError):
        ExperimentCandidateMetric(
            model_name="random_forest",
            model_version="1.0.0",
            passed=True,
            accuracy=0.95,
            f1_macro=0.94,
            poor_recall=0.96,
            prediction_latency_seconds=-0.01,
            training_duration_seconds=1.50,
        )


def test_negative_training_duration_is_rejected(
) -> None:
    """
    Vérifie que la durée d'entraînement
    ne peut pas être négative.
    """

    with pytest.raises(ValidationError):
        ExperimentCandidateMetric(
            model_name="random_forest",
            model_version="1.0.0",
            passed=True,
            accuracy=0.95,
            f1_macro=0.94,
            poor_recall=0.96,
            prediction_latency_seconds=0.01,
            training_duration_seconds=-1.0,
        )


def test_valid_completed_experiment_record(
) -> None:
    """
    Vérifie une expérience terminée valide.
    """

    experiment = (
        create_valid_experiment_record()
    )

    assert experiment.experiment_id == (
        "run_20261008_120500_ab12cd34"
    )

    assert experiment.status == "COMPLETED"
    assert experiment.random_seed == 42
    assert experiment.candidate_count == 1

    assert experiment.champion_model_name == (
        "random_forest"
    )

    assert experiment.champion_model_version == (
        "1.0.0"
    )

    assert len(experiment.candidates) == 1
    assert len(experiment.artifacts) == 1
    assert experiment.error_message is None


def test_valid_running_experiment_record() -> None:
    """
    Vérifie qu'une expérience en cours
    peut ne pas avoir de champion.
    """

    experiment = ExperimentRecord(
        experiment_id="run_test",
        created_at=(
            "2026-10-08T12:05:00+00:00"
        ),
        completed_at=None,
        status="RUNNING",
        random_seed=42,
        dataset_path=(
            "data/processed/X_train.csv"
        ),
        candidate_count=0,
        champion_model_name=None,
        champion_model_version=None,
        candidates=[],
        artifacts=[],
        error_message=None,
    )

    assert experiment.status == "RUNNING"
    assert experiment.completed_at is None
    assert experiment.champion_model_name is None


def test_valid_failed_experiment_record() -> None:
    """
    Vérifie qu'une expérience échouée peut
    contenir un message d'erreur.
    """

    experiment = ExperimentRecord(
        experiment_id="run_failed",
        created_at=(
            "2026-10-08T12:05:00+00:00"
        ),
        completed_at=(
            "2026-10-08T12:05:01+00:00"
        ),
        status="FAILED",
        random_seed=42,
        dataset_path=(
            "data/processed/X_train.csv"
        ),
        candidate_count=0,
        champion_model_name=None,
        champion_model_version=None,
        candidates=[],
        artifacts=[],
        error_message=(
            "Fichier d'entraînement introuvable."
        ),
    )

    assert experiment.status == "FAILED"

    assert experiment.error_message == (
        "Fichier d'entraînement introuvable."
    )


def test_invalid_experiment_status_is_rejected(
) -> None:
    """
    Vérifie qu'un statut inconnu est refusé.
    """

    experiment_data = (
        create_valid_experiment_record()
        .model_dump()
    )

    experiment_data["status"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            experiment_data
        )


def test_empty_experiment_id_is_rejected() -> None:
    """
    Vérifie que l'identifiant est obligatoire.
    """

    experiment_data = (
        create_valid_experiment_record()
        .model_dump()
    )

    experiment_data["experiment_id"] = ""

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            experiment_data
        )


def test_negative_random_seed_is_rejected() -> None:
    """
    Vérifie que la seed ne peut pas
    être négative.
    """

    experiment_data = (
        create_valid_experiment_record()
        .model_dump()
    )

    experiment_data["random_seed"] = -1

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            experiment_data
        )


def test_negative_candidate_count_is_rejected(
) -> None:
    """
    Vérifie que le nombre de candidats
    ne peut pas être négatif.
    """

    experiment_data = (
        create_valid_experiment_record()
        .model_dump()
    )

    experiment_data["candidate_count"] = -1

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            experiment_data
        )


def test_unknown_experiment_field_is_rejected(
) -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    experiment_data = (
        create_valid_experiment_record()
        .model_dump()
    )

    experiment_data[
        "unknown_field"
    ] = "unexpected"

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            experiment_data
        )


def test_valid_index_entry() -> None:
    """
    Vérifie une entrée valide dans l'index.
    """

    entry = create_valid_index_entry()

    assert entry.experiment_id == (
        "run_20261008_120500_ab12cd34"
    )

    assert entry.status == "COMPLETED"

    assert entry.champion_model_name == (
        "random_forest"
    )

    assert entry.experiment_path.endswith(
        "experiment.json"
    )


def test_running_index_entry_without_champion(
) -> None:
    """
    Vérifie une entrée d'index sans champion.
    """

    entry = ExperimentIndexEntry(
        experiment_id="run_running",
        created_at=(
            "2026-10-08T12:05:00+00:00"
        ),
        completed_at=None,
        status="RUNNING",
        champion_model_name=None,
        champion_model_version=None,
        experiment_path=(
            "models/registry/runs/"
            "run_running/experiment.json"
        ),
    )

    assert entry.status == "RUNNING"
    assert entry.completed_at is None
    assert entry.champion_model_name is None


def test_default_registry_index() -> None:
    """
    Vérifie l'index vide par défaut.
    """

    registry_index = ExperimentRegistryIndex()

    assert registry_index.registry_version == (
        "1.0.0"
    )

    assert registry_index.experiment_count == 0
    assert registry_index.experiments == []


def test_registry_index_with_experiment() -> None:
    """
    Vérifie un index contenant une expérience.
    """

    entry = create_valid_index_entry()

    registry_index = ExperimentRegistryIndex(
        registry_version="1.0.0",
        experiment_count=1,
        experiments=[
            entry
        ],
    )

    assert registry_index.experiment_count == 1
    assert len(registry_index.experiments) == 1

    assert (
        registry_index
        .experiments[0]
        .experiment_id
        == entry.experiment_id
    )


def test_negative_experiment_count_is_rejected(
) -> None:
    """
    Vérifie que le compteur global
    ne peut pas être négatif.
    """

    with pytest.raises(ValidationError):
        ExperimentRegistryIndex(
            registry_version="1.0.0",
            experiment_count=-1,
            experiments=[],
        )


def test_unknown_registry_field_is_rejected(
) -> None:
    """
    Vérifie extra='forbid' sur l'index.
    """

    registry_data = {
        "registry_version": "1.0.0",
        "experiment_count": 0,
        "experiments": [],
        "unknown_field": "unexpected",
    }

    with pytest.raises(ValidationError):
        ExperimentRegistryIndex.model_validate(
            registry_data
        )