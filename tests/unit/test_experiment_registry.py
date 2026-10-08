import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.experiment_registry import (
    EXPERIMENT_FILENAME,
    REGISTRY_INDEX_FILENAME,
    RUNS_DIRECTORY_NAME,
    add_experiment_to_index,
    generate_experiment_id,
    get_experiment_directory,
    get_registry_index_path,
    list_experiments,
    load_experiment_record,
    load_registry_index,
    register_experiment,
    save_experiment_record,
    save_registry_index,
    utc_now_iso,
)
from app.core.registry_schemas import (
    ExperimentArtifact,
    ExperimentCandidateMetric,
    ExperimentRecord,
    ExperimentRegistryIndex,
)


def create_candidate_metric(
    model_name: str = "random_forest",
) -> ExperimentCandidateMetric:
    """
    Crée les métriques valides d'un modèle candidat.
    """

    return ExperimentCandidateMetric(
        model_name=model_name,
        model_version="1.0.0",
        passed=True,
        accuracy=0.95,
        f1_macro=0.94,
        poor_recall=0.96,
        prediction_latency_seconds=0.01,
        training_duration_seconds=1.50,
    )


def create_artifact() -> ExperimentArtifact:
    """
    Crée un artefact associé à une expérience.
    """

    return ExperimentArtifact(
        artifact_name="comparison_report",
        artifact_type="json",
        artifact_path=(
            "reports/evaluation/"
            "candidates/comparison.json"
        ),
    )


def create_experiment(
    experiment_id: str = "run_test_001",
    champion_model_name: str = "random_forest",
) -> ExperimentRecord:
    """
    Crée une expérience terminée valide.
    """

    return ExperimentRecord(
        experiment_id=experiment_id,
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
            champion_model_name
        ),
        champion_model_version="1.0.0",
        candidates=[
            create_candidate_metric(
                champion_model_name
            )
        ],
        artifacts=[
            create_artifact()
        ],
        error_message=None,
    )


def test_utc_now_iso_returns_value() -> None:
    """
    Vérifie que la date UTC est générée.
    """

    current_time = utc_now_iso()

    assert isinstance(
        current_time,
        str,
    )

    assert current_time
    assert "+00:00" in current_time


def test_generate_experiment_id() -> None:
    """
    Vérifie le format et l'unicité des identifiants.
    """

    first_identifier = generate_experiment_id()
    second_identifier = generate_experiment_id()

    assert first_identifier.startswith(
        "run_"
    )

    assert second_identifier.startswith(
        "run_"
    )

    assert first_identifier != (
        second_identifier
    )


def test_get_registry_index_path(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chemin de l'index global.
    """

    index_path = get_registry_index_path(
        tmp_path
    )

    assert index_path == (
        tmp_path.resolve()
        / REGISTRY_INDEX_FILENAME
    )


def test_get_experiment_directory(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chemin du dossier d'une expérience.
    """

    experiment_directory = (
        get_experiment_directory(
            tmp_path,
            "run_test_001",
        )
    )

    assert experiment_directory == (
        tmp_path.resolve()
        / RUNS_DIRECTORY_NAME
        / "run_test_001"
    )


def test_empty_experiment_id_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un identifiant vide est refusé.
    """

    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        get_experiment_directory(
            tmp_path,
            "",
        )


def test_load_missing_registry_returns_empty_index(
    tmp_path: Path,
) -> None:
    """
    Un registre inexistant doit retourner
    un index vide.
    """

    registry_index = load_registry_index(
        tmp_path
    )

    assert isinstance(
        registry_index,
        ExperimentRegistryIndex,
    )

    assert registry_index.experiment_count == 0
    assert registry_index.experiments == []


def test_save_and_load_registry_index(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'écriture et la lecture de l'index.
    """

    registry_index = ExperimentRegistryIndex()

    index_path = save_registry_index(
        tmp_path,
        registry_index,
    )

    assert index_path.exists()

    loaded_index = load_registry_index(
        tmp_path
    )

    assert loaded_index == registry_index


def test_saved_registry_is_valid_json(
    tmp_path: Path,
) -> None:
    """
    Vérifie que le fichier d'index contient
    un document JSON valide.
    """

    registry_index = ExperimentRegistryIndex()

    index_path = save_registry_index(
        tmp_path,
        registry_index,
    )

    raw_content = json.loads(
        index_path.read_text(
            encoding="utf-8",
        )
    )

    assert raw_content[
        "registry_version"
    ] == "1.0.0"

    assert raw_content[
        "experiment_count"
    ] == 0

    assert raw_content[
        "experiments"
    ] == []


def test_invalid_registry_json_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les données incompatibles
    avec le schéma sont refusées.
    """

    index_path = (
        tmp_path
        / REGISTRY_INDEX_FILENAME
    )

    index_path.write_text(
        json.dumps(
            {
                "registry_version": "1.0.0",
                "experiment_count": -1,
                "experiments": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError
    ):
        load_registry_index(
            tmp_path
        )


def test_save_experiment_record(
    tmp_path: Path,
) -> None:
    """
    Vérifie la sauvegarde d'une expérience.
    """

    experiment = create_experiment()

    experiment_path = (
        save_experiment_record(
            tmp_path,
            experiment,
        )
    )

    assert experiment_path.exists()

    assert experiment_path.name == (
        EXPERIMENT_FILENAME
    )

    assert experiment_path.parent.name == (
        experiment.experiment_id
    )


def test_save_and_load_experiment_record(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une expérience sauvegardée
    peut être rechargée.
    """

    experiment = create_experiment()

    save_experiment_record(
        tmp_path,
        experiment,
    )

    loaded_experiment = (
        load_experiment_record(
            tmp_path,
            experiment.experiment_id,
        )
    )

    assert loaded_experiment == experiment


def test_missing_experiment_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une expérience absente
    produit une erreur claire.
    """

    with pytest.raises(
        FileNotFoundError,
        match="Expérience introuvable",
    ):
        load_experiment_record(
            tmp_path,
            "run_missing",
        )


def test_add_experiment_to_index(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'ajout d'une expérience à l'index.
    """

    experiment = create_experiment()

    experiment_path = (
        save_experiment_record(
            tmp_path,
            experiment,
        )
    )

    registry_index = (
        add_experiment_to_index(
            tmp_path,
            experiment,
            experiment_path,
        )
    )

    assert registry_index.experiment_count == 1

    assert len(
        registry_index.experiments
    ) == 1

    index_entry = (
        registry_index.experiments[0]
    )

    assert index_entry.experiment_id == (
        experiment.experiment_id
    )

    assert index_entry.status == "COMPLETED"

    assert index_entry.champion_model_name == (
        "random_forest"
    )


def test_register_experiment(
    tmp_path: Path,
) -> None:
    """
    Vérifie le processus complet d'enregistrement.
    """

    experiment = create_experiment()

    experiment_path = register_experiment(
        tmp_path,
        experiment,
    )

    assert experiment_path.exists()

    index_path = (
        tmp_path
        / REGISTRY_INDEX_FILENAME
    )

    assert index_path.exists()

    registry_index = load_registry_index(
        tmp_path
    )

    assert registry_index.experiment_count == 1

    assert (
        registry_index
        .experiments[0]
        .experiment_id
        == experiment.experiment_id
    )


def test_register_same_experiment_updates_index(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une expérience existante est mise
    à jour sans créer de doublon.
    """

    initial_experiment = create_experiment(
        experiment_id="run_test_001",
        champion_model_name="random_forest",
    )

    register_experiment(
        tmp_path,
        initial_experiment,
    )

    updated_experiment = (
        initial_experiment.model_copy(
            update={
                "champion_model_name": (
                    "gradient_boosting"
                ),
                "champion_model_version": (
                    "2.0.0"
                ),
            }
        )
    )

    register_experiment(
        tmp_path,
        updated_experiment,
    )

    registry_index = load_registry_index(
        tmp_path
    )

    assert registry_index.experiment_count == 1

    assert len(
        registry_index.experiments
    ) == 1

    index_entry = (
        registry_index.experiments[0]
    )

    assert index_entry.champion_model_name == (
        "gradient_boosting"
    )

    assert index_entry.champion_model_version == (
        "2.0.0"
    )

    loaded_experiment = (
        load_experiment_record(
            tmp_path,
            "run_test_001",
        )
    )

    assert (
        loaded_experiment
        .champion_model_name
        == "gradient_boosting"
    )


def test_register_multiple_experiments(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'enregistrement de plusieurs expériences.
    """

    first_experiment = create_experiment(
        experiment_id="run_first",
        champion_model_name="random_forest",
    )

    second_experiment = create_experiment(
        experiment_id="run_second",
        champion_model_name=(
            "gradient_boosting"
        ),
    )

    register_experiment(
        tmp_path,
        first_experiment,
    )

    register_experiment(
        tmp_path,
        second_experiment,
    )

    registry_index = load_registry_index(
        tmp_path
    )

    assert registry_index.experiment_count == 2

    assert len(
        registry_index.experiments
    ) == 2


def test_list_experiments_returns_newest_first(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la dernière expérience enregistrée
    apparaît en première position.
    """

    first_experiment = create_experiment(
        experiment_id="run_first",
        champion_model_name="random_forest",
    )

    second_experiment = create_experiment(
        experiment_id="run_second",
        champion_model_name=(
            "gradient_boosting"
        ),
    )

    register_experiment(
        tmp_path,
        first_experiment,
    )

    register_experiment(
        tmp_path,
        second_experiment,
    )

    experiments = list_experiments(
        tmp_path
    )

    assert len(experiments) == 2

    assert experiments[0].experiment_id == (
        "run_second"
    )

    assert experiments[1].experiment_id == (
        "run_first"
    )


def test_list_experiments_for_empty_registry(
    tmp_path: Path,
) -> None:
    """
    Vérifie la liste d'un registre vide.
    """

    experiments = list_experiments(
        tmp_path
    )

    assert experiments == []