import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.registry_schemas import (
    ExperimentIndexEntry,
    ExperimentRecord,
    ExperimentRegistryIndex,
)


REGISTRY_INDEX_FILENAME = "experiments.json"
RUNS_DIRECTORY_NAME = "runs"
EXPERIMENT_FILENAME = "experiment.json"


def utc_now_iso() -> str:
    """
    Retourne la date UTC actuelle au format ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def generate_experiment_id() -> str:
    """
    Génère un identifiant unique et lisible.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    short_identifier = uuid4().hex[:8]

    return (
        f"run_{timestamp}_"
        f"{short_identifier}"
    )


def get_registry_index_path(
    registry_directory: Path,
) -> Path:
    """
    Retourne le chemin de l'index global.
    """

    return (
        registry_directory.resolve()
        / REGISTRY_INDEX_FILENAME
    )


def get_experiment_directory(
    registry_directory: Path,
    experiment_id: str,
) -> Path:
    """
    Retourne le dossier d'une expérience.
    """

    if not experiment_id.strip():
        raise ValueError(
            "L'identifiant de l'expérience "
            "ne peut pas être vide."
        )

    return (
        registry_directory.resolve()
        / RUNS_DIRECTORY_NAME
        / experiment_id
    )


def load_registry_index(
    registry_directory: Path,
) -> ExperimentRegistryIndex:
    """
    Charge l'index du registre.

    Si le registre n'existe pas encore,
    retourne un index vide.
    """

    index_path = get_registry_index_path(
        registry_directory
    )

    if not index_path.exists():
        return ExperimentRegistryIndex()

    if not index_path.is_file():
        raise ValueError(
            "Le chemin de l'index du registre "
            "ne correspond pas à un fichier."
        )

    raw_content = json.loads(
        index_path.read_text(
            encoding="utf-8"
        )
    )

    return ExperimentRegistryIndex.model_validate(
        raw_content
    )


def save_registry_index(
    registry_directory: Path,
    registry_index: ExperimentRegistryIndex,
) -> Path:
    """
    Sauvegarde l'index global du registre.
    """

    registry_directory = (
        registry_directory.resolve()
    )

    registry_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = get_registry_index_path(
        registry_directory
    )

    index_path.write_text(
        registry_index.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return index_path


def save_experiment_record(
    registry_directory: Path,
    experiment: ExperimentRecord,
) -> Path:
    """
    Sauvegarde l'enregistrement complet
    d'une expérience.
    """

    experiment_directory = (
        get_experiment_directory(
            registry_directory,
            experiment.experiment_id,
        )
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_path = (
        experiment_directory
        / EXPERIMENT_FILENAME
    )

    experiment_path.write_text(
        experiment.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return experiment_path


def load_experiment_record(
    registry_directory: Path,
    experiment_id: str,
) -> ExperimentRecord:
    """
    Charge une expérience précise.
    """

    experiment_path = (
        get_experiment_directory(
            registry_directory,
            experiment_id,
        )
        / EXPERIMENT_FILENAME
    )

    if not experiment_path.exists():
        raise FileNotFoundError(
            "Expérience introuvable : "
            f"{experiment_id}"
        )

    raw_content = json.loads(
        experiment_path.read_text(
            encoding="utf-8"
        )
    )

    return ExperimentRecord.model_validate(
        raw_content
    )


def add_experiment_to_index(
    registry_directory: Path,
    experiment: ExperimentRecord,
    experiment_path: Path,
) -> ExperimentRegistryIndex:
    """
    Ajoute ou remplace une expérience
    dans l'index global.
    """

    registry_index = load_registry_index(
        registry_directory
    )

    remaining_entries = []

    for entry in registry_index.experiments:
        if (
            entry.experiment_id
            != experiment.experiment_id
        ):
            remaining_entries.append(
                entry
            )

    new_entry = ExperimentIndexEntry(
        experiment_id=(
            experiment.experiment_id
        ),
        created_at=experiment.created_at,
        completed_at=experiment.completed_at,
        status=experiment.status,
        champion_model_name=(
            experiment.champion_model_name
        ),
        champion_model_version=(
            experiment.champion_model_version
        ),
        experiment_path=str(
            experiment_path.resolve()
        ),
    )

    updated_entries = [
        new_entry,
        *remaining_entries,
    ]

    updated_index = ExperimentRegistryIndex(
        registry_version=(
            registry_index.registry_version
        ),
        experiment_count=len(
            updated_entries
        ),
        experiments=updated_entries,
    )

    save_registry_index(
        registry_directory,
        updated_index,
    )

    return updated_index


def register_experiment(
    registry_directory: Path,
    experiment: ExperimentRecord,
) -> Path:
    """
    Sauvegarde une expérience et met à jour l'index.
    """

    experiment_path = save_experiment_record(
        registry_directory,
        experiment,
    )

    add_experiment_to_index(
        registry_directory,
        experiment,
        experiment_path,
    )

    return experiment_path


def list_experiments(
    registry_directory: Path,
) -> list[ExperimentIndexEntry]:
    """
    Retourne les expériences enregistrées,
    de la plus récente à la plus ancienne.
    """

    registry_index = load_registry_index(
        registry_directory
    )

    return list(
        registry_index.experiments
    )