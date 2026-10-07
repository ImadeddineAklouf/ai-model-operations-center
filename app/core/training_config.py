from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.training_schemas import TrainingConfig


DEFAULT_TRAINING_CONFIG_PATH = (
    settings.project_root
    / "config"
    / "training.yaml"
)


def resolve_project_path(
    configured_path: str | Path,
) -> Path:
    """
    Résout un chemin à partir de la racine du projet.

    Un chemin déjà absolu est conservé.
    Un chemin relatif est rattaché à settings.project_root.
    """

    path = Path(
        configured_path
    )

    if not path.is_absolute():
        path = (
            settings.project_root
            / path
        )

    return path.resolve()


def load_yaml_file(
    configuration_path: str | Path,
) -> dict[str, Any]:
    """
    Charge un fichier YAML et vérifie que sa racine
    contient bien un dictionnaire.
    """

    resolved_path = Path(
        configuration_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configuration d'entraînement introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin de configuration ne correspond "
            f"pas à un fichier : {resolved_path}"
        )

    with resolved_path.open(
        mode="r",
        encoding="utf-8",
    ) as configuration_file:
        raw_configuration = yaml.safe_load(
            configuration_file
        )

    if not isinstance(
        raw_configuration,
        dict,
    ):
        raise ValueError(
            "La configuration YAML doit contenir "
            "un objet à sa racine."
        )

    return raw_configuration


def normalize_training_paths(
    training_configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Convertit les chemins relatifs de la configuration
    en chemins absolus.

    Le dictionnaire d'origine n'est pas modifié.
    """

    if "output" not in training_configuration:
        raise ValueError(
            "La section 'output' est absente "
            "de la configuration d'entraînement."
        )

    output_configuration = training_configuration[
        "output"
    ]

    if not isinstance(
        output_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'output' doit être un objet."
        )

    required_output_paths = [
        "processed_data_directory",
        "artifacts_directory",
        "reports_directory",
    ]

    missing_output_paths = [
        field_name
        for field_name in required_output_paths
        if field_name not in output_configuration
    ]

    if missing_output_paths:
        raise ValueError(
            "Chemin(s) de sortie absent(s) : "
            f"{missing_output_paths}"
        )

    if "dataset_path" not in training_configuration:
        raise ValueError(
            "Le champ 'dataset_path' est absent "
            "de la configuration d'entraînement."
        )

    return {
        **training_configuration,
        "dataset_path": resolve_project_path(
            training_configuration[
                "dataset_path"
            ]
        ),
        "output": {
            **output_configuration,
            "processed_data_directory": resolve_project_path(
                output_configuration[
                    "processed_data_directory"
                ]
            ),
            "artifacts_directory": resolve_project_path(
                output_configuration[
                    "artifacts_directory"
                ]
            ),
            "reports_directory": resolve_project_path(
                output_configuration[
                    "reports_directory"
                ]
            ),
        },
    }


def load_training_config(
    configuration_path: str | Path = (
        DEFAULT_TRAINING_CONFIG_PATH
    ),
) -> TrainingConfig:
    """
    Charge, normalise et valide la configuration
    d'entraînement.

    Returns:
        Une instance immutable de TrainingConfig.
    """

    raw_configuration = load_yaml_file(
        configuration_path
    )

    if "training" not in raw_configuration:
        raise ValueError(
            "La section 'training' est absente "
            "de la configuration."
        )

    training_configuration = raw_configuration[
        "training"
    ]

    if not isinstance(
        training_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'training' doit être un objet."
        )

    normalized_configuration = (
        normalize_training_paths(
            training_configuration
        )
    )

    return TrainingConfig.model_validate(
        normalized_configuration
    )