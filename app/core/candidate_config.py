from pathlib import Path
from typing import Any

import yaml

from app.core.candidate_schemas import (
    CandidateExperimentConfig,
)
from app.core.config import settings


DEFAULT_CANDIDATE_CONFIG_PATH = (
    settings.project_root
    / "config"
    / "candidate_models.yaml"
)


def resolve_candidate_path(
    configured_path: str | Path,
) -> Path:
    """
    Résout un chemin depuis la racine du projet.

    Un chemin absolu est conservé.
    Un chemin relatif est rattaché à la racine du projet.
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


def load_candidate_yaml(
    configuration_path: str | Path,
) -> dict[str, Any]:
    """
    Charge un fichier YAML de configuration.

    Raises:
        FileNotFoundError:
            Si le fichier n'existe pas.

        ValueError:
            Si le chemin ne désigne pas un fichier
            ou si la racine YAML n'est pas un objet.
    """

    resolved_path = Path(
        configuration_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configuration des modèles candidats "
            f"introuvable : {resolved_path}"
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


def normalize_candidate_paths(
    candidate_configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Convertit les chemins relatifs de la configuration
    en chemins absolus.

    Le dictionnaire reçu n'est pas modifié.
    """

    if "input" not in candidate_configuration:
        raise ValueError(
            "La section 'input' est absente "
            "de la configuration des candidats."
        )

    if "output" not in candidate_configuration:
        raise ValueError(
            "La section 'output' est absente "
            "de la configuration des candidats."
        )

    input_configuration = candidate_configuration[
        "input"
    ]

    output_configuration = candidate_configuration[
        "output"
    ]

    if not isinstance(
        input_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'input' doit être un objet."
        )

    if not isinstance(
        output_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'output' doit être un objet."
        )

    if (
        "processed_data_directory"
        not in input_configuration
    ):
        raise ValueError(
            "Le champ 'processed_data_directory' "
            "est absent de la section 'input'."
        )

    if (
        "candidates_directory"
        not in output_configuration
    ):
        raise ValueError(
            "Le champ 'candidates_directory' "
            "est absent de la section 'output'."
        )

    if (
        "comparison_directory"
        not in output_configuration
    ):
        raise ValueError(
            "Le champ 'comparison_directory' "
            "est absent de la section 'output'."
        )

    normalized_input = {
        **input_configuration,
        "processed_data_directory": (
            resolve_candidate_path(
                input_configuration[
                    "processed_data_directory"
                ]
            )
        ),
    }

    normalized_output = {
        **output_configuration,
        "candidates_directory": (
            resolve_candidate_path(
                output_configuration[
                    "candidates_directory"
                ]
            )
        ),
        "comparison_directory": (
            resolve_candidate_path(
                output_configuration[
                    "comparison_directory"
                ]
            )
        ),
    }

    normalized_configuration = {
        **candidate_configuration,
        "input": normalized_input,
        "output": normalized_output,
    }

    return normalized_configuration

def load_candidate_config(
    configuration_path: str | Path = DEFAULT_CANDIDATE_CONFIG_PATH,
) -> CandidateExperimentConfig:
    """
    Charge, normalise et valide la configuration
    des modèles candidats.

    Args:
        configuration_path:
            Chemin du fichier YAML de configuration.

    Returns:
        Configuration validée des modèles candidats.

    Raises:
        ValueError:
            Si la section candidate_models est absente
            ou si sa structure est incorrecte.
    """

    raw_configuration = load_candidate_yaml(
        configuration_path
    )

    if "candidate_models" not in raw_configuration:
        raise ValueError(
            "La section 'candidate_models' "
            "est absente de la configuration."
        )

    candidate_configuration = raw_configuration[
        "candidate_models"
    ]

    if not isinstance(
        candidate_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'candidate_models' "
            "doit être un objet."
        )

    normalized_configuration = normalize_candidate_paths(
        candidate_configuration
    )

    return CandidateExperimentConfig.model_validate(
        normalized_configuration
    )