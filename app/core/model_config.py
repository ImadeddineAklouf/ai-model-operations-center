from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.model_schemas import BaselineModelConfig


DEFAULT_BASELINE_CONFIG_PATH = (
    settings.project_root
    / "config"
    / "baseline_model.yaml"
)


def resolve_model_path(
    configured_path: str | Path,
) -> Path:
    """
    Résout un chemin depuis la racine du projet.
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


def load_baseline_model_config(
    configuration_path: str | Path = (
        DEFAULT_BASELINE_CONFIG_PATH
    ),
) -> BaselineModelConfig:
    """
    Charge et valide la configuration du modèle baseline.
    """

    resolved_path = Path(
        configuration_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configuration du modèle introuvable : "
            f"{resolved_path}"
        )

    with resolved_path.open(
        mode="r",
        encoding="utf-8",
    ) as configuration_file:
        raw_configuration: Any = yaml.safe_load(
            configuration_file
        )

    if not isinstance(
        raw_configuration,
        dict,
    ):
        raise ValueError(
            "La configuration YAML doit être un objet."
        )

    if "baseline_model" not in raw_configuration:
        raise ValueError(
            "La section 'baseline_model' est absente."
        )

    model_configuration = raw_configuration[
        "baseline_model"
    ]

    if not isinstance(
        model_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'baseline_model' "
            "doit être un objet."
        )

    input_configuration = model_configuration[
        "input"
    ]

    output_configuration = model_configuration[
        "output"
    ]

    normalized_configuration = {
        **model_configuration,
        "input": {
            **input_configuration,
            "processed_data_directory": (
                resolve_model_path(
                    input_configuration[
                        "processed_data_directory"
                    ]
                )
            ),
        },
        "output": {
            **output_configuration,
            "model_directory": resolve_model_path(
                output_configuration[
                    "model_directory"
                ]
            ),
            "evaluation_directory": (
                resolve_model_path(
                    output_configuration[
                        "evaluation_directory"
                    ]
                )
            ),
        },
    }

    return BaselineModelConfig.model_validate(
        normalized_configuration
    )