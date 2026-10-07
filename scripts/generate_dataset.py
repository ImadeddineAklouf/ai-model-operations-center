from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)


CONFIGURATION_PATH = (
    settings.project_root
    / "config"
    / "dataset.yaml"
)


def load_configuration(
    configuration_path: Path,
) -> tuple[DatasetGenerationConfig, Path]:
    """
    Charge le YAML et retourne :

    - la configuration Pydantic ;
    - le chemin absolu du fichier CSV.
    """

    if not configuration_path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : "
            f"{configuration_path}"
        )

    with configuration_path.open(
        mode="r",
        encoding="utf-8",
    ) as configuration_file:
        raw_configuration: dict[
            str,
            Any,
        ] = yaml.safe_load(
            configuration_file
        )

    dataset_configuration = raw_configuration[
        "dataset"
    ]

    generation_configuration = raw_configuration[
        "generation"
    ]

    class_distribution = generation_configuration[
        "class_distribution"
    ]

    config = DatasetGenerationConfig(
        number_of_records=generation_configuration[
            "number_of_records"
        ],
        random_seed=generation_configuration[
            "random_seed"
        ],
        good_ratio=class_distribution[
            "good"
        ],
        acceptable_ratio=class_distribution[
            "acceptable"
        ],
        poor_ratio=class_distribution[
            "poor"
        ],
    )

    output_path = Path(
        dataset_configuration[
            "output_file"
        ]
    )

    if not output_path.is_absolute():
        output_path = (
            settings.project_root
            / output_path
        )

    return config, output_path.resolve()


def main() -> None:
    """
    Génère et écrit le dataset au format CSV.
    """

    config, output_path = load_configuration(
        CONFIGURATION_PATH
    )

    dataframe = generate_quality_dataset(
        config
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Dataset généré : {output_path}"
    )

    print(
        f"Dimensions : {dataframe.shape}"
    )

    print(
        "Distribution :"
    )

    print(
        dataframe[
            "quality_label"
        ].value_counts()
    )


if __name__ == "__main__":
    main()