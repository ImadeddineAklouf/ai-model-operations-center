from collections.abc import Callable

import numpy as np
import pandas as pd

from app.core.quality_schemas import (
    DatasetGenerationConfig,
    DatasetQualityRecord,
    QualityLabel,
)


GeneratorFunction = Callable[
    [np.random.Generator],
    DatasetQualityRecord,
]


RATIO_COLUMNS = [
    "missing_value_ratio",
    "duplicate_row_ratio",
    "invalid_type_ratio",
    "primary_key_uniqueness",
    "foreign_key_match_ratio",
    "outlier_ratio",
]


EXPECTED_COLUMNS = [
"row_count",
"column_count",
"missing_value_ratio",
"duplicate_row_ratio",
"invalid_type_ratio",
"primary_key_uniqueness",
"foreign_key_match_ratio",
"schema_change_count",
"outlier_ratio",
"freshness_delay_hours",
"quality_label",
]

def clip_ratio(value: float) -> float:
    """
    Limite un ratio dans l'intervalle [0, 1].

    Les distributions normales peuvent produire
    exceptionnellement des valeurs négatives ou
    supérieures à 1.
    """

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def generate_good_record(
    generator: np.random.Generator,
) -> DatasetQualityRecord:
    """
    Génère un enregistrement de bonne qualité.
    """

    return DatasetQualityRecord(
        row_count=int(
            generator.integers(
                low=5_000,
                high=500_001,
            )
        ),
        column_count=int(
            generator.integers(
                low=5,
                high=51,
            )
        ),
        missing_value_ratio=clip_ratio(
            generator.normal(
                loc=0.02,
                scale=0.015,
            )
        ),
        duplicate_row_ratio=clip_ratio(
            generator.normal(
                loc=0.01,
                scale=0.01,
            )
        ),
        invalid_type_ratio=clip_ratio(
            generator.normal(
                loc=0.005,
                scale=0.005,
            )
        ),
        primary_key_uniqueness=clip_ratio(
            generator.normal(
                loc=0.995,
                scale=0.005,
            )
        ),
        foreign_key_match_ratio=clip_ratio(
            generator.normal(
                loc=0.99,
                scale=0.01,
            )
        ),
        schema_change_count=int(
            generator.integers(
                low=0,
                high=2,
            )
        ),
        outlier_ratio=clip_ratio(
            generator.normal(
                loc=0.02,
                scale=0.015,
            )
        ),
        freshness_delay_hours=max(
            0.0,
            float(
                generator.normal(
                    loc=2.0,
                    scale=1.5,
                )
            ),
        ),
        quality_label=QualityLabel.GOOD,
    )


def generate_acceptable_record(
    generator: np.random.Generator,
) -> DatasetQualityRecord:
    """
    Génère un enregistrement de qualité acceptable.
    """

    return DatasetQualityRecord(
        row_count=int(
            generator.integers(
                low=1_000,
                high=250_001,
            )
        ),
        column_count=int(
            generator.integers(
                low=5,
                high=61,
            )
        ),
        missing_value_ratio=clip_ratio(
            generator.normal(
                loc=0.12,
                scale=0.05,
            )
        ),
        duplicate_row_ratio=clip_ratio(
            generator.normal(
                loc=0.06,
                scale=0.03,
            )
        ),
        invalid_type_ratio=clip_ratio(
            generator.normal(
                loc=0.04,
                scale=0.025,
            )
        ),
        primary_key_uniqueness=clip_ratio(
            generator.normal(
                loc=0.95,
                scale=0.03,
            )
        ),
        foreign_key_match_ratio=clip_ratio(
            generator.normal(
                loc=0.90,
                scale=0.05,
            )
        ),
        schema_change_count=int(
            generator.integers(
                low=0,
                high=4,
            )
        ),
        outlier_ratio=clip_ratio(
            generator.normal(
                loc=0.08,
                scale=0.04,
            )
        ),
        freshness_delay_hours=max(
            0.0,
            float(
                generator.normal(
                    loc=18.0,
                    scale=8.0,
                )
            ),
        ),
        quality_label=QualityLabel.ACCEPTABLE,
    )


def generate_poor_record(
    generator: np.random.Generator,
) -> DatasetQualityRecord:
    """
    Génère un enregistrement de mauvaise qualité.
    """

    return DatasetQualityRecord(
        row_count=int(
            generator.integers(
                low=100,
                high=100_001,
            )
        ),
        column_count=int(
            generator.integers(
                low=3,
                high=81,
            )
        ),
        missing_value_ratio=clip_ratio(
            generator.normal(
                loc=0.40,
                scale=0.12,
            )
        ),
        duplicate_row_ratio=clip_ratio(
            generator.normal(
                loc=0.22,
                scale=0.10,
            )
        ),
        invalid_type_ratio=clip_ratio(
            generator.normal(
                loc=0.18,
                scale=0.08,
            )
        ),
        primary_key_uniqueness=clip_ratio(
            generator.normal(
                loc=0.72,
                scale=0.12,
            )
        ),
        foreign_key_match_ratio=clip_ratio(
            generator.normal(
                loc=0.58,
                scale=0.16,
            )
        ),
        schema_change_count=int(
            generator.integers(
                low=2,
                high=9,
            )
        ),
        outlier_ratio=clip_ratio(
            generator.normal(
                loc=0.28,
                scale=0.12,
            )
        ),
        freshness_delay_hours=max(
            0.0,
            float(
                generator.normal(
                    loc=96.0,
                    scale=40.0,
                )
            ),
        ),
        quality_label=QualityLabel.POOR,
    )


def calculate_class_counts(
    config: DatasetGenerationConfig,
) -> dict[QualityLabel, int]:
    """
    Calcule le nombre de lignes de chaque classe.

    POOR reçoit le reste pour garantir que le total
    correspond exactement à number_of_records.
    """

    good_count = round(
        config.number_of_records
        * config.good_ratio
    )

    acceptable_count = round(
        config.number_of_records
        * config.acceptable_ratio
    )

    poor_count = (
        config.number_of_records
        - good_count
        - acceptable_count
    )

    return {
        QualityLabel.GOOD: good_count,
        QualityLabel.ACCEPTABLE: acceptable_count,
        QualityLabel.POOR: poor_count,
    }


def generate_quality_dataset(
    config: DatasetGenerationConfig,
) -> pd.DataFrame:
    """
    Génère le dataset complet en mémoire.

    Cette fonction ne crée pas encore de fichier CSV.
    Elle retourne uniquement un DataFrame Pandas.
    """

    random_generator = np.random.default_rng(
        config.random_seed
    )

    class_counts = calculate_class_counts(
        config
    )

    generator_functions: dict[
        QualityLabel,
        GeneratorFunction,
    ] = {
        QualityLabel.GOOD: generate_good_record,
        QualityLabel.ACCEPTABLE: generate_acceptable_record,
        QualityLabel.POOR: generate_poor_record,
    }

    records: list[dict[str, object]] = []

    for quality_label, record_count in class_counts.items():
        record_generator = generator_functions[
            quality_label
        ]

        for _ in range(record_count):
            record = record_generator(
                random_generator
            )

            records.append(
                record.model_dump(
                    mode="json",
                )
            )

    dataframe = pd.DataFrame(
        records,
        columns=EXPECTED_COLUMNS,
    )

    dataframe = dataframe.sample(
        frac=1.0,
        random_state=config.random_seed,
    ).reset_index(
        drop=True
    )

    return dataframe