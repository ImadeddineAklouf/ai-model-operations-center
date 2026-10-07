import numpy as np
import pandas as pd
import pytest

from app.core.dataset_generator import (
    RATIO_COLUMNS,
    calculate_class_counts,
    clip_ratio,
    generate_acceptable_record,
    generate_good_record,
    generate_poor_record,
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
    DatasetQualityRecord,
    QualityLabel,
)


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


def build_generation_config(
    *,
    number_of_records: int = 100,
    random_seed: int = 42,
    good_ratio: float = 0.40,
    acceptable_ratio: float = 0.35,
    poor_ratio: float = 0.25,
) -> DatasetGenerationConfig:
    """
    Crée une configuration de génération adaptée
    aux tests unitaires.
    """

    return DatasetGenerationConfig(
        number_of_records=number_of_records,
        random_seed=random_seed,
        good_ratio=good_ratio,
        acceptable_ratio=acceptable_ratio,
        poor_ratio=poor_ratio,
    )


@pytest.mark.parametrize(
    ("input_value", "expected_value"),
    [
        (-1.0, 0.0),
        (-0.01, 0.0),
        (0.0, 0.0),
        (0.25, 0.25),
        (1.0, 1.0),
        (1.01, 1.0),
        (2.0, 1.0),
    ],
)
def test_clip_ratio(
    input_value: float,
    expected_value: float,
) -> None:
    """
    Vérifie que clip_ratio limite correctement
    les valeurs dans l'intervalle [0, 1].
    """

    result = clip_ratio(
        input_value
    )

    assert result == pytest.approx(
        expected_value
    )


def test_generate_good_record() -> None:
    """
    Vérifie la génération d'un enregistrement GOOD.
    """

    random_generator = np.random.default_rng(
        42
    )

    record = generate_good_record(
        random_generator
    )

    assert isinstance(
        record,
        DatasetQualityRecord,
    )

    assert record.quality_label is QualityLabel.GOOD
    assert record.row_count >= 1
    assert record.column_count >= 1


def test_generate_acceptable_record() -> None:
    """
    Vérifie la génération d'un enregistrement ACCEPTABLE.
    """

    random_generator = np.random.default_rng(
        42
    )

    record = generate_acceptable_record(
        random_generator
    )

    assert isinstance(
        record,
        DatasetQualityRecord,
    )

    assert (
        record.quality_label
        is QualityLabel.ACCEPTABLE
    )


def test_generate_poor_record() -> None:
    """
    Vérifie la génération d'un enregistrement POOR.
    """

    random_generator = np.random.default_rng(
        42
    )

    record = generate_poor_record(
        random_generator
    )

    assert isinstance(
        record,
        DatasetQualityRecord,
    )

    assert record.quality_label is QualityLabel.POOR


@pytest.mark.parametrize(
    "generation_function",
    [
        generate_good_record,
        generate_acceptable_record,
        generate_poor_record,
    ],
)
def test_generated_record_contains_valid_ratios(
    generation_function,
) -> None:
    """
    Vérifie que chaque type de générateur produit
    des ratios compris entre zéro et un.
    """

    random_generator = np.random.default_rng(
        42
    )

    record = generation_function(
        random_generator
    )

    for column_name in RATIO_COLUMNS:
        value = getattr(
            record,
            column_name,
        )

        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(
    "generation_function",
    [
        generate_good_record,
        generate_acceptable_record,
        generate_poor_record,
    ],
)
def test_generated_record_contains_valid_counts(
    generation_function,
) -> None:
    """
    Vérifie les compteurs et le délai générés.
    """

    random_generator = np.random.default_rng(
        42
    )

    record = generation_function(
        random_generator
    )

    assert record.row_count >= 1
    assert record.column_count >= 1

    assert (
        record.schema_change_count
        >= 0
    )

    assert (
        record.freshness_delay_hours
        >= 0.0
    )


def test_calculate_class_counts() -> None:
    """
    Vérifie la distribution 40/35/25
    pour un dataset de 100 observations.
    """

    config = build_generation_config(
        number_of_records=100
    )

    result = calculate_class_counts(
        config
    )

    assert result == {
        QualityLabel.GOOD: 40,
        QualityLabel.ACCEPTABLE: 35,
        QualityLabel.POOR: 25,
    }


def test_class_counts_match_total_for_non_round_number() -> None:
    """
    Vérifie que les arrondis ne modifient pas
    le nombre total demandé.
    """

    config = build_generation_config(
        number_of_records=101
    )

    result = calculate_class_counts(
        config
    )

    assert sum(result.values()) == 101


def test_generate_quality_dataset_returns_dataframe() -> None:
    """
    Vérifie le type de retour principal.
    """

    dataframe = generate_quality_dataset(
        build_generation_config()
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )


def test_generated_dataset_has_expected_shape() -> None:
    """
    Vérifie le nombre de lignes et de colonnes.
    """

    dataframe = generate_quality_dataset(
        build_generation_config(
            number_of_records=100
        )
    )

    assert dataframe.shape == (
        100,
        len(EXPECTED_COLUMNS),
    )


def test_generated_dataset_has_expected_columns() -> None:
    """
    Vérifie les noms et l'ordre des colonnes.
    """

    dataframe = generate_quality_dataset(
        build_generation_config()
    )

    assert list(
        dataframe.columns
    ) == EXPECTED_COLUMNS


def test_generated_dataset_has_expected_distribution() -> None:
    """
    Vérifie le nombre de lignes de chaque classe.
    """

    dataframe = generate_quality_dataset(
        build_generation_config(
            number_of_records=100
        )
    )

    class_counts = (
        dataframe["quality_label"]
        .value_counts()
        .to_dict()
    )

    assert class_counts == {
        QualityLabel.GOOD.value: 40,
        QualityLabel.ACCEPTABLE.value: 35,
        QualityLabel.POOR.value: 25,
    }


def test_all_generated_ratios_are_valid() -> None:
    """
    Vérifie tous les ratios du DataFrame.
    """

    dataframe = generate_quality_dataset(
        build_generation_config(
            number_of_records=300
        )
    )

    for column_name in RATIO_COLUMNS:
        assert (
            dataframe[column_name]
            .between(
                0.0,
                1.0,
                inclusive="both",
            )
            .all()
        )


def test_all_generated_counts_and_delays_are_valid() -> None:
    """
    Vérifie les champs numériques non ratio.
    """

    dataframe = generate_quality_dataset(
        build_generation_config(
            number_of_records=300
        )
    )

    assert (
        dataframe["row_count"] >= 1
    ).all()

    assert (
        dataframe["column_count"] >= 1
    ).all()

    assert (
        dataframe["schema_change_count"] >= 0
    ).all()

    assert (
        dataframe["freshness_delay_hours"] >= 0.0
    ).all()


def test_generated_dataset_has_no_missing_values() -> None:
    """
    Le dataset synthétique initial ne doit contenir
    aucune cellule vide.
    """

    dataframe = generate_quality_dataset(
        build_generation_config()
    )

    assert not dataframe.isna().any().any()


def test_generation_is_reproducible() -> None:
    """
    Une même seed doit produire exactement
    le même DataFrame.
    """

    config = build_generation_config(
        random_seed=42
    )

    first_dataframe = generate_quality_dataset(
        config
    )

    second_dataframe = generate_quality_dataset(
        config
    )

    pd.testing.assert_frame_equal(
        first_dataframe,
        second_dataframe,
    )


def test_different_seeds_produce_different_datasets() -> None:
    """
    Deux seeds différentes doivent produire
    des observations différentes.
    """

    first_dataframe = generate_quality_dataset(
        build_generation_config(
            random_seed=42
        )
    )

    second_dataframe = generate_quality_dataset(
        build_generation_config(
            random_seed=43
        )
    )

    assert not first_dataframe.equals(
        second_dataframe
    )


def test_good_class_has_better_average_metrics_than_poor_class() -> None:
    """
    Vérifie la cohérence métier globale du dataset.

    GOOD doit statistiquement avoir moins d'erreurs
    et de meilleures métriques d'intégrité que POOR.
    """

    config = build_generation_config(
        number_of_records=1_000,
        good_ratio=0.50,
        acceptable_ratio=0.00,
        poor_ratio=0.50,
    )

    dataframe = generate_quality_dataset(
        config
    )

    good_records = dataframe.loc[
        dataframe["quality_label"]
        == QualityLabel.GOOD.value
    ]

    poor_records = dataframe.loc[
        dataframe["quality_label"]
        == QualityLabel.POOR.value
    ]

    assert (
        good_records[
            "missing_value_ratio"
        ].mean()
        <
        poor_records[
            "missing_value_ratio"
        ].mean()
    )

    assert (
        good_records[
            "duplicate_row_ratio"
        ].mean()
        <
        poor_records[
            "duplicate_row_ratio"
        ].mean()
    )

    assert (
        good_records[
            "invalid_type_ratio"
        ].mean()
        <
        poor_records[
            "invalid_type_ratio"
        ].mean()
    )

    assert (
        good_records[
            "primary_key_uniqueness"
        ].mean()
        >
        poor_records[
            "primary_key_uniqueness"
        ].mean()
    )

    assert (
        good_records[
            "foreign_key_match_ratio"
        ].mean()
        >
        poor_records[
            "foreign_key_match_ratio"
        ].mean()
    )

    assert (
        good_records[
            "freshness_delay_hours"
        ].mean()
        <
        poor_records[
            "freshness_delay_hours"
        ].mean()
    )


def test_dataset_contains_only_expected_labels() -> None:
    """
    Vérifie qu'aucune classe inconnue n'est produite.
    """

    dataframe = generate_quality_dataset(
        build_generation_config()
    )

    actual_labels = set(
        dataframe[
            "quality_label"
        ].unique()
    )

    expected_labels = {
        QualityLabel.GOOD.value,
        QualityLabel.ACCEPTABLE.value,
        QualityLabel.POOR.value,
    }

    assert actual_labels == expected_labels