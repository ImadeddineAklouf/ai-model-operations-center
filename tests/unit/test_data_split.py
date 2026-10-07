import pandas as pd
import pytest

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)
from app.core.training_schemas import DataSplitConfig
from training.data_split import (
    calculate_split_sizes,
    separate_features_and_target,
    split_dataset,
)


def create_test_dataframe(
    number_of_records: int = 1000,
) -> pd.DataFrame:
    """
    Génère un dataset reproductible.
    """

    return generate_quality_dataset(
        DatasetGenerationConfig(
            number_of_records=number_of_records,
            random_seed=42,
            good_ratio=0.40,
            acceptable_ratio=0.35,
            poor_ratio=0.25,
        )
    )


def create_split_config() -> DataSplitConfig:
    """
    Retourne la configuration 70/15/15.
    """

    return DataSplitConfig(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify=True,
    )


def test_separate_features_and_target() -> None:
    dataframe = create_test_dataframe(
        number_of_records=100
    )

    features, target = separate_features_and_target(
        dataframe=dataframe,
        target_column="quality_label",
    )

    assert features.shape == (100, 10)
    assert target.shape == (100,)
    assert "quality_label" not in features.columns
    assert target.name == "quality_label"


def test_empty_dataframe_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="vide",
    ):
        separate_features_and_target(
            dataframe=pd.DataFrame(),
            target_column="quality_label",
        )


def test_missing_target_column_is_rejected() -> None:
    dataframe = create_test_dataframe(
        number_of_records=100
    )

    with pytest.raises(
        ValueError,
        match="colonne cible",
    ):
        separate_features_and_target(
            dataframe=dataframe,
            target_column="unknown_target",
        )


def test_dataset_without_features_is_rejected() -> None:
    dataframe = pd.DataFrame(
        {
            "quality_label": [
                "GOOD",
                "POOR",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="aucune feature",
    ):
        separate_features_and_target(
            dataframe=dataframe,
            target_column="quality_label",
        )


def test_split_has_expected_sizes() -> None:
    splits = split_dataset(
        dataframe=create_test_dataframe(
            number_of_records=1000
        ),
        target_column="quality_label",
        config=create_split_config(),
    )

    assert splits.X_train.shape == (700, 10)

    assert splits.X_validation.shape == (
        150,
        10,
    )

    assert splits.X_test.shape == (150, 10)

    assert splits.y_train.shape == (700,)

    assert splits.y_validation.shape == (
        150,
    )

    assert splits.y_test.shape == (150,)


def test_features_and_targets_are_aligned() -> None:
    splits = split_dataset(
        dataframe=create_test_dataframe(),
        target_column="quality_label",
        config=create_split_config(),
    )

    assert len(splits.X_train) == len(
        splits.y_train
    )

    assert len(splits.X_validation) == len(
        splits.y_validation
    )

    assert len(splits.X_test) == len(
        splits.y_test
    )


def test_all_splits_contain_all_classes() -> None:
    splits = split_dataset(
        dataframe=create_test_dataframe(),
        target_column="quality_label",
        config=create_split_config(),
    )

    expected_labels = {
        "GOOD",
        "ACCEPTABLE",
        "POOR",
    }

    assert set(
        splits.y_train.unique()
    ) == expected_labels

    assert set(
        splits.y_validation.unique()
    ) == expected_labels

    assert set(
        splits.y_test.unique()
    ) == expected_labels


def test_split_is_reproducible() -> None:
    dataframe = create_test_dataframe()
    config = create_split_config()

    first_splits = split_dataset(
        dataframe=dataframe,
        target_column="quality_label",
        config=config,
    )

    second_splits = split_dataset(
        dataframe=dataframe,
        target_column="quality_label",
        config=config,
    )

    pd.testing.assert_frame_equal(
        first_splits.X_train,
        second_splits.X_train,
    )

    pd.testing.assert_series_equal(
        first_splits.y_train,
        second_splits.y_train,
    )

    pd.testing.assert_frame_equal(
        first_splits.X_validation,
        second_splits.X_validation,
    )

    pd.testing.assert_frame_equal(
        first_splits.X_test,
        second_splits.X_test,
    )


def test_train_split_preserves_class_distribution() -> None:
    dataframe = create_test_dataframe()

    splits = split_dataset(
        dataframe=dataframe,
        target_column="quality_label",
        config=create_split_config(),
    )

    original_distribution = (
        dataframe["quality_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    train_distribution = (
        splits.y_train
        .value_counts(normalize=True)
        .sort_index()
    )

    for label in original_distribution.index:
        assert train_distribution[
            label
        ] == pytest.approx(
            original_distribution[label],
            abs=0.01,
        )

def test_calculate_split_sizes_for_one_hundred_rows() -> None:
    """
    Vérifie la séparation exacte de 100 observations.
    """

    sizes = calculate_split_sizes(
        number_of_records=100,
        config=create_split_config(),
    )

    assert sizes.train == 70
    assert sizes.validation == 15
    assert sizes.test == 15

    assert (
        sizes.train
        + sizes.validation
        + sizes.test
        == 100
    )


def test_calculate_split_sizes_for_three_thousand_rows() -> None:
    """
    Vérifie la séparation du dataset réel.
    """

    sizes = calculate_split_sizes(
        number_of_records=3000,
        config=create_split_config(),
    )

    assert sizes.train == 2100
    assert sizes.validation == 450
    assert sizes.test == 450


def test_split_sizes_always_match_total() -> None:
    """
    Vérifie le comportement avec un nombre non divisible.
    """

    sizes = calculate_split_sizes(
        number_of_records=101,
        config=create_split_config(),
    )

    assert (
        sizes.train
        + sizes.validation
        + sizes.test
        == 101
    )


def test_empty_record_count_is_rejected() -> None:
    """
    Vérifie qu'un nombre nul est refusé.
    """

    with pytest.raises(
        ValueError,
        match="doit être positif",
    ):
        calculate_split_sizes(
            number_of_records=0,
            config=create_split_config(),
        )