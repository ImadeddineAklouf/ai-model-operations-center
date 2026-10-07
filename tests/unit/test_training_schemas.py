from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.training_schemas import (
    DataPreparationReport,
    DataSplitConfig,
    DatasetSplitSummary,
    PreprocessingConfig,
    TrainingConfig,
    TrainingOutputConfig,
)


def create_valid_output_config() -> TrainingOutputConfig:
    """
    Crée une configuration valide des dossiers de sortie.
    """

    return TrainingOutputConfig(
        processed_data_directory=Path(
            "data/processed"
        ),
        artifacts_directory=Path(
            "models/preprocessing"
        ),
        reports_directory=Path(
            "reports/training"
        ),
    )


def create_valid_training_config() -> TrainingConfig:
    """
    Crée une configuration complète et valide.
    """

    return TrainingConfig(
        dataset_path=Path(
            "data/raw/dataset_quality.csv"
        ),
        target_column="quality_label",
        split=DataSplitConfig(),
        preprocessing=PreprocessingConfig(),
        output=create_valid_output_config(),
    )


def create_valid_split_summary(
    row_count: int = 100,
) -> DatasetSplitSummary:
    """
    Crée un résumé valide d'un sous-ensemble.
    """

    return DatasetSplitSummary(
        row_count=row_count,
        feature_count=10,
        class_distribution={
            "GOOD": 40,
            "ACCEPTABLE": 35,
            "POOR": 25,
        },
    )


def test_default_data_split_config() -> None:
    """
    Vérifie les valeurs par défaut de la séparation.
    """

    config = DataSplitConfig()

    assert config.train_ratio == pytest.approx(
        0.70
    )

    assert config.validation_ratio == pytest.approx(
        0.15
    )

    assert config.test_ratio == pytest.approx(
        0.15
    )

    assert config.random_seed == 42
    assert config.stratify is True


def test_custom_data_split_config() -> None:
    """
    Vérifie une autre répartition valide.
    """

    config = DataSplitConfig(
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        random_seed=100,
        stratify=False,
    )

    assert config.train_ratio == pytest.approx(
        0.60
    )

    assert config.validation_ratio == pytest.approx(
        0.20
    )

    assert config.test_ratio == pytest.approx(
        0.20
    )

    assert config.random_seed == 100
    assert config.stratify is False


def test_invalid_ratio_sum_is_rejected() -> None:
    """
    Vérifie que la somme des ratios doit être égale à 1.
    """

    with pytest.raises(
        ValidationError,
        match="somme des proportions",
    ):
        DataSplitConfig(
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
        )


def test_zero_train_ratio_is_rejected() -> None:
    """
    Vérifie que le ratio d'entraînement ne peut pas valoir 0.
    """

    with pytest.raises(ValidationError):
        DataSplitConfig(
            train_ratio=0.0,
            validation_ratio=0.50,
            test_ratio=0.50,
        )


def test_zero_validation_ratio_is_rejected() -> None:
    """
    Vérifie que le ratio de validation ne peut pas valoir 0.
    """

    with pytest.raises(ValidationError):
        DataSplitConfig(
            train_ratio=0.80,
            validation_ratio=0.0,
            test_ratio=0.20,
        )


def test_zero_test_ratio_is_rejected() -> None:
    """
    Vérifie que le ratio de test ne peut pas valoir 0.
    """

    with pytest.raises(ValidationError):
        DataSplitConfig(
            train_ratio=0.80,
            validation_ratio=0.20,
            test_ratio=0.0,
        )


def test_negative_random_seed_is_rejected() -> None:
    """
    Vérifie que la seed ne peut pas être négative.
    """

    with pytest.raises(ValidationError):
        DataSplitConfig(
            random_seed=-1,
        )


def test_unknown_split_parameter_is_rejected() -> None:
    """
    Vérifie extra='forbid' sur DataSplitConfig.
    """

    configuration = {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
        "stratify": True,
        "unknown_parameter": "value",
    }

    with pytest.raises(ValidationError):
        DataSplitConfig.model_validate(
            configuration
        )


def test_data_split_config_is_immutable() -> None:
    """
    Vérifie frozen=True.
    """

    config = DataSplitConfig()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "random_seed",
            100,
        )


def test_default_preprocessing_config() -> None:
    """
    Vérifie la configuration de prétraitement par défaut.
    """

    config = PreprocessingConfig()

    assert config.imputation_strategy == "median"
    assert config.scale_features is True


@pytest.mark.parametrize(
    "strategy",
    [
        "mean",
        "median",
        "most_frequent",
    ],
)
def test_valid_imputation_strategy(
    strategy: str,
) -> None:
    """
    Vérifie les stratégies d'imputation autorisées.
    """

    config = PreprocessingConfig.model_validate(
        {
            "imputation_strategy": strategy,
            "scale_features": True,
        }
    )

    assert config.imputation_strategy == strategy


def test_invalid_imputation_strategy_is_rejected() -> None:
    """
    Vérifie qu'une stratégie inconnue est refusée.
    """

    with pytest.raises(ValidationError):
        PreprocessingConfig.model_validate(
            {
                "imputation_strategy": "random",
                "scale_features": True,
            }
        )


def test_unknown_preprocessing_parameter_is_rejected() -> None:
    """
    Vérifie extra='forbid' sur PreprocessingConfig.
    """

    with pytest.raises(ValidationError):
        PreprocessingConfig.model_validate(
            {
                "imputation_strategy": "median",
                "scale_features": True,
                "unknown_parameter": "value",
            }
        )


def test_preprocessing_config_is_immutable() -> None:
    """
    Vérifie frozen=True sur PreprocessingConfig.
    """

    config = PreprocessingConfig()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "scale_features",
            False,
        )


def test_valid_training_output_config() -> None:
    """
    Vérifie les chemins de sortie.
    """

    config = create_valid_output_config()

    assert config.processed_data_directory == Path(
        "data/processed"
    )

    assert config.artifacts_directory == Path(
        "models/preprocessing"
    )

    assert config.reports_directory == Path(
        "reports/training"
    )


def test_missing_output_directory_is_rejected() -> None:
    """
    Vérifie que les trois chemins de sortie
    sont obligatoires.
    """

    incomplete_configuration = {
        "processed_data_directory": (
            "data/processed"
        ),
        "artifacts_directory": (
            "models/preprocessing"
        ),
    }

    with pytest.raises(ValidationError):
        TrainingOutputConfig.model_validate(
            incomplete_configuration
        )


def test_unknown_output_parameter_is_rejected() -> None:
    """
    Vérifie extra='forbid' sur TrainingOutputConfig.
    """

    configuration = {
        "processed_data_directory": (
            "data/processed"
        ),
        "artifacts_directory": (
            "models/preprocessing"
        ),
        "reports_directory": (
            "reports/training"
        ),
        "unknown_directory": "unknown",
    }

    with pytest.raises(ValidationError):
        TrainingOutputConfig.model_validate(
            configuration
        )


def test_training_output_config_is_immutable() -> None:
    """
    Vérifie frozen=True sur TrainingOutputConfig.
    """

    config = create_valid_output_config()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "reports_directory",
            Path("another/directory"),
        )


def test_valid_training_config() -> None:
    """
    Vérifie la configuration complète.
    """

    config = create_valid_training_config()

    assert config.dataset_path == Path(
        "data/raw/dataset_quality.csv"
    )

    assert config.target_column == "quality_label"

    assert config.split.train_ratio == pytest.approx(
        0.70
    )

    assert (
        config.preprocessing.imputation_strategy
        == "median"
    )

    assert config.output.artifacts_directory == Path(
        "models/preprocessing"
    )


def test_empty_target_column_is_rejected() -> None:
    """
    Vérifie que target_column ne peut pas être vide.
    """

    with pytest.raises(ValidationError):
        TrainingConfig(
            dataset_path=Path(
                "data/raw/dataset_quality.csv"
            ),
            target_column="",
            split=DataSplitConfig(),
            preprocessing=PreprocessingConfig(),
            output=create_valid_output_config(),
        )


def test_missing_dataset_path_is_rejected() -> None:
    """
    Vérifie que dataset_path est obligatoire.
    """

    incomplete_configuration = {
        "target_column": "quality_label",
        "split": {
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "stratify": True,
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "scale_features": True,
        },
        "output": {
            "processed_data_directory": (
                "data/processed"
            ),
            "artifacts_directory": (
                "models/preprocessing"
            ),
            "reports_directory": (
                "reports/training"
            ),
        },
    }

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            incomplete_configuration
        )


def test_unknown_training_parameter_is_rejected() -> None:
    """
    Vérifie extra='forbid' sur TrainingConfig.
    """

    configuration = (
        create_valid_training_config()
        .model_dump()
    )

    configuration[
        "unknown_parameter"
    ] = "value"

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            configuration
        )


def test_training_config_is_immutable() -> None:
    """
    Vérifie frozen=True sur TrainingConfig.
    """

    config = create_valid_training_config()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "target_column",
            "new_target",
        )


def test_valid_dataset_split_summary() -> None:
    """
    Vérifie un résumé valide.
    """

    summary = create_valid_split_summary()

    assert summary.row_count == 100
    assert summary.feature_count == 10

    assert summary.class_distribution == {
        "GOOD": 40,
        "ACCEPTABLE": 35,
        "POOR": 25,
    }


def test_negative_summary_row_count_is_rejected() -> None:
    """
    Vérifie que row_count ne peut pas être négatif.
    """

    with pytest.raises(ValidationError):
        DatasetSplitSummary(
            row_count=-1,
            feature_count=10,
            class_distribution={},
        )


def test_negative_summary_feature_count_is_rejected() -> None:
    """
    Vérifie que feature_count ne peut pas être négatif.
    """

    with pytest.raises(ValidationError):
        DatasetSplitSummary(
            row_count=100,
            feature_count=-1,
            class_distribution={},
        )


def test_unknown_summary_parameter_is_rejected() -> None:
    """
    Vérifie extra='forbid' sur DatasetSplitSummary.
    """

    with pytest.raises(ValidationError):
        DatasetSplitSummary.model_validate(
            {
                "row_count": 100,
                "feature_count": 10,
                "class_distribution": {
                    "GOOD": 40,
                    "ACCEPTABLE": 35,
                    "POOR": 25,
                },
                "unknown_parameter": "value",
            }
        )


def test_valid_data_preparation_report() -> None:
    """
    Vérifie un rapport complet valide.
    """

    train_summary = DatasetSplitSummary(
        row_count=2100,
        feature_count=10,
        class_distribution={
            "GOOD": 840,
            "ACCEPTABLE": 735,
            "POOR": 525,
        },
    )

    validation_summary = DatasetSplitSummary(
        row_count=450,
        feature_count=10,
        class_distribution={
            "GOOD": 180,
            "ACCEPTABLE": 157,
            "POOR": 113,
        },
    )

    test_summary = DatasetSplitSummary(
        row_count=450,
        feature_count=10,
        class_distribution={
            "GOOD": 180,
            "ACCEPTABLE": 158,
            "POOR": 112,
        },
    )

    report = DataPreparationReport(
        source_dataset_path=(
            "data/raw/dataset_quality.csv"
        ),
        target_column="quality_label",
        random_seed=42,
        stratified=True,
        feature_names=[
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
        ],
        train=train_summary,
        validation=validation_summary,
        test=test_summary,
    )

    assert report.source_dataset_path == (
        "data/raw/dataset_quality.csv"
    )

    assert report.target_column == "quality_label"
    assert report.random_seed == 42
    assert report.stratified is True

    assert report.train.row_count == 2100
    assert report.validation.row_count == 450
    assert report.test.row_count == 450

    assert len(report.feature_names) == 10


def test_data_preparation_report_rejects_unknown_field() -> None:
    """
    Vérifie extra='forbid' sur DataPreparationReport.
    """

    summary = create_valid_split_summary()

    configuration = {
        "source_dataset_path": (
            "data/raw/dataset_quality.csv"
        ),
        "target_column": "quality_label",
        "random_seed": 42,
        "stratified": True,
        "feature_names": [
            "row_count",
            "column_count",
        ],
        "train": summary.model_dump(),
        "validation": summary.model_dump(),
        "test": summary.model_dump(),
        "unknown_parameter": "value",
    }

    with pytest.raises(ValidationError):
        DataPreparationReport.model_validate(
            configuration
        )