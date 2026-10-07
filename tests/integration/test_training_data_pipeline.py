from pathlib import Path

import joblib
import pandas as pd

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)
from app.core.training_schemas import (
    DataSplitConfig,
    PreprocessingConfig,
    TrainingConfig,
    TrainingOutputConfig,
)
from pipelines.prepare_training_data import (
    PREPROCESSOR_FILENAME,
    REPORT_FILENAME,
    X_TEST_FILENAME,
    X_TRAIN_FILENAME,
    X_VALIDATION_FILENAME,
    Y_TEST_FILENAME,
    Y_TRAIN_FILENAME,
    Y_VALIDATION_FILENAME,
    prepare_training_data,
)


def create_test_configuration(
    tmp_path: Path,
) -> TrainingConfig:
    """
    Crée une configuration isolée dans tmp_path.
    """

    dataset_path = (
        tmp_path
        / "data"
        / "raw"
        / "dataset_quality.csv"
    )

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_quality_dataset(
        DatasetGenerationConfig(
            number_of_records=100,
            random_seed=42,
            good_ratio=0.40,
            acceptable_ratio=0.35,
            poor_ratio=0.25,
        )
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
        encoding="utf-8",
    )

    return TrainingConfig(
        dataset_path=dataset_path,
        target_column="quality_label",
        split=DataSplitConfig(
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            stratify=True,
        ),
        preprocessing=PreprocessingConfig(
            imputation_strategy="median",
            scale_features=True,
        ),
        output=TrainingOutputConfig(
            processed_data_directory=(
                tmp_path
                / "data"
                / "processed"
            ),
            artifacts_directory=(
                tmp_path
                / "models"
                / "preprocessing"
            ),
            reports_directory=(
                tmp_path
                / "reports"
                / "training"
            ),
        ),
    )


def test_complete_training_data_pipeline(
    tmp_path: Path,
) -> None:
    """
    Vérifie le pipeline complet dans un environnement isolé.
    """

    config = create_test_configuration(
        tmp_path
    )

    report = prepare_training_data(
        config
    )

    processed_directory = (
        config.output.processed_data_directory
    )

    artifacts_directory = (
        config.output.artifacts_directory
    )

    reports_directory = (
        config.output.reports_directory
    )

    expected_processed_files = [
        X_TRAIN_FILENAME,
        X_VALIDATION_FILENAME,
        X_TEST_FILENAME,
        Y_TRAIN_FILENAME,
        Y_VALIDATION_FILENAME,
        Y_TEST_FILENAME,
    ]

    for filename in expected_processed_files:
        assert (
            processed_directory
            / filename
        ).exists()

    preprocessor_path = (
        artifacts_directory
        / PREPROCESSOR_FILENAME
    )

    report_path = (
        reports_directory
        / REPORT_FILENAME
    )

    assert preprocessor_path.exists()
    assert report_path.exists()

    X_train = pd.read_csv(
        processed_directory
        / X_TRAIN_FILENAME
    )

    X_validation = pd.read_csv(
        processed_directory
        / X_VALIDATION_FILENAME
    )

    X_test = pd.read_csv(
        processed_directory
        / X_TEST_FILENAME
    )

    y_train = pd.read_csv(
        processed_directory
        / Y_TRAIN_FILENAME
    )

    y_validation = pd.read_csv(
        processed_directory
        / Y_VALIDATION_FILENAME
    )

    y_test = pd.read_csv(
        processed_directory
        / Y_TEST_FILENAME
    )

    assert X_train.shape == (70, 10)
    assert X_validation.shape == (15, 10)
    assert X_test.shape == (15, 10)

    assert y_train.shape == (70, 1)
    assert y_validation.shape == (15, 1)
    assert y_test.shape == (15, 1)

    assert report.train.row_count == 70
    assert report.validation.row_count == 15
    assert report.test.row_count == 15

    fitted_preprocessor = joblib.load(
        preprocessor_path
    )

    assert hasattr(
        fitted_preprocessor,
        "transformers_",
    )


def test_pipeline_is_reproducible(
    tmp_path: Path,
) -> None:
    """
    Deux exécutions avec la même configuration doivent
    produire exactement les mêmes ensembles.
    """

    config = create_test_configuration(
        tmp_path
    )

    prepare_training_data(
        config
    )

    first_X_train = pd.read_csv(
        config.output.processed_data_directory
        / X_TRAIN_FILENAME
    )

    prepare_training_data(
        config
    )

    second_X_train = pd.read_csv(
        config.output.processed_data_directory
        / X_TRAIN_FILENAME
    )

    pd.testing.assert_frame_equal(
        first_X_train,
        second_X_train,
    )