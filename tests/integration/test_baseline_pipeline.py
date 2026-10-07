import json
from pathlib import Path

import joblib
import pandas as pd

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.model_schemas import (
    BaselineModelConfig,
    BaselineTrainingConfig,
    ModelEvaluationConfig,
    ModelInputConfig,
    ModelOutputConfig,
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
    prepare_training_data,
)
from pipelines.train_baseline import (
    CONFUSION_MATRIX_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    train_baseline_pipeline,
)


def create_baseline_test_config(
    tmp_path: Path,
) -> BaselineModelConfig:
    """
    Crée les données et les configurations nécessaires
    au test du pipeline baseline.
    """

    raw_dataset_path = (
        tmp_path
        / "data"
        / "raw"
        / "dataset_quality.csv"
    )

    raw_dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_quality_dataset(
        DatasetGenerationConfig(
            number_of_records=300,
            random_seed=42,
            good_ratio=0.40,
            acceptable_ratio=0.35,
            poor_ratio=0.25,
        )
    )

    dataframe.to_csv(
        raw_dataset_path,
        index=False,
        encoding="utf-8",
    )

    processed_directory = (
        tmp_path
        / "data"
        / "processed"
    )

    preprocessing_artifacts = (
        tmp_path
        / "models"
        / "preprocessing"
    )

    preparation_reports = (
        tmp_path
        / "reports"
        / "training"
    )

    training_config = TrainingConfig(
        dataset_path=raw_dataset_path,
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
                processed_directory
            ),
            artifacts_directory=(
                preprocessing_artifacts
            ),
            reports_directory=(
                preparation_reports
            ),
        ),
    )

    prepare_training_data(
        training_config
    )

    return BaselineModelConfig(
        model_name="logistic_regression",
        model_version="0.1.0",
        training=BaselineTrainingConfig(
            maximum_iterations=1000,
            random_seed=42,
            class_weight=None,
        ),
        evaluation=ModelEvaluationConfig(
            primary_metric="f1_macro",
            minimum_f1_macro=0.50,
            minimum_poor_recall=0.50,
        ),
        input=ModelInputConfig(
            processed_data_directory=(
                processed_directory
            ),
        ),
        output=ModelOutputConfig(
            model_directory=(
                tmp_path
                / "models"
                / "candidates"
                / "logistic_regression"
            ),
            evaluation_directory=(
                tmp_path
                / "reports"
                / "evaluation"
                / "baseline"
            ),
        ),
    )


def test_complete_baseline_pipeline(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'entraînement, l'évaluation
    et la sauvegarde des artefacts.
    """

    config = create_baseline_test_config(
        tmp_path
    )

    result = train_baseline_pipeline(
        config
    )

    model_path = (
        config.output.model_directory
        / MODEL_FILENAME
    )

    metadata_path = (
        config.output.model_directory
        / METADATA_FILENAME
    )

    metrics_path = (
        config.output.evaluation_directory
        / METRICS_FILENAME
    )