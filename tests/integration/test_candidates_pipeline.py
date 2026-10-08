import json
from pathlib import Path

import pandas as pd

from app.core.candidate_schemas import (
    CandidateEvaluationConfig,
    CandidateExperimentConfig,
    CandidateInputConfig,
    CandidateModelsConfig,
    CandidateOutputConfig,
    GradientBoostingCandidateConfig,
    LogisticRegressionCandidateConfig,
    RandomForestCandidateConfig,
)
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
    prepare_training_data,
)
from pipelines.train_candidates import (
    CHAMPION_FILENAME,
    COMPARISON_CSV_FILENAME,
    COMPARISON_JSON_FILENAME,
    MODEL_FILENAME,
    train_candidates_pipeline,
)


def create_test_config(
    tmp_path: Path,
) -> CandidateExperimentConfig:
    """
    Prépare les données nécessaires au test.
    """

    data_directory = (
        tmp_path
        / "data"
    )

    raw_directory = (
        data_directory
        / "raw"
    )

    processed_directory = (
        data_directory
        / "processed"
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        raw_directory
        / "dataset_quality.csv"
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
        dataset_path,
        index=False,
        encoding="utf-8",
    )

    training_config = TrainingConfig(
        dataset_path=dataset_path,
        target_column="quality_label",
        split=DataSplitConfig(),
        preprocessing=PreprocessingConfig(),
        output=TrainingOutputConfig(
            processed_data_directory=(
                processed_directory
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

    prepare_training_data(
        training_config
    )

    return CandidateExperimentConfig(
        random_seed=42,
        models=CandidateModelsConfig(
            logistic_regression=(
                LogisticRegressionCandidateConfig(
                    maximum_iterations=500
                )
            ),
            random_forest=(
                RandomForestCandidateConfig(
                    number_of_estimators=20
                )
            ),
            gradient_boosting=(
                GradientBoostingCandidateConfig(
                    number_of_estimators=20
                )
            ),
        ),
        evaluation=CandidateEvaluationConfig(
            minimum_f1_macro=0.50,
            minimum_poor_recall=0.50,
        ),
        input=CandidateInputConfig(
            processed_data_directory=(
                processed_directory
            )
        ),
        output=CandidateOutputConfig(
            candidates_directory=(
                tmp_path
                / "models"
                / "candidates"
            ),
            comparison_directory=(
                tmp_path
                / "reports"
                / "evaluation"
                / "candidates"
            ),
        ),
    )


def test_complete_candidates_pipeline(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'entraînement et la comparaison
    des trois modèles.
    """

    config = create_test_config(
        tmp_path
    )

    report = train_candidates_pipeline(
        config
    )

    assert report.candidate_count == 3
    assert len(report.ranking) == 3

    candidate_names = {
        candidate.model_name
        for candidate in report.ranking
    }

    assert candidate_names == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }

    for model_name in candidate_names:
        model_path = (
            config.output.candidates_directory
            / model_name
            / MODEL_FILENAME
        )

        assert model_path.exists()

    comparison_directory = (
        config.output.comparison_directory
    )

    assert (
        comparison_directory
        / COMPARISON_JSON_FILENAME
    ).exists()

    assert (
        comparison_directory
        / COMPARISON_CSV_FILENAME
    ).exists()

    assert (
        comparison_directory
        / CHAMPION_FILENAME
    ).exists()

    comparison_dataframe = pd.read_csv(
        comparison_directory
        / COMPARISON_CSV_FILENAME
    )

    assert comparison_dataframe.shape[0] == 3

    champion_data = json.loads(
        (
            comparison_directory
            / CHAMPION_FILENAME
        ).read_text(
            encoding="utf-8"
        )
    )

    assert "champion_selected" in (
        champion_data
    )

    if (
        report.champion_model_name
        is not None
    ):
        assert (
            champion_data[
                "champion_selected"
            ]
            is True
        )

        assert (
            champion_data[
                "model_name"
            ]
            == report.champion_model_name
        )