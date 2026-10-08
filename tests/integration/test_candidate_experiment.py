from pathlib import Path

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
from app.core.experiment_registry import (
    list_experiments,
    load_experiment_record,
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
from pipelines.run_candidate_experiment import (
    run_candidate_experiment,
)


def create_test_configuration(
    tmp_path: Path,
) -> CandidateExperimentConfig:
    """
    Crée les données préparées et la configuration
    des modèles candidats.
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

    training_config = TrainingConfig(
        dataset_path=raw_dataset_path,
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


def test_complete_candidate_experiment(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'entraînement puis l'enregistrement
    de l'expérience dans le registre.
    """

    config = create_test_configuration(
        tmp_path
    )

    registry_directory = (
        tmp_path
        / "models"
        / "registry"
    )

    experiment = run_candidate_experiment(
        config,
        registry_directory,
    )

    assert experiment.status == "COMPLETED"
    assert experiment.completed_at is not None
    assert experiment.candidate_count == 3
    assert len(experiment.candidates) == 3

    assert experiment.error_message is None

    index_entries = list_experiments(
        registry_directory
    )

    assert len(index_entries) == 1

    assert (
        index_entries[0].experiment_id
        == experiment.experiment_id
    )

    loaded_experiment = (
        load_experiment_record(
            registry_directory,
            experiment.experiment_id,
        )
    )

    assert (
        loaded_experiment.experiment_id
        == experiment.experiment_id
    )

    assert loaded_experiment.status == (
        "COMPLETED"
    )

    assert (
        loaded_experiment.candidate_count
        == 3
    )

    assert len(
        loaded_experiment.artifacts
    ) == 6