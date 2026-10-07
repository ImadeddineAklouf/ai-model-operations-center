import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from app.core.dataset_validation import validate_dataset_file
from app.core.training_config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    load_training_config,
)
from app.core.training_schemas import (
    DataPreparationReport,
    DatasetSplitSummary,
    TrainingConfig,
)
from training.data_split import (
    DatasetSplits,
    split_dataset,
)
from training.preprocessing import (
    build_numeric_preprocessor,
    fit_and_transform_splits,
)


X_TRAIN_FILENAME = "X_train.csv"
X_VALIDATION_FILENAME = "X_validation.csv"
X_TEST_FILENAME = "X_test.csv"

Y_TRAIN_FILENAME = "y_train.csv"
Y_VALIDATION_FILENAME = "y_validation.csv"
Y_TEST_FILENAME = "y_test.csv"

PREPROCESSOR_FILENAME = "preprocessor.joblib"
REPORT_FILENAME = "data_preparation_report.json"


def load_validated_dataset(
    config: TrainingConfig,
) -> pd.DataFrame:
    """
    Charge le dataset seulement après validation.

    Raises:
        ValueError:
            Si le dataset contient au moins une erreur
            bloquante.
    """

    validation_report = validate_dataset_file(
        config.dataset_path
    )

    if not validation_report.valid:
        error_codes = [
            issue.code
            for issue in validation_report.issues
            if issue.severity == "ERROR"
        ]

        raise ValueError(
            "Le dataset d'entraînement est invalide. "
            f"Erreurs détectées : {error_codes}"
        )

    return pd.read_csv(
        config.dataset_path
    )


def create_split_summary(
    features: pd.DataFrame,
    target: pd.Series,
) -> DatasetSplitSummary:
    """
    Crée le résumé d'un sous-ensemble ML.
    """

    class_distribution = {
        str(label): int(count)
        for label, count in (
            target
            .value_counts()
            .to_dict()
            .items()
        )
    }

    return DatasetSplitSummary(
        row_count=len(features),
        feature_count=len(features.columns),
        class_distribution=class_distribution,
    )


def create_preparation_report(
    *,
    config: TrainingConfig,
    transformed_X_train: pd.DataFrame,
    transformed_X_validation: pd.DataFrame,
    transformed_X_test: pd.DataFrame,
    splits: DatasetSplits,
) -> DataPreparationReport:
    """
    Construit le rapport final de préparation.
    """

    return DataPreparationReport(
        source_dataset_path=str(
            config.dataset_path
        ),
        target_column=config.target_column,
        random_seed=config.split.random_seed,
        stratified=config.split.stratify,
        feature_names=[
            str(column_name)
            for column_name
            in transformed_X_train.columns
        ],
        train=create_split_summary(
            features=transformed_X_train,
            target=splits.y_train,
        ),
        validation=create_split_summary(
            features=transformed_X_validation,
            target=splits.y_validation,
        ),
        test=create_split_summary(
            features=transformed_X_test,
            target=splits.y_test,
        ),
    )


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Sauvegarde un DataFrame en CSV.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        resolved_path,
        index=False,
        encoding="utf-8",
    )

    return resolved_path


def save_target(
    target: pd.Series,
    output_path: Path,
    target_column: str,
) -> Path:
    """
    Sauvegarde une variable cible dans un CSV.

    Le nom de la colonne cible est conservé.
    """

    target_dataframe = target.to_frame(
        name=target_column
    )

    return save_dataframe(
        dataframe=target_dataframe,
        output_path=output_path,
    )


def save_json_report(
    report: DataPreparationReport,
    output_path: Path,
) -> Path:
    """
    Sauvegarde le rapport de préparation en JSON.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        report.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )

    return resolved_path


def prepare_training_data(
    config: TrainingConfig,
) -> DataPreparationReport:
    """
    Exécute le pipeline complet de préparation.

    Étapes :
    1. Validation et chargement du dataset.
    2. Séparation train, validation et test.
    3. Construction du préprocesseur.
    4. Fit uniquement sur X_train.
    5. Transformation des trois ensembles.
    6. Sauvegarde des datasets.
    7. Sauvegarde du préprocesseur.
    8. Génération du rapport.
    """

    dataframe = load_validated_dataset(
        config
    )

    splits = split_dataset(
        dataframe=dataframe,
        target_column=config.target_column,
        config=config.split,
    )

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=config.preprocessing,
    )

    (
        fitted_preprocessor,
        transformed_X_train,
        transformed_X_validation,
        transformed_X_test,
    ) = fit_and_transform_splits(
        preprocessor=preprocessor,
        X_train=splits.X_train,
        X_validation=splits.X_validation,
        X_test=splits.X_test,
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

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataframe(
        transformed_X_train,
        processed_directory
        / X_TRAIN_FILENAME,
    )

    save_dataframe(
        transformed_X_validation,
        processed_directory
        / X_VALIDATION_FILENAME,
    )

    save_dataframe(
        transformed_X_test,
        processed_directory
        / X_TEST_FILENAME,
    )

    save_target(
        target=splits.y_train,
        output_path=(
            processed_directory
            / Y_TRAIN_FILENAME
        ),
        target_column=config.target_column,
    )

    save_target(
        target=splits.y_validation,
        output_path=(
            processed_directory
            / Y_VALIDATION_FILENAME
        ),
        target_column=config.target_column,
    )

    save_target(
        target=splits.y_test,
        output_path=(
            processed_directory
            / Y_TEST_FILENAME
        ),
        target_column=config.target_column,
    )

    joblib.dump(
        fitted_preprocessor,
        artifacts_directory
        / PREPROCESSOR_FILENAME,
    )

    report = create_preparation_report(
        config=config,
        transformed_X_train=(
            transformed_X_train
        ),
        transformed_X_validation=(
            transformed_X_validation
        ),
        transformed_X_test=(
            transformed_X_test
        ),
        splits=splits,
    )

    save_json_report(
        report=report,
        output_path=(
            reports_directory
            / REPORT_FILENAME
        ),
    )

    return report


def print_preparation_summary(
    report: DataPreparationReport,
) -> None:
    """
    Affiche un résumé lisible du pipeline.
    """

    print()
    print("=" * 60)
    print("TRAINING DATA PREPARATION")
    print("=" * 60)

    print(
        f"Dataset source : "
        f"{report.source_dataset_path}"
    )

    print(
        f"Colonne cible : "
        f"{report.target_column}"
    )

    print(
        f"Seed : {report.random_seed}"
    )

    print(
        f"Stratification : "
        f"{report.stratified}"
    )

    print(
        f"Nombre de features : "
        f"{len(report.feature_names)}"
    )

    print()
    print(
        f"Train : "
        f"{report.train.row_count} lignes"
    )

    print(
        f"Validation : "
        f"{report.validation.row_count} lignes"
    )

    print(
        f"Test : "
        f"{report.test.row_count} lignes"
    )

    print()
    print("Distribution Train :")

    for label, count in (
        report.train.class_distribution.items()
    ):
        print(
            f"  - {label}: {count}"
        )

    print()
    print("=" * 60)


def main() -> int:
    """
    Point d'entrée exécutable du pipeline.

    Returns:
        0 si la préparation réussit.
        1 si elle échoue.
    """

    try:
        config = load_training_config(
            DEFAULT_TRAINING_CONFIG_PATH
        )

        report = prepare_training_data(
            config
        )

        print_preparation_summary(
            report
        )

        print(
            "Préparation terminée avec succès."
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
    ) as error:
        print(
            "Échec de la préparation des données : "
            f"{error}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )