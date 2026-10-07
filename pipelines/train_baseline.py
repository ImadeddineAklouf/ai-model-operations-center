import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.core.model_config import (
    DEFAULT_BASELINE_CONFIG_PATH,
    load_baseline_model_config,
)
from app.core.model_schemas import (
    BaselineModelConfig,
    ModelEvaluationResult,
)
from training.baseline_model import (
    build_baseline_model,
    train_baseline_model,
)
from training.evaluation import evaluate_classifier


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "validation_metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.csv"

X_TRAIN_FILENAME = "X_train.csv"
Y_TRAIN_FILENAME = "y_train.csv"
X_VALIDATION_FILENAME = "X_validation.csv"
Y_VALIDATION_FILENAME = "y_validation.csv"

TARGET_COLUMN = "quality_label"


def load_features(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Charge et valide un fichier CSV contenant les features.
    """

    resolved_path = file_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} introuvable : {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Le chemin de {dataset_name} "
            "ne correspond pas à un fichier."
        )

    dataframe = pd.read_csv(
        resolved_path
    )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} ne peut pas être vide."
        )

    if dataframe.columns.duplicated().any():
        raise ValueError(
            f"{dataset_name} contient des colonnes dupliquées."
        )

    if dataframe.isna().any().any():
        raise ValueError(
            f"{dataset_name} contient des valeurs manquantes."
        )

    non_numeric_columns = []

    for column_name in dataframe.columns:
        if not pd.api.types.is_numeric_dtype(
            dataframe[column_name]
        ):
            non_numeric_columns.append(
                str(column_name)
            )

    if non_numeric_columns:
        raise ValueError(
            f"{dataset_name} contient des features "
            f"non numériques : {non_numeric_columns}"
        )

    return dataframe


def load_target(
    file_path: Path,
    dataset_name: str,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """
    Charge et valide un fichier contenant la variable cible.
    """

    resolved_path = file_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} introuvable : {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Le chemin de {dataset_name} "
            "ne correspond pas à un fichier."
        )

    dataframe = pd.read_csv(
        resolved_path
    )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} ne peut pas être vide."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"La colonne cible '{target_column}' "
            f"est absente de {dataset_name}."
        )

    if len(dataframe.columns) != 1:
        raise ValueError(
            f"{dataset_name} doit contenir uniquement "
            f"la colonne '{target_column}'."
        )

    target = dataframe[target_column].copy()

    if target.empty:
        raise ValueError(
            f"{dataset_name} ne peut pas être vide."
        )

    if target.isna().any():
        raise ValueError(
            f"{dataset_name} contient des valeurs manquantes."
        )

    return target


def load_training_data(
    config: BaselineModelConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Charge les données d'entraînement et de validation.
    """

    processed_directory = (
        config.input.processed_data_directory
    )

    X_train = load_features(
        processed_directory / X_TRAIN_FILENAME,
        "X_train",
    )

    y_train = load_target(
        processed_directory / Y_TRAIN_FILENAME,
        "y_train",
        TARGET_COLUMN,
    )

    X_validation = load_features(
        processed_directory / X_VALIDATION_FILENAME,
        "X_validation",
    )

    y_validation = load_target(
        processed_directory / Y_VALIDATION_FILENAME,
        "y_validation",
        TARGET_COLUMN,
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train et y_train doivent contenir "
            "le même nombre d'observations."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation et y_validation doivent contenir "
            "le même nombre d'observations."
        )

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "X_train et X_validation doivent avoir "
            "les mêmes features dans le même ordre."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "y_train doit contenir au moins "
            "deux classes différentes."
        )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def save_model(
    model: LogisticRegression,
    output_path: Path,
) -> Path:
    """
    Sauvegarde le modèle entraîné avec Joblib.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        resolved_path,
    )

    return resolved_path


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Sauvegarde un dictionnaire dans un fichier JSON.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_content = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )

    resolved_path.write_text(
        json_content,
        encoding="utf-8",
    )

    return resolved_path


def save_evaluation_result(
    result: ModelEvaluationResult,
    output_path: Path,
) -> Path:
    """
    Sauvegarde le rapport d'évaluation au format JSON.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        result.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return resolved_path


def save_confusion_matrix(
    result: ModelEvaluationResult,
    output_path: Path,
) -> Path:
    """
    Sauvegarde la matrice de confusion au format CSV.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_names = []

    for label in result.labels:
        row_names.append(
            f"actual_{label}"
        )

    column_names = []

    for label in result.labels:
        column_names.append(
            f"predicted_{label}"
        )

    matrix_dataframe = pd.DataFrame(
        result.confusion_matrix,
        index=row_names,
        columns=column_names,
    )

    matrix_dataframe.index.name = "actual_label"

    matrix_dataframe.to_csv(
        resolved_path,
        index=True,
        encoding="utf-8",
    )

    return resolved_path


def create_metadata(
    config: BaselineModelConfig,
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_duration_seconds: float,
    evaluation_result: ModelEvaluationResult,
) -> dict[str, Any]:
    """
    Construit les métadonnées du modèle entraîné.
    """

    class_counts = (
        y_train
        .value_counts()
        .to_dict()
    )

    class_distribution = {}

    for label, count in class_counts.items():
        class_distribution[
            str(label)
        ] = int(count)

    feature_names = []

    for column_name in X_train.columns:
        feature_names.append(
            str(column_name)
        )

    model_classes = []

    for label in model.classes_:
        model_classes.append(
            str(label)
        )

    poor_recall = (
        evaluation_result
        .metrics
        .metrics_by_class["POOR"]
        .recall
    )

    if evaluation_result.passed:
        status = "VALIDATED"
    else:
        status = "REJECTED"

    metadata = {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "model_type": type(model).__name__,
        "status": status,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "training": {
            "duration_seconds": float(
                training_duration_seconds
            ),
            "row_count": len(X_train),
            "feature_count": len(
                X_train.columns
            ),
            "feature_names": feature_names,
            "class_distribution": (
                class_distribution
            ),
            "random_seed": (
                config.training.random_seed
            ),
            "maximum_iterations": (
                config
                .training
                .maximum_iterations
            ),
            "class_weight": (
                config.training.class_weight
            ),
        },
        "evaluation": {
            "evaluated_split": (
                evaluation_result.evaluated_split
            ),
            "passed": (
                evaluation_result.passed
            ),
            "accuracy": (
                evaluation_result
                .metrics
                .accuracy
            ),
            "precision_macro": (
                evaluation_result
                .metrics
                .precision_macro
            ),
            "recall_macro": (
                evaluation_result
                .metrics
                .recall_macro
            ),
            "f1_macro": (
                evaluation_result
                .metrics
                .f1_macro
            ),
            "poor_recall": poor_recall,
            "rejection_reasons": (
                evaluation_result
                .rejection_reasons
            ),
        },
        "model_classes": model_classes,
    }

    return metadata


def train_baseline_pipeline(
    config: BaselineModelConfig,
) -> ModelEvaluationResult:
    """
    Exécute le pipeline complet du modèle baseline.
    """

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = load_training_data(
        config
    )

    model = build_baseline_model(
        config.training
    )

    training_start = perf_counter()

    fitted_model = train_baseline_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    training_duration_seconds = (
        perf_counter()
        - training_start
    )

    evaluation_result = evaluate_classifier(
        model=fitted_model,
        X_evaluation=X_validation,
        y_evaluation=y_validation,
        model_name=config.model_name,
        model_version=config.model_version,
        evaluation_config=config.evaluation,
        evaluated_split="validation",
    )

    model_directory = (
        config.output.model_directory
    )

    evaluation_directory = (
        config.output.evaluation_directory
    )

    model_path = (
        model_directory
        / MODEL_FILENAME
    )

    metadata_path = (
        model_directory
        / METADATA_FILENAME
    )

    metrics_path = (
        evaluation_directory
        / METRICS_FILENAME
    )

    confusion_matrix_path = (
        evaluation_directory
        / CONFUSION_MATRIX_FILENAME
    )

    save_model(
        fitted_model,
        model_path,
    )

    metadata = create_metadata(
        config,
        fitted_model,
        X_train,
        y_train,
        training_duration_seconds,
        evaluation_result,
    )

    save_json(
        metadata,
        metadata_path,
    )

    save_evaluation_result(
        evaluation_result,
        metrics_path,
    )

    save_confusion_matrix(
        evaluation_result,
        confusion_matrix_path,
    )

    return evaluation_result


def print_evaluation_summary(
    result: ModelEvaluationResult,
) -> None:
    """
    Affiche un résumé de l'évaluation.
    """

    poor_recall = (
        result
        .metrics
        .metrics_by_class["POOR"]
        .recall
    )

    print()
    print("=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Modèle : {result.model_name}"
    )

    print(
        f"Version : {result.model_version}"
    )

    print(
        f"Ensemble évalué : "
        f"{result.evaluated_split}"
    )

    print()
    print(
        f"Accuracy : "
        f"{result.metrics.accuracy:.4f}"
    )

    print(
        f"Precision macro : "
        f"{result.metrics.precision_macro:.4f}"
    )

    print(
        f"Recall macro : "
        f"{result.metrics.recall_macro:.4f}"
    )

    print(
        f"F1 macro : "
        f"{result.metrics.f1_macro:.4f}"
    )

    print(
        f"Recall POOR : "
        f"{poor_recall:.4f}"
    )

    print(
        f"Nombre de prédictions : "
        f"{result.metrics.prediction_count}"
    )

    print(
        f"Latence totale : "
        f"{result.metrics.prediction_latency_seconds:.6f} s"
    )

    print()
    print(
        f"Seuils respectés : "
        f"{result.passed}"
    )

    if result.rejection_reasons:
        print()
        print("Raisons du rejet :")

        for reason in result.rejection_reasons:
            print(
                f"  - {reason}"
            )

    print()
    print("Matrice de confusion :")

    index = 0

    while index < len(result.labels):
        label = result.labels[index]
        row = result.confusion_matrix[index]

        print(
            f"  Réel {label}: {row}"
        )

        index += 1

    print("=" * 60)
    print()


def main() -> int:
    """
    Exécute le pipeline depuis la ligne de commande.
    """

    try:
        config = load_baseline_model_config(
            DEFAULT_BASELINE_CONFIG_PATH
        )

        result = train_baseline_pipeline(
            config
        )

        print_evaluation_summary(
            result
        )

        if result.passed:
            print(
                "Le modèle baseline respecte "
                "les seuils d'acceptation."
            )
        else:
            print(
                "Le modèle baseline a été entraîné, "
                "mais ne respecte pas tous les seuils."
            )

        return 0

    except FileNotFoundError as error:
        print(
            "Fichier requis introuvable : "
            f"{error}"
        )

        return 1

    except ValueError as error:
        print(
            "Données ou configuration invalides : "
            f"{error}"
        )

        return 1

    except TypeError as error:
        print(
            "Type incompatible : "
            f"{error}"
        )

        return 1

    except KeyError as error:
        print(
            "Champ obligatoire absent : "
            f"{error}"
        )

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)