import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd

from app.core.candidate_config import (
    DEFAULT_CANDIDATE_CONFIG_PATH,
    load_candidate_config,
)
from app.core.candidate_schemas import (
    CandidateComparisonReport,
    CandidateExperimentConfig,
    CandidateModelResult,
)
from app.core.model_schemas import (
    ModelEvaluationConfig,
    ModelEvaluationResult,
)
from training.candidate_models import (
    build_enabled_candidates,
    get_candidate_version,
)
from training.evaluation import evaluate_classifier
from training.model_comparison import (
    build_comparison_report,
    create_candidate_result,
)


X_TRAIN_FILENAME = "X_train.csv"
Y_TRAIN_FILENAME = "y_train.csv"
X_VALIDATION_FILENAME = "X_validation.csv"
Y_VALIDATION_FILENAME = "y_validation.csv"

TARGET_COLUMN = "quality_label"

MODEL_FILENAME = "model.joblib"
EVALUATION_FILENAME = "validation_metrics.json"
METADATA_FILENAME = "metadata.json"

COMPARISON_JSON_FILENAME = "comparison.json"
COMPARISON_CSV_FILENAME = "comparison.csv"
CHAMPION_FILENAME = "champion.json"


def load_features(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Charge et valide un fichier de features.
    """

    resolved_path = file_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} introuvable : "
            f"{resolved_path}"
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
            f"{dataset_name} contient des "
            "colonnes dupliquées."
        )

    if dataframe.isna().any().any():
        raise ValueError(
            f"{dataset_name} contient des "
            "valeurs manquantes."
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
) -> pd.Series:
    """
    Charge et valide un fichier contenant la cible.
    """

    resolved_path = file_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} introuvable : "
            f"{resolved_path}"
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

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"La colonne cible '{TARGET_COLUMN}' "
            f"est absente de {dataset_name}."
        )

    if len(dataframe.columns) != 1:
        raise ValueError(
            f"{dataset_name} doit contenir uniquement "
            f"la colonne '{TARGET_COLUMN}'."
        )

    target = dataframe[
        TARGET_COLUMN
    ].copy()

    if target.isna().any():
        raise ValueError(
            f"{dataset_name} contient des "
            "valeurs manquantes."
        )

    return target


def load_candidate_datasets(
    config: CandidateExperimentConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Charge les données utilisées par tous les candidats.
    """

    processed_directory = (
        config.input.processed_data_directory
    )

    X_train = load_features(
        processed_directory
        / X_TRAIN_FILENAME,
        "X_train",
    )

    y_train = load_target(
        processed_directory
        / Y_TRAIN_FILENAME,
        "y_train",
    )

    X_validation = load_features(
        processed_directory
        / X_VALIDATION_FILENAME,
        "X_validation",
    )

    y_validation = load_target(
        processed_directory
        / Y_VALIDATION_FILENAME,
        "y_validation",
    )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train et y_train ne sont pas alignés."
        )

    if len(X_validation) != len(
        y_validation
    ):
        raise ValueError(
            "X_validation et y_validation "
            "ne sont pas alignés."
        )

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "X_train et X_validation doivent contenir "
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


def create_evaluation_config(
    config: CandidateExperimentConfig,
) -> ModelEvaluationConfig:
    """
    Convertit les seuils des candidats vers la
    configuration commune utilisée par l'évaluateur.
    """

    return ModelEvaluationConfig(
        primary_metric="f1_macro",
        minimum_f1_macro=(
            config.evaluation.minimum_f1_macro
        ),
        minimum_poor_recall=(
            config.evaluation.minimum_poor_recall
        ),
    )


def train_candidate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, float]:
    """
    Entraîne un candidat et retourne sa durée
    d'entraînement.
    """

    training_start = perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_duration_seconds = (
        perf_counter()
        - training_start
    )

    return (
        model,
        float(training_duration_seconds),
    )


def save_json_dictionary(
    data: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Sauvegarde un dictionnaire au format JSON.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return resolved_path


def save_evaluation_result(
    result: ModelEvaluationResult,
    output_path: Path,
) -> Path:
    """
    Sauvegarde une évaluation Pydantic en JSON.
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


def save_candidate_artifacts(
    model_name: str,
    model_version: str,
    model: Any,
    training_duration_seconds: float,
    evaluation_result: ModelEvaluationResult,
    config: CandidateExperimentConfig,
) -> None:
    """
    Sauvegarde le modèle, ses métriques
    et ses métadonnées.
    """

    candidate_directory = (
        config.output.candidates_directory
        / model_name
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        candidate_directory
        / MODEL_FILENAME,
    )

    save_evaluation_result(
        evaluation_result,
        candidate_directory
        / EVALUATION_FILENAME,
    )

    if evaluation_result.passed:
        status = "VALIDATED"
    else:
        status = "REJECTED"

    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": type(model).__name__,
        "status": status,
        "training_duration_seconds": (
            float(training_duration_seconds)
        ),
        "validation": {
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
            "poor_recall": (
                evaluation_result
                .metrics
                .metrics_by_class[
                    "POOR"
                ]
                .recall
            ),
            "prediction_latency_seconds": (
                evaluation_result
                .metrics
                .prediction_latency_seconds
            ),
            "passed": (
                evaluation_result.passed
            ),
            "rejection_reasons": (
                evaluation_result
                .rejection_reasons
            ),
        },
    }

    save_json_dictionary(
        metadata,
        candidate_directory
        / METADATA_FILENAME,
    )


def save_comparison_report(
    report: CandidateComparisonReport,
    comparison_directory: Path,
) -> None:
    """
    Sauvegarde le classement en JSON et en CSV.
    """

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_json_path = (
        comparison_directory
        / COMPARISON_JSON_FILENAME
    )

    comparison_json_path.write_text(
        report.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    ranking_rows = []

    rank = 1

    for candidate in report.ranking:
        ranking_rows.append(
            {
                "rank": rank,
                "model_name": (
                    candidate.model_name
                ),
                "model_version": (
                    candidate.model_version
                ),
                "passed": candidate.passed,
                "accuracy": (
                    candidate.accuracy
                ),
                "precision_macro": (
                    candidate.precision_macro
                ),
                "recall_macro": (
                    candidate.recall_macro
                ),
                "f1_macro": (
                    candidate.f1_macro
                ),
                "poor_recall": (
                    candidate.poor_recall
                ),
                "prediction_latency_seconds": (
                    candidate
                    .prediction_latency_seconds
                ),
                "training_duration_seconds": (
                    candidate
                    .training_duration_seconds
                ),
            }
        )

        rank += 1

    ranking_dataframe = pd.DataFrame(
        ranking_rows
    )

    ranking_dataframe.to_csv(
        comparison_directory
        / COMPARISON_CSV_FILENAME,
        index=False,
        encoding="utf-8",
    )


def save_champion_report(
    report: CandidateComparisonReport,
    comparison_directory: Path,
) -> None:
    """
    Sauvegarde l'identité et les métriques
    du modèle champion.
    """

    champion_data: dict[str, Any]

    if report.champion_model_name is None:
        champion_data = {
            "champion_selected": False,
            "model_name": None,
            "model_version": None,
            "reason": (
                "Aucun modèle candidat ne respecte "
                "les seuils obligatoires."
            ),
        }

    else:
        champion_result = None

        for candidate in report.ranking:
            if (
                candidate.model_name
                == report.champion_model_name
                and candidate.model_version
                == report.champion_model_version
            ):
                champion_result = candidate
                break

        if champion_result is None:
            raise RuntimeError(
                "Le champion est absent du classement."
            )

        champion_data = {
            "champion_selected": True,
            "model_name": (
                champion_result.model_name
            ),
            "model_version": (
                champion_result.model_version
            ),
            "f1_macro": (
                champion_result.f1_macro
            ),
            "poor_recall": (
                champion_result.poor_recall
            ),
            "accuracy": (
                champion_result.accuracy
            ),
            "prediction_latency_seconds": (
                champion_result
                .prediction_latency_seconds
            ),
            "training_duration_seconds": (
                champion_result
                .training_duration_seconds
            ),
        }

    save_json_dictionary(
        champion_data,
        comparison_directory
        / CHAMPION_FILENAME,
    )


def train_candidates_pipeline(
    config: CandidateExperimentConfig,
) -> CandidateComparisonReport:
    """
    Entraîne, évalue et compare tous les modèles activés.
    """

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = load_candidate_datasets(
        config
    )

    candidate_models = build_enabled_candidates(
        config
    )

    evaluation_config = (
        create_evaluation_config(
            config
        )
    )

    candidate_results: list[
        CandidateModelResult
    ] = []

    for model_name, model in (
        candidate_models.items()
    ):
        model_version = get_candidate_version(
            model_name,
            config,
        )

        (
            fitted_model,
            training_duration_seconds,
        ) = train_candidate_model(
            model,
            X_train,
            y_train,
        )

        evaluation_result = evaluate_classifier(
            model=fitted_model,
            X_evaluation=X_validation,
            y_evaluation=y_validation,
            model_name=model_name,
            model_version=model_version,
            evaluation_config=(
                evaluation_config
            ),
            evaluated_split="validation",
        )

        candidate_result = create_candidate_result(
            model_name,
            model_version,
            training_duration_seconds,
            evaluation_result,
        )

        candidate_results.append(
            candidate_result
        )

        save_candidate_artifacts(
            model_name,
            model_version,
            fitted_model,
            training_duration_seconds,
            evaluation_result,
            config,
        )

    comparison_report = build_comparison_report(
        candidate_results
    )

    save_comparison_report(
        comparison_report,
        config.output.comparison_directory,
    )

    save_champion_report(
        comparison_report,
        config.output.comparison_directory,
    )

    return comparison_report


def print_comparison_summary(
    report: CandidateComparisonReport,
) -> None:
    """
    Affiche le classement des modèles candidats.
    """

    print()
    print("=" * 80)
    print("CANDIDATE MODEL COMPARISON")
    print("=" * 80)

    rank = 1

    for candidate in report.ranking:
        print(
            f"{rank}. {candidate.model_name} "
            f"version {candidate.model_version}"
        )

        print(
            f"   Validé : {candidate.passed}"
        )

        print(
            f"   F1 macro : "
            f"{candidate.f1_macro:.4f}"
        )

        print(
            f"   Recall POOR : "
            f"{candidate.poor_recall:.4f}"
        )

        print(
            f"   Accuracy : "
            f"{candidate.accuracy:.4f}"
        )

        print(
            f"   Latence : "
            f"{candidate.prediction_latency_seconds:.6f} s"
        )

        print(
            f"   Entraînement : "
            f"{candidate.training_duration_seconds:.6f} s"
        )

        print()

        rank += 1

    if report.champion_model_name is None:
        print(
            "Aucun champion sélectionné."
        )
    else:
        print(
            "Champion : "
            f"{report.champion_model_name} "
            f"version "
            f"{report.champion_model_version}"
        )

    print("=" * 80)


def main() -> int:
    """
    Point d'entrée du pipeline des candidats.
    """

    try:
        config = load_candidate_config(
            DEFAULT_CANDIDATE_CONFIG_PATH
        )

        report = train_candidates_pipeline(
            config
        )

        print_comparison_summary(
            report
        )

        return 0

    except FileNotFoundError as error:
        print(
            f"Fichier introuvable : {error}"
        )

        return 1

    except ValueError as error:
        print(
            f"Configuration ou données invalides : "
            f"{error}"
        )

        return 1

    except RuntimeError as error:
        print(
            f"Erreur d'exécution : {error}"
        )

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)