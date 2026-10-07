from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from app.core.dataset_generator import (
    EXPECTED_COLUMNS,
    RATIO_COLUMNS,
)
from app.core.quality_schemas import (
    DatasetQualityRecord,
    DatasetValidationIssue,
    DatasetValidationReport,
    QualityLabel,
    ValidationSeverity,
)


def create_issue(
    *,
    code: str,
    severity: ValidationSeverity,
    message: str,
    column_name: str | None = None,
) -> DatasetValidationIssue:
    """
    Crée un problème de validation structuré.
    """

    return DatasetValidationIssue(
        code=code,
        severity=severity,
        message=message,
        column_name=column_name,
    )


def validate_dataset_dataframe(
    dataframe: pd.DataFrame,
    dataset_path: str = "<memory>",
) -> DatasetValidationReport:
    """
    Valide un DataFrame contenant des rapports de qualité.
    """

    issues: list[DatasetValidationIssue] = []

    if dataframe.empty:
        issues.append(
            create_issue(
                code="EMPTY_DATASET",
                severity="ERROR",
                message="Le dataset ne contient aucune ligne.",
            )
        )

    actual_columns = list(
        dataframe.columns
    )

    missing_columns = [
        column_name
        for column_name in EXPECTED_COLUMNS
        if column_name not in actual_columns
    ]

    unexpected_columns = [
        column_name
        for column_name in actual_columns
        if column_name not in EXPECTED_COLUMNS
    ]

    for column_name in missing_columns:
        issues.append(
            create_issue(
                code="MISSING_COLUMN",
                severity="ERROR",
                message=(
                    f"La colonne obligatoire "
                    f"'{column_name}' est absente."
                ),
                column_name=column_name,
            )
        )

    for column_name in unexpected_columns:
        issues.append(
            create_issue(
                code="UNEXPECTED_COLUMN",
                severity="WARNING",
                message=(
                    f"La colonne '{column_name}' "
                    f"n'est pas attendue."
                ),
                column_name=column_name,
            )
        )

    missing_value_count = int(
        dataframe.isna().sum().sum()
    )

    if missing_value_count > 0:
        issues.append(
            create_issue(
                code="MISSING_VALUES",
                severity="ERROR",
                message=(
                    f"Le dataset contient "
                    f"{missing_value_count} valeur(s) manquante(s)."
                ),
            )
        )

    duplicate_row_count = int(
        dataframe.duplicated().sum()
    )

    if duplicate_row_count > 0:
        issues.append(
            create_issue(
                code="DUPLICATE_ROWS",
                severity="WARNING",
                message=(
                    f"Le dataset contient "
                    f"{duplicate_row_count} ligne(s) dupliquée(s)."
                ),
            )
        )

    if not missing_columns:
        validation_dataframe = dataframe.loc[
            :,
            EXPECTED_COLUMNS,
        ].copy()

        for row_index, row in validation_dataframe.iterrows():
            row_data = row.to_dict()

            try:
                DatasetQualityRecord.model_validate(
                    row_data
                )

            except ValidationError as error:
                issues.append(
                    create_issue(
                        code="INVALID_RECORD",
                        severity="ERROR",
                        message=(
                            f"Ligne {row_index} invalide : "
                            f"{error.errors()}"
                        ),
                    )
                )            

    class_distribution: dict[str, int] = {}

    if "quality_label" in dataframe.columns:
        class_distribution = {
            str(label): int(count)
            for label, count in (
                dataframe["quality_label"]
                .value_counts()
                .to_dict()
                .items()
            )
        }

        expected_labels = {
            label.value
            for label in QualityLabel
        }

        actual_labels = set(
            class_distribution
        )

        unknown_labels = (
            actual_labels
            - expected_labels
        )

        for label in sorted(unknown_labels):
            issues.append(
                create_issue(
                    code="UNKNOWN_LABEL",
                    severity="ERROR",
                    message=(
                        f"Le label '{label}' "
                        f"n'est pas autorisé."
                    ),
                    column_name="quality_label",
                )
            )

        missing_labels = (
            expected_labels
            - actual_labels
        )

        for label in sorted(missing_labels):
            issues.append(
                create_issue(
                    code="MISSING_CLASS",
                    severity="ERROR",
                    message=(
                        f"La classe '{label}' "
                        f"est absente du dataset."
                    ),
                    column_name="quality_label",
                )
            )

    if len(dataframe) < 30:
        issues.append(
            create_issue(
                code="INSUFFICIENT_ROWS",
                severity="ERROR",
                message=(
                    "Le dataset doit contenir "
                    "au moins 30 observations."
                ),
            )
        )

    valid = not any(
        issue.severity == "ERROR"
        for issue in issues
    )

    return DatasetValidationReport(
        dataset_path=dataset_path,
        valid=valid,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_row_count=duplicate_row_count,
        missing_value_count=missing_value_count,
        class_distribution=class_distribution,
        issues=issues,
    )


def validate_dataset_file(
    dataset_path: str | Path,
) -> DatasetValidationReport:
    """
    Charge et valide un fichier CSV.
    """

    resolved_path = Path(
        dataset_path
    ).resolve()

    if not resolved_path.exists():
        return DatasetValidationReport(
            dataset_path=str(resolved_path),
            valid=False,
            row_count=0,
            column_count=0,
            duplicate_row_count=0,
            missing_value_count=0,
            class_distribution={},
            issues=[
                create_issue(
                    code="FILE_NOT_FOUND",
                    severity="ERROR",
                    message=(
                        f"Le fichier n'existe pas : "
                        f"{resolved_path}"
                    ),
                )
            ],
        )

    try:
        dataframe = pd.read_csv(
            resolved_path
        )
    except pd.errors.EmptyDataError:
        dataframe = pd.DataFrame()

    return validate_dataset_dataframe(
        dataframe=dataframe,
        dataset_path=str(resolved_path),
    )