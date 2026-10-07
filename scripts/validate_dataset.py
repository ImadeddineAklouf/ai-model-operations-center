import sys
from pathlib import Path

from app.core.config import settings
from app.core.dataset_validation import (
    validate_dataset_file,
)
from app.core.quality_schemas import (
    DatasetValidationReport,
)


DEFAULT_DATASET_PATH = (
    settings.raw_data_directory
    / "dataset_quality.csv"
)

DEFAULT_REPORT_PATH = (
    settings.reports_directory
    / "data_validation"
    / "dataset_quality_validation.json"
)


def write_validation_report(
    report: DatasetValidationReport,
    report_path: Path,
) -> Path:
    """
    Enregistre un rapport de validation au format JSON.

    Args:
        report:
            Rapport Pydantic à enregistrer.

        report_path:
            Chemin du fichier JSON de sortie.

    Returns:
        Chemin absolu du rapport écrit.
    """

    resolved_report_path = report_path.resolve()

    resolved_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_report_path.write_text(
        report.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )

    return resolved_report_path


def run_dataset_validation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> DatasetValidationReport:
    """
    Exécute la validation du dataset et écrit le rapport.

    Args:
        dataset_path:
            Chemin du CSV à contrôler.

        report_path:
            Chemin du rapport JSON à produire.

    Returns:
        Rapport complet de validation.
    """

    report = validate_dataset_file(
        dataset_path
    )

    write_validation_report(
        report=report,
        report_path=report_path,
    )

    return report


def print_validation_summary(
    report: DatasetValidationReport,
    report_path: Path,
) -> None:
    """
    Affiche un résumé lisible de la validation.
    """

    print()
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(
        f"Dataset : {report.dataset_path}"
    )

    print(
        f"Statut : "
        f"{'VALID' if report.valid else 'INVALID'}"
    )

    print(
        f"Nombre de lignes : {report.row_count}"
    )

    print(
        f"Nombre de colonnes : {report.column_count}"
    )

    print(
        f"Valeurs manquantes : "
        f"{report.missing_value_count}"
    )

    print(
        f"Lignes dupliquées : "
        f"{report.duplicate_row_count}"
    )

    print()
    print("Distribution des classes :")

    if report.class_distribution:
        for label, count in (
            report.class_distribution.items()
        ):
            print(
                f"  - {label}: {count}"
            )
    else:
        print(
            "  Aucune distribution disponible."
        )

    print()
    print("Problèmes détectés :")

    if report.issues:
        for issue in report.issues:
            column_information = ""

            if issue.column_name is not None:
                column_information = (
                    f" | colonne={issue.column_name}"
                )

            print(
                f"  - [{issue.severity}] "
                f"{issue.code}"
                f"{column_information}"
            )

            print(
                f"    {issue.message}"
            )
    else:
        print(
            "  Aucun problème détecté."
        )

    print()
    print(
        f"Rapport JSON : "
        f"{report_path.resolve()}"
    )

    print("=" * 60)
    print()


def main() -> int:
    """
    Point d'entrée du script.

    Returns:
        0 si le dataset est valide.
        1 si le dataset est invalide.
    """

    report = run_dataset_validation(
        dataset_path=DEFAULT_DATASET_PATH,
        report_path=DEFAULT_REPORT_PATH,
    )

    print_validation_summary(
        report=report,
        report_path=DEFAULT_REPORT_PATH,
    )

    if report.valid:
        print(
            "Le dataset peut continuer "
            "dans le pipeline d'entraînement."
        )

        return 0

    print(
        "Le dataset ne doit pas être utilisé "
        "pour l'entraînement."
    )

    return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )