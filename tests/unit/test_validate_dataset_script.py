import json
from pathlib import Path

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)
from scripts.validate_dataset import (
    run_dataset_validation,
    write_validation_report,
)


def build_valid_dataset_file(
    tmp_path: Path,
) -> Path:
    """
    Crée temporairement un dataset CSV valide.
    """

    config = DatasetGenerationConfig(
        number_of_records=100,
        random_seed=42,
        good_ratio=0.40,
        acceptable_ratio=0.35,
        poor_ratio=0.25,
    )

    dataframe = generate_quality_dataset(
        config
    )

    dataset_path = (
        tmp_path
        / "dataset_quality.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
        encoding="utf-8",
    )

    return dataset_path


def test_write_validation_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un rapport JSON est correctement écrit.
    """

    dataset_path = build_valid_dataset_file(
        tmp_path
    )

    report_path = (
        tmp_path
        / "reports"
        / "validation.json"
    )

    report = run_dataset_validation(
        dataset_path=dataset_path,
        report_path=report_path,
    )

    assert report.valid is True
    assert report_path.exists()

    report_content = json.loads(
        report_path.read_text(
            encoding="utf-8",
        )
    )

    assert report_content["valid"] is True
    assert report_content["row_count"] == 100
    assert report_content["column_count"] == 11
    assert report_content["issues"] == []


def test_run_dataset_validation_returns_valid_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie le parcours complet avec un dataset valide.
    """

    dataset_path = build_valid_dataset_file(
        tmp_path
    )

    report_path = (
        tmp_path
        / "validation.json"
    )

    report = run_dataset_validation(
        dataset_path=dataset_path,
        report_path=report_path,
    )

    assert report.valid is True
    assert report.row_count == 100

    assert report.class_distribution == {
        "GOOD": 40,
        "ACCEPTABLE": 35,
        "POOR": 25,
    }

    assert report_path.exists()


def test_run_dataset_validation_returns_invalid_report_for_missing_file(
    tmp_path: Path,
) -> None:
    """
    Vérifie que le script produit aussi un rapport
    JSON lorsqu'un dataset est absent.
    """

    missing_dataset_path = (
        tmp_path
        / "missing.csv"
    )

    report_path = (
        tmp_path
        / "missing_validation.json"
    )

    report = run_dataset_validation(
        dataset_path=missing_dataset_path,
        report_path=report_path,
    )

    assert report.valid is False
    assert report.row_count == 0
    assert report_path.exists()

    issue_codes = {
        issue.code
        for issue in report.issues
    }

    assert "FILE_NOT_FOUND" in issue_codes


def test_report_parent_directory_is_created(
    tmp_path: Path,
) -> None:
    """
    Vérifie que l'arborescence du rapport
    est créée automatiquement.
    """

    dataset_path = build_valid_dataset_file(
        tmp_path
    )

    report_path = (
        tmp_path
        / "nested"
        / "reports"
        / "validation.json"
    )

    assert not report_path.parent.exists()

    run_dataset_validation(
        dataset_path=dataset_path,
        report_path=report_path,
    )

    assert report_path.parent.exists()
    assert report_path.exists()