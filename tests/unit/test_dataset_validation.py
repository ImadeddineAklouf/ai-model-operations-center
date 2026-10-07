from pathlib import Path

import pandas as pd

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.dataset_validation import (
    validate_dataset_dataframe,
    validate_dataset_file,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)


def build_valid_dataframe(
    number_of_records: int = 100,
) -> pd.DataFrame:
    """
    Génère un DataFrame valide utilisé par les tests.
    """

    config = DatasetGenerationConfig(
        number_of_records=number_of_records,
        random_seed=42,
        good_ratio=0.40,
        acceptable_ratio=0.35,
        poor_ratio=0.25,
    )

    return generate_quality_dataset(
        config
    )


def get_issue_codes(
    dataframe: pd.DataFrame,
) -> set:
    """
    Valide un DataFrame et retourne seulement
    les codes des problèmes détectés.
    """

    report = validate_dataset_dataframe(
        dataframe
    )

    return {
        issue.code
        for issue in report.issues
    }


def test_valid_dataframe_is_accepted() -> None:
    dataframe = build_valid_dataframe()

    report = validate_dataset_dataframe(
        dataframe,
        dataset_path="<test-memory>",
    )

    assert report.valid is True
    assert report.dataset_path == "<test-memory>"
    assert report.row_count == 100
    assert report.column_count == 11
    assert report.duplicate_row_count == 0
    assert report.missing_value_count == 0
    assert report.issues == []


def test_valid_dataframe_contains_expected_distribution() -> None:
    dataframe = build_valid_dataframe()

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.class_distribution == {
        "GOOD": 40,
        "ACCEPTABLE": 35,
        "POOR": 25,
    }


def test_empty_dataframe_is_rejected() -> None:
    dataframe = pd.DataFrame()

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False
    assert report.row_count == 0
    assert "EMPTY_DATASET" in {
        issue.code
        for issue in report.issues
    }


def test_dataframe_with_missing_column_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe = dataframe.drop(
        columns=["missing_value_ratio"]
    )

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False

    matching_issues = [
        issue
        for issue in report.issues
        if issue.code == "MISSING_COLUMN"
    ]

    assert len(matching_issues) == 1

    assert (
        matching_issues[0].column_name
        == "missing_value_ratio"
    )


def test_unexpected_column_creates_warning() -> None:
    dataframe = build_valid_dataframe()
    dataframe["unexpected_column"] = "value"

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is True

    matching_issues = [
        issue
        for issue in report.issues
        if issue.code == "UNEXPECTED_COLUMN"
    ]

    assert len(matching_issues) == 1

    assert (
        matching_issues[0].severity
        == "WARNING"
    )

    assert (
        matching_issues[0].column_name
        == "unexpected_column"
    )


def test_missing_value_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "missing_value_ratio",
    ] = None

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False
    assert report.missing_value_count == 1

    issue_codes = {
        issue.code
        for issue in report.issues
    }

    assert "MISSING_VALUES" in issue_codes
    assert "INVALID_RECORD" in issue_codes


def test_duplicate_row_creates_warning() -> None:
    dataframe = build_valid_dataframe()

    duplicate = dataframe.iloc[[0]].copy()

    dataframe = pd.concat(
        [
            dataframe,
            duplicate,
        ],
        ignore_index=True,
    )

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is True
    assert report.duplicate_row_count == 1

    matching_issues = [
        issue
        for issue in report.issues
        if issue.code == "DUPLICATE_ROWS"
    ]

    assert len(matching_issues) == 1
    assert matching_issues[0].severity == "WARNING"


def test_ratio_above_one_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "missing_value_ratio",
    ] = 1.5

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False

    issue_codes = {
        issue.code
        for issue in report.issues
    }

    assert "INVALID_RECORD" in issue_codes


def test_negative_freshness_delay_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "freshness_delay_hours",
    ] = -5.0

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False
    assert "INVALID_RECORD" in get_issue_codes(
        dataframe
    )


def test_unknown_label_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "quality_label",
    ] = "EXCELLENT"

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False

    issue_codes = {
        issue.code
        for issue in report.issues
    }

    assert "UNKNOWN_LABEL" in issue_codes
    assert "INVALID_RECORD" in issue_codes


def test_missing_class_is_rejected() -> None:
    dataframe = build_valid_dataframe()

    dataframe = dataframe.loc[
        dataframe["quality_label"] != "POOR"
    ].reset_index(
        drop=True
    )

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False

    matching_issues = [
        issue
        for issue in report.issues
        if issue.code == "MISSING_CLASS"
    ]

    assert len(matching_issues) == 1
    assert matching_issues[0].column_name == "quality_label"
    assert "POOR" in matching_issues[0].message


def test_dataset_with_fewer_than_thirty_rows_is_rejected() -> None:
    dataframe = build_valid_dataframe(
        number_of_records=30
    ).head(
        29
    )

    report = validate_dataset_dataframe(
        dataframe
    )

    assert report.valid is False
    assert "INSUFFICIENT_ROWS" in {
        issue.code
        for issue in report.issues
    }


def test_existing_valid_csv_is_accepted(
    tmp_path: Path,
) -> None:
    dataframe = build_valid_dataframe()

    dataset_path = (
        tmp_path
        / "valid_dataset.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
        encoding="utf-8",
    )

    report = validate_dataset_file(
        dataset_path
    )

    assert report.valid is True
    assert report.row_count == 100
    assert report.column_count == 11
    assert report.issues == []


def test_missing_file_returns_structured_report(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "missing.csv"
    )

    report = validate_dataset_file(
        missing_path
    )

    assert report.valid is False
    assert report.row_count == 0
    assert report.column_count == 0
    assert len(report.issues) == 1
    assert report.issues[0].code == "FILE_NOT_FOUND"


def test_physically_empty_csv_is_rejected(
    tmp_path: Path,
) -> None:
    empty_path = (
        tmp_path
        / "empty.csv"
    )

    empty_path.write_text(
        "",
        encoding="utf-8",
    )

    report = validate_dataset_file(
        empty_path
    )

    assert report.valid is False

    issue_codes = {
        issue.code
        for issue in report.issues
    }

    assert "EMPTY_DATASET" in issue_codes
