from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.training_config import (
    load_training_config,
    resolve_project_path,
)


VALID_TRAINING_YAML = """
training:
  dataset_path: "data/raw/dataset_quality.csv"
  target_column: "quality_label"

  split:
    train_ratio: 0.70
    validation_ratio: 0.15
    test_ratio: 0.15
    random_seed: 42
    stratify: true

  preprocessing:
    imputation_strategy: "median"
    scale_features: true

  output:
    processed_data_directory: "data/processed"
    artifacts_directory: "models/preprocessing"
    reports_directory: "reports/training"
"""


def write_yaml(
    path: Path,
    content: str,
) -> Path:
    """
    Écrit un fichier YAML temporaire.
    """

    path.write_text(
        content.strip(),
        encoding="utf-8",
    )

    return path


def test_resolve_relative_project_path() -> None:
    """
    Un chemin relatif doit devenir absolu.
    """

    resolved_path = resolve_project_path(
        "data/raw/dataset_quality.csv"
    )

    assert resolved_path.is_absolute()

    assert resolved_path.name == (
        "dataset_quality.csv"
    )


def test_resolve_absolute_path(
    tmp_path: Path,
) -> None:
    """
    Un chemin absolu doit rester absolu.
    """

    absolute_path = (
        tmp_path
        / "dataset.csv"
    )

    resolved_path = resolve_project_path(
        absolute_path
    )

    assert resolved_path == (
        absolute_path.resolve()
    )


def test_load_valid_training_config(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chargement d'une configuration valide.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        VALID_TRAINING_YAML,
    )

    config = load_training_config(
        configuration_path
    )

    assert config.target_column == "quality_label"

    assert config.split.train_ratio == pytest.approx(
        0.70
    )

    assert config.split.validation_ratio == pytest.approx(
        0.15
    )

    assert config.split.test_ratio == pytest.approx(
        0.15
    )

    assert config.split.random_seed == 42
    assert config.split.stratify is True

    assert (
        config.preprocessing.imputation_strategy
        == "median"
    )

    assert config.preprocessing.scale_features is True

    assert config.dataset_path.is_absolute()

    assert (
        config.output.processed_data_directory
        .is_absolute()
    )

    assert (
        config.output.artifacts_directory
        .is_absolute()
    )

    assert (
        config.output.reports_directory
        .is_absolute()
    )


def test_missing_configuration_file_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie le comportement pour un fichier absent.
    """

    missing_path = (
        tmp_path
        / "missing.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="introuvable",
    ):
        load_training_config(
            missing_path
        )


def test_missing_training_section_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la section training est obligatoire.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        """
other_section:
  value: 1
""",
    )

    with pytest.raises(
        ValueError,
        match="section 'training'",
    ):
        load_training_config(
            configuration_path
        )


def test_training_section_must_be_object(
    tmp_path: Path,
) -> None:
    """
    Vérifie que training doit être un objet YAML.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        """
training: "invalid"
""",
    )

    with pytest.raises(
        ValueError,
        match="doit être un objet",
    ):
        load_training_config(
            configuration_path
        )


def test_missing_output_section_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la section output est obligatoire.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        """
training:
  dataset_path: "data/raw/dataset_quality.csv"
  target_column: "quality_label"

  split:
    train_ratio: 0.70
    validation_ratio: 0.15
    test_ratio: 0.15
    random_seed: 42
    stratify: true

  preprocessing:
    imputation_strategy: "median"
    scale_features: true
""",
    )

    with pytest.raises(
        ValueError,
        match="section 'output'",
    ):
        load_training_config(
            configuration_path
        )


def test_invalid_ratio_sum_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que Pydantic valide les ratios du YAML.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        """
training:
  dataset_path: "data/raw/dataset_quality.csv"
  target_column: "quality_label"

  split:
    train_ratio: 0.70
    validation_ratio: 0.20
    test_ratio: 0.20
    random_seed: 42
    stratify: true

  preprocessing:
    imputation_strategy: "median"
    scale_features: true

  output:
    processed_data_directory: "data/processed"
    artifacts_directory: "models/preprocessing"
    reports_directory: "reports/training"
""",
    )

    with pytest.raises(
        ValidationError,
        match="somme des proportions",
    ):
        load_training_config(
            configuration_path
        )


def test_unknown_training_field_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les champs YAML inconnus sont refusés.
    """

    configuration_path = write_yaml(
        tmp_path / "training.yaml",
        """
training:
  dataset_path: "data/raw/dataset_quality.csv"
  target_column: "quality_label"
  unknown_field: "unexpected"

  split:
    train_ratio: 0.70
    validation_ratio: 0.15
    test_ratio: 0.15
    random_seed: 42
    stratify: true

  preprocessing:
    imputation_strategy: "median"
    scale_features: true

  output:
    processed_data_directory: "data/processed"
    artifacts_directory: "models/preprocessing"
    reports_directory: "reports/training"
""",
    )

    with pytest.raises(ValidationError):
        load_training_config(
            configuration_path
        )