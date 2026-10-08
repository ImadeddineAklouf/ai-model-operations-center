from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.candidate_config import (
    load_candidate_config,
    load_candidate_yaml,
    normalize_candidate_paths,
    resolve_candidate_path,
)


VALID_CANDIDATE_YAML = """
candidate_models:
  random_seed: 42

  models:
    logistic_regression:
      enabled: true
      version: "1.0.0"
      maximum_iterations: 1000
      class_weight: null

    random_forest:
      enabled: true
      version: "1.0.0"
      number_of_estimators: 200
      maximum_depth: null
      minimum_samples_split: 2
      minimum_samples_leaf: 1
      class_weight: null

    gradient_boosting:
      enabled: true
      version: "1.0.0"
      number_of_estimators: 100
      learning_rate: 0.10
      maximum_depth: 3

  evaluation:
    minimum_f1_macro: 0.80
    minimum_poor_recall: 0.85

  input:
    processed_data_directory: "data/processed"

  output:
    candidates_directory: "models/candidates"
    comparison_directory: "reports/evaluation/candidates"
"""


def write_yaml_file(
    file_path: Path,
    content: str,
) -> Path:
    """
    Écrit un fichier YAML utilisé par les tests.
    """

    file_path.write_text(
        content.strip(),
        encoding="utf-8",
    )

    return file_path


def test_resolve_relative_candidate_path() -> None:
    """
    Vérifie qu'un chemin relatif est transformé
    en chemin absolu.
    """

    resolved_path = resolve_candidate_path(
        "data/processed"
    )

    assert resolved_path.is_absolute()
    assert resolved_path.name == "processed"


def test_resolve_absolute_candidate_path(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un chemin absolu reste inchangé.
    """

    absolute_path = (
        tmp_path
        / "data"
        / "processed"
    )

    resolved_path = resolve_candidate_path(
        absolute_path
    )

    assert resolved_path == (
        absolute_path.resolve()
    )


def test_load_candidate_yaml(
    tmp_path: Path,
) -> None:
    """
    Vérifie la lecture du fichier YAML.
    """

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        VALID_CANDIDATE_YAML,
    )

    result = load_candidate_yaml(
        configuration_path
    )

    assert isinstance(
        result,
        dict,
    )

    assert "candidate_models" in result


def test_missing_yaml_file_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie le comportement avec un fichier absent.
    """

    missing_path = (
        tmp_path
        / "missing.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="introuvable",
    ):
        load_candidate_yaml(
            missing_path
        )


def test_yaml_path_must_be_file(
    tmp_path: Path,
) -> None:
    """
    Vérifie que le chemin doit correspondre
    à un fichier et non à un dossier.
    """

    directory_path = (
        tmp_path
        / "configuration"
    )

    directory_path.mkdir()

    with pytest.raises(
        ValueError,
        match="ne correspond pas à un fichier",
    ):
        load_candidate_yaml(
            directory_path
        )


def test_yaml_root_must_be_dictionary(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la racine YAML doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        """
- first
- second
""",
    )

    with pytest.raises(
        ValueError,
        match="objet à sa racine",
    ):
        load_candidate_yaml(
            configuration_path
        )


def test_normalize_candidate_paths() -> None:
    """
    Vérifie que les chemins sont normalisés
    et que la configuration est conservée.
    """

    configuration = {
        "random_seed": 42,
        "models": {
            "logistic_regression": {
                "enabled": True,
                "version": "1.0.0",
                "maximum_iterations": 1000,
                "class_weight": None,
            },
            "random_forest": {
                "enabled": True,
                "version": "1.0.0",
                "number_of_estimators": 200,
                "maximum_depth": None,
                "minimum_samples_split": 2,
                "minimum_samples_leaf": 1,
                "class_weight": None,
            },
            "gradient_boosting": {
                "enabled": True,
                "version": "1.0.0",
                "number_of_estimators": 100,
                "learning_rate": 0.10,
                "maximum_depth": 3,
            },
        },
        "evaluation": {
            "minimum_f1_macro": 0.80,
            "minimum_poor_recall": 0.85,
        },
        "input": {
            "processed_data_directory": (
                "data/processed"
            ),
        },
        "output": {
            "candidates_directory": (
                "models/candidates"
            ),
            "comparison_directory": (
                "reports/evaluation/candidates"
            ),
        },
    }

    result = normalize_candidate_paths(
        configuration
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["random_seed"] == 42

    assert (
        result["input"][
            "processed_data_directory"
        ].is_absolute()
    )

    assert (
        result["output"][
            "candidates_directory"
        ].is_absolute()
    )

    assert (
        result["output"][
            "comparison_directory"
        ].is_absolute()
    )


def test_missing_input_section_is_rejected() -> None:
    """
    Vérifie que la section input est obligatoire.
    """

    configuration = {
        "random_seed": 42,
        "models": {},
        "evaluation": {},
        "output": {
            "candidates_directory": (
                "models/candidates"
            ),
            "comparison_directory": (
                "reports/evaluation/candidates"
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="section 'input'",
    ):
        normalize_candidate_paths(
            configuration
        )


def test_missing_output_section_is_rejected() -> None:
    """
    Vérifie que la section output est obligatoire.
    """

    configuration = {
        "random_seed": 42,
        "models": {},
        "evaluation": {},
        "input": {
            "processed_data_directory": (
                "data/processed"
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="section 'output'",
    ):
        normalize_candidate_paths(
            configuration
        )


def test_missing_processed_directory_is_rejected() -> None:
    """
    Vérifie la présence du répertoire des données.
    """

    configuration = {
        "random_seed": 42,
        "models": {},
        "evaluation": {},
        "input": {},
        "output": {
            "candidates_directory": (
                "models/candidates"
            ),
            "comparison_directory": (
                "reports/evaluation/candidates"
            ),
        },
    }

    with pytest.raises(
        ValueError,
        match="processed_data_directory",
    ):
        normalize_candidate_paths(
            configuration
        )


def test_load_valid_candidate_config(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chargement complet d'une configuration.
    """

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        VALID_CANDIDATE_YAML,
    )

    config = load_candidate_config(
        configuration_path
    )

    assert config.random_seed == 42

    assert (
        config.models.logistic_regression.enabled
        is True
    )

    assert (
        config.models.random_forest.enabled
        is True
    )

    assert (
        config.models.gradient_boosting.enabled
        is True
    )

    assert (
        config.models.random_forest
        .number_of_estimators
        == 200
    )

    assert (
        config.models.gradient_boosting
        .learning_rate
        == pytest.approx(0.10)
    )

    assert (
        config.evaluation.minimum_f1_macro
        == pytest.approx(0.80)
    )

    assert (
        config.evaluation.minimum_poor_recall
        == pytest.approx(0.85)
    )

    assert (
        config.input.processed_data_directory
        .is_absolute()
    )

    assert (
        config.output.candidates_directory
        .is_absolute()
    )

    assert (
        config.output.comparison_directory
        .is_absolute()
    )


def test_missing_candidate_models_section_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la section candidate_models
    est obligatoire.
    """

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        """
other_section:
  value: 1
""",
    )

    with pytest.raises(
        ValueError,
        match="candidate_models",
    ):
        load_candidate_config(
            configuration_path
        )


def test_candidate_models_section_must_be_object(
    tmp_path: Path,
) -> None:
    """
    Vérifie que candidate_models doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        """
candidate_models: invalid
""",
    )

    with pytest.raises(
        ValueError,
        match="doit être un objet",
    ):
        load_candidate_config(
            configuration_path
        )


def test_invalid_evaluation_threshold_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les seuils sont validés par Pydantic.
    """

    invalid_yaml = (
        VALID_CANDIDATE_YAML.replace(
            "minimum_f1_macro: 0.80",
            "minimum_f1_macro: 1.50",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_candidate_config(
            configuration_path
        )


def test_unknown_configuration_field_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    invalid_yaml = (
        VALID_CANDIDATE_YAML.replace(
            "  random_seed: 42",
            (
                "  random_seed: 42\n"
                "  unknown_field: unexpected"
            ),
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "candidate_models.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_candidate_config(
            configuration_path
        )