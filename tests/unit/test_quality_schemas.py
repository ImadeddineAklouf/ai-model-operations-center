import pytest
from pydantic import ValidationError

from app.core.quality_schemas import (
    DatasetGenerationConfig,
    DatasetQualityFeatures,
    DatasetQualityRecord,
    QualityLabel,
)


def build_valid_features() -> dict[str, int | float]:
    """
    Retourne un jeu de features valide réutilisable
    dans les différents tests.
    """

    return {
        "row_count": 10_000,
        "column_count": 12,
        "missing_value_ratio": 0.02,
        "duplicate_row_ratio": 0.01,
        "invalid_type_ratio": 0.005,
        "primary_key_uniqueness": 0.99,
        "foreign_key_match_ratio": 0.98,
        "schema_change_count": 0,
        "outlier_ratio": 0.03,
        "freshness_delay_hours": 2.0,
    }


def test_quality_label_values() -> None:
    """
    Vérifie les trois classes autorisées.
    """

    assert QualityLabel.GOOD.value == "GOOD"

    assert (
        QualityLabel.ACCEPTABLE.value
        == "ACCEPTABLE"
    )

    assert QualityLabel.POOR.value == "POOR"


def test_valid_quality_features_are_created() -> None:
    """
    Vérifie qu'un ensemble de features valide
    produit correctement un modèle Pydantic.
    """

    features = DatasetQualityFeatures(
        **build_valid_features()
    )

    assert features.row_count == 10_000
    assert features.column_count == 12

    assert (
        features.missing_value_ratio
        == pytest.approx(0.02)
    )

    assert (
        features.primary_key_uniqueness
        == pytest.approx(0.99)
    )

    assert (
        features.foreign_key_match_ratio
        == pytest.approx(0.98)
    )


def test_valid_quality_record_is_created() -> None:
    """
    Vérifie la construction d'un enregistrement
    contenant les features et la variable cible.
    """

    record = DatasetQualityRecord(
        **build_valid_features(),
        quality_label=QualityLabel.GOOD,
    )

    assert record.quality_label is QualityLabel.GOOD
    assert record.row_count == 10_000


def test_string_label_is_converted_to_enum() -> None:
    """
    Vérifie que Pydantic convertit automatiquement
    la chaîne GOOD vers QualityLabel.GOOD.
    """

    record = DatasetQualityRecord(
        **build_valid_features(),
        quality_label="GOOD",
    )

    assert record.quality_label is QualityLabel.GOOD


def test_invalid_quality_label_is_rejected() -> None:
    """
    Vérifie qu'une classe inconnue est refusée.
    """

    with pytest.raises(ValidationError):
        DatasetQualityRecord(
            **build_valid_features(),
            quality_label="EXCELLENT",
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "missing_value_ratio",
        "duplicate_row_ratio",
        "invalid_type_ratio",
        "primary_key_uniqueness",
        "foreign_key_match_ratio",
        "outlier_ratio",
    ],
)
def test_ratio_above_one_is_rejected(
    field_name: str,
) -> None:
    """
    Vérifie que chaque ratio est limité à 1.
    """

    features = build_valid_features()
    features[field_name] = 1.01

    with pytest.raises(ValidationError):
        DatasetQualityFeatures(
            **features
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "missing_value_ratio",
        "duplicate_row_ratio",
        "invalid_type_ratio",
        "primary_key_uniqueness",
        "foreign_key_match_ratio",
        "outlier_ratio",
    ],
)
def test_negative_ratio_is_rejected(
    field_name: str,
) -> None:
    """
    Vérifie que chaque ratio est supérieur
    ou égal à zéro.
    """

    features = build_valid_features()
    features[field_name] = -0.01

    with pytest.raises(ValidationError):
        DatasetQualityFeatures(
            **features
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "missing_value_ratio",
        "duplicate_row_ratio",
        "invalid_type_ratio",
        "primary_key_uniqueness",
        "foreign_key_match_ratio",
        "outlier_ratio",
    ],
)
def test_ratio_boundaries_are_accepted(
    field_name: str,
) -> None:
    """
    Vérifie que les bornes exactes 0 et 1
    sont acceptées.
    """

    features_at_zero = build_valid_features()
    features_at_zero[field_name] = 0.0

    zero_result = DatasetQualityFeatures(
        **features_at_zero
    )

    assert getattr(
        zero_result,
        field_name,
    ) == pytest.approx(0.0)

    features_at_one = build_valid_features()
    features_at_one[field_name] = 1.0

    one_result = DatasetQualityFeatures(
        **features_at_one
    )

    assert getattr(
        one_result,
        field_name,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("row_count", 0),
        ("column_count", 0),
        ("schema_change_count", -1),
        ("freshness_delay_hours", -0.1),
    ],
)
def test_invalid_count_or_delay_is_rejected(
    field_name: str,
    invalid_value: int | float,
) -> None:
    """
    Vérifie les contraintes sur les compteurs
    et le délai de fraîcheur.
    """

    features = build_valid_features()
    features[field_name] = invalid_value

    with pytest.raises(ValidationError):
        DatasetQualityFeatures(
            **features
        )


def test_unknown_feature_is_rejected() -> None:
    """
    Vérifie le comportement extra='forbid'.
    """

    features = build_valid_features()

    with pytest.raises(ValidationError):
        DatasetQualityFeatures(
            **features,
            unknown_metric=42,
        )


def test_required_feature_cannot_be_missing() -> None:
    """
    Vérifie qu'une feature obligatoire absente
    provoque une erreur.
    """

    features = build_valid_features()

    del features["missing_value_ratio"]

    with pytest.raises(ValidationError):
        DatasetQualityFeatures(
            **features
        )


def test_valid_generation_configuration() -> None:
    """
    Vérifie une configuration explicite valide.
    """

    config = DatasetGenerationConfig(
        number_of_records=3_000,
        random_seed=42,
        good_ratio=0.40,
        acceptable_ratio=0.35,
        poor_ratio=0.25,
    )

    assert config.number_of_records == 3_000
    assert config.random_seed == 42

    ratio_sum = (
        config.good_ratio
        + config.acceptable_ratio
        + config.poor_ratio
    )

    assert ratio_sum == pytest.approx(1.0)


def test_default_generation_configuration() -> None:
    """
    Vérifie les valeurs par défaut.
    """

    config = DatasetGenerationConfig()

    assert config.number_of_records == 3_000
    assert config.random_seed == 42

    assert config.good_ratio == pytest.approx(
        0.40
    )

    assert config.acceptable_ratio == pytest.approx(
        0.35
    )

    assert config.poor_ratio == pytest.approx(
        0.25
    )


def test_generation_config_requires_minimum_records() -> None:
    """
    Vérifie la contrainte number_of_records >= 30.
    """

    with pytest.raises(ValidationError):
        DatasetGenerationConfig(
            number_of_records=29,
        )


def test_negative_random_seed_is_rejected() -> None:
    """
    Vérifie que la seed ne peut pas être négative.
    """

    with pytest.raises(ValidationError):
        DatasetGenerationConfig(
            random_seed=-1,
        )


def test_invalid_class_distribution_is_rejected() -> None:
    """
    Vérifie que la somme des ratios doit valoir 1.
    """

    with pytest.raises(
        ValidationError,
        match="somme des proportions",
    ):
        DatasetGenerationConfig(
            number_of_records=3_000,
            random_seed=42,
            good_ratio=0.50,
            acceptable_ratio=0.40,
            poor_ratio=0.30,
        )


def test_generation_config_rejects_unknown_field() -> None:
    """
    Vérifie le comportement extra='forbid'
    sur la configuration.
    """

    with pytest.raises(ValidationError):
        DatasetGenerationConfig(
            unknown_parameter="value",
        )


def test_generation_config_is_immutable() -> None:
    """
    Vérifie le comportement frozen=True.

    setattr est utilisé volontairement pour éviter
    que l'analyseur Python considère l'affectation
    comme une erreur statique avant le test.
    """

    config = DatasetGenerationConfig()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "random_seed",
            100,
        )