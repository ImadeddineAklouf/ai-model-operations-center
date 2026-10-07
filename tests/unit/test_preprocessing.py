import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

from app.core.dataset_generator import (
    generate_quality_dataset,
)
from app.core.quality_schemas import (
    DatasetGenerationConfig,
)
from app.core.training_schemas import (
    DataSplitConfig,
    PreprocessingConfig,
)
from training.data_split import split_dataset
from training.preprocessing import (
    build_numeric_preprocessor,
    fit_and_transform_splits,
    fit_preprocessor,
    transform_features,
    validate_feature_dataframe,
    validate_feature_names,
)


def create_dataset_splits():
    """
    Produit des ensembles reproductibles pour les tests.
    """

    dataframe = generate_quality_dataset(
        DatasetGenerationConfig(
            number_of_records=100,
            random_seed=42,
            good_ratio=0.40,
            acceptable_ratio=0.35,
            poor_ratio=0.25,
        )
    )

    return split_dataset(
        dataframe=dataframe,
        target_column="quality_label",
        config=DataSplitConfig(
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            stratify=True,
        ),
    )


def test_validate_feature_names() -> None:
    """
    Vérifie la validation d'une liste correcte.
    """

    feature_names = [
        "feature_a",
        "feature_b",
    ]

    result = validate_feature_names(
        feature_names
    )

    assert result == feature_names


def test_empty_feature_names_are_rejected() -> None:
    """
    Vérifie qu'au moins une feature est obligatoire.
    """

    with pytest.raises(
        ValueError,
        match="Au moins une feature",
    ):
        validate_feature_names(
            []
        )


def test_blank_feature_name_is_rejected() -> None:
    """
    Vérifie qu'un nom vide est refusé.
    """

    with pytest.raises(
        ValueError,
        match="ne peuvent pas être vides",
    ):
        validate_feature_names(
            [
                "feature_a",
                "",
            ]
        )


def test_duplicate_feature_names_are_rejected() -> None:
    """
    Vérifie que les noms doivent être uniques.
    """

    with pytest.raises(
        ValueError,
        match="doivent être uniques",
    ):
        validate_feature_names(
            [
                "feature_a",
                "feature_a",
            ]
        )


def test_build_numeric_preprocessor() -> None:
    """
    Vérifie le type du préprocesseur construit.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(),
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )


def test_scaler_is_present_when_enabled() -> None:
    """
    Vérifie que le scaler est ajouté si demandé.
    """

    preprocessor = build_numeric_preprocessor(
        feature_names=[
            "feature_a",
            "feature_b",
        ],
        config=PreprocessingConfig(
            scale_features=True,
        ),
    )

    numeric_pipeline = (
        preprocessor.transformers[0][1]
    )

    assert "imputer" in (
        numeric_pipeline.named_steps
    )

    assert "scaler" in (
        numeric_pipeline.named_steps
    )


def test_scaler_is_absent_when_disabled() -> None:
    """
    Vérifie que le scaler peut être désactivé.
    """

    preprocessor = build_numeric_preprocessor(
        feature_names=[
            "feature_a",
            "feature_b",
        ],
        config=PreprocessingConfig(
            scale_features=False,
        ),
    )

    numeric_pipeline = (
        preprocessor.transformers[0][1]
    )

    assert "imputer" in (
        numeric_pipeline.named_steps
    )

    assert "scaler" not in (
        numeric_pipeline.named_steps
    )


def test_empty_feature_dataframe_is_rejected() -> None:
    """
    Vérifie qu'un DataFrame vide est refusé.
    """

    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        validate_feature_dataframe(
            pd.DataFrame(),
            dataframe_name="X_train",
        )


def test_non_numeric_feature_is_rejected() -> None:
    """
    Vérifie que toutes les features doivent être numériques.
    """

    dataframe = pd.DataFrame(
        {
            "numeric_feature": [
                1.0,
                2.0,
            ],
            "text_feature": [
                "a",
                "b",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="non numériques",
    ):
        validate_feature_dataframe(
            dataframe,
            dataframe_name="X_train",
        )


def test_preprocessor_is_fitted_on_train() -> None:
    """
    Vérifie que le préprocesseur peut être entraîné.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(),
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=splits.X_train,
    )

    assert hasattr(
        fitted_preprocessor,
        "transformers_",
    )


def test_transform_before_fit_is_rejected() -> None:
    """
    Vérifie qu'un préprocesseur non entraîné
    ne peut pas transformer des données.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(),
    )

    with pytest.raises(NotFittedError):
        transform_features(
            preprocessor=preprocessor,
            features=splits.X_train,
            dataframe_name="X_train",
        )


def test_transformed_train_has_expected_shape() -> None:
    """
    Vérifie que la transformation conserve
    les dimensions.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(),
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=splits.X_train,
    )

    transformed = transform_features(
        preprocessor=fitted_preprocessor,
        features=splits.X_train,
        dataframe_name="X_train",
    )

    assert transformed.shape == (
        splits.X_train.shape
    )

    assert list(
        transformed.columns
    ) == list(
        splits.X_train.columns
    )


def test_scaled_training_features_have_zero_mean() -> None:
    """
    Après StandardScaler, les moyennes de X_train
    doivent être proches de zéro.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(
            scale_features=True,
        ),
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=splits.X_train,
    )

    transformed = transform_features(
        preprocessor=fitted_preprocessor,
        features=splits.X_train,
        dataframe_name="X_train",
    )

    assert np.allclose(
        transformed.mean().to_numpy(),
        0.0,
        atol=1e-7,
    )


def test_scaled_training_features_have_unit_standard_deviation() -> None:
    """
    Après StandardScaler, les écarts-types population
    de X_train doivent être proches de 1.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(
            scale_features=True,
        ),
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=splits.X_train,
    )

    transformed = transform_features(
        preprocessor=fitted_preprocessor,
        features=splits.X_train,
        dataframe_name="X_train",
    )

    assert np.allclose(
        transformed.std(
            ddof=0
        ).to_numpy(),
        1.0,
        atol=1e-7,
    )


def test_missing_values_are_imputed() -> None:
    """
    Vérifie que l'imputeur supprime les valeurs manquantes.
    """

    splits = create_dataset_splits()

    X_train_with_missing_value = (
        splits.X_train.copy()
    )

    X_train_with_missing_value.loc[
        0,
        "missing_value_ratio",
    ] = np.nan

    preprocessor = build_numeric_preprocessor(
        feature_names=(
            X_train_with_missing_value.columns
        ),
        config=PreprocessingConfig(
            imputation_strategy="median",
        ),
    )

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=X_train_with_missing_value,
    )

    transformed = transform_features(
        preprocessor=fitted_preprocessor,
        features=X_train_with_missing_value,
        dataframe_name="X_train",
    )

    assert not transformed.isna().any().any()


def test_fit_and_transform_all_splits() -> None:
    """
    Vérifie le parcours complet sur les trois ensembles.
    """

    splits = create_dataset_splits()

    preprocessor = build_numeric_preprocessor(
        feature_names=splits.X_train.columns,
        config=PreprocessingConfig(),
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

    assert hasattr(
        fitted_preprocessor,
        "transformers_",
    )

    assert transformed_X_train.shape == (
        splits.X_train.shape
    )

    assert transformed_X_validation.shape == (
        splits.X_validation.shape
    )

    assert transformed_X_test.shape == (
        splits.X_test.shape
    )

    assert list(
        transformed_X_train.columns
    ) == list(
        transformed_X_validation.columns
    )

    assert list(
        transformed_X_train.columns
    ) == list(
        transformed_X_test.columns
    )