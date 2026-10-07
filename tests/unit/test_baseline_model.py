import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.core.model_schemas import BaselineTrainingConfig
from training.baseline_model import (
    build_baseline_model,
    train_baseline_model,
)


def create_training_features() -> pd.DataFrame:
    """
    Crée un petit jeu de features numériques
    destiné aux tests unitaires.
    """

    return pd.DataFrame(
        {
            "missing_value_ratio": [
                0.40,
                0.35,
                0.12,
                0.10,
                0.02,
                0.01,
            ],
            "duplicate_row_ratio": [
                0.25,
                0.20,
                0.07,
                0.05,
                0.01,
                0.00,
            ],
            "primary_key_uniqueness": [
                0.65,
                0.70,
                0.92,
                0.95,
                0.99,
                1.00,
            ],
        }
    )


def create_training_target() -> pd.Series:
    """
    Crée une cible contenant les trois classes.
    """

    return pd.Series(
        [
            "POOR",
            "POOR",
            "ACCEPTABLE",
            "ACCEPTABLE",
            "GOOD",
            "GOOD",
        ],
        name="quality_label",
    )


def test_build_baseline_model() -> None:
    """
    Vérifie la construction de la régression logistique.
    """

    config = BaselineTrainingConfig(
        maximum_iterations=500,
        random_seed=42,
        class_weight=None,
    )

    model = build_baseline_model(
        config
    )

    assert isinstance(
        model,
        LogisticRegression,
    )

    assert model.max_iter == 500
    assert model.random_state == 42
    assert model.class_weight is None
    assert model.solver == "lbfgs"


def test_build_baseline_model_with_balanced_weights() -> None:
    """
    Vérifie la prise en charge de class_weight='balanced'.
    """

    config = BaselineTrainingConfig(
        maximum_iterations=1000,
        random_seed=42,
        class_weight="balanced",
    )

    model = build_baseline_model(
        config
    )

    assert model.class_weight == "balanced"


def test_train_baseline_model() -> None:
    """
    Vérifie que le modèle peut être entraîné.
    """

    X_train = create_training_features()
    y_train = create_training_target()

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    fitted_model = train_baseline_model(
        model,
        X_train,
        y_train,
    )

    assert isinstance(
        fitted_model,
        LogisticRegression,
    )

    assert hasattr(
        fitted_model,
        "classes_",
    )

    assert set(
        fitted_model.classes_
    ) == {
        "GOOD",
        "ACCEPTABLE",
        "POOR",
    }

    assert fitted_model.n_features_in_ == 3


def test_trained_model_can_predict() -> None:
    """
    Vérifie qu'un modèle entraîné produit une prédiction
    pour chaque observation.
    """

    X_train = create_training_features()
    y_train = create_training_target()

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    fitted_model = train_baseline_model(
        model,
        X_train,
        y_train,
    )

    predictions = fitted_model.predict(
        X_train
    )

    assert len(predictions) == len(
        X_train
    )

    assert set(predictions).issubset(
        {
            "GOOD",
            "ACCEPTABLE",
            "POOR",
        }
    )


def test_empty_training_features_are_rejected() -> None:
    """
    Vérifie qu'un X_train vide est refusé.
    """

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="X_train",
    ):
        train_baseline_model(
            model,
            pd.DataFrame(),
            create_training_target(),
        )


def test_empty_training_target_is_rejected() -> None:
    """
    Vérifie qu'un y_train vide est refusé.
    """

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="y_train",
    ):
        train_baseline_model(
            model,
            create_training_features(),
            pd.Series(
                dtype=str,
                name="quality_label",
            ),
        )


def test_misaligned_training_data_are_rejected() -> None:
    """
    Vérifie que X_train et y_train doivent avoir
    la même longueur.
    """

    X_train = create_training_features()

    y_train = create_training_target().iloc[
        :-1
    ]

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="même nombre",
    ):
        train_baseline_model(
            model,
            X_train,
            y_train,
        )


def test_single_training_class_is_rejected() -> None:
    """
    Vérifie que l'entraînement nécessite
    au moins deux classes.
    """

    X_train = create_training_features()

    y_train = pd.Series(
        [
            "GOOD",
            "GOOD",
            "GOOD",
            "GOOD",
            "GOOD",
            "GOOD",
        ],
        name="quality_label",
    )

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="deux classes",
    ):
        train_baseline_model(
            model,
            X_train,
            y_train,
        )


def test_training_features_with_missing_values_are_rejected() -> None:
    """
    Vérifie que les données préparées ne contiennent
    aucune valeur manquante.
    """

    X_train = create_training_features()

    X_train.loc[
        0,
        "missing_value_ratio",
    ] = None

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="valeurs manquantes",
    ):
        train_baseline_model(
            model,
            X_train,
            create_training_target(),
        )


def test_non_numeric_training_feature_is_rejected() -> None:
    """
    Vérifie que toutes les features sont numériques.
    """

    X_train = create_training_features()

    X_train["text_feature"] = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]

    model = build_baseline_model(
        BaselineTrainingConfig()
    )

    with pytest.raises(
        ValueError,
        match="non numériques",
    ):
        train_baseline_model(
            model,
            X_train,
            create_training_target(),
        )