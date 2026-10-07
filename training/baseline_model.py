import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.core.model_schemas import BaselineTrainingConfig


def build_baseline_model(
    config: BaselineTrainingConfig,
) -> LogisticRegression:
    """
    Construit une régression logistique non entraînée.

    Args:
        config:
            Paramètres de configuration du modèle.

    Returns:
        Une instance non entraînée de LogisticRegression.
    """

    return LogisticRegression(
        max_iter=config.maximum_iterations,
        random_state=config.random_seed,
        class_weight=config.class_weight,
        solver="lbfgs",
    )


def train_baseline_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Valide les données puis entraîne le modèle baseline.

    Args:
        model:
            Modèle de régression logistique à entraîner.

        X_train:
            Features d'entraînement.

        y_train:
            Variable cible d'entraînement.

    Returns:
        Le modèle entraîné.

    Raises:
        ValueError:
            Si les données sont vides, désalignées,
            incomplètes ou incompatibles.
    """

    if X_train.empty:
        raise ValueError(
            "X_train ne peut pas être vide."
        )

    if y_train.empty:
        raise ValueError(
            "y_train ne peut pas être vide."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train et y_train doivent contenir "
            "le même nombre d'observations."
        )

    if X_train.columns.duplicated().any():
        raise ValueError(
            "X_train contient des colonnes dupliquées."
        )

    if X_train.isna().any().any():
        raise ValueError(
            "X_train contient des valeurs manquantes."
        )

    if y_train.isna().any():
        raise ValueError(
            "y_train contient des valeurs manquantes."
        )

    non_numeric_columns = []

    for column_name in X_train.columns:
        if not pd.api.types.is_numeric_dtype(
            X_train[column_name]
        ):
            non_numeric_columns.append(
                str(column_name)
            )

    if non_numeric_columns:
        raise ValueError(
            "X_train contient des features "
            f"non numériques : {non_numeric_columns}"
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "L'entraînement nécessite au moins "
            "deux classes différentes."
        )

    model.fit(
        X_train,
        y_train,
    )

    return model