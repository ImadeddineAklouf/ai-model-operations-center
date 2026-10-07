from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.training_schemas import PreprocessingConfig


def validate_feature_names(
    feature_names: Sequence[str],
) -> list:
    """
    Valide et normalise les noms des features.

    Args:
        feature_names:
            Noms des colonnes utilisées comme features.

    Returns:
        Liste validée des noms de features.

    Raises:
        ValueError:
            Si aucune feature n'est fournie,
            si un nom est vide ou si un nom est dupliqué.
    """

    normalized_feature_names = [
        str(feature_name)
        for feature_name in feature_names
    ]

    if not normalized_feature_names:
        raise ValueError(
            "Au moins une feature est nécessaire."
        )

    empty_feature_names = [
        feature_name
        for feature_name in normalized_feature_names
        if not feature_name.strip()
    ]

    if empty_feature_names:
        raise ValueError(
            "Les noms des features ne peuvent pas être vides."
        )

    if (
        len(normalized_feature_names)
        != len(set(normalized_feature_names))
    ):
        raise ValueError(
            "Les noms des features doivent être uniques."
        )

    return normalized_feature_names


def build_numeric_preprocessor(
    feature_names: Sequence[str],
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """
    Construit le préprocesseur numérique.

    Le pipeline applique :

    1. une imputation des valeurs manquantes ;
    2. éventuellement une standardisation.

    Le préprocesseur n'est pas encore entraîné à ce stade.
    """

    validated_feature_names = validate_feature_names(
        feature_names
    )

    numeric_steps: list[
        tuple[str, object]
    ] = [
        (
            "imputer",
            SimpleImputer(
                strategy=config.imputation_strategy,
            ),
        )
    ]

    if config.scale_features:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                validated_feature_names,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataframe_name: str,
) -> None:
    """
    Vérifie qu'un DataFrame de features peut être utilisé.

    Args:
        dataframe:
            DataFrame à contrôler.

        dataframe_name:
            Nom utilisé dans les messages d'erreur.
    """

    if dataframe.empty:
        raise ValueError(
            f"{dataframe_name} ne peut pas être vide."
        )

    if dataframe.columns.duplicated().any():
        raise ValueError(
            f"{dataframe_name} contient "
            "des colonnes dupliquées."
        )

    non_numeric_columns = [
        column_name
        for column_name in dataframe.columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column_name]
        )
    ]

    if non_numeric_columns:
        raise ValueError(
            f"{dataframe_name} contient des features "
            f"non numériques : {non_numeric_columns}"
        )


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """
    Entraîne le préprocesseur uniquement sur X_train.

    Cette règle évite que les ensembles de validation
    ou de test influencent les médianes, moyennes ou
    écarts-types du prétraitement.
    """

    validate_feature_dataframe(
        X_train,
        dataframe_name="X_train",
    )

    preprocessor.fit(
        X_train
    )

    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    features: pd.DataFrame,
    *,
    dataframe_name: str = "features",
) -> pd.DataFrame:
    """
    Transforme des features avec un préprocesseur entraîné.

    Args:
        preprocessor:
            Préprocesseur déjà entraîné.

        features:
            Features à transformer.

        dataframe_name:
            Nom logique du DataFrame.

    Returns:
        DataFrame transformé avec les noms de colonnes.
    """

    validate_feature_dataframe(
        features,
        dataframe_name=dataframe_name,
    )

    transformed_values = preprocessor.transform(
        features
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    transformed_array = np.asarray(
        transformed_values
    )

    return pd.DataFrame(
        transformed_array,
        columns=feature_names,
    ).reset_index(
        drop=True
    )


def fit_and_transform_splits(
    *,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    ColumnTransformer,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Entraîne le préprocesseur sur X_train uniquement,
    puis transforme les trois ensembles.
    """

    fitted_preprocessor = fit_preprocessor(
        preprocessor=preprocessor,
        X_train=X_train,
    )

    transformed_X_train = transform_features(
        preprocessor=fitted_preprocessor,
        features=X_train,
        dataframe_name="X_train",
    )

    transformed_X_validation = transform_features(
        preprocessor=fitted_preprocessor,
        features=X_validation,
        dataframe_name="X_validation",
    )

    transformed_X_test = transform_features(
        preprocessor=fitted_preprocessor,
        features=X_test,
        dataframe_name="X_test",
    )

    return (
        fitted_preprocessor,
        transformed_X_train,
        transformed_X_validation,
        transformed_X_test,
    )