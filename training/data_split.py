from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.training_schemas import DataSplitConfig


@dataclass(frozen=True)
class DatasetSplits:
    """
    Contient les ensembles d'entraînement,
    de validation et de test.
    """

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class SplitSizes:
    """
    Nombre exact d'observations attendu dans chaque ensemble.
    """

    train: int
    validation: int
    test: int


def calculate_split_sizes(
    number_of_records: int,
    config: DataSplitConfig,
) -> SplitSizes:
    """
    Calcule le nombre exact d'observations par ensemble.

    Le nombre de lignes de test et de validation est calculé
    explicitement. Le jeu d'entraînement reçoit le reste afin
    de garantir que la somme finale correspond exactement au
    nombre total d'observations.
    """

    if number_of_records <= 0:
        raise ValueError(
            "Le nombre d'observations doit être positif."
        )

    test_size = round(
        number_of_records
        * config.test_ratio
    )

    validation_size = round(
        number_of_records
        * config.validation_ratio
    )

    train_size = (
        number_of_records
        - validation_size
        - test_size
    )

    if train_size <= 0:
        raise ValueError(
            "La configuration ne laisse aucune observation "
            "pour l'ensemble d'entraînement."
        )

    if validation_size <= 0:
        raise ValueError(
            "La configuration ne laisse aucune observation "
            "pour l'ensemble de validation."
        )

    if test_size <= 0:
        raise ValueError(
            "La configuration ne laisse aucune observation "
            "pour l'ensemble de test."
        )

    return SplitSizes(
        train=train_size,
        validation=validation_size,
        test=test_size,
    )


def separate_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features X de la cible y.
    """

    if dataframe.empty:
        raise ValueError(
            "Le dataset ne peut pas être vide."
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"La colonne cible '{target_column}' "
            "est absente du dataset."
        )

    feature_columns = [
        column_name
        for column_name in dataframe.columns
        if column_name != target_column
    ]

    if not feature_columns:
        raise ValueError(
            "Le dataset ne contient aucune feature."
        )

    features = dataframe.loc[
        :,
        feature_columns,
    ].copy()

    target = dataframe.loc[
        :,
        target_column,
    ].copy()

    return features, target


def split_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    config: DataSplitConfig,
) -> DatasetSplits:
    """
    Sépare les données en entraînement, validation et test.

    Les tailles sont transmises à Scikit-learn sous forme
    d'entiers afin d'éviter les erreurs d'arrondi liées aux
    nombres flottants.
    """

    features, target = separate_features_and_target(
        dataframe=dataframe,
        target_column=target_column,
    )

    split_sizes = calculate_split_sizes(
        number_of_records=len(dataframe),
        config=config,
    )

    first_stratification_target = (
        target
        if config.stratify
        else None
    )

    (
        X_train_validation,
        X_test,
        y_train_validation,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=split_sizes.test,
        random_state=config.random_seed,
        shuffle=True,
        stratify=first_stratification_target,
    )

    second_stratification_target = (
        y_train_validation
        if config.stratify
        else None
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=split_sizes.validation,
        random_state=config.random_seed,
        shuffle=True,
        stratify=second_stratification_target,
    )

    result = DatasetSplits(
        X_train=X_train.reset_index(
            drop=True
        ),
        X_validation=X_validation.reset_index(
            drop=True
        ),
        X_test=X_test.reset_index(
            drop=True
        ),
        y_train=y_train.reset_index(
            drop=True
        ),
        y_validation=y_validation.reset_index(
            drop=True
        ),
        y_test=y_test.reset_index(
            drop=True
        ),
    )

    validate_split_sizes(
        splits=result,
        expected_sizes=split_sizes,
    )

    return result


def validate_split_sizes(
    *,
    splits: DatasetSplits,
    expected_sizes: SplitSizes,
) -> None:
    """
    Vérifie que les tailles réellement générées
    correspondent aux tailles attendues.
    """

    actual_sizes = SplitSizes(
        train=len(splits.X_train),
        validation=len(splits.X_validation),
        test=len(splits.X_test),
    )

    if actual_sizes != expected_sizes:
        raise RuntimeError(
            "Les tailles générées ne correspondent pas "
            f"aux tailles attendues. "
            f"Attendu : {expected_sizes}. "
            f"Obtenu : {actual_sizes}."
        )

    "aux tailles attendues. "
    f"Attendu : {expected_sizes}. "
    f"Obtenu : {actual_sizes}."

    if len(splits.X_train) != len(
        splits.y_train
    ):
        raise RuntimeError(
            "X_train et y_train ne sont pas alignés."
        )

    if len(splits.X_validation) != len(
        splits.y_validation
    ):
        raise RuntimeError(
            "X_validation et y_validation "
            "ne sont pas alignés."
        )

    if len(splits.X_test) != len(
        splits.y_test
    ):
        raise RuntimeError(
            "X_test et y_test ne sont pas alignés."
        )