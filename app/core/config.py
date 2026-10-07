from functools import lru_cache
from pathlib import Path

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    """
    Retourne le chemin absolu de la racine du projet.

    Le fichier courant se trouve dans :
        app/core/config.py

    parents[0] = app/core
    parents[1] = app
    parents[2] = racine du projet
    """
    return Path(__file__).resolve().parents[2]


class ApplicationSettings(BaseSettings):
    """
    Configuration générale de l'application.

    Les valeurs peuvent être surchargées avec des variables
    d'environnement ou avec un fichier .env local.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    application_name: str = Field(
        default="ai-model-operations-center",
        validation_alias="APP_NAME",
        min_length=1,
    )

    application_version: str = Field(
        default="0.1.0",
        validation_alias="APP_VERSION",
        min_length=1,
    )

    environment: str = Field(
        default="development",
        validation_alias="APP_ENV",
        min_length=1,
    )

    host: str = Field(
        default="127.0.0.1",
        validation_alias="APP_HOST",
        min_length=1,
    )

    port: int = Field(
        default=8002,
        validation_alias="APP_PORT",
        ge=1,
        le=65535,
    )

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        min_length=1,
    )

    project_root: Path = Field(
        default_factory=get_project_root,
    )

    @property
    def data_directory(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_directory(self) -> Path:
        return self.data_directory / "raw"

    @property
    def processed_data_directory(self) -> Path:
        return self.data_directory / "processed"

    @property
    def reference_data_directory(self) -> Path:
        return self.data_directory / "reference"

    @property
    def models_directory(self) -> Path:
        return self.project_root / "models"

    @property
    def registry_directory(self) -> Path:
        return self.models_directory / "registry"

    @property
    def production_models_directory(self) -> Path:
        return self.models_directory / "production"

    @property
    def reports_directory(self) -> Path:
        return self.project_root / "reports"

    @property
    def logs_directory(self) -> Path:
        return self.project_root / "logs"


@lru_cache
def get_settings() -> ApplicationSettings:
    """
    Retourne une instance mise en cache de la configuration.
    """

    return ApplicationSettings()


settings = get_settings()