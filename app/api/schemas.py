from typing import Literal

from pydantic import BaseModel, ConfigDict


ServiceStatus = Literal["healthy", "unhealthy"]


class HealthResponse(BaseModel):
    """
    Réponse de la route de santé.
    """

    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus
    service: str
    version: str
    environment: str


class RootResponse(BaseModel):
    """
    Informations générales sur l'application.
    """

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    environment: str
    documentation: str
    health: str