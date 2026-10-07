from fastapi import APIRouter

from app.api.schemas import HealthResponse, RootResponse
from app.core.config import settings


router = APIRouter()


@router.get(
    "/",
    response_model=RootResponse,
    tags=["Root"],
    summary="Informations générales sur l'application",
)
def root() -> RootResponse:
    """
    Retourne les informations générales sur le service.
    """

    return RootResponse(
        service=settings.application_name,
        version=settings.application_version,
        environment=settings.environment,
        documentation="/docs",
        health="/health",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Vérifier la santé du service",
)
def health() -> HealthResponse:
    """
    Vérifie que l'application est correctement démarrée.

    Plus tard, cette route vérifiera également le registre,
    la base de monitoring et le modèle de production.
    """

    return HealthResponse(
        status="healthy",
        service=settings.application_name,
        version=settings.application_version,
        environment=settings.environment,
    )