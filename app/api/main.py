from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    Gère le cycle de vie de l'application.

    Au démarrage, cette fonction peut notamment :

    - créer les dossiers nécessaires ;
    - initialiser le Model Registry ;
    - créer laitoring ;
    - charger le modèle de production.

    À l'arrêt, cette fonction pourra ultérieurement :

    - fermer les connexions à la base de données ;
    - libérer les ressources utilisées ;
    - enregistrer les dernières métriques.
    """

    del application

    settings.logs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    yield


def create_application() -> FastAPI:
    """
    Construit et configure l'application FastAPI.

    Returns:
        L'instance configurée de l'application FastAPI.
    """

    application = FastAPI(
        title="AI Model Operations Center API",
        description=(
            "Plateforme MLOps permettant d'entraîner, comparer, "
            "versionner, déployer et superviser des modèles "
            "de machine learning."
        ),
        version=settings.application_version,
        lifespan=lifespan,
    )

    application.include_router(router)

    return application


app = create_application()