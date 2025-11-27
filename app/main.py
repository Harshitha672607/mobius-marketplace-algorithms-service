from fastapi import FastAPI
from app.api.v1.routes.valuation_route import router as valuation_router
from app.core.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )

    app.include_router(valuation_router)
    return app

app = create_app()
