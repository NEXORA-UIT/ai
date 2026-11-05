"""HTTP process entrypoint."""

from fastapi import FastAPI

from src.api.documents import router as documents_router


def create_app() -> FastAPI:
    app = FastAPI(title="Nexora AI Service", version="0.1.0")
    app.include_router(documents_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
