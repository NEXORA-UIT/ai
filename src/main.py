"""HTTP process entrypoint."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Nexora AI Service", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
