from asyncio import run
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_health_is_independent_of_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "invalid://unreachable")

    from src.main import create_app

    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_warmup_finishes_before_api_serves(monkeypatch):
    from src import main

    warmup = MagicMock()
    monkeypatch.setenv("AI_WARMUP_MODELS", "1")
    monkeypatch.setattr(main, "_warmup_models", warmup, raising=False)

    app = main.create_app()

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            warmup.assert_called_once_with()

    run(run_lifespan())
