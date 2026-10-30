from fastapi.testclient import TestClient


def test_health_is_independent_of_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "invalid://unreachable")

    from src.main import create_app

    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
