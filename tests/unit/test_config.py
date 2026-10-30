import pytest

from nexora_ai.settings import Settings


def test_worker_requires_redis_and_express_without_leaking_credentials():
    settings = Settings(redis_url=None, express_base_url=None, database_url=None)
    with pytest.raises(ValueError) as error:
        settings.require_worker()
    assert "REDIS_URL" in str(error.value)
    assert "EXPRESS_BASE_URL" in str(error.value)


def test_secret_url_is_masked():
    settings = Settings(redis_url="redis://user:secret@localhost:6379")
    assert "secret" not in repr(settings)
    assert settings.redis_url.get_secret_value().endswith(":6379")
