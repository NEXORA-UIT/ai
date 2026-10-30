"""Configuration shared by the HTTP and worker processes."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: SecretStr | None = None
    express_base_url: str | None = None
    database_url: SecretStr | None = None

    def require_worker(self) -> tuple[str, str]:
        missing = [
            name
            for name, value in (
                ("REDIS_URL", self.redis_url),
                ("EXPRESS_BASE_URL", self.express_base_url),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Missing worker configuration: " + ", ".join(missing))
        return self.redis_url.get_secret_value(), self.express_base_url

    def require_database(self) -> str:
        if self.database_url is None:
            raise ValueError("Missing database configuration: DATABASE_URL")
        return self.database_url.get_secret_value()
