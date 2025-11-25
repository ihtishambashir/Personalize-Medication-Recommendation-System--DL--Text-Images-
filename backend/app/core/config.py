from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PMRS Demo API"
    api_prefix: str = "/api"
    cors_origins: list[str] = ["*"]  # For demo only; restrict in production.

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
