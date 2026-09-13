"""Application configuration.

All settings can be overridden through environment variables or a `.env`
file located in the `backend/` directory.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

# Load backend/.env (if present) before pydantic reads the environment.
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # --- LLM provider -----------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.3
    request_timeout: float = 120.0

    # --- RAG pipeline -----------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    max_history: int = 10

    # --- Uploads ----------------------------------------------------------
    max_upload_mb: int = 25

    # --- Server -----------------------------------------------------------
    app_name: str = "Unfold"
    app_version: str = "3.0.0"
    cors_origins: str = "*"
    log_level: str = "INFO"

    # --- Storage ----------------------------------------------------------
    data_dir: Path = BASE_DIR / "storage"
    upload_dir: Path = BASE_DIR / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
