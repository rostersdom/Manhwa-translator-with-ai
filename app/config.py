import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Manga Translator API"
    debug: bool = False

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/translations.db")

    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    output_dir: str = os.getenv("OUTPUT_DIR", "./data/outputs")

    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    max_pages_per_job: int = int(os.getenv("MAX_PAGES_PER_JOB", "100"))

    translation_cache_size: int = 512
    lama_tile_size: int = 1024

    api_keys: list[str] = []
    api_key_enabled: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.api_keys:
            self.api_key_enabled = True


settings = Settings()
