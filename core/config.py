from __future__ import annotations

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    app_name: str = "AshAI"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    hf_token: str = os.getenv("HF_TOKEN", "")
    default_model: str = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    class Config:
        extra = "allow"

_config: Config | None = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config

