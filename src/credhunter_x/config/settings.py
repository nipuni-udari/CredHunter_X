from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from credhunter_x.models.treatment import Treatment


class Settings(BaseSettings):
    """Secrets and paths, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gitleaks_binary_path: str = "gitleaks"


class Mode(StrEnum):
    SINGLE = "single"
    AGENTIC = "agentic"


class ScanConfig(BaseModel):
    """User-selectable scan behaviour, loaded from .secretscan.yml.

    Defaults to the privacy-safe choice — raw must be opted into explicitly
    and is never the shipped default.
    """

    mode: Mode = Mode.SINGLE
    treatment: Treatment = Treatment.MASKED


def load_scan_config(path: Path = Path(".secretscan.yml")) -> ScanConfig:
    if not path.exists():
        return ScanConfig()
    data = yaml.safe_load(path.read_text()) or {}
    return ScanConfig(**data)
