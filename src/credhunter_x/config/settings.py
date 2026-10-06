from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from credhunter_x.models.treatment import Treatment

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Secrets and paths, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # litellm "provider/model" string, e.g. "anthropic/claude-sonnet-5".
    # Required, so a missing value fails at startup.
    llm_model: str
    llm_api_key: str = ""
    # Only used by reasoning models; leaving it unset keeps the model's default.
    llm_reasoning_effort: str | None = None
    gitleaks_binary_path: str = "gitleaks"
    trufflehog_binary_path: str = "trufflehog3"


class Mode(StrEnum):
    SINGLE = "single"
    AGENTIC = "agentic"


class ScanConfig(BaseModel):
    """Scan settings from .secretscan.yml. Defaults to masked (raw has to be
    asked for) and agentic (more thorough, worth the cost in CI)."""

    mode: Mode = Mode.AGENTIC
    treatment: Treatment = Treatment.MASKED


def load_scan_config(path: Path = Path(".secretscan.yml")) -> ScanConfig:
    """Loads scan settings for the CLI. The treatment is always masked here,
    whatever the file says; only Python callers can choose another."""
    if not path.exists():
        return ScanConfig()
    data = yaml.safe_load(path.read_text()) or {}
    if "treatment" in data:
        logger.warning(
            "treatment is not configurable via .secretscan.yml and will be ignored; "
            "the shipped CLI always uses masked treatment"
        )
        data.pop("treatment")
    return ScanConfig(**data)
