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

    # litellm-style "provider/model" string, e.g. "anthropic/claude-sonnet-5".
    # Switching providers is a config change, not a code change. Required,
    # not defaulted -- a missing LLM_MODEL should fail loudly at startup,
    # not silently fall back to an arbitrary unused model.
    llm_model: str
    llm_api_key: str = ""
    # Only meaningful for reasoning-capable models; ignored by others.
    # Unset leaves the model's own default reasoning behaviour in place.
    llm_reasoning_effort: str | None = None
    gitleaks_binary_path: str = "gitleaks"
    trufflehog_binary_path: str = "trufflehog3"


class Mode(StrEnum):
    SINGLE = "single"
    AGENTIC = "agentic"


class ScanConfig(BaseModel):
    """User-selectable scan behaviour, loaded from .secretscan.yml.
    Defaults to masked (raw must be opted in) and agentic (more thorough,
    worth the extra cost for a CI gate)."""

    mode: Mode = Mode.AGENTIC
    treatment: Treatment = Treatment.MASKED


def load_scan_config(path: Path = Path(".secretscan.yml")) -> ScanConfig:
    """Loads scan behaviour for the shipped CLI. `treatment` isn't
    user-configurable here -- always masked, regardless of what the file
    says; only Python callers can request otherwise."""
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
