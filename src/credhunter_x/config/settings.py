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

    # litellm-style "provider/model" string (e.g. "openai/gpt-5.2-mini",
    # "anthropic/claude-sonnet-5") plus the one API key for whichever
    # provider that selects. Switching providers is a config change only —
    # llm/litellm_client.py is the single adapter for every provider.
    llm_model: str = "gemini/gemini-flash-latest"
    llm_api_key: str = ""
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
    """Loads user-selectable behaviour for the shipped CLI. `treatment` is
    deliberately not one of those user-selectable things — the research
    scope document's own .secretscan.yml example only ever shows `mode`,
    and raw treatment is explicitly "never used in the shipped tool." Any
    `treatment` key in the file is ignored so the CLI path always resolves
    to ScanConfig's default (masked) regardless of what a repo's config
    file says — only Python callers (e.g. the research evaluation harness)
    can request a different treatment, by constructing ScanConfig directly.
    """
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
