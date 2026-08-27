"""
Configuration loading for the Azure Action Monitoring prototype.

All settings are read from environment variables (optionally loaded from a
.env file) so no secrets ever need to be hard-coded or committed. See
.env.example for the full list of variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can also be set directly.
    pass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    subscription_id: str
    resource_group: str
    account_name: str
    account_id: str
    location: str

    # Polling behaviour
    poll_interval_seconds: float = 10.0
    processing_timeout_seconds: float = 1800.0  # 30 minutes

    @staticmethod
    def from_env() -> Settings:
        required = {
            "AVI_SUBSCRIPTION_ID": "subscription_id",
            "AVI_RESOURCE_GROUP": "resource_group",
            "AVI_ACCOUNT_NAME": "account_name",
            "AVI_ACCOUNT_ID": "account_id",
            "AVI_LOCATION": "location",
        }

        values: dict[str, str] = {}
        missing: list[str] = []
        for env_var, field_name in required.items():
            value = os.environ.get(env_var, "").strip()
            if not value:
                missing.append(env_var)
            values[field_name] = value

        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill in your Azure AI "
                "Video Indexer account details, or export them in your shell."
            )

        poll_interval = float(os.environ.get("AVI_POLL_INTERVAL_SECONDS", str(10.0)))
        timeout = float(os.environ.get("AVI_PROCESSING_TIMEOUT_SECONDS", str(1800.0)))

        return Settings(
            subscription_id=values["subscription_id"],
            resource_group=values["resource_group"],
            account_name=values["account_name"],
            account_id=values["account_id"],
            location=values["location"],
            poll_interval_seconds=poll_interval,
            processing_timeout_seconds=timeout,
        )
