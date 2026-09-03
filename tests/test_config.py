"""
Unit tests for src/config.py's Settings.from_env(): the required-variable
validation, the optional numeric overrides (with their defaults), and the
ConfigError hardening around malformed/non-positive numeric env vars.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, Settings

REQUIRED_VARS = {
    "AVI_SUBSCRIPTION_ID": "sub-id",
    "AVI_RESOURCE_GROUP": "rg",
    "AVI_ACCOUNT_NAME": "account",
    "AVI_ACCOUNT_ID": "account-id",
    "AVI_LOCATION": "eastus",
}

ALL_ENV_VARS = (
    *REQUIRED_VARS,
    "AVI_POLL_INTERVAL_SECONDS",
    "AVI_PROCESSING_TIMEOUT_SECONDS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with none of this module's env vars set, so tests
    can't leak into each other or pick up the real shell's environment."""
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in REQUIRED_VARS.items():
        monkeypatch.setenv(var, value)


# ----------------------------------------------------------------------
# Required variables
# ----------------------------------------------------------------------


def test_from_env_succeeds_with_all_required_vars_set(monkeypatch: pytest.MonkeyPatch):
    _set_required(monkeypatch)

    settings = Settings.from_env()

    assert settings.subscription_id == "sub-id"
    assert settings.resource_group == "rg"
    assert settings.account_name == "account"
    assert settings.account_id == "account-id"
    assert settings.location == "eastus"


@pytest.mark.parametrize("missing_var", list(REQUIRED_VARS))
def test_from_env_raises_when_one_required_var_is_missing(
    monkeypatch: pytest.MonkeyPatch, missing_var: str
):
    _set_required(monkeypatch)
    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env()

    assert missing_var in str(exc_info.value)


def test_from_env_raises_when_required_var_is_blank(monkeypatch: pytest.MonkeyPatch):
    """A var that's set but empty/whitespace-only is treated the same as
    missing -- it's a common way for a shell/CI config to leave a variable
    "set" but effectively unconfigured."""
    _set_required(monkeypatch)
    monkeypatch.setenv("AVI_LOCATION", "   ")

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env()

    assert "AVI_LOCATION" in str(exc_info.value)


def test_from_env_lists_all_missing_vars_at_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AVI_SUBSCRIPTION_ID", "sub-id")
    # Everything else left unset.

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env()

    message = str(exc_info.value)
    assert "AVI_RESOURCE_GROUP" in message
    assert "AVI_ACCOUNT_NAME" in message
    assert "AVI_ACCOUNT_ID" in message
    assert "AVI_LOCATION" in message
    assert "AVI_SUBSCRIPTION_ID" not in message


# ----------------------------------------------------------------------
# Numeric overrides: defaults
# ----------------------------------------------------------------------


def test_from_env_uses_default_poll_interval_and_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_required(monkeypatch)

    settings = Settings.from_env()

    assert settings.poll_interval_seconds == 10.0
    assert settings.processing_timeout_seconds == 1800.0


def test_from_env_honours_valid_numeric_overrides(monkeypatch: pytest.MonkeyPatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("AVI_POLL_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("AVI_PROCESSING_TIMEOUT_SECONDS", "60")

    settings = Settings.from_env()

    assert settings.poll_interval_seconds == 5.0
    assert settings.processing_timeout_seconds == 60.0


# ----------------------------------------------------------------------
# Numeric overrides: hardened parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var", ["AVI_POLL_INTERVAL_SECONDS", "AVI_PROCESSING_TIMEOUT_SECONDS"]
)
def test_from_env_raises_config_error_for_malformed_numeric_value(
    monkeypatch: pytest.MonkeyPatch, env_var: str
):
    """A malformed numeric env var must surface as a ConfigError, not a raw
    ValueError -- callers of from_env() only expect to handle ConfigError."""
    _set_required(monkeypatch)
    monkeypatch.setenv(env_var, "not-a-number")

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env()

    assert env_var in str(exc_info.value)


@pytest.mark.parametrize(
    "env_var", ["AVI_POLL_INTERVAL_SECONDS", "AVI_PROCESSING_TIMEOUT_SECONDS"]
)
@pytest.mark.parametrize("bad_value", ["0", "-1", "-0.001"])
def test_from_env_raises_config_error_for_non_positive_numeric_value(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
):
    _set_required(monkeypatch)
    monkeypatch.setenv(env_var, bad_value)

    with pytest.raises(ConfigError) as exc_info:
        Settings.from_env()

    assert env_var in str(exc_info.value)


def test_from_env_treats_blank_numeric_override_as_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    """A blank override (e.g. an empty .env line) falls back to the
    default rather than failing float() parsing on an empty string."""
    _set_required(monkeypatch)
    monkeypatch.setenv("AVI_POLL_INTERVAL_SECONDS", "   ")

    settings = Settings.from_env()

    assert settings.poll_interval_seconds == 10.0


def test_settings_can_still_be_constructed_directly_with_zero_values():
    """Settings itself places no constraint on these fields -- only
    from_env()'s parsing does -- so direct construction (as tests elsewhere
    use to exercise a zero-timeout edge case) is unaffected."""
    settings = Settings(
        subscription_id="sub-id",
        resource_group="rg",
        account_name="account",
        account_id="account-id",
        location="eastus",
        poll_interval_seconds=0.0,
        processing_timeout_seconds=0.0,
    )

    assert settings.poll_interval_seconds == 0.0
    assert settings.processing_timeout_seconds == 0.0
