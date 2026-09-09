"""
Unit tests for explore_insights.py's CLI wiring: argument validation,
--check-auth, the --video-id-reuse vs --video-upload branching, and error
handling -- the same "no live account needed" fake-client approach as the
other CLI test files in this project.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import explore_insights
from src.config import ConfigError, Settings
from src.video_indexer_client import ProcessingFailedError, VideoIndexerError

FAKE_SETTINGS = Settings(
    subscription_id="sub-id",
    resource_group="rg",
    account_name="account",
    account_id="account-id",
    location="eastus",
)

SAMPLE_INDEX: dict[str, Any] = {
    "videos": [
        {
            "insights": {
                "duration": "0:00:30.0",
                "sourceLanguage": "en-US",
                "faces": [
                    {
                        "name": "Jane Doe",
                        "confidence": 0.9,
                        "instances": [{"start": "0:00:01.0", "end": "0:00:02.0"}],
                    }
                ],
            }
        }
    ]
}


class FakeVideoIndexerClient:
    """Stands in for VideoIndexerClient: canned results/errors per method,
    plus a record of how each was called so tests can assert on it."""

    def __init__(
        self,
        settings: Settings,
        *,
        access_token_error: Exception | None = None,
        index: dict[str, Any] | None = None,
        wait_for_processing_error: Exception | None = None,
        uploaded_video_id: str = "uploaded-id",
    ):
        self.settings = settings
        self._access_token_error = access_token_error
        self._index = index if index is not None else {}
        self._wait_for_processing_error = wait_for_processing_error
        self.uploaded_video_id = uploaded_video_id
        self.upload_calls: list[tuple[Any, str | None, str]] = []
        self.wait_for_processing_calls: list[str] = []

    def get_access_token(self) -> str:
        if self._access_token_error:
            raise self._access_token_error
        return "fake-token"

    def upload_video(
        self,
        video_path: Path,
        name: str | None = None,
        indexing_preset: str = "Default",
    ) -> str:
        self.upload_calls.append((video_path, name, indexing_preset))
        return self.uploaded_video_id

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        self.wait_for_processing_calls.append(video_id)
        if self._wait_for_processing_error:
            raise self._wait_for_processing_error
        return self._index


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        explore_insights.Settings, "from_env", staticmethod(lambda: FAKE_SETTINGS)
    )


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> FakeVideoIndexerClient:
    fake = FakeVideoIndexerClient(FAKE_SETTINGS, **kwargs)
    monkeypatch.setattr(explore_insights, "VideoIndexerClient", lambda settings: fake)
    return fake


# ----------------------------------------------------------------------
# _validate_args
# ----------------------------------------------------------------------


def test_validate_args_requires_video_or_video_id():
    args = explore_insights.build_arg_parser().parse_args([])
    assert explore_insights._validate_args(args) is not None


def test_validate_args_rejects_missing_video_file(tmp_path: Path):
    args = explore_insights.build_arg_parser().parse_args(
        ["--video", str(tmp_path / "missing.mp4")]
    )
    error = explore_insights._validate_args(args)
    assert error is not None and "not found" in error


def test_validate_args_accepts_check_auth_alone():
    args = explore_insights.build_arg_parser().parse_args(["--check-auth"])
    assert explore_insights._validate_args(args) is None


def test_validate_args_accepts_video_id_alone():
    args = explore_insights.build_arg_parser().parse_args(["--video-id", "abc"])
    assert explore_insights._validate_args(args) is None


def test_default_indexing_preset_is_advanced():
    args = explore_insights.build_arg_parser().parse_args(["--video-id", "abc"])
    assert args.indexing_preset == "Advanced"


# ----------------------------------------------------------------------
# main() end to end
# ----------------------------------------------------------------------


def test_main_returns_2_for_invalid_args(tmp_path: Path):
    exit_code = explore_insights.main(["--video", str(tmp_path / "missing.mp4")])
    assert exit_code == 2


def test_main_returns_2_when_config_is_missing(monkeypatch: pytest.MonkeyPatch):
    def raise_config_error() -> Settings:
        raise ConfigError(
            "Missing required environment variable(s): AVI_SUBSCRIPTION_ID"
        )

    monkeypatch.setattr(
        explore_insights.Settings, "from_env", staticmethod(raise_config_error)
    )

    exit_code = explore_insights.main(["--video-id", "abc"])
    assert exit_code == 2


def test_main_check_auth_success(monkeypatch: pytest.MonkeyPatch):
    _install_fake_client(monkeypatch)
    assert explore_insights.main(["--check-auth"]) == 0


def test_main_check_auth_failure(monkeypatch: pytest.MonkeyPatch):
    _install_fake_client(
        monkeypatch, access_token_error=VideoIndexerError("no permission")
    )
    assert explore_insights.main(["--check-auth"]) == 1


def test_main_reuses_video_id_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _install_fake_client(monkeypatch, index=SAMPLE_INDEX)

    exit_code = explore_insights.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert fake.wait_for_processing_calls == ["abc123"]
    assert fake.upload_calls == []  # reused, never uploaded

    json_path = tmp_path / "abc123.capabilities.json"
    html_path = tmp_path / "abc123.capabilities.html"
    assert json_path.is_file()
    assert html_path.is_file()

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["coverage"]["faces"] == 1


def test_main_uploads_fresh_video_with_advanced_preset_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _install_fake_client(
        monkeypatch, index=SAMPLE_INDEX, uploaded_video_id="new-id"
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not a real video, just needs to exist")

    exit_code = explore_insights.main(
        ["--video", str(video_path), "--out-dir", str(tmp_path / "out")]
    )

    assert exit_code == 0
    assert fake.upload_calls == [(video_path, None, "Advanced")]
    assert fake.wait_for_processing_calls == ["new-id"]
    assert (tmp_path / "out" / "clip.capabilities.json").is_file()


def test_main_returns_1_when_processing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(
        monkeypatch,
        wait_for_processing_error=ProcessingFailedError("indexing failed"),
    )

    exit_code = explore_insights.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 1


def test_main_save_raw_insights_writes_raw_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(monkeypatch, index=SAMPLE_INDEX)

    exit_code = explore_insights.main(
        [
            "--video-id",
            "abc123",
            "--out-dir",
            str(tmp_path),
            "--save-raw-insights",
        ]
    )

    assert exit_code == 0
    raw_path = tmp_path / "abc123.raw_insights.json"
    assert raw_path.is_file()
