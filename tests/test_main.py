"""
Unit tests for main.py's CLI wiring: argument validation, --check-auth,
the --video-id-reuse vs --video-upload branching, and error handling.

None of this needs a live Azure account: VideoIndexerClient and
Settings.from_env are replaced with fakes, the same "no live account
needed" approach as the rest of this project's tests. This closes a real
gap -- despite its name, tests/test_detect_all_actions.py never actually
imports detect_all_actions.py, and main.py had no CLI-level tests at all,
so the argument validation and upload/reuse branching in `main()` (the
most complex, least-tested code in the project) previously ran completely
uncovered by the test suite.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from src.config import ConfigError, Settings
from src.video_indexer_client import (
    ProcessingFailedError,
    VideoIndexerError,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_insights_jumping.json"

FAKE_SETTINGS = Settings(
    subscription_id="sub-id",
    resource_group="rg",
    account_name="account",
    account_id="account-id",
    location="eastus",
)


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
        upload_video_error: Exception | None = None,
        uploaded_video_id: str = "uploaded-id",
    ):
        self.settings = settings
        self._access_token_error = access_token_error
        self._index = index if index is not None else {}
        self._wait_for_processing_error = wait_for_processing_error
        self._upload_video_error = upload_video_error
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
        if self._upload_video_error:
            raise self._upload_video_error
        self.upload_calls.append((video_path, name, indexing_preset))
        return self.uploaded_video_id

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        self.wait_for_processing_calls.append(video_id)
        if self._wait_for_processing_error:
            raise self._wait_for_processing_error
        return self._index


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a working Settings.from_env() by default -- tests
    that specifically want to exercise the ConfigError path override this."""
    monkeypatch.setattr(main.Settings, "from_env", staticmethod(lambda: FAKE_SETTINGS))


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> FakeVideoIndexerClient:
    fake = FakeVideoIndexerClient(FAKE_SETTINGS, **kwargs)
    monkeypatch.setattr(main, "VideoIndexerClient", lambda settings: fake)
    return fake


def _sample_index() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# _validate_args
# ----------------------------------------------------------------------


def test_validate_args_requires_video_or_video_id():
    args = main.build_arg_parser().parse_args([])
    assert main._validate_args(args) is not None


def test_validate_args_rejects_missing_video_file(tmp_path: Path):
    args = main.build_arg_parser().parse_args(
        ["--video", str(tmp_path / "missing.mp4")]
    )
    error = main._validate_args(args)
    assert error is not None
    assert "not found" in error


def test_validate_args_rejects_negative_min_overlap_seconds():
    args = main.build_arg_parser().parse_args(
        ["--video-id", "abc", "--min-overlap-seconds", "-1"]
    )
    error = main._validate_args(args)
    assert error is not None
    assert "min-overlap-seconds" in error


def test_validate_args_rejects_negative_merge_gap_seconds():
    args = main.build_arg_parser().parse_args(
        ["--video-id", "abc", "--merge-gap-seconds", "-0.5"]
    )
    error = main._validate_args(args)
    assert error is not None
    assert "merge-gap-seconds" in error


@pytest.mark.parametrize("bad_value", ["-0.1", "1.1"])
def test_validate_args_rejects_out_of_range_min_temporal_iou(bad_value: str):
    args = main.build_arg_parser().parse_args(
        ["--video-id", "abc", "--min-temporal-iou", bad_value]
    )
    error = main._validate_args(args)
    assert error is not None
    assert "min-temporal-iou" in error


def test_validate_args_accepts_video_id_alone():
    args = main.build_arg_parser().parse_args(["--video-id", "abc"])
    assert main._validate_args(args) is None


# ----------------------------------------------------------------------
# main() end to end
# ----------------------------------------------------------------------


def test_main_returns_2_for_invalid_args(tmp_path: Path):
    exit_code = main.main(["--video", str(tmp_path / "missing.mp4")])
    assert exit_code == 2


def test_main_returns_2_when_config_is_missing(monkeypatch: pytest.MonkeyPatch):
    def raise_config_error() -> Settings:
        raise ConfigError(
            "Missing required environment variable(s): AVI_SUBSCRIPTION_ID"
        )

    monkeypatch.setattr(main.Settings, "from_env", staticmethod(raise_config_error))

    exit_code = main.main(["--video-id", "abc"])
    assert exit_code == 2


def test_main_check_auth_success(monkeypatch: pytest.MonkeyPatch):
    _install_fake_client(monkeypatch)
    assert main.main(["--check-auth"]) == 0


def test_main_check_auth_failure(monkeypatch: pytest.MonkeyPatch):
    _install_fake_client(
        monkeypatch, access_token_error=VideoIndexerError("no permission")
    )
    assert main.main(["--check-auth"]) == 1


def test_main_reuses_video_id_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake = _install_fake_client(monkeypatch, index=_sample_index())

    exit_code = main.main(
        [
            "--video-id",
            "abc123",
            "--action",
            "jumping",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert fake.wait_for_processing_calls == ["abc123"]
    assert fake.upload_calls == []  # reused, never uploaded

    json_path = tmp_path / "abc123.report.json"
    html_path = tmp_path / "abc123.report.html"
    assert json_path.is_file()
    assert html_path.is_file()

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["occurrence_count"] == 3
    assert written["action"] == "jumping"


def test_main_uploads_fresh_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake = _install_fake_client(
        monkeypatch, index=_sample_index(), uploaded_video_id="new-id"
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not a real video, just needs to exist")

    exit_code = main.main(
        [
            "--video",
            str(video_path),
            "--action",
            "jumping",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert len(fake.upload_calls) == 1
    assert fake.upload_calls[0][0] == video_path
    assert fake.wait_for_processing_calls == ["new-id"]
    assert (tmp_path / "out" / "clip.report.json").is_file()


def test_main_returns_1_when_processing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(
        monkeypatch,
        wait_for_processing_error=ProcessingFailedError("indexing failed"),
    )

    exit_code = main.main(["--video-id", "abc123", "--out-dir", str(tmp_path)])

    assert exit_code == 1


def test_main_save_raw_insights_writes_raw_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(monkeypatch, index=_sample_index())

    exit_code = main.main(
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
    assert json.loads(raw_path.read_text(encoding="utf-8"))["id"] == "abc123"
