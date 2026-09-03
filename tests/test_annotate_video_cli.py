"""
Unit tests for annotate_video.py's CLI wiring: argument validation, the
--report / --video-id / --video branching (including the "no client
needed at all" case when --report and --video are both given), frame-source
resolution, --only-composite, and error handling.

VideoIndexerClient, Settings.from_env, and render_annotated_video are all
replaced with fakes/spies here -- the same "no live account, no real video
file" approach as the rest of this project's tests. This closes a real
gap: annotate_video.py's main() previously had no CLI-level test coverage
at all (only _load_actions_from_report/_filter_only_composite were tested,
in tests/test_annotate_video.py).
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import annotate_video
from src.action_analyzer import DetectedAction
from src.config import ConfigError, Settings
from src.video_annotator import VideoAnnotationError
from src.video_indexer_client import ProcessingFailedError, VideoIndexerError

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
        index: dict[str, Any] | None = None,
        wait_for_processing_error: Exception | None = None,
        upload_video_error: Exception | None = None,
        download_video_error: Exception | None = None,
        uploaded_video_id: str = "uploaded-id",
    ):
        self.settings = settings
        self._index = index if index is not None else {}
        self._wait_for_processing_error = wait_for_processing_error
        self._upload_video_error = upload_video_error
        self._download_video_error = download_video_error
        self.uploaded_video_id = uploaded_video_id
        self.upload_calls: list[Any] = []
        self.wait_for_processing_calls: list[str] = []
        self.download_calls: list[tuple[str, Path]] = []

    def upload_video(self, video_path: Path, name: str | None = None) -> str:
        if self._upload_video_error:
            raise self._upload_video_error
        self.upload_calls.append(video_path)
        return self.uploaded_video_id

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        self.wait_for_processing_calls.append(video_id)
        if self._wait_for_processing_error:
            raise self._wait_for_processing_error
        return self._index

    def download_video(self, video_id: str, dest_path: Path) -> Path:
        self.download_calls.append((video_id, dest_path))
        if self._download_video_error:
            raise self._download_video_error
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake source video")
        return dest_path


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> FakeVideoIndexerClient:
    fake = FakeVideoIndexerClient(FAKE_SETTINGS, **kwargs)
    monkeypatch.setattr(annotate_video, "VideoIndexerClient", lambda settings: fake)
    monkeypatch.setattr(
        annotate_video.Settings, "from_env", staticmethod(lambda: FAKE_SETTINGS)
    )
    return fake


def _fail_if_settings_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert Settings.from_env is never called -- used for the --report +
    --video combination, which needs no Azure config or client at all."""

    def _unexpected() -> Settings:
        raise AssertionError("Settings.from_env should not be called here")

    monkeypatch.setattr(annotate_video.Settings, "from_env", staticmethod(_unexpected))


class FakeRenderer:
    """Spy standing in for render_annotated_video: records how it was
    called and returns a canned output path instead of touching cv2/ffmpeg."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Path:
        self.calls.append(kwargs)
        return kwargs["output_path"]


@pytest.fixture()
def fake_renderer(monkeypatch: pytest.MonkeyPatch) -> FakeRenderer:
    renderer = FakeRenderer()
    monkeypatch.setattr(annotate_video, "render_annotated_video", renderer)
    return renderer


def _write_all_actions_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "name": "person using phone",
                        "source": "derived",
                        "start_seconds": 3.0,
                        "end_seconds": 6.0,
                        "confidence": 0.8,
                    },
                    {
                        "name": "person",
                        "source": "labels",
                        "start_seconds": 0.0,
                        "end_seconds": 10.0,
                        "confidence": 1.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# _validate_args
# ----------------------------------------------------------------------


def test_validate_args_requires_video_or_video_id():
    args = annotate_video.build_arg_parser().parse_args([])
    assert annotate_video._validate_args(args) is not None


def test_validate_args_rejects_missing_video_file(tmp_path: Path):
    args = annotate_video.build_arg_parser().parse_args(
        ["--video", str(tmp_path / "missing.mp4")]
    )
    error = annotate_video._validate_args(args)
    assert error is not None and "not found" in error


@pytest.mark.parametrize("bad_value", ["-0.1", "1.5"])
def test_validate_args_rejects_out_of_range_min_confidence(bad_value: str):
    args = annotate_video.build_arg_parser().parse_args(
        ["--video-id", "abc", "--min-confidence", bad_value]
    )
    error = annotate_video._validate_args(args)
    assert error is not None and "min-confidence" in error


def test_validate_args_rejects_missing_report_file(tmp_path: Path):
    args = annotate_video.build_arg_parser().parse_args(
        ["--video-id", "abc", "--report", str(tmp_path / "missing.json")]
    )
    error = annotate_video._validate_args(args)
    assert error is not None and "Report file not found" in error


def test_validate_args_accepts_video_id_alone():
    args = annotate_video.build_arg_parser().parse_args(["--video-id", "abc"])
    assert annotate_video._validate_args(args) is None


# ----------------------------------------------------------------------
# _build_client_if_needed
# ----------------------------------------------------------------------


def test_build_client_not_needed_when_report_and_video_both_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fail_if_settings_needed(monkeypatch)
    report_path = tmp_path / "video.all_actions.json"
    _write_all_actions_report(report_path)
    args = annotate_video.build_arg_parser().parse_args(
        ["--video", str(tmp_path / "clip.mp4"), "--report", str(report_path)]
    )

    assert annotate_video._build_client_if_needed(args) is None


def test_build_client_needed_when_video_id_alone(monkeypatch: pytest.MonkeyPatch):
    _install_fake_client(monkeypatch)
    args = annotate_video.build_arg_parser().parse_args(["--video-id", "abc"])

    client = annotate_video._build_client_if_needed(args)

    assert isinstance(client, FakeVideoIndexerClient)


# ----------------------------------------------------------------------
# _resolve_frame_source
# ----------------------------------------------------------------------


def test_resolve_frame_source_prefers_local_video(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    args = annotate_video.build_arg_parser().parse_args(["--video", str(video_path)])

    assert annotate_video._resolve_frame_source(None, args) == video_path


def test_resolve_frame_source_downloads_when_only_video_id_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = _install_fake_client(monkeypatch)
    args = annotate_video.build_arg_parser().parse_args(
        ["--video-id", "abc123", "--out-dir", str(tmp_path)]
    )

    frame_source = annotate_video._resolve_frame_source(fake, args)

    assert frame_source == tmp_path / "abc123.source.mp4"
    assert fake.download_calls == [("abc123", tmp_path / "abc123.source.mp4")]


# ----------------------------------------------------------------------
# main() end to end
# ----------------------------------------------------------------------


def test_main_returns_2_for_invalid_args(fake_renderer: FakeRenderer):
    assert annotate_video.main([]) == 2
    assert fake_renderer.calls == []


def test_main_returns_2_when_config_is_missing(
    monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    def raise_config_error() -> Settings:
        raise ConfigError("Missing required environment variable(s): AVI_ACCOUNT_ID")

    monkeypatch.setattr(
        annotate_video.Settings, "from_env", staticmethod(raise_config_error)
    )

    assert annotate_video.main(["--video-id", "abc"]) == 2
    assert fake_renderer.calls == []


def test_main_report_plus_video_needs_no_azure_calls_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    _fail_if_settings_needed(monkeypatch)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    report_path = tmp_path / "clip.all_actions.json"
    _write_all_actions_report(report_path)

    exit_code = annotate_video.main(
        [
            "--video",
            str(video_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(fake_renderer.calls) == 1
    call = fake_renderer.calls[0]
    assert call["video_path"] == video_path
    assert {a.name for a in call["actions"]} == {"person using phone", "person"}
    assert call["output_path"] == tmp_path / "clip.annotated.mp4"


def test_main_video_id_alone_downloads_and_annotates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    index = {
        "videos": [
            {
                "insights": {
                    "labels": [
                        {
                            "name": "person",
                            "instances": [
                                {
                                    "confidence": 1.0,
                                    "start": "0:00:00",
                                    "end": "0:00:05",
                                }
                            ],
                        }
                    ]
                }
            }
        ]
    }
    fake = _install_fake_client(monkeypatch, index=index)

    exit_code = annotate_video.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert fake.wait_for_processing_calls == ["abc123"]
    assert fake.download_calls == [("abc123", tmp_path / "abc123.source.mp4")]
    call = fake_renderer.calls[0]
    assert call["video_path"] == tmp_path / "abc123.source.mp4"
    assert call["output_path"] == tmp_path / "abc123.annotated.mp4"


def test_main_only_composite_filters_actions_and_suffixes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    report_path = tmp_path / "clip.all_actions.json"
    _write_all_actions_report(report_path)

    exit_code = annotate_video.main(
        [
            "--video",
            str(video_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path),
            "--only-composite",
        ]
    )

    assert exit_code == 0
    call = fake_renderer.calls[0]
    assert {a.name for a in call["actions"]} == {"person using phone"}
    assert call["output_path"] == tmp_path / "clip.composite_actions.annotated.mp4"


def test_main_only_composite_with_no_derived_actions_warns_but_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    report_path = tmp_path / "clip.all_actions.json"
    report_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "name": "person",
                        "source": "labels",
                        "start_seconds": 0.0,
                        "end_seconds": 10.0,
                        "confidence": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = annotate_video.main(
        [
            "--video",
            str(video_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path),
            "--only-composite",
        ]
    )

    assert exit_code == 0
    assert fake_renderer.calls[0]["actions"] == []


def test_main_returns_1_when_acquiring_actions_fails(
    monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    _install_fake_client(
        monkeypatch,
        wait_for_processing_error=ProcessingFailedError("indexing failed"),
    )

    exit_code = annotate_video.main(["--video-id", "abc123"])

    assert exit_code == 1
    assert fake_renderer.calls == []


def test_main_returns_1_when_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    _install_fake_client(
        monkeypatch,
        index={"videos": [{"insights": {}}]},
        download_video_error=VideoIndexerError("download failed"),
    )

    exit_code = annotate_video.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path)]
    )

    assert exit_code == 1
    assert fake_renderer.calls == []


def test_main_returns_1_when_render_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    report_path = tmp_path / "clip.all_actions.json"
    _write_all_actions_report(report_path)

    def raise_render_error(**kwargs: Any) -> Path:
        raise VideoAnnotationError("opencv-python is required")

    monkeypatch.setattr(annotate_video, "render_annotated_video", raise_render_error)

    exit_code = annotate_video.main(
        [
            "--video",
            str(video_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1


def test_main_uploads_when_only_video_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    fake = _install_fake_client(
        monkeypatch,
        index={"videos": [{"insights": {}}]},
        uploaded_video_id="new-id",
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")

    exit_code = annotate_video.main(
        ["--video", str(video_path), "--out-dir", str(tmp_path)]
    )

    assert exit_code == 0
    assert fake.upload_calls == [video_path]
    assert fake.wait_for_processing_calls == ["new-id"]
    # A local --video means no download is needed for frames either.
    assert fake.download_calls == []
    assert fake_renderer.calls[0]["video_path"] == video_path


def test_main_passes_through_render_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_renderer: FakeRenderer
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"local file")
    report_path = tmp_path / "clip.all_actions.json"
    _write_all_actions_report(report_path)

    exit_code = annotate_video.main(
        [
            "--video",
            str(video_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path),
            "--min-confidence",
            "0.75",
            "--include-unscored",
            "--max-lines",
            "3",
            "--no-audio",
        ]
    )

    assert exit_code == 0
    call = fake_renderer.calls[0]
    assert call["min_confidence"] == pytest.approx(0.75)
    assert call["include_unscored"] is True
    assert call["max_lines"] == 3
    assert call["keep_audio"] is False


def test_load_actions_from_report_returns_detected_actions(tmp_path: Path):
    report_path = tmp_path / "clip.all_actions.json"
    _write_all_actions_report(report_path)

    actions = annotate_video._load_actions_from_report(report_path)

    assert all(isinstance(a, DetectedAction) for a in actions)
    assert {a.name for a in actions} == {"person using phone", "person"}
