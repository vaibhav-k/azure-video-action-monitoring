"""
Unit tests for analyze_video.py's CLI wiring: argument validation, the
--metadata / --video-id branching (an explicit --metadata file always
wins; --video-id prefers a cached <out-dir>/<video-id>.raw_insights.json,
falling back to a fresh Video Indexer fetch which then gets cached), the
insights -> analyze_all -> summarize_insights -> stdout pipeline, and error
handling. No real Video Indexer calls are made -- VideoIndexerClient is
monkeypatched to a fake, the same "no live account needed" approach the
rest of this project's CLI tests use.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_video
from src.config import Settings
from src.video_indexer_client import VideoIndexerError

FAKE_SETTINGS = Settings(
    subscription_id="sub-id",
    resource_group="rg",
    account_name="account",
    account_id="account-id",
    location="eastus",
)


def _insights_payload(*, kiss: bool = True) -> dict[str, Any]:
    labels = [
        {
            "name": "outdoor",
            "instances": [{"start": "0:00:00", "end": "0:00:30", "confidence": 0.99}],
        }
    ]
    if kiss:
        labels.append(
            {
                "name": "kiss",
                "instances": [
                    {"start": "0:00:09.8", "end": "0:00:10.6", "confidence": 0.97}
                ],
            }
        )
    return {
        "durationInSeconds": 30.0,
        "videos": [{"insights": {"duration": "30.0", "labels": labels}}],
    }


class FakeVideoIndexerClient:
    """Stands in for VideoIndexerClient: a canned wait_for_processing()
    result/error, plus a record of how it was called."""

    def __init__(
        self,
        settings: Settings,
        *,
        index: dict[str, Any] | None = None,
        wait_for_processing_error: Exception | None = None,
    ):
        self.settings = settings
        self._index = index if index is not None else _insights_payload()
        self._wait_for_processing_error = wait_for_processing_error
        self.wait_for_processing_calls: list[str] = []

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        self.wait_for_processing_calls.append(video_id)
        if self._wait_for_processing_error:
            raise self._wait_for_processing_error
        return self._index


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> FakeVideoIndexerClient:
    fake = FakeVideoIndexerClient(FAKE_SETTINGS, **kwargs)
    monkeypatch.setattr(analyze_video, "VideoIndexerClient", lambda settings: fake)
    monkeypatch.setattr(
        analyze_video.Settings, "from_env", staticmethod(lambda: FAKE_SETTINGS)
    )
    return fake


def _fail_if_settings_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert Settings.from_env is never called -- used whenever a cached
    file or explicit --metadata should make a Video Indexer account
    entirely unnecessary."""

    def _unexpected() -> Settings:
        raise AssertionError("Settings.from_env should not be called here")

    monkeypatch.setattr(analyze_video.Settings, "from_env", staticmethod(_unexpected))


@pytest.fixture
def metadata_file(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_insights_payload()), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------


def test_main_returns_2_when_neither_metadata_nor_video_id_given():
    assert analyze_video.main([]) == 2


def test_main_returns_2_for_missing_metadata_file(tmp_path: Path):
    assert analyze_video.main(["--metadata", str(tmp_path / "nope.json")]) == 2


def test_main_returns_2_for_out_of_range_min_confidence(metadata_file: Path):
    assert (
        analyze_video.main(
            ["--metadata", str(metadata_file), "--min-confidence", "1.5"]
        )
        == 2
    )
    assert (
        analyze_video.main(
            ["--metadata", str(metadata_file), "--min-confidence", "-0.1"]
        )
        == 2
    )


def test_main_returns_2_for_malformed_metadata_json(tmp_path: Path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")

    result = analyze_video.main(["--metadata", str(bad_path)])

    assert result == 2


# ----------------------------------------------------------------------
# --metadata: no Azure account needed at all
# ----------------------------------------------------------------------


def test_main_metadata_alone_needs_no_video_indexer_settings(
    monkeypatch: pytest.MonkeyPatch, metadata_file: Path
):
    _fail_if_settings_needed(monkeypatch)

    result = analyze_video.main(["--metadata", str(metadata_file)])

    assert result == 0


def test_main_happy_path_prints_scene_understanding(
    monkeypatch: pytest.MonkeyPatch, metadata_file: Path, capsys: pytest.CaptureFixture
):
    _fail_if_settings_needed(monkeypatch)

    result = analyze_video.main(["--metadata", str(metadata_file)])

    assert result == 0
    out = capsys.readouterr().out
    assert out.startswith("Scene understanding:\n\n")
    assert "kiss" in out
    assert (
        "\n" not in out.split("\n\n", 1)[1].strip()
    )  # the paragraph itself is one line


def test_main_passes_min_confidence_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
):
    """A high --min-confidence drops the weaker 'kiss' label's absence
    doesn't apply here (kiss is 0.97) -- instead verify a low-confidence
    label gets filtered out while outdoor (0.99) survives."""
    payload = _insights_payload(kiss=False)
    payload["videos"][0]["insights"]["labels"].append(
        {
            "name": "weak-thing",
            "instances": [{"start": "0:00:01", "end": "0:00:02", "confidence": 0.2}],
        }
    )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _fail_if_settings_needed(monkeypatch)

    result = analyze_video.main(["--metadata", str(path), "--min-confidence", "0.5"])

    assert result == 0
    out = capsys.readouterr().out
    assert "weak-thing" not in out


def test_main_writes_output_file_when_given(
    monkeypatch: pytest.MonkeyPatch, metadata_file: Path, tmp_path: Path
):
    _fail_if_settings_needed(monkeypatch)
    out_path = tmp_path / "out" / "result.txt"

    result = analyze_video.main(
        ["--metadata", str(metadata_file), "--output", str(out_path)]
    )

    assert result == 0
    assert out_path.is_file()
    assert "kiss" in out_path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# --video-id: cached raw insights vs. fresh fetch
# ----------------------------------------------------------------------


def test_main_video_id_fetches_fresh_when_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_client = _install_fake_client(monkeypatch)
    out_dir = tmp_path / "output"

    result = analyze_video.main(["--video-id", "abc123", "--out-dir", str(out_dir)])

    assert result == 0
    assert fake_client.wait_for_processing_calls == ["abc123"]


def test_main_video_id_fetch_is_cached_for_next_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(monkeypatch)
    out_dir = tmp_path / "output"

    analyze_video.main(["--video-id", "abc123", "--out-dir", str(out_dir)])

    cached_path = out_dir / "abc123.raw_insights.json"
    assert cached_path.is_file()
    saved = json.loads(cached_path.read_text(encoding="utf-8"))
    assert saved == _insights_payload()


def test_main_video_id_uses_cache_instead_of_fetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    cached_path = out_dir / "abc123.raw_insights.json"
    cached_path.write_text(json.dumps(_insights_payload()), encoding="utf-8")
    _fail_if_settings_needed(monkeypatch)

    result = analyze_video.main(["--video-id", "abc123", "--out-dir", str(out_dir)])

    assert result == 0


def test_main_explicit_metadata_overrides_video_id_cache_and_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    cached_path = out_dir / "abc123.raw_insights.json"
    cached_path.write_text(json.dumps(_insights_payload(kiss=False)), encoding="utf-8")
    metadata_path = tmp_path / "explicit.json"
    metadata_path.write_text(json.dumps(_insights_payload(kiss=True)), encoding="utf-8")
    _fail_if_settings_needed(monkeypatch)

    result = analyze_video.main(
        [
            "--video-id",
            "abc123",
            "--out-dir",
            str(out_dir),
            "--metadata",
            str(metadata_path),
        ]
    )

    assert result == 0


def test_main_returns_1_when_video_id_fetch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_fake_client(
        monkeypatch, wait_for_processing_error=VideoIndexerError("boom")
    )

    result = analyze_video.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path / "output")]
    )

    assert result == 1


def test_main_returns_2_when_video_indexer_settings_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """No cached file, no --metadata, and Settings.from_env() fails (no
    Azure config) -- surfaces as a config error, not a crash."""
    from src.config import ConfigError

    def _raise() -> Settings:
        raise ConfigError("missing config")

    monkeypatch.setattr(analyze_video.Settings, "from_env", staticmethod(_raise))

    result = analyze_video.main(
        ["--video-id", "abc123", "--out-dir", str(tmp_path / "output")]
    )

    assert result == 2
