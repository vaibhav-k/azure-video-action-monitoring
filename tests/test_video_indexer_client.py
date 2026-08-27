"""
Unit tests for VideoIndexerClient.get_video_download_url / download_video,
using a fake requests.Session so this can be verified without a live Azure
account or network access -- the same "no live account needed" approach as
the rest of this project's tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.video_indexer_client import VideoIndexerClient, VideoIndexerError

SAS_URL = (
    "https://fakestorage.blob.core.windows.net/container/video.mp4?sv=fake-sas-token"
)


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", chunks=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json = json_data
        self.text = text
        self._chunks = chunks or []

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1024 * 1024):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Maps a URL substring to a canned FakeResponse (or a list consumed in order)."""

    def __init__(self, responses: dict):
        self._responses = responses
        self.requested_urls: list[str] = []

    def get(self, url, params=None, timeout=None, stream=None):
        self.requested_urls.append(url)
        for substring, response in self._responses.items():
            if substring in url:
                return response
        raise AssertionError(f"Unexpected URL requested: {url}")


@pytest.fixture()
def settings():
    return Settings(
        subscription_id="sub-id",
        resource_group="rg",
        account_name="account",
        account_id="account-id",
        location="eastus",
    )


def _client_with_token(settings, session) -> VideoIndexerClient:
    client = VideoIndexerClient(settings, session=session)
    client._vi_access_token = "fake-token"  # skip Azure AD / ARM auth entirely
    return client


def test_get_video_download_url_returns_sas_url(settings):
    session = FakeSession({"SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL)})
    client = _client_with_token(settings, session)

    url = client.get_video_download_url("video-123")

    assert url == SAS_URL
    assert any("video-123/SourceFile/DownloadUrl" in u for u in session.requested_urls)


def test_get_video_download_url_raises_on_failure(settings):
    session = FakeSession(
        {"SourceFile/DownloadUrl": FakeResponse(status_code=404, text="not found")}
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.get_video_download_url("missing-video")


def test_get_video_download_url_raises_on_unexpected_body(settings):
    session = FakeSession(
        {"SourceFile/DownloadUrl": FakeResponse(json_data={"not": "a url"})}
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.get_video_download_url("video-123")


def test_download_video_writes_streamed_content_to_dest(settings, tmp_path):
    session = FakeSession(
        {
            "SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL),
            "fakestorage.blob.core.windows.net": FakeResponse(
                chunks=[b"hello ", b"world"]
            ),
        }
    )
    client = _client_with_token(settings, session)
    dest = tmp_path / "downloaded" / "video.mp4"

    written = client.download_video("video-123", dest)

    assert written == dest
    assert dest.read_bytes() == b"hello world"


def test_download_video_raises_when_blob_request_fails(settings, tmp_path):
    session = FakeSession(
        {
            "SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL),
            "fakestorage.blob.core.windows.net": FakeResponse(
                status_code=403, text="forbidden"
            ),
        }
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.download_video("video-123", tmp_path / "video.mp4")


def test_download_video_raises_on_empty_download(settings, tmp_path):
    session = FakeSession(
        {
            "SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL),
            "fakestorage.blob.core.windows.net": FakeResponse(chunks=[]),
        }
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.download_video("video-123", tmp_path / "video.mp4")
