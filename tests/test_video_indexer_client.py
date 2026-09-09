"""
Unit tests for VideoIndexerClient, using a fake requests.Session (and a
fake Azure credential) so all of this can be verified without a live Azure
account or network access -- the same "no live account needed" approach as
the rest of this project's tests.

Originally this file only covered get_video_download_url/download_video;
the rest of the client (auth, upload, index/poll, and the
generateAccessToken failure-diagnosis messages) previously had no test
coverage at all despite being the most failure-prone code in the project
(the one part that talks to a real network and a real Azure account).
"""

import sys
from pathlib import Path
from typing import Any

import pytest
from azure.core.exceptions import ClientAuthenticationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.video_indexer_client import (
    AuthenticationFailedError,
    ProcessingFailedError,
    ProcessingTimeoutError,
    UploadFailedError,
    VideoIndexerClient,
    VideoIndexerError,
    _build_session,
)

SAS_URL = (
    "https://fakestorage.blob.core.windows.net/container/video.mp4?sv=fake-sas-token"
)


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        chunks: list[bytes] | None = None,
    ):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json = json_data
        self.text = text
        self._chunks = chunks or []

    def json(self):
        return self._json

    def iter_content(self, chunk_size: int = 1024 * 1024):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Maps a URL substring to a canned FakeResponse (or a list consumed in
    order, for endpoints polled more than once). Records every GET/POST so
    tests can assert on what was actually sent."""

    def __init__(self, responses: dict[str, FakeResponse | list[FakeResponse]]):
        self._responses = responses
        self.requested_urls: list[str] = []
        self.post_calls: list[dict[str, Any]] = []

    def _resolve(self, url: str) -> FakeResponse:
        self.requested_urls.append(url)
        # Every VideoIndexerClient endpoint URL is built as an f-string with
        # no query string attached (params are passed separately), so each
        # one's distinguishing key is a *suffix* -- e.g. ".../Videos" for
        # upload vs. ".../Videos/{id}/Index" for polling, which would
        # otherwise collide under plain substring matching ("/Videos" is a
        # substring of the Index URL too, and longer than "/Index"). Only
        # the blob-storage download URL (an opaque SAS URL, not one of this
        # client's own endpoints) has no fixed suffix, so that one still
        # falls back to substring matching.
        suffix_matches = [
            (key, response)
            for key, response in self._responses.items()
            if url.endswith(key)
        ]
        matches = suffix_matches or [
            (key, response) for key, response in self._responses.items() if key in url
        ]
        if not matches:
            raise AssertionError(f"Unexpected URL requested: {url}")
        _key, response = max(matches, key=lambda pair: len(pair[0]))
        if isinstance(response, list):
            return response.pop(0) if len(response) > 1 else response[0]
        return response

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
        stream: bool | None = None,
    ):
        return self._resolve(url)

    def post(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        timeout: int | None = None,
    ):
        self.post_calls.append(
            {
                "url": url,
                "params": params,
                "json": json,
                "files": files,
                "headers": headers,
            }
        )
        return self._resolve(url)


class FakeCredential:
    """Stands in for azure.identity.DefaultAzureCredential."""

    def __init__(self, token: str = "arm-token", error: Exception | None = None):
        self._token = token
        self._error = error
        self.requested_scopes: list[str] = []

    def get_token(self, scope: str):
        self.requested_scopes.append(scope)
        if self._error:
            raise self._error
        return type("Token", (), {"token": self._token})()


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        subscription_id="sub-id",
        resource_group="rg",
        account_name="account",
        account_id="account-id",
        location="eastus",
    )


def _client_with_token(settings: Settings, session: FakeSession) -> VideoIndexerClient:
    client = VideoIndexerClient(settings, session=session)
    client._vi_access_token = "fake-token"  # skip Azure AD / ARM auth entirely
    return client


def test_get_video_download_url_returns_sas_url(settings: Settings):
    session = FakeSession({"SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL)})
    client = _client_with_token(settings, session)

    url = client.get_video_download_url("video-123")

    assert url == SAS_URL
    assert any("video-123/SourceFile/DownloadUrl" in u for u in session.requested_urls)


def test_get_video_download_url_raises_on_failure(settings: Settings):
    session = FakeSession(
        {"SourceFile/DownloadUrl": FakeResponse(status_code=404, text="not found")}
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.get_video_download_url("missing-video")


def test_get_video_download_url_raises_on_unexpected_body(settings: Settings):
    session = FakeSession(
        {"SourceFile/DownloadUrl": FakeResponse(json_data={"not": "a url"})}
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.get_video_download_url("video-123")


def test_download_video_writes_streamed_content_to_dest(
    settings: Settings, tmp_path: Path
):
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


def test_download_video_raises_when_blob_request_fails(
    settings: Settings, tmp_path: Path
):
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


def test_download_video_raises_on_empty_download(settings: Settings, tmp_path: Path):
    session = FakeSession(
        {
            "SourceFile/DownloadUrl": FakeResponse(json_data=SAS_URL),
            "fakestorage.blob.core.windows.net": FakeResponse(chunks=[]),
        }
    )
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.download_video("video-123", tmp_path / "video.mp4")


# ----------------------------------------------------------------------
# _build_session
# ----------------------------------------------------------------------


def test_build_session_retries_transient_status_codes():
    session = _build_session(total_retries=4)

    adapter = session.get_adapter("https://api.videoindexer.ai/")
    retry = adapter.max_retries

    assert retry.total == 4
    assert set(retry.status_forcelist) == {429, 500, 502, 503, 504}
    assert "GET" in retry.allowed_methods
    # POST is deliberately excluded: upload_video's POST isn't idempotent,
    # so an automatic retry could re-upload the same file. See
    # _build_session's docstring in src/video_indexer_client.py.
    assert "POST" not in retry.allowed_methods


# ----------------------------------------------------------------------
# Authentication: _get_arm_token / get_access_token / _ensure_token
# ----------------------------------------------------------------------


def _client_with_credential(
    settings: Settings, session: FakeSession, credential: FakeCredential
) -> VideoIndexerClient:
    client = VideoIndexerClient(settings, session=session)
    client._credential = credential  # type: ignore[assignment]
    return client


def test_get_arm_token_returns_token_on_success(settings: Settings):
    client = _client_with_credential(
        settings, FakeSession({}), FakeCredential(token="arm-token-123")
    )

    assert client._get_arm_token() == "arm-token-123"


def test_get_arm_token_raises_authentication_failed_on_client_auth_error(
    settings: Settings,
):
    credential = FakeCredential(error=ClientAuthenticationError("no credentials"))
    client = _client_with_credential(settings, FakeSession({}), credential)

    with pytest.raises(AuthenticationFailedError):
        client._get_arm_token()


def test_get_access_token_returns_and_caches_token(settings: Settings):
    session = FakeSession(
        {"generateAccessToken": FakeResponse(json_data={"accessToken": "vi-token"})}
    )
    client = _client_with_credential(settings, session, FakeCredential())

    token = client.get_access_token()

    assert token == "vi-token"
    assert client._vi_access_token == "vi-token"
    assert session.post_calls[0]["json"] == {
        "permissionType": "Contributor",
        "scope": "Account",
    }


def test_get_access_token_video_scope_requires_video_id(settings: Settings):
    client = _client_with_credential(settings, FakeSession({}), FakeCredential())

    with pytest.raises(ValueError):
        client.get_access_token(scope="Video")


def test_get_access_token_video_scope_includes_video_id_in_body(settings: Settings):
    session = FakeSession(
        {"generateAccessToken": FakeResponse(json_data={"accessToken": "vi-token"})}
    )
    client = _client_with_credential(settings, session, FakeCredential())

    client.get_access_token(scope="Video", video_id="video-123")

    assert session.post_calls[0]["json"] == {
        "permissionType": "Contributor",
        "scope": "Video",
        "videoId": "video-123",
    }


def test_get_access_token_diagnoses_subscription_not_found(settings: Settings):
    session = FakeSession(
        {
            "generateAccessToken": FakeResponse(
                status_code=404, text='{"error": "SubscriptionNotFound"}'
            )
        }
    )
    client = _client_with_credential(settings, session, FakeCredential())

    with pytest.raises(AuthenticationFailedError) as exc_info:
        client.get_access_token()
    assert "AVI_SUBSCRIPTION_ID" in str(exc_info.value)


def test_get_access_token_diagnoses_forbidden(settings: Settings):
    session = FakeSession(
        {
            "generateAccessToken": FakeResponse(
                status_code=403, text="AuthorizationFailed"
            )
        }
    )
    client = _client_with_credential(settings, session, FakeCredential())

    with pytest.raises(AuthenticationFailedError) as exc_info:
        client.get_access_token()
    assert "Contributor" in str(exc_info.value)


def test_get_access_token_raises_when_response_has_no_token(settings: Settings):
    session = FakeSession({"generateAccessToken": FakeResponse(json_data={})})
    client = _client_with_credential(settings, session, FakeCredential())

    with pytest.raises(AuthenticationFailedError):
        client.get_access_token()


def test_ensure_token_reuses_cached_token_without_a_new_call(settings: Settings):
    session = FakeSession({})  # no generateAccessToken response registered at all
    client = _client_with_credential(settings, session, FakeCredential())
    client._vi_access_token = "already-have-one"

    assert client._ensure_token() == "already-have-one"
    assert session.post_calls == []


def test_ensure_token_fetches_when_missing(settings: Settings):
    session = FakeSession(
        {"generateAccessToken": FakeResponse(json_data={"accessToken": "fresh-token"})}
    )
    client = _client_with_credential(settings, session, FakeCredential())

    assert client._ensure_token() == "fresh-token"
    assert client._vi_access_token == "fresh-token"


# ----------------------------------------------------------------------
# upload_video
# ----------------------------------------------------------------------


def test_upload_video_returns_video_id(settings: Settings, tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")
    session = FakeSession({"/Videos": FakeResponse(json_data={"id": "video-1"})})
    client = _client_with_token(settings, session)

    video_id = client.upload_video(video_path, indexing_preset="Advanced")

    assert video_id == "video-1"
    call = session.post_calls[0]
    assert call["params"]["indexingPreset"] == "Advanced"
    assert call["params"]["name"] == "clip"
    assert call["files"]["file"][0] == "clip.mp4"


def test_upload_video_raises_file_not_found(settings: Settings, tmp_path: Path):
    client = _client_with_token(settings, FakeSession({}))

    with pytest.raises(FileNotFoundError):
        client.upload_video(tmp_path / "missing.mp4")


def test_upload_video_raises_with_already_failed_hint(
    settings: Settings, tmp_path: Path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")
    session = FakeSession(
        {"/Videos": FakeResponse(status_code=409, text="VIDEO_ALREADY_FAILED")}
    )
    client = _client_with_token(settings, session)

    with pytest.raises(UploadFailedError) as exc_info:
        client.upload_video(video_path)
    assert "different --name" in str(exc_info.value)


def test_upload_video_raises_when_response_has_no_id(
    settings: Settings, tmp_path: Path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")
    session = FakeSession({"/Videos": FakeResponse(json_data={})})
    client = _client_with_token(settings, session)

    with pytest.raises(UploadFailedError):
        client.upload_video(video_path)


# ----------------------------------------------------------------------
# get_video_index / wait_for_processing / index_video
# ----------------------------------------------------------------------


def test_get_video_index_returns_payload(settings: Settings):
    session = FakeSession({"/Index": FakeResponse(json_data={"state": "Processed"})})
    client = _client_with_token(settings, session)

    assert client.get_video_index("video-1") == {"state": "Processed"}


def test_get_video_index_raises_on_failure(settings: Settings):
    session = FakeSession({"/Index": FakeResponse(status_code=500, text="oops")})
    client = _client_with_token(settings, session)

    with pytest.raises(VideoIndexerError):
        client.get_video_index("video-1")


def test_wait_for_processing_returns_index_once_processed(settings: Settings):
    session = FakeSession(
        {"/Index": FakeResponse(json_data={"state": "Processed", "id": "video-1"})}
    )
    client = _client_with_token(settings, session)

    index = client.wait_for_processing("video-1")

    assert index["state"] == "Processed"


def test_wait_for_processing_raises_on_failed_state(settings: Settings):
    session = FakeSession(
        {
            "/Index": FakeResponse(
                json_data={"state": "Failed", "processingProgress": "bad codec"}
            )
        }
    )
    client = _client_with_token(settings, session)

    with pytest.raises(ProcessingFailedError) as exc_info:
        client.wait_for_processing("video-1")
    assert "bad codec" in str(exc_info.value)


def test_wait_for_processing_raises_on_timeout(settings: Settings):
    # A zero timeout with a never-terminal state means the deadline check
    # trips on the very first iteration -- no real waiting in this test.
    timed_out_settings = Settings(
        subscription_id=settings.subscription_id,
        resource_group=settings.resource_group,
        account_name=settings.account_name,
        account_id=settings.account_id,
        location=settings.location,
        poll_interval_seconds=0.0,
        processing_timeout_seconds=0.0,
    )
    session = FakeSession({"/Index": FakeResponse(json_data={"state": "Processing"})})
    client = _client_with_token(timed_out_settings, session)

    with pytest.raises(ProcessingTimeoutError):
        client.wait_for_processing("video-1")


def test_index_video_uploads_then_waits_for_processing(
    settings: Settings, tmp_path: Path
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")
    session = FakeSession(
        {
            "/Videos": FakeResponse(json_data={"id": "video-1"}),
            "/Index": FakeResponse(json_data={"state": "Processed", "id": "video-1"}),
        }
    )
    client = _client_with_token(settings, session)

    index = client.index_video(video_path)

    assert index["id"] == "video-1"
