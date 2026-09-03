"""
Thin, production-minded client for Azure AI Video Indexer using the current
ARM-based authentication flow (Azure AD -> ARM generateAccessToken -> data
plane API), which is the supported path for accounts created after the
classic API-key trial model was retired for new accounts.

Flow:
    1. Acquire an Azure AD token for the ARM audience (management.azure.com)
       via azure-identity's DefaultAzureCredential. This picks up `az login`
       sessions, managed identities, or service-principal env vars
       automatically -- whatever is available in the environment running
       this code.
    2. Call the Video Indexer ARM control-plane operation
       `generateAccessToken` to mint a short-lived Video Indexer data-plane
       token scoped to this account.
    3. Use that token against the api.videoindexer.ai data-plane API to
       upload a video, poll processing state, and fetch the insights index.

Reference docs (fetched 2026-08-27):
    - ARM generateAccessToken:
      POST https://management.azure.com/subscriptions/{sub}/resourceGroups/
      {rg}/providers/Microsoft.VideoIndexer/accounts/{account}/
      generateAccessToken?api-version=2025-04-01
    - Data-plane upload / index:
      https://api.videoindexer.ai/{location}/Accounts/{accountId}/Videos...
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .constants import (
    ARM_API_VERSION,
    ARM_BASE_URL,
    DATA_PLANE_BASE_URL,
    STATE_FAILED,
    STATE_PROCESSED,
)

logger = logging.getLogger(__name__)


class VideoIndexerError(RuntimeError):
    """Base class for all Video Indexer client errors."""


class AuthenticationFailedError(VideoIndexerError):
    """Raised when an Azure AD or Video Indexer token could not be obtained."""


class UploadFailedError(VideoIndexerError):
    """Raised when the video upload request itself fails."""


class ProcessingFailedError(VideoIndexerError):
    """Raised when Video Indexer finishes processing with a Failed state."""


class ProcessingTimeoutError(VideoIndexerError):
    """Raised when processing does not complete within the configured timeout."""


def _build_session(total_retries: int = 4) -> requests.Session:
    """A requests.Session with sane retry/backoff for transient failures."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class VideoIndexerClient:
    """Client for uploading videos and retrieving insights from Video Indexer."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self._settings = settings
        self._session = session or _build_session()
        self._credential = DefaultAzureCredential()
        self._vi_access_token: str | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def _get_arm_token(self) -> str:
        try:
            token = self._credential.get_token("https://management.azure.com/.default")
        except ClientAuthenticationError as exc:
            raise AuthenticationFailedError(
                "Could not obtain an Azure AD token. Make sure you're logged "
                "in (e.g. `az login`) or that service-principal environment "
                "variables (AZURE_CLIENT_ID / AZURE_TENANT_ID / "
                "AZURE_CLIENT_SECRET) are set, and that the identity has "
                "access to the Video Indexer account's resource group."
            ) from exc
        return token.token

    def get_access_token(
        self, scope: str = "Account", video_id: str | None = None
    ) -> str:
        """Mint a Video Indexer data-plane access token via the ARM control plane."""
        arm_token = self._get_arm_token()
        url = (
            f"{ARM_BASE_URL}/subscriptions/{self._settings.subscription_id}"
            f"/resourceGroups/{self._settings.resource_group}"
            f"/providers/Microsoft.VideoIndexer/accounts/{self._settings.account_name}"
            f"/generateAccessToken"
        )
        body: dict[str, Any] = {"permissionType": "Contributor", "scope": scope}
        if scope == "Video":
            if not video_id:
                raise ValueError("video_id is required when scope='Video'")
            body["videoId"] = video_id

        response = self._session.post(
            url,
            params={"api-version": ARM_API_VERSION},
            json=body,
            headers={"Authorization": f"Bearer {arm_token}"},
            timeout=30,
        )
        if not response.ok:
            raise AuthenticationFailedError(
                self._diagnose_generate_token_failure(response)
            )

        token = response.json().get("accessToken")
        if not token:
            raise AuthenticationFailedError(
                f"generateAccessToken response did not contain an accessToken: {response.text}"
            )
        self._vi_access_token = token
        return token

    def _diagnose_generate_token_failure(self, response: requests.Response) -> str:
        """Turn a failed generateAccessToken call into an actionable message.

        The two failure modes people actually hit in practice are: the
        AVI_SUBSCRIPTION_ID / AVI_ACCOUNT_ID values got swapped in .env (both
        are GUIDs, easy to mix up), or `az login` authenticated into a
        different Azure AD tenant than the one that owns the subscription
        the Video Indexer resource lives in (common with multi-tenant
        accounts). Both surface as a 404 SubscriptionNotFound from ARM, even
        though the subscription does exist.
        """
        base = f"generateAccessToken failed ({response.status_code}): {response.text}"
        body_lower = response.text.lower()

        if response.status_code == 404 and "subscriptionnotfound" in body_lower:
            return (
                base + "\n\nThis almost always means one of two things:\n"
                "  1. AVI_SUBSCRIPTION_ID and AVI_ACCOUNT_ID got swapped or "
                "mistyped in .env (both are GUIDs) -- reopen the Video "
                "Indexer resource in the Azure portal and re-copy the "
                "Subscription ID from its Overview page's 'Essentials' "
                "section (not the Account ID).\n"
                "  2. `az login` authenticated into a different Azure AD "
                "tenant than the one that owns this subscription (common "
                "with multi-tenant/work+personal accounts). Run "
                "`az account show` to see which tenant/subscription is "
                "currently active, and `az account list --all -o table` to "
                "see every subscription your account can reach. If the "
                "right one is under a different tenant, run "
                "`az login --tenant <tenant-id>` and retry."
            )
        if response.status_code in (401, 403):
            return (
                base + "\n\nYour identity authenticated, but doesn't have "
                "permission to call generateAccessToken on this resource. "
                "Grant it at least 'Contributor' on the Video Indexer "
                "account or its resource group (Azure Portal -> the "
                "resource -> Access control (IAM) -> Add role assignment)."
            )
        return base

    def _ensure_token(self) -> str:
        if not self._vi_access_token:
            return self.get_access_token()
        return self._vi_access_token

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_video(
        self,
        video_path: Path,
        name: str | None = None,
        indexing_preset: str = "Default",
        privacy: str = "Private",
    ) -> str:
        """Upload a local video file and return the Video Indexer video ID."""
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        token = self._ensure_token()
        url = (
            f"{DATA_PLANE_BASE_URL}/{self._settings.location}"
            f"/Accounts/{self._settings.account_id}/Videos"
        )
        params = {
            "accessToken": token,
            "name": name or video_path.stem,
            "privacy": privacy,
            "indexingPreset": indexing_preset,
        }

        logger.info(
            "Uploading %s to Video Indexer (this can take a while for large files)...",
            video_path,
        )
        with video_path.open("rb") as fh:
            files = {"file": (video_path.name, fh, "application/octet-stream")}
            response = self._session.post(url, params=params, files=files, timeout=600)

        if not response.ok:
            hint = ""
            if (
                response.status_code == 409
                and "video_already_failed" in response.text.lower()
            ):
                hint = (
                    "\n\nVideo Indexer is rate-limiting retries against a "
                    "previously-failed upload with this same name (common "
                    "right after fixing a permissions issue that caused the "
                    "original failure). Try again with a different "
                    "--name/name=..., or wait out the cooldown the error "
                    "message states."
                )
            raise UploadFailedError(
                f"Upload failed ({response.status_code}): {response.text}{hint}"
            )

        video_id = response.json().get("id")
        if not video_id:
            raise UploadFailedError(
                f"Upload response did not contain a video id: {response.text}"
            )

        logger.info("Uploaded video, id=%s", video_id)
        return video_id

    # ------------------------------------------------------------------
    # Processing / insights
    # ------------------------------------------------------------------
    def get_video_index(self, video_id: str) -> dict[str, Any]:
        """Fetch the current index/insights payload for a video, as-is (any state)."""
        token = self._ensure_token()
        url = (
            f"{DATA_PLANE_BASE_URL}/{self._settings.location}/Accounts/"
            f"{self._settings.account_id}/Videos/{video_id}/Index"
        )
        response = self._session.get(url, params={"accessToken": token}, timeout=60)
        if not response.ok:
            raise VideoIndexerError(
                f"Get index failed ({response.status_code}): {response.text}"
            )
        return response.json()

    def wait_for_processing(self, video_id: str) -> dict[str, Any]:
        """Poll the video index until it reaches a terminal state."""
        deadline = time.monotonic() + self._settings.processing_timeout_seconds
        last_state = None

        while True:
            index = self.get_video_index(video_id)
            state = index.get("state")
            if state != last_state:
                logger.info("Video %s state: %s", video_id, state)
                last_state = state

            if state == STATE_PROCESSED:
                return index
            if state == STATE_FAILED:
                raise ProcessingFailedError(
                    f"Video Indexer failed to process video {video_id}: "
                    f"{index.get('processingProgress', 'no details provided')}"
                )

            if time.monotonic() >= deadline:
                raise ProcessingTimeoutError(
                    f"Timed out after {self._settings.processing_timeout_seconds:.0f}s "
                    f"waiting for video {video_id} to finish processing (last state: {state})."
                )

            time.sleep(self._settings.poll_interval_seconds)

    # ------------------------------------------------------------------
    # Source file download (lets a video be reused by --video-id alone,
    # without also needing the original file on hand)
    # ------------------------------------------------------------------
    def get_video_download_url(self, video_id: str) -> str:
        """Get a short-lived SAS URL for downloading the original source
        video file Video Indexer stored for `video_id`."""
        token = self._ensure_token()
        url = (
            f"{DATA_PLANE_BASE_URL}/{self._settings.location}/Accounts/"
            f"{self._settings.account_id}/Videos/{video_id}/SourceFile/DownloadUrl"
        )
        response = self._session.get(url, params={"accessToken": token}, timeout=30)
        if not response.ok:
            raise VideoIndexerError(
                f"Get source download URL failed ({response.status_code}): {response.text}"
            )
        download_url = response.json()
        if not isinstance(download_url, str) or not download_url:
            raise VideoIndexerError(
                f"Unexpected source download URL response: {response.text}"
            )
        return download_url

    def download_video(self, video_id: str, dest_path: Path) -> Path:
        """Download the original source video file for `video_id` to
        `dest_path`, streaming so large files don't need to fit in memory.
        Returns `dest_path`."""
        download_url = self.get_video_download_url(video_id)
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading source video for %s to %s ...", video_id, dest_path)
        # The SAS URL is pre-authenticated (blob storage), not a Video
        # Indexer data-plane endpoint, so no accessToken is sent here.
        with self._session.get(download_url, stream=True, timeout=600) as response:
            if not response.ok:
                raise VideoIndexerError(
                    f"Downloading source video failed ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            with dest_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

        size = dest_path.stat().st_size
        if size == 0:
            raise VideoIndexerError(f"Downloaded source video for {video_id} is empty.")
        logger.info("Downloaded source video (%d bytes)", size)
        return dest_path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def index_video(self, video_path: Path, name: str | None = None) -> dict[str, Any]:
        """Upload a video and block until its full insights index is ready."""
        video_id = self.upload_video(video_path, name=name)
        return self.wait_for_processing(video_id)
