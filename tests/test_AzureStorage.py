"""Tests for Azure Storage module."""

import os
import uuid
from datetime import datetime

import pytest
import pytz
from azure.identity import DefaultAzureCredential

from engine import AzureStorage
from engine.AzureStorage import (
    EngineOutputDownloader,
    EngineOutputUploader,
)

# Manually update this to be the 'output' folder.
EXPECTED_FILE_COUNT = 9


def test_get_credential_uses_default_azure_credential() -> None:
    """Storage uses the shared Microsoft Entra credential."""
    assert isinstance(AzureStorage.get_credential(), DefaultAzureCredential)


def test_build_blob_client_requires_account(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage account name must be configured explicitly."""
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)

    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT_NAME"):
        AzureStorage.build_blob_service_client()


@pytest.mark.xdist_group("azure_outputs")
def test_upload_outputs() -> None:
    """Test uploading outputs to Azure storage."""
    uploader = EngineOutputUploader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        pytest.output_config["storage"]["container"],
        pytest.output_config["folder_name"],
    )

    uploader.upload_latest_output()

    blobs = uploader.list_blobs(
        name_starts_with=datetime.now(pytz.timezone("Pacific/Auckland")).strftime(
            "%Y-%m-%d_%H"
        )
    )

    assert len(blobs) > 0


@pytest.mark.xdist_group("azure_outputs")
def test_download_outputs(tmpdir: object) -> None:
    """Test downloading outputs from Azure storage."""
    downloader = EngineOutputDownloader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        pytest.output_config["storage"]["container"],
        tmpdir,
    )

    downloader.download_latest_output()

    downloaded_files = [len(f[2]) for f in os.walk(tmpdir.strpath)]
    assert sum(downloaded_files) == EXPECTED_FILE_COUNT


def test_upload_pdf_to_pdf_container(test_pdf_storage_manager: object) -> None:
    """Upload a small fake PDF to the test PDF container and verify it exists."""
    report_id = f"TEST_UPLOAD_{uuid.uuid4().hex[:8]}"
    pdf_bytes = b"%PDF-1.4\n%Fake PDF for tests\n%%EOF"

    uploaded = test_pdf_storage_manager.upload_pdf(report_id, pdf_bytes, overwrite=True)
    assert uploaded is True

    pdfs = test_pdf_storage_manager.list_pdfs()
    assert report_id in pdfs

    downloaded = test_pdf_storage_manager.download_pdf(report_id)
    assert downloaded is not None
    assert downloaded.startswith(b"%PDF") or pdf_bytes in downloaded
