import os
import uuid
from datetime import datetime

import pytest
import pytz

from engine.utils.AzureStorage import (
    EngineOutputDownloader,
    EngineOutputUploader,
)


def test_upload_outputs():
    uploader = EngineOutputUploader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
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


def test_download_outputs(tmpdir):
    downloader = EngineOutputDownloader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        pytest.output_config["storage"]["container"],
        tmpdir,
    )

    downloader.download_latest_output()

    downloaded_files = [len(f[2]) for f in os.walk(tmpdir.strpath)]
    assert sum(downloaded_files) == 40


def test_upload_pdf_to_pdf_container(test_pdf_storage_manager):
    """Upload a small fake PDF to the test PDF container and verify it exists."""
    report_id = f"TEST_UPLOAD_{uuid.uuid4().hex[:8]}"
    pdf_bytes = b"%PDF-1.4\n%Fake PDF for tests\n%%EOF"

    uploaded = test_pdf_storage_manager.upload_pdf(
        report_id, pdf_bytes, overwrite=True
    )
    assert uploaded is True

    pdfs = test_pdf_storage_manager.list_pdfs()
    assert report_id in pdfs

    downloaded = test_pdf_storage_manager.download_pdf(report_id)
    assert downloaded is not None
    assert downloaded.startswith(b"%PDF") or pdf_bytes in downloaded
