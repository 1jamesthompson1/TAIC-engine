"""Azure Blob Storage functionality for the TAIC engine.

This module provides tools for managing files in Azure Blob Storage,
including uploading, downloading, and organizing blobs.
"""

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pytz
from azure.storage.blob import BlobServiceClient
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from engine.Logging import get_logger

logger = get_logger(__name__)


class AzureStorageBase:
    """Base class for Azure Blob Storage operations.

    Provides common functionality for container management and blob operations.
    """

    def __init__(
        self,
        storage_account_name: str,
        storage_account_key: str,
        container_name: str,
    ) -> None:
        """Initialize the Azure storage client.

        Args:
            storage_account_name: The Azure storage account name
            storage_account_key: The Azure storage account key
            container_name: The container name
        """
        self.storage_account_name = storage_account_name
        self.storage_account_key = storage_account_key
        self.container_name = container_name
        self.container = self._get_container()

    def _get_container(self) -> object:
        """Initialize and return the container client.

        Returns:
            The container client object
        """
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.storage_account_name};AccountKey={self.storage_account_key}==;EndpointSuffix=core.windows.net"

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container = blob_service_client.get_container_client(self.container_name)

        if not container.exists():
            container.create_container()

        return container

    def blob_exists(self, blob_name: str) -> bool:
        """Check if a blob exists in the container.

        Args:
            blob_name: The blob name

        Returns:
            True if the blob exists, False otherwise
        """
        blob_client = self.container.get_blob_client(blob_name)
        return blob_client.exists()

    def upload_blob(self, blob_name: str, data: bytes, overwrite: bool = False) -> bool:
        """Upload data to a blob.

        Args:
            blob_name: The blob name
            data: The data to upload as bytes
            overwrite: Whether to overwrite existing blobs

        Returns:
            True if uploaded, False if skipped (already exists and overwrite=False)
        """
        if not overwrite and self.blob_exists(blob_name):
            return False

        blob_client = self.container.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=overwrite)
        return True

    def download_blob(self, blob_name: str) -> bytes | None:
        """Download a blob from the container.

        Args:
            blob_name: The blob name

        Returns:
            Blob content or None if not found
        """
        try:
            blob_client = self.container.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception:
            return None

    def delete_blob(self, blob_name: str) -> bool:
        """Delete a blob from the container.

        Args:
            blob_name: The blob name

        Returns:
            True if deleted, False if not found
        """
        try:
            blob_client = self.container.get_blob_client(blob_name)
            blob_client.delete_blob()
        except Exception:
            return False
        return True

    def get_blob_info(self, blob_name: str) -> dict | None:
        """Get information about a blob in the container.

        Args:
            blob_name: The blob name

        Returns:
            Blob metadata or None if not found
        """
        try:
            blob_client = self.container.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()
        except Exception:
            return None
        return {
            "name": blob_name,
            "size": properties.size,
            "last_modified": properties.last_modified,
            "etag": properties.etag,
        }

    def list_blobs(self, name_starts_with: str | None = None) -> list[str]:
        """List all blob names in the container.

        Args:
            name_starts_with: Optional prefix filter

        Returns:
            List of blob names
        """
        blob_names = self.container.list_blob_names(name_starts_with=name_starts_with)
        return list(blob_names)

    def upload_file(
        self, file_path: str, blob_name: str, overwrite: bool = False
    ) -> bool:
        """Upload a file to a blob.

        Args:
            file_path: Path to the file to upload
            blob_name: The blob name to upload to
            overwrite: Whether to overwrite existing blobs

        Returns:
            True if uploaded, False if skipped
        """
        if not overwrite and self.blob_exists(blob_name):
            return False

        blob_client = self.container.get_blob_client(blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)
        return True

    def download_to_file(self, blob_name: str, file_path: str) -> bool:
        """Download a blob to a file.

        Args:
            blob_name: The blob name to download
            file_path: Path where to save the file

        Returns:
            True if downloaded, False if not found
        """
        try:
            blob_client = self.container.get_blob_client(blob_name)

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as data:
                blob_client.download_blob().readinto(data)
        except Exception:
            return False
        return True


class PDFStorageManager(AzureStorageBase):
    """Manages PDF storage in Azure Blob Storage.

    Provides specialized methods for PDF operations with streaming capabilities.
    """

    def __init__(
        self,
        storage_account_name: str,
        storage_account_key: str,
        container_name: str,
    ) -> None:
        """Initialize the PDF storage manager.

        Args:
            storage_account_name: The Azure storage account name
            storage_account_key: The Azure storage account key
            container_name: The container name
        """
        super().__init__(storage_account_name, storage_account_key, container_name)
        self.pdf_list = [b for b in self.list_blobs() if b.endswith(".pdf")]

    @staticmethod
    def _report_id_to_blob_name(report_id: str) -> str:
        """Convert a report_id to the corresponding blob name.

        Args:
            report_id: The report ID

        Returns:
            The blob name
        """
        return f"{report_id}.pdf"

    def pdf_exists(self, report_id: str) -> bool:
        """Check if a PDF already exists in the cached list of blobs.

        Args:
            report_id: The report ID

        Returns:
            True if the PDF exists, False otherwise
        """
        blob_name = self._report_id_to_blob_name(report_id)
        return blob_name in self.pdf_list

    def upload_pdf(
        self, report_id: str, pdf_data: bytes, overwrite: bool = False
    ) -> bool:
        """Upload a PDF to the container.

        Args:
            report_id: The report ID
            pdf_data: The PDF data as bytes
            overwrite: Whether to overwrite existing PDFs

        Returns:
            True if uploaded, False if skipped
        """
        blob_name = self._report_id_to_blob_name(report_id)
        return self.upload_blob(blob_name, pdf_data, overwrite)

    def download_pdf(self, report_id: str) -> bytes | None:
        """Download a PDF from the container.

        Args:
            report_id: The report ID

        Returns:
            The PDF data or None if not found
        """
        blob_name = self._report_id_to_blob_name(report_id)
        return self.download_blob(blob_name)

    def stream_pdf_to_temp_file(self, report_id: str) -> str | None:
        """Stream a PDF to a temporary file and return the file path.

        The caller is responsible for cleaning up the temporary file.

        Args:
            report_id: The report ID

        Returns:
            The path to the temporary file, or None if PDF not found
        """
        pdf_data = self.download_pdf(report_id)
        if pdf_data is None:
            return None

        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix=f"{report_id}_")
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(pdf_data)
        except Exception:
            with contextlib.suppress(Exception):
                os.unlink(temp_path)
            raise
        else:
            return temp_path

    def list_pdfs(self) -> list[str]:
        """List all PDF report IDs in the container.

        Returns:
            List of PDF report IDs
        """
        blob_names = self.list_blobs()
        return [name[:-4] for name in blob_names if name.endswith(".pdf")]

    def delete_pdf(self, report_id: str) -> bool:
        """Delete a PDF from the container.

        Args:
            report_id: The report ID

        Returns:
            True if deleted, False if not found
        """
        blob_name = self._report_id_to_blob_name(report_id)
        return self.delete_blob(blob_name)

    def get_pdf_info(self, report_id: str) -> dict | None:
        """Get information about a PDF in the container.

        Args:
            report_id: The report ID

        Returns:
            PDF metadata or None if not found
        """
        blob_name = self._report_id_to_blob_name(report_id)
        info = self.get_blob_info(blob_name)
        if info:
            info["report_id"] = report_id
        return info


class EngineOutputManager(AzureStorageBase):
    """Manages engine output storage in Azure Blob Storage.

    Provides specialized methods for organizing outputs by date and managing
    vector databases.
    """

    def __init__(
        self,
        storage_account_name: str,
        storage_account_key: str,
        container_name: str,
    ) -> None:
        """Initialize the engine output manager.

        Args:
            storage_account_name: The Azure storage account name
            storage_account_key: The Azure storage account key
            container_name: The container name
        """
        super().__init__(storage_account_name, storage_account_key, container_name)
        self.output_date_format = "%Y-%m-%d_%H:%M:%S"

    def _get_latest_output(self) -> str | None:
        """Get the latest output folder based on date.

        Returns:
            The latest output folder name, or None if no outputs exist
        """
        logger.debug("Getting latest output folder...")
        blob_names = self.list_blobs()
        folders = list({f.split("/")[0] for f in blob_names if "/" in f})

        dates = [
            datetime.strptime(date, self.output_date_format).astimezone(
                pytz.timezone("Pacific/Auckland")
            )
            for date in folders
        ]
        if len(dates) == 0:
            return None

        return folders[dates.index(max(dates))]


class EngineOutputUploader(EngineOutputManager):
    """Handles uploading engine outputs."""

    def __init__(
        self,
        storage_account_name: str,
        storage_account_key: str,
        container_name: str,
        folder_to_upload: str,
    ) -> None:
        """Initialize the engine output uploader.

        Args:
            storage_account_name: The Azure storage account name
            storage_account_key: The Azure storage account key
            container_name: The container name
            folder_to_upload: The local folder to upload
        """
        super().__init__(storage_account_name, storage_account_key, container_name)

        self.folder_to_upload = folder_to_upload

    def _upload_folder(self, uploaded_folder_name: str) -> None:
        """Upload a folder to the container with improved reliability and speed.

        Args:
            uploaded_folder_name: The destination folder name in the container

        Raises:
            RuntimeError: If the upload fails
        """
        start_time = time.time()

        # Check if Azure CLI is available
        if shutil.which("az"):
            logger.info(
                f"Using Azure CLI to upload folder {self.folder_to_upload} to {uploaded_folder_name}"
            )
            try:
                cmd = [
                    "az",
                    "storage",
                    "blob",
                    "upload-batch",
                    "--destination",
                    self.container_name,
                    "--source",
                    self.folder_to_upload,
                    "--destination-path",
                    uploaded_folder_name,
                    "--account-name",
                    self.storage_account_name,
                    "--account-key",
                    self.storage_account_key,
                    "--overwrite",
                    "--max-connections",
                    "10",
                    # "--no-progress"  # Reduce output noise
                ]

                subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(
                    f"Azure CLI upload completed in {time.time() - start_time:.2f} seconds"
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Azure CLI upload failed: {e.stderr}")
                logger.info("Falling back to Python implementation...")

        # Fallback to improved Python implementation
        logger.info(
            f"Using Python implementation to upload folder {self.folder_to_upload} to {uploaded_folder_name}"
        )
        _, failed = self._upload_folder_python_parallel(
            uploaded_folder_name, start_time
        )

        if failed > 0:
            msg = f"Upload failed with {failed} errors. Please check the logs for details."
            raise RuntimeError(msg)

    def _upload_folder_python_parallel(
        self, uploaded_folder_name: str, start_time: float
    ) -> tuple[int, int]:
        """Parallel Python implementation with retry logic.

        Args:
            uploaded_folder_name: The destination folder name
            start_time: The start time for timing

        Returns:
            A tuple of (successful count, failed count)
        """
        # Collect all files to upload
        files_to_upload = []
        for root, _, files in os.walk(self.folder_to_upload):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                blob_name = os.path.join(
                    uploaded_folder_name,
                    os.path.relpath(file_path, self.folder_to_upload),
                ).replace(os.sep, "/")  # Ensure forward slashes for blob names
                files_to_upload.append((file_path, blob_name))

        logger.info(f"Found {len(files_to_upload)} files to upload")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        def upload_single_file(file_info: tuple[str, str]) -> bool:
            file_path, blob_name = file_info
            return self.upload_file(file_path, blob_name, overwrite=True)

        successful = 0
        failed = 0

        # Use ThreadPoolExecutor for parallel uploads
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(upload_single_file, file_info): file_info
                for file_info in files_to_upload
            }

            # Process completed uploads with progress bar
            for future in tqdm(
                as_completed(future_to_file),
                total=len(files_to_upload),
                desc="Uploading files",
            ):
                file_info = future_to_file[future]
                try:
                    future.result()
                    successful += 1
                except Exception:
                    logger.exception(f"Failed to upload {file_info[0]}")
                    failed += 1

        duration = time.time() - start_time
        logger.info(
            f"Upload completed: {successful} successful, {failed} failed in {duration:.2f} seconds"
        )
        return successful, failed

    def upload_latest_output(self) -> None:
        """Upload the latest output folder."""
        logger.info(f"=== Uploading latest output from {self.folder_to_upload} ===")
        output_date_str = datetime.now(pytz.timezone("Pacific/Auckland")).strftime(
            self.output_date_format
        )
        logger.info(f"Destination: {output_date_str}")

        self._upload_folder(output_date_str)


class EngineOutputDownloader(EngineOutputManager):
    """Handles downloading engine outputs from Azure Blob Storage."""

    def __init__(
        self,
        storage_account_name: str,
        storage_account_key: str,
        container_name: str,
        downloaded_folder_name: str,
    ) -> None:
        """Initialize the engine output downloader.

        Args:
            storage_account_name: The Azure storage account name
            storage_account_key: The Azure storage account key
            container_name: The container name
            downloaded_folder_name: The local folder to download to
        """
        super().__init__(storage_account_name, storage_account_key, container_name)
        self.downloaded_folder_name = downloaded_folder_name

    def download_latest_output(self) -> None:
        """Download the latest output to the specified folder."""
        logger.info(
            f"=== Downloading latest output to {self.downloaded_folder_name} ==="
        )
        latest_output_folder_name = self._get_latest_output()
        if latest_output_folder_name is None:
            logger.info("No latest output found")
            return

        blobs = self.list_blobs(name_starts_with=latest_output_folder_name)

        logger.info(f"Downloading {len(blobs)} blobs from {latest_output_folder_name}")

        successful = 0
        failed = 0
        start_time = time.time()
        for blob_name in blobs:
            if successful % 1000 == 0:
                logger.info(f"Downloaded {successful} files so far...")
            download_path = os.path.join(
                self.downloaded_folder_name,
                os.path.relpath(blob_name, latest_output_folder_name),
            )
            try:
                self.download_to_file(blob_name, download_path)
            except Exception:
                logger.exception(f"Failed to download {blob_name}")
                failed += 1
                continue
            successful += 1

        duration = time.time() - start_time
        logger.info(
            f"Downloaded {successful} files successfully, {failed} failed in {duration:.2f}s."
        )
