import argparse  # noqa: INP001
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

logging.basicConfig(format="%(source)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run_command(cmd, capture_output):
    """Run a shell command and return output."""
    try:
        shell_mode = os.name == "nt"
        if capture_output:
            result = subprocess.run(  # noqa: S603
                cmd,
                shell=shell_mode,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        result = subprocess.run(cmd, shell=shell_mode, check=True)  # noqa: S603
        return result.returncode  # noqa: TRY300
    except subprocess.CalledProcessError as e:
        logger.debug("Error running command", extra={"source": e.stderr})
        sys.exit(1)


def main():
    """Main function to download vector database using Azure CLI and AzCopy."""
    parser = argparse.ArgumentParser(
        description="Download vector database from Azure Blob Storage using AzCopy.",
    )
    parser.add_argument(
        "--account-name",
        type=str,
        help="Azure Storage account name",
        default="taicdocumentsearcherdata",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        help="Azure Storage container name",
        default="vectordb",
    )
    parser.add_argument(
        "--blob-path",
        type=str,
        help="Path to the blob in Azure Storage",
        default="prod/all_document_types.lance",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory to save the downloaded files",
        default="./vectordb",
    )
    parser.add_argument("--log-level", type=str, help="Logging level", default="INFO")

    args = parser.parse_args()

    # Set expiry to 1 day from now in UTC
    expiry = (datetime.now(timezone.utc) + timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    logger.debug("Using SAS expiry time: %(expiry)s", extra={"expiry": expiry})
    # Generate SAS token command
    sas_cmd = [
        "az",
        "storage",
        "container",
        "generate-sas",
        "--account-name",
        args.account_name,
        "--name",
        args.container_name,
        "--permissions",
        "rl",
        "--expiry",
        expiry,
        "--output",
        "tsv",
    ]

    logger.debug(
        "Generating SAS token...\nWith command:\n%(command)s",
        extra={"command": " ".join(sas_cmd)},
    )

    sas_token = run_command(sas_cmd, capture_output=True)
    sas_token = sas_token.strip()

    args.blob_path = args.blob_path.removeprefix("/")

    blob_url = f"https://{args.account_name}.blob.core.windows.net/{args.container_name}/{args.blob_path}?{sas_token}"

    logger.debug("Blob URL with SAS token:\n%(url)s", extra={"url": blob_url})

    azcopy_cmd = [
        "azcopy",
        "copy",
        blob_url,
        args.local_dir,
        "--recursive",
        "--log-level",
        args.log_level,
    ]

    logger.debug("Starting azcopy download...")
    run_command(azcopy_cmd, capture_output=False)
    logger.debug("Download completed.")


if __name__ == "__main__":
    main()
