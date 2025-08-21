import subprocess
import sys
import os
from datetime import datetime, timedelta, timezone
import argparse

def run_command(cmd, capture_output):
    """Run a shell command and return output."""
    try:
        
        shell_mode = os.name == 'nt'
        if capture_output:
            result = subprocess.run(cmd, shell=shell_mode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=shell_mode, check=True)
            return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        sys.exit(1)

def main():
    """Main function to download vector database using Azure CLI and AzCopy."""
    parser = argparse.ArgumentParser(description="Download vector database from Azure Blob Storage using AzCopy.")
    parser.add_argument('--account-name', type=str, help='Azure Storage account name', default="taicdocumentsearcherdata")
    parser.add_argument('--container-name', type=str, help='Azure Storage container name', default="vectordb")
    parser.add_argument('--blob-path', type=str, help='Path to the blob in Azure Storage', default="prod/all_document_types.lance")
    parser.add_argument('--local-dir', type=str, help='Local directory to save the downloaded files', default="./vectordb")
    parser.add_argument('--log-level', type=str, help='Logging level', default="INFO")

    args = parser.parse_args()

    # Set expiry to 1 day from now in UTC
    expiry = (datetime.now(timezone.utc) + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"Using SAS expiry time: {expiry}")
    # Generate SAS token command
    sas_cmd = [
        "az", "storage", "container", "generate-sas",
        "--account-name", args.account_name,
        "--name", args.container_name,
        "--permissions", "rl",
        "--expiry", expiry,
        "--output", "tsv",
    ]

    print(f"Generating SAS token...\nWith command:\n{' '.join(sas_cmd)}")

    sas_token = run_command(sas_cmd, capture_output=True)
    sas_token = sas_token.strip()

    if args.blob_path.startswith('/'):
        args.blob_path = args.blob_path[1:]

    blob_url = f"https://{args.account_name}.blob.core.windows.net/{args.container_name}/{args.blob_path}?{sas_token}"

    print(f"Blob URL with SAS token:\n{blob_url}")

    azcopy_cmd = [
        "azcopy", "copy",
        blob_url,
        args.local_dir,
        "--recursive",
        "--log-level", args.log_level
    ]

    print("Starting azcopy download...")
    run_command(azcopy_cmd, capture_output=False)
    print("Download completed.")

if __name__ == "__main__":
    main()
