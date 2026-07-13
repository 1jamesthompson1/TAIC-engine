"""This is designed for TAIC staff to easily download/copy/delete blobs. There are some assumtpions made about storage structure.

Why:
- In containers, `azcopy login` can fail (no keyring).
- This script generates a short-lived SAS token and uses `azcopy` with it.

It supports three operations:
- Download to local folder: provide '--local'
- Copy within a container: provide `--dst` (without `--local`)
- Delete from container: omit `--dst` (asks for confirmation)

Expected `.env` (repo root):
- AZURE_STORAGE_ACCOUNT_NAME
- AZURE_STORAGE_ACCOUNT_KEY

Usage examples:
- Download the latest output run:
  uv run azure --latest-output

- Download the latest production DB:
- uv run azure --prod-db

- Download a run folder into local output/:
  uv run azure --container engineoutput --src 2025-11-17_10:09:19 --local output

- Copy within same container:
  uv run azure --src some/prefix --dst other/prefix --container vectordb

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import dotenv

from engine.AzureStorage import EngineOutputManager


def _repo_root() -> Path:
    # notebooks/admin/easy_azure.py -> repo root is two parents up
    return Path(__file__).resolve().parents[2]


# Load repo-root .env (works no matter where you run from)
dotenv.load_dotenv(_repo_root() / ".env")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        msg = (
            f"Missing required environment variable {name}. "
            f"Create {_repo_root() / '.env'} and set it, or export it in your shell."
        )
        raise SystemExit(msg)
    return val


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def _generate_sas(
    *, account: str, key: str, container: str, expiry_hours: int = 1
) -> str:
    expiry = (datetime.now(UTC) + timedelta(hours=expiry_hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    cmd = [
        "az",
        "storage",
        "container",
        "generate-sas",
        "--account-name",
        account,
        "--name",
        container,
        "--permissions",
        "rlwd",
        "--expiry",
        expiry,
        "--account-key",
        key,
        "--output",
        "tsv",
    ]

    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except FileNotFoundError as e:
        msg = "`az` (Azure CLI) not found. Install Azure CLI inside the dev machine."
        raise SystemExit(msg) from e
    except subprocess.CalledProcessError as e:
        msg = f"Failed to generate SAS token (exit {e.returncode})."
        raise SystemExit(msg) from e

    if not out:
        msg = "Azure CLI returned an empty SAS token."
        raise SystemExit(msg)

    return out


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915
    """Generate SAS token and execute azcopy operations.

    Supports download, copy, and delete operations on Azure blob storage.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code from azcopy command or 0 for successful completion.

    Raises:
        SystemExit: If required environment variables are missing, Azure CLI fails,
                   or no engine output folders are found.
        ValueError: If invalid argument combinations are provided.
    """
    parser = argparse.ArgumentParser(
        prog="easy_azure",
        description="Generate SAS and run azcopy (download/copy/delete).",
    )
    parser.add_argument(
        "--latest-output",
        action="store_true",
        help="Use latest engine output folder as source",
    )
    parser.add_argument(
        "--prod-db",
        action="store_true",
        help="Use latest production vectordb folder as source",
    )
    parser.add_argument("--src", help="Source folder/prefix inside container")
    parser.add_argument("--dst", help="Destiniation container prefix)")
    parser.add_argument(
        "--local",
        help="Local path to download to.",
    )
    parser.add_argument(
        "--container",
        help="Container name",
    )
    parser.add_argument(
        "--expiry-hours",
        type=int,
        default=1,
        help="SAS token expiry in hours (default: 1)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with delete)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but don't execute them",
    )

    args = parser.parse_args(argv)

    if args.latest_output or args.prod_db:
        if any([args.src, args.dst, args.container]):
            msg = "--latest-output and --prod-db cannot be combined with --src, --dst, or --container."
            raise ValueError(msg)
        if args.local is None:
            msg = "--latest-output and --prod-db require --local to be specified."
            raise ValueError(msg)

    if args.local and args.dst:
        msg = "Only Local or dst can be specified, not both."
        raise ValueError(msg)
    if not args.local and not args.dst:
        msg = "Either --local or --dst must be specified."
        raise ValueError(msg)

    account = _require_env("AZURE_STORAGE_ACCOUNT_NAME")
    key = _require_env("AZURE_STORAGE_ACCOUNT_KEY")

    if args.latest_output:
        output_container = "engineoutput"
        manager = EngineOutputManager(account, key, output_container)
        latest_folder = manager._get_latest_output()
        if not latest_folder:
            msg = "No engine output folders found in storage."
            raise SystemExit(msg)
        args.container = output_container
        args.src = latest_folder

    if args.prod_db:
        args.containers = "vectordb"
        args.src = "production_db"

    sas = _generate_sas(
        account=account,
        key=key,
        container=args.container,
        expiry_hours=args.expiry_hours,
    )

    src_prefix = args.src.strip("/")

    if args.local:
        # Local download
        dst_path = Path(args.local)
        dst_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "azcopy",
            "copy",
            f"https://{account}.blob.core.windows.net/{args.container}/{src_prefix}/*?{sas}",
            str(dst_path),
            "--recursive=true",
        ]
        if args.dry_run:
            return 0
        return _run(cmd)

    if args.dst:
        # Copy within container
        dst_prefix = args.dst.strip("/")
        cmd = [
            "azcopy",
            "copy",
            f"https://{account}.blob.core.windows.net/{args.container}/{src_prefix}/*?{sas}",
            f"https://{account}.blob.core.windows.net/{args.container}/{dst_prefix}?{sas}",
            "--exclude-pattern",
            "*embeddings*",
            "--recursive=true",
        ]
        if args.dry_run:
            return 0
        return _run(cmd)

    # Delete operation
    if not args.yes:
        confirm = input(
            f"Are you sure you want to delete {src_prefix}? [y/N]: "
        ).strip()
        if confirm.lower() != "y":
            return 0

    cmd = [
        "azcopy",
        "remove",
        f"https://{account}.blob.core.windows.net/{args.container}/{src_prefix}?{sas}",
        "--recursive=true",
    ]
    if args.dry_run:
        return 0
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
