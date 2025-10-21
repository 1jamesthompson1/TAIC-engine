"""
Simple version tracking utility for TAIC smart assistant.
Reads version from pyproject.toml and provides compatibility checking.
"""

import re
from pathlib import Path


def get_current_version() -> str:
    """Get current version from pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with pyproject_path.open() as f:
        content = f.read()

    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if version_match:
        return version_match.group(1)
    return "0.1.0"


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch)"""
    try:
        parts = version.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return (0, 1, 0)


def is_compatible(
    stored_version: str,
    current_version: str | None = None,
) -> tuple[bool, str]:
    """
    Check if stored version is compatible with current version.

    Rules:
    - No version = incompatible (error)
    - Major version differences = incompatible
    - Minor/patch differences = compatible but warn

    Returns: (is_compatible, message)
    """
    if current_version is None:
        current_version = get_current_version()

    if not stored_version:
        return (
            False,
            "No version information found. This data was created before version tracking was implemented. Please migrate or recreate this data.",
        )

    stored_major, stored_minor, stored_patch = parse_version(stored_version)
    current_major, current_minor, current_patch = parse_version(current_version)

    if stored_major != current_major:
        return (
            False,
            f"Incompatible version: stored={stored_version}, current={current_version}. Major version differences may cause issues.",
        )

    if stored_minor != current_minor or stored_patch != current_patch:
        return (
            True,
            f"Version difference detected: stored={stored_version}, current={current_version}. Data should load but may have minor compatibility issues.",
        )

    return True, ""


# Module-level constants
CURRENT_VERSION = get_current_version()
