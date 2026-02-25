"""Mode enumeration for different report types."""

import re
from enum import Enum


class Mode(Enum):
    """Enumeration of report modes: Aviation, Rail, and Marine."""

    a = 0  # Aviation
    r = 1  # Rail
    m = 2  # Marine

    @classmethod
    def as_string(cls, mode):
        """Convert mode to human-readable string.

        Args:
            mode: The mode to convert.

        Returns:
            String representation of the mode ("Aviation", "Rail", or "Marine"), or None if invalid.
        """
        if mode == cls.a:
            return "Aviation"
        if mode == cls.r:
            return "Rail"
        if mode == cls.m:
            return "Marine"
        return None

    @classmethod
    def as_char(cls, mode):
        """Convert mode to single character.

        Args:
            mode: The mode to convert.

        Returns:
            Single character representation of the mode ('a', 'r', or 'm').
        """
        return Mode.as_string(mode).lower()[0]


all_modes = [Mode.a, Mode.r, Mode.m]


def get_report_mode_from_id(report_id: str):
    """Extract report mode from report ID string.

    Args:
        report_id: The report ID string.

    Returns:
        The Mode enum value extracted from the ID, or None if no valid mode found.
    """
    if match := re.search(r"_([amr])_", report_id):
        return Mode[match.group(1)]
    # Leaving in the old id format for backwards compatibility. Once integration of ATSB and TSB is complete and the test sets are updated this can be removed.
    if match := re.search(r"(\d{4})_(\d{3})", report_id):
        return Mode(int(match.group(2)[0]))
    return None
