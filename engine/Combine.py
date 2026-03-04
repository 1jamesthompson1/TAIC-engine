"""A module that takes in all the individual data outputs of the engine and formats them into a single long dataframe."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LongDataFormatPaths:
    """Group file paths required for creating long format report data.

    This structure encapsulates all engine output paths required to combine
    reports into a single long dataframe for downstream embedding and analysis.
    """

    parsed_reports_path: Path
    report_titles_path: Path
    extracted_reports_path: Path
    atsb_safety_issues_path: Path
    tsb_recommendations_path: Path
    taic_recommendations_path: Path
    report_event_types_path: Path


def create_long_data_format(paths: LongDataFormatPaths, output_path: Path) -> None:
    """Collate all engine output data into a long format for easier embedding and analysis.

    The output will be a dataframe that is in a long format, where each row corresponds to a single docuemnt (which could be of any time, e.g section, safety issues, recommendation, summary, etc). Along with the document its id it will also contain metadata for the report it is associated with.

    Args:
        paths: Grouped paths to the parsed reports, report titles, extracted reports,
            safety issues, recommendations, and report event types.
        output_path: Path where the collated long format dataframe will be saved.

    Returns:
        None. However it will output a file to the specified output path containing the collated long format dataframe.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
