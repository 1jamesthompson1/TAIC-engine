"""A module for extracting structured information from accident investigation reports.

This includes extracting safety issues, recommendations, and chunking the report into sections based on page numbers. The safety issue and recommendation extracting is done by AI. While chunking is a simple recursive text splitter.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import regex as re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tqdm import tqdm

from engine import Modes
from engine.AICaller import ai_caller
from engine.ExtractionModels import (
    build_extraction_output_model,
)
from engine.ExtractionPrompts import PromptBuilder
from engine.Logging import get_logger
from engine.SavedDataFrames import ExtractedReports, ParsedReports, ReportTitles

logger = get_logger(__name__)


class InvalidExtractionConfigError(ValueError):
    """Raised when none of the extraction flags (safety_issues, recommendations, metadata) are True."""

    def __init__(self) -> None:
        """Initialize the exception with an appropriate error message."""
        super().__init__(
            "At least one of safety_issues, recommendations, or metadata must be True"
        )


def load_event_type_taxonomy(
    event_types_csv_path: Path,
) -> dict[Modes.Mode, list[dict[str, str]]]:
    """Load and group allowed event types (with descriptions) by mode.

    Args:
        event_types_csv_path: Path to the taxonomy CSV (for example, data/event_types.csv).

    Returns:
        dict[Modes.Mode, list[dict[str, str]]]:
            Allowed event types by report mode, where each entry contains:
            - event_type: taxonomy value
            - description: short guidance text for the value
    """
    if not event_types_csv_path.exists():
        logger.warning("Event taxonomy file not found at %s", event_types_csv_path)
        return {}

    event_types_df = pd.read_csv(event_types_csv_path)
    required_columns = {"Value", "mode", "description"}
    if not required_columns.issubset(event_types_df.columns):
        logger.warning(
            "Event taxonomy is missing required columns %s in %s",
            sorted(required_columns),
            event_types_csv_path,
        )
        return {}

    mode_name_map = {
        "aviation": Modes.Mode.a,
        "rail": Modes.Mode.r,
        "marine": Modes.Mode.m,
    }

    grouped: dict[Modes.Mode, list[dict[str, str]]] = {}
    for mode_name, rows in event_types_df.groupby(event_types_df["mode"].str.lower()):
        mode = mode_name_map.get(mode_name)
        if mode is None:
            continue

        entries: list[dict[str, str]] = []
        seen_event_types: set[str] = set()
        for _, row in rows.iterrows():
            event_type = str(row.get("Value", "")).strip()
            if not event_type or event_type in seen_event_types:
                continue

            entries.append(
                {
                    "event_type": event_type,
                    "description": str(row["description"].strip()),
                }
            )
            seen_event_types.add(event_type)

        if entries:
            grouped[mode] = entries

    return grouped


def ai_read_report(  # noqa: PLR0913, PLR0917
    agency_name: str,
    report_text: str,
    safety_issues: bool,
    recommendations: bool,
    metadata: bool = True,
    report_mode: Modes.Mode | None = None,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[dict[str, str]]] | None = None,
    report_id: str | None = None,
    agency_id: str | None = None,
) -> BaseModel:
    """Use AI to read the report and extract safety issues, recommendations, and metadata as needed.

    Args:
        agency_name: The name of the investigation agency (e.g., 'TAIC', 'TSB', 'ATSB').
        report_text: The full text of the report.
        safety_issues: Whether to extract safety issues using AI.
        recommendations: Whether to extract recommendations using AI.
        metadata: Whether to extract occurrence metadata using AI (default: True).
        report_mode: Optional mode parsed from report ID to constrain taxonomy.
        event_type_taxonomy_by_mode: Optional taxonomy entries keyed by mode.
        report_id: The identifier of the report being processed (e.g., 'TAIC_a_2023_011').
        agency_id: Agency-native occurrence identifier from report titles (for example,
            ATSB short occurrence ID) used to disambiguate multi-occurrence bulletins.

    Returns:
        A BaseModel containing the extracted safety issues, recommendations, and/or metadata,
        depending on the input flags.

    Raises:
        InvalidExtractionConfigError: If none of safety_issues, recommendations, or metadata is True.
        RuntimeError: If the AI query fails or if the output model cannot be built.

    """
    if not any([safety_issues, recommendations, metadata]):
        raise InvalidExtractionConfigError()

    # Build prompt using the prompt builder
    builder = PromptBuilder(
        agency_name=agency_name,
        report_id=report_id,
        agency_id=agency_id,
        safety_issues=safety_issues,
        recommendations=recommendations,
        metadata=metadata,
    )

    # Get taxonomy for metadata extraction
    taxonomy = None
    if metadata and report_mode and event_type_taxonomy_by_mode:
        taxonomy = event_type_taxonomy_by_mode.get(report_mode)

    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(report_mode, report_text, taxonomy)

    # Build output model
    try:
        output_model = build_extraction_output_model(
            safety_issues_enabled=safety_issues,
            recommendations_enabled=recommendations,
            metadata_enabled=metadata,
            report_mode=report_mode,
            event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        )
    except Exception as e:
        msg = f"Failed to build output model for agency '{agency_name}' with report ID '{report_id}'"
        raise RuntimeError(msg) from e

    try:
        response = ai_caller.query(
            model="gpt-5.4-mini",
            reasoning="high",
            system=system_prompt,
            user=user_prompt,
            output_structure=output_model,
        )
    except Exception as e:
        msg = f"AI extraction failed for agency '{agency_name}': {type(e).__name__}"
        raise RuntimeError(msg) from e

    return response


def chunk_report_into_sections(report_text: str) -> list[dict[str, str]]:
    """Chunks the report into sections based on headings and returns section dicts.

    Args:
        report_text (str): The full text of the report.

    Returns:
        list[dict[str, str]]: A list of section records, each with "page" and "text" keys.

    """
    page_regex = re.compile(r"<< Page (\d+|[xvi]+) >>")
    page_matches = list(page_regex.finditer(report_text))

    position_to_page = {}
    for match in page_matches:
        position_to_page[match.start()] = match.group(1)

    reporter_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    sections = reporter_splitter.split_text(report_text)

    chunks_with_pages = []
    page_for_sections = []
    search_from = 0

    for section in sections:
        section_start = report_text.find(section, search_from)
        if section_start == -1:
            section_start = report_text.find(section)
        else:
            search_from = section_start + len(section)

        current_page = "pre"
        for pos in sorted(position_to_page.keys()):
            if pos <= section_start:
                current_page = position_to_page[pos]
            else:
                break

        page_for_sections.append((current_page, section))

    page_totals = {}
    for page, _section in page_for_sections:
        page_totals[page] = page_totals.get(page, 0) + 1

    page_counts = {}
    for page, section in page_for_sections:
        page_counts[page] = page_counts.get(page, 0) + 1
        suffix = page_counts[page]
        page_key = f"page_{page}.{suffix}" if page_totals[page] > 1 else f"page_{page}"
        chunks_with_pages.append({"section": page_key, "text": section})

    return chunks_with_pages


def extract_report(
    report_row: pd.Series | dict,
    ai_extraction_config: dict,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[dict[str, str]]] | None = None,
) -> dict:
    """Extract safety issues, recommendations, metadata, and sections from a report row.

    Args:
        report_row: Row containing 'report_id' and 'report_text'.
        ai_extraction_config: Configuration dictionary specifying which extraction tasks
            (safety_issues, recommendations, metadata) to perform for each agency.
        event_type_taxonomy_by_mode: Allowed occurrence-type entries per mode.

    Returns:
        dict: Extracted fields with keys 'report_id', 'safety_issues',
              'recommendations', 'metadata', and 'sections'.

    Raises:
        ValueError: If extracted metadata appears invalid or incomplete such that
            the report text is likely not properly scraped.

    """
    report_id = report_row["report_id"]
    report_text = report_row["text"]
    agency = report_id.split("_")[0]
    report_mode = Modes.get_report_mode_from_id(report_id)
    agency_id = report_row.get("agency_id")

    extraction_config = ai_extraction_config[agency].copy()

    if agency == "ATSB":
        report_year = int(report_id.split("_")[2])
        atsb_mode_add_dates = {
            Modes.Mode.a: 2008,
            Modes.Mode.r: 2007,
            Modes.Mode.m: 2009,
        }
        if report_year < atsb_mode_add_dates[Modes.Mode.m]:
            extraction_config["safety_issues"] = True

    extracted_data = ai_read_report(
        agency_name=agency,
        report_text=report_text,
        report_mode=report_mode,
        report_id=report_id,
        agency_id=agency_id,
        event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        **extraction_config,
    )

    safety_issues = getattr(extracted_data, "safety_issues", [])
    recommendations = getattr(extracted_data, "recommendations", [])
    metadata = getattr(extracted_data, "metadata", None)

    safety_issues = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in safety_issues
    ]
    recommendations = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in recommendations
    ]

    if metadata and hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    elif metadata and not isinstance(metadata, dict):
        metadata = None
    else:
        metadata = metadata or {}

    # Make sure that atleat the accident type, location and date are simultanously non-None treat it as a failed metadata extraction and asume the report text is actually not scrapped properly.
    required_metadata_fields = [
        "occurrence_type",
        "total_persons_involved",
        "who_may_benefit",
    ]
    occurrence_metadata = metadata["occurrence"] if metadata else {}
    if (
        occurrence_metadata
        and all(
            field in occurrence_metadata and occurrence_metadata[field] is None
            for field in required_metadata_fields
        )
        and occurrence_metadata.get("location", {}).get("standardised_location") is None
    ):
        logger.warning(
            f"Metadata for report {report_id} is missing required fields or has None values. Treating as failed extraction. Extracted metadata: {occurrence_metadata}"
        )
        raise ValueError(  # noqa: TRY003
            f"Report {report_id} has invalid metadata extraction which likely indicates the report text was not properly scraped. Metadata: {occurrence_metadata}"
        )

    sections = chunk_report_into_sections(report_text)

    return {
        "report_id": report_id,
        "safety_issues": safety_issues if len(safety_issues) > 0 else None,
        "recommendations": recommendations if len(recommendations) > 0 else None,
        "metadata": metadata,
        "sections": sections,
    }


def process_reports_parallel(  # noqa: PLR0912, PLR0913, PLR0917
    parsed_reports_dc: ParsedReports,
    extracted_reports_dc: ExtractedReports,
    ai_extraction_config: dict,
    report_titles_dc: ReportTitles | None = None,
    event_types_csv_path: Path = Path("data/event_types.csv"),
    max_workers: int | None = None,
    save_interval: int = 100,
) -> pd.DataFrame:
    """Process reports in parallel, skipping those already extracted.

    This function loads input data from disk, identifies reports that need
    processing, processes new reports in parallel using ThreadPoolExecutor,
    and writes the updated extraction results back to disk.

    Args:
        parsed_reports_dc: ParsedReports instance for reading parsed reports.
        extracted_reports_dc: ExtractedReports instance for reading/writing
            extracted reports.
        ai_extraction_config: Configuration dictionary specifying which extraction tasks
            (safety_issues, recommendations, metadata) to perform for each agency.
        report_titles_dc: Optional ReportTitles data container. When provided,
            this function reads it and merges 'agency_id' into parsed reports by
            'report_id'. Used for disambiguating multi-occurrence bulletins (for example, ATSB) during AI extraction.
        event_types_csv_path: Path to taxonomy CSV used for occurrence_type values.
        max_workers: Maximum number of parallel workers. Adjust
            based on your API rate limits and system resources.
        save_interval: Number of reports to process before saving progress to disk.

    Returns:
        pd.DataFrame: Updated DataFrame containing both previously extracted
        and newly extracted reports. Columns include:
            - report_id: Unique identifier for the report
            - safety_issues: List of extracted safety issues
            - recommendations: List of extracted recommendations
            - metadata: Dictionary of extracted occurrence and mode-specific metadata
            - sections: Dictionary of report sections

    Example:
        >>> result = process_reports_parallel(
        ...     ParsedReports(Path("output")),
        ...     ExtractedReports(Path("output")),
        ...     max_workers=8,
        ... )
        >>> # Only new reports are processed, existing data preserved

    """
    reports_df = parsed_reports_dc.read()

    if report_titles_dc is not None:
        report_titles_df = report_titles_dc.read()
    else:
        report_titles_df = pd.DataFrame(columns=["report_id", "agency_id"])

    if len(report_titles_df) > 0:
        titles_for_merge = report_titles_df[["report_id", "agency_id"]].drop_duplicates(
            subset=["report_id"],
            keep="last",
        )
        reports_df = reports_df.merge(titles_for_merge, on="report_id", how="left")
    else:
        reports_df = reports_df.copy()
        reports_df["agency_id"] = None

    event_type_taxonomy_by_mode = load_event_type_taxonomy(event_types_csv_path)

    current_extracted_df = extracted_reports_dc.read_or_create()

    if len(current_extracted_df) > 0:
        already_processed = set(current_extracted_df["report_id"])
        reports_to_process = reports_df[
            ~reports_df["report_id"].isin(already_processed)
        ]
    else:
        reports_to_process = reports_df

    logger.welcome(
        "Extracting Report Data",
        {
            "Output file": str(extracted_reports_dc.path),
            "Reports to process": str(len(reports_to_process)),
            "Already processed": str(len(reports_df) - len(reports_to_process)),
        },
    )

    if len(reports_to_process) == 0:
        return current_extracted_df

    results = []

    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_report = {
            executor.submit(
                extract_report,
                row,
                ai_extraction_config,
                event_type_taxonomy_by_mode,
            ): row["report_id"]
            for _, row in reports_to_process.iterrows()
        }

        for future in tqdm(
            as_completed(future_to_report),
            total=len(future_to_report),
            desc="Processing",
        ):
            report_id = future_to_report[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(f"Error processing {report_id}: {exc}", exc_info=True)
                errors.append(f"{report_id}: {exc}")
                continue
            if result is not None:
                results.append(result)

            if len(results) > 0 and len(results) % save_interval == 0:
                _save_progress(results, current_extracted_df, extracted_reports_dc)

    _save_progress(results, current_extracted_df, extracted_reports_dc)

    if len(errors) > 0:
        for err in errors:
            logger.warning(err)
        logger.warning(f"Completed with {len(errors)} missed reports due to errors.")

    return extracted_reports_dc.read()


def _save_progress(
    results: list,
    current_extracted_df: pd.DataFrame,
    extracted_reports_dc: ExtractedReports,
) -> pd.DataFrame | None:
    if len(results) == 0:
        return None

    new_df = pd.DataFrame(results)
    if len(current_extracted_df) > 0:
        combined = pd.concat([current_extracted_df, new_df], ignore_index=True)
    else:
        combined = new_df

    extracted_reports_dc.save(combined)
    logger.info(
        f"Saved {len(combined)} total reports ({len(new_df)} new in this batch)"
    )
    return combined
