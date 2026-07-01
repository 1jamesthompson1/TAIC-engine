"""PDF Parser module using PyMuPDF for reliable text extraction with page tracking.

This module provides tools to extract text from PDFs with accurate page number tracking.
Uses pymupdf4llm for better structure preservation and markdown output.
"""

import logging
import re
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pymupdf4llm
from tqdm import tqdm

from engine.AzureStorage import PDFStorageManager
from engine.SavedDataFrames import ParsedReports

# Configure logger
logger = logging.getLogger(__name__)

# Batch size for saving parsed reports to disk
SAVE_BATCH_SIZE = 50

# Minimum length of extracted text to consider it valid (to filter out failed extractions)
MIN_EXTRACTED_TEXT_LENGTH = 1000

PDF_PAGE_MARKER_REGEX = re.compile(r"<< PDF Page (\d+) start >>")


def process_all_pdfs_into_text(
    parsed_reports_dc: ParsedReports,
    refresh: bool,
    pdf_storage_manager: PDFStorageManager,
    max_workers: int | None = None,
) -> None:
    """Convert PDFs to text and save as dataframe.

    Args:
        parsed_reports_dc: The ParsedReports data frame manager instance
        refresh: Whether to reprocess all PDFs
        pdf_storage_manager: Azure storage manager for accessing PDFs
        max_workers: Maximum number of parallel workers for PDF processing
    """
    logger.welcome(
        "Converting PDFs to text",
        {
            "Output file": parsed_reports_dc.path,
            "Refresh": refresh,
        },
    )

    # Get PDFs from cloud storage
    report_ids = pdf_storage_manager.list_pdfs()
    if not report_ids:
        logger.warning("No PDFs found in storage container.")
        return
    logger.info(f"Found {len(report_ids)} PDFs in storage container")

    if refresh:
        parsed_reports_df = parsed_reports_dc.create_empty()
    else:
        parsed_reports_df = parsed_reports_dc.read_or_create()

    logger.info(
        f"Parsing {len(report_ids)} reports, there are currently {len(parsed_reports_df)} reports in the parsed reports dataframe"
    )

    # Filter out already processed reports
    reports_to_process = [
        rid
        for rid in report_ids
        if rid not in parsed_reports_df["report_id"].to_numpy()
    ]
    logger.info(f"Need to process {len(reports_to_process)} new reports")

    if not reports_to_process:
        logger.info(f"Completed: {len(parsed_reports_df)} total reports in dataframe")
        return

    batch_reports = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_report = {
            executor.submit(pdf_to_text, pdf_storage_manager, report_id): report_id
            for report_id in reports_to_process
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_report),
            total=len(reports_to_process),
            desc="Processing PDFs",
        ):
            report_id = future_to_report[future]
            parsed_reports_df = _process_future_result(
                future,
                report_id,
                batch_reports,
                parsed_reports_df,
                parsed_reports_dc,
            )

    # Save any remaining reports
    if len(batch_reports) > 0:
        parsed_reports_df = pd.concat(
            [parsed_reports_df, pd.DataFrame(batch_reports)],
            ignore_index=True,
        )
        parsed_reports_dc.save(parsed_reports_df)
        logger.debug(f"Saved final batch of {len(batch_reports)} reports")

    logger.info(f"Completed: {len(parsed_reports_df)} total reports in dataframe")


def _process_future_result(
    future: Future,
    report_id: str,
    batch_reports: list[dict[str, str]],
    parsed_reports_df: pd.DataFrame,
    parsed_reports_dc: ParsedReports,
) -> pd.DataFrame:
    """Process the result of a single PDF processing future.

    Returns:
        Updated dataframe with the processed report added.
    """
    try:
        report_text = future.result()
        if report_text is None:
            return parsed_reports_df

        batch_reports.append(
            {
                "report_id": report_id,
                "text": report_text,
            }
        )

        # Save in batches
        if len(batch_reports) >= SAVE_BATCH_SIZE:
            parsed_reports_df = pd.concat(
                [parsed_reports_df, pd.DataFrame(batch_reports)],
                ignore_index=True,
            )
            parsed_reports_dc.save(parsed_reports_df)
            logger.debug(f"Saved batch of {len(batch_reports)} reports")
            batch_reports.clear()

    except Exception:
        logger.exception(f"Error processing {report_id}")

    return parsed_reports_df


def pdf_to_text(pdf_manager: PDFStorageManager, report_id: str) -> str | None:
    """Helper function to convert a single PDF to text.

    Args:
        pdf_manager: PDFStorageManager instance for accessing PDFs
        report_id: ID of the report/PDF to convert

    Returns:
        Extracted text from the PDF
    """
    try:
        if not pdf_manager.pdf_exists(report_id):
            logger.error(f"PDF {report_id} does not exist in storage")
            return None
        temp_pdf_path = pdf_manager.stream_pdf_to_temp_file(report_id)

        if temp_pdf_path is None:
            logger.error(f"Failed to download {report_id} from storage")
            return None

        # check pdf is not empty
        if Path(temp_pdf_path).stat().st_size == 0:
            logger.error(f"Downloaded PDF {report_id} is empty")
            return None

        text = extract_text_from_pdf(temp_pdf_path)

        if len(text) < MIN_EXTRACTED_TEXT_LENGTH:
            logger.warning(
                f"Extracted text from {report_id} is unusually short ({len(text)} characters) and is being ignored"
            )
            return None
        return clean_extracted_text(text)
    except Exception:
        logger.exception(f"Error processing {report_id}")
        return None
    finally:
        if "temp_pdf_path" in locals() and temp_pdf_path:
            temp_path = Path(temp_pdf_path)
            if temp_path.exists():
                temp_path.unlink()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF with accurate page tracking.

    This uses PyMuPDF's markdown output to preserve structure. Then it uses a series of custom regex replacements to clean up the page numbers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Extracted text with page markers and markdown formatting
    """
    try:
        text_md = pymupdf4llm.to_markdown(pdf_path, page_separators=True)

        # MOve the end of page markers to be start of page markers ("--- end of page=n ---") to ("--- start of page=(n+1) ---")
        text_with_start_of_page_markers = re.sub(
            r"--- end of page=(\d+) ---",
            lambda match: f"<< PDF Page {int(match.group(1)) + 1} start >>",
            text_md,
        )

        # Add a start of page marker at the very beginning of the document
        text_with_complete_start_of_page_markers = (
            "<< PDF Page 0 start >>\n" + text_with_start_of_page_markers
        )

        # Remove the last page marker as it is not a start of page marker anyomre.
        return re.sub(
            r"--- Page \d+ start ---\s*$",
            "",
            text_with_complete_start_of_page_markers,
        )

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return ""


def clean_extracted_text(text: str) -> str:
    """Clean unusual characters from the extracted text.

    Replaces special Unicode characters with their ASCII equivalents for consistency.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text with standardized characters
    """
    characters_to_replace = [
        ("–", "-"),  # noqa: RUF001
        ("’", "'"),  # noqa: RUF001
        ("‘", "'"),  # noqa: RUF001
        ("“", '"'),
        ("”", '"'),
        ("›", ">"),  # noqa: RUF001
        ("‹", "<"),  # noqa: RUF001
    ]

    for old_char, new_char in characters_to_replace:
        text = text.replace(old_char, new_char)

    return text
