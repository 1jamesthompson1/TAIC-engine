"""PDF Parser module using PyMuPDF for reliable text extraction with page tracking.

This module provides tools to extract text from PDFs with accurate page number tracking.
Uses pymupdf4llm for better structure preservation and markdown output.
"""

import logging
import os
import re

import pandas as pd
import pymupdf4llm
from tqdm import tqdm

from ..utils.AzureStorage import PDFStorageManager

# Configure logger
logger = logging.getLogger(__name__)

# Batch size for saving parsed reports to disk
BATCH_SAVE_SIZE = 50

PDF_PAGE_MARKER_REGEX = re.compile(r"<< PDF Page (\d+) start >>")


def process_all_pdfs_into_text(
    parsed_reports_df_file_name, refresh, pdf_storage_manager: PDFStorageManager
):
    """Convert PDFs to text and save as dataframe.

    Args:
        parsed_reports_df_file_name: Path to save/load the pickle dataframe
        refresh: Whether to reprocess all PDFs
        pdf_storage_manager: Azure storage manager for accessing PDFs
    """
    logger.info(
        "=" * 150
        + "\n"
        + "|" * 150
        + "\n"
        + "Converting PDFs to text"
        + "|" * 150
        + "\n"
        + "=" * 150
    )

    logger.debug("Using PDF storage manager (streaming from cloud)")
    logger.debug(f"Output file: {parsed_reports_df_file_name}")
    logger.debug(f"Refresh: {refresh}")

    # Get PDFs from cloud storage
    report_ids = pdf_storage_manager.list_pdfs()
    if not report_ids:
        logger.warning("No PDFs found in storage container.")
        return
    logger.info(f"Found {len(report_ids)} PDFs in storage container")

    if os.path.exists(parsed_reports_df_file_name) and not refresh:
        parsed_reports_df = pd.read_pickle(parsed_reports_df_file_name)
    else:
        parsed_reports_df = pd.DataFrame(columns=["report_id", "text"])

    new_parsed_reports = []

    logger.info(
        f"Parsing {len(report_ids)} reports, there are currently {len(parsed_reports_df)} reports in the parsed reports dataframe"
    )

    for report_id in (pbar := tqdm(report_ids)):
        pbar.set_description(
            f"Extracting text from report PDFs, currently processing {report_id}"
        )

        # Skip if already processed
        if report_id in parsed_reports_df["report_id"].to_numpy():
            continue

        report_text = pdf_to_text(pdf_storage_manager, report_id)

        if report_text is None:
            logger.error(f"Failed to extract text from {report_id}, skipping")
            continue

        new_parsed_reports.append(
            {
                "report_id": report_id,
                "text": report_text,
            }
        )

        if len(new_parsed_reports) > BATCH_SAVE_SIZE:
            parsed_reports_df = pd.concat(
                [parsed_reports_df, pd.DataFrame(new_parsed_reports)],
                ignore_index=True,
            )
            parsed_reports_df.to_pickle(parsed_reports_df_file_name)
            logger.debug(
                f"Saving {len(new_parsed_reports)} reports to {parsed_reports_df_file_name}. There are now {len(parsed_reports_df)} reports in the parsed dataframe."
            )
            new_parsed_reports = []

    if len(new_parsed_reports) > 0:
        parsed_reports_df = pd.concat(
            [parsed_reports_df, pd.DataFrame(new_parsed_reports)], ignore_index=True
        )
        parsed_reports_df.to_pickle(parsed_reports_df_file_name)

    logger.info(f"Completed: {len(parsed_reports_df)} total reports in dataframe")


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

        text = extract_text_from_pdf(temp_pdf_path)
        return clean_extracted_text(text)
    except Exception:
        logger.exception(f"Error processing {report_id}")
        return None
    finally:
        if "temp_pdf_path" in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


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
            "--- Page 0 start ---\n" + text_with_start_of_page_markers
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
