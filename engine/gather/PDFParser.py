"""PDF Parser module using PyMuPDF for reliable text extraction with page tracking.

This module provides tools to extract text from PDFs with accurate page number tracking.
Uses pymupdf4llm for better structure preservation and markdown output.
"""

import logging
import os
import re
from typing import Literal

import pandas as pd
import pymupdf4llm
import roman
from tqdm import tqdm

from ..utils.AzureStorage import PDFStorageManager

# Configure logger
logger = logging.getLogger(__name__)


# Pre-compiled regex patterns
class _RegexPatterns:
    """Container for pre-compiled regex patterns used in PDF parsing."""

    # Page marker patterns
    PDF_PAGE_MARKER = re.compile(r"^--- Page (\d+) start ---$", re.MULTILINE)

    # ATSB page number patterns
    # First one: "^ +- (\d{1,3}|[XVI]{1,6}) $"
    #
    ATSB_PRIMARY = re.compile(
        r"^ ?[->][ ]{0,2}((\d{1,3})|([XVI]{1,4}))[ ]{0,2}[-<]",
        re.IGNORECASE | re.MULTILINE,
    )
    ATSB_FALLBACK_INT = re.compile(
        r"^ ?(\d{1,3}|[XVI]{1,4}) ?$", re.MULTILINE | re.IGNORECASE
    )
    ATSB_FALLBACK_LOWER_ROMAN = re.compile(
        r"^ ?([xvi]{2,8}) ?[\w\W]{0,10}$", re.MULTILINE
    )
    ATSB_SECONDARY_ROMAN = re.compile(
        r"^ ?([XVI]{1,8}) ?$", re.MULTILINE | re.IGNORECASE
    )

    # TSB page number patterns
    TSB_PRIMARY = re.compile(r"^ ?- ?(\d+) ?$", re.IGNORECASE | re.MULTILINE)
    TSB_FALLBACK = re.compile(r"[\|■] {0,2}(\d+) ?$", re.IGNORECASE | re.MULTILINE)

    # TAIC page number pattern
    TAIC_PRIMARY = re.compile(
        r"Page {1,3}(\d+|[ivx]+) ?\|?(?!.*start)",
        re.IGNORECASE | re.MULTILINE,
    )


class PageNumberSynchronizer:
    """Synchronizes PDF page markers with document page numbers."""

    @staticmethod
    def get_atsb_page_numbers(text: str) -> list:
        """Extract page numbers from ATSB reports.

        Args:
            text (str): Extracted text from the PDF report

        Returns:
            list: List of regex match objects for page numbers
        """
        page_number_matches = list(_RegexPatterns.ATSB_PRIMARY.finditer(text))

        if not page_number_matches:
            page_number_matches = list(_RegexPatterns.ATSB_FALLBACK_INT.finditer(text))
            if all(match.group(1).isnumeric() for match in page_number_matches):
                page_number_matches.extend(
                    _RegexPatterns.ATSB_FALLBACK_LOWER_ROMAN.finditer(text)
                )
        else:
            page_number_matches.extend(
                _RegexPatterns.ATSB_SECONDARY_ROMAN.finditer(text)
            )

        return page_number_matches

    @staticmethod
    def get_tsb_page_numbers(text: str) -> list:
        """Extract page numbers from TSB reports.

        Args:
            text (str): Extracted text from the PDF report

        Returns:
            list: List of regex match objects for page numbers
        """
        page_number_matches = list(_RegexPatterns.TSB_PRIMARY.finditer(text))
        if not page_number_matches:
            page_number_matches = list(_RegexPatterns.TSB_FALLBACK.finditer(text))

        return page_number_matches

    @staticmethod
    def get_taic_page_numbers(text: str) -> list:
        """Extract page numbers from TAIC reports.

        Args:
            text (str): Extracted text from the PDF report

        Returns:
            list: List of regex match objects for page numbers
        """
        return list(_RegexPatterns.TAIC_PRIMARY.finditer(text))

    @staticmethod
    def get_page_numbers(text: str, agency: Literal["ATSB", "TSB", "TAIC"]) -> list:
        """Get page numbers for the specified agency.

        Args:
            text (str): Document text
            agency (str): Agency type (ATSB, TSB, or TAIC)

        Returns:
            list: List of regex match objects

        Raises:
            ValueError: If unknown agency type
        """
        match agency:
            case "ATSB":
                return PageNumberSynchronizer.get_atsb_page_numbers(text)
            case "TSB":
                return PageNumberSynchronizer.get_tsb_page_numbers(text)
            case "TAIC":
                return PageNumberSynchronizer.get_taic_page_numbers(text)
            case _:
                raise ValueError(f"Unknown agency: {agency}")

    @staticmethod
    def sync_page_numbers(page_number_matches: list, pdf_page_matches: list) -> list:
        """Synchronize the PDF page numbers with the internal page numbers mentioned in the document.

        Parameters:
            page_number_matches (list): List of regex matches from the document's internal page numbers
            pdf_page_matches (list): List of regex matches from the PDF page numbers

        Returns:
            list: A list of replacement values for pdf_page_matches to update the PDF page numbers
        """
        # Default to just labelling the pages 1,n
        if len(page_number_matches) == 0:
            return [num + 1 for num in range(len(pdf_page_matches))]

        # Line up pdf page numbers with read page numbers. I need to figure out if there are pages before page one?

        # Filter out regex matched pages that are higher than actual PDF.
        page_number_matches = [
            match
            for match in page_number_matches
            if not match.group(1).isdecimal()
            or int(match.group(1))
            <= max(
                [
                    int(match.group(1)) + 10 if match.group(1).isdecimal() else 0
                    for match in pdf_page_matches
                ]
            )
        ]
        # Remove roman numerals that are found after multiple consecutive integers
        is_int = [match.group(1).isdecimal() for match in page_number_matches]

        found_start_int = False
        indices_to_delete = []
        for i in range(1, len(is_int)):
            if is_int[i] and is_int[i - 1]:
                if (
                    int(page_number_matches[i - 1].group(1))
                    == int(page_number_matches[i].group(1)) - 1
                ):
                    found_start_int = True

            if not is_int[i] and found_start_int:
                indices_to_delete.append(i)

        if len(indices_to_delete) > 0:
            page_number_matches = [
                page_number_matches[i]
                for i in range(len(page_number_matches))
                if i not in indices_to_delete
            ]

        # Remove integers that are found before a roman numeral
        is_int = [match.group(1).isdecimal() for match in page_number_matches]
        if not all(is_int):
            last_numeral = max([i for i, x in enumerate(is_int) if not x])

            kept_indicies = [
                i
                for i, x in enumerate(is_int)
                if (x and i > last_numeral) or (not x and i <= last_numeral)
            ]
            page_number_matches = [page_number_matches[i] for i in kept_indicies]

        # Remove any duplicate matches that happen on the same page
        for i in range(len(pdf_page_matches) - 1):
            page_start = pdf_page_matches[i].start()
            page_end = pdf_page_matches[i + 1].start()

            matches_in_page = [
                match
                for match in page_number_matches
                if match.span()[0] >= page_start and match.span()[1] <= page_end
            ]
            if len(matches_in_page) > 1:
                for j in range(1, len(matches_in_page)):
                    page_number_matches.remove(matches_in_page[j])

        # Find the first and last roman numeral and integer
        last_numeral = (
            max(
                i
                for i, match in enumerate(page_number_matches)
                if not match.group(1).isdecimal()
            )
            if not all(is_int)
            else -1
        )
        anchors = []
        if last_numeral != -1:
            anchors = [page_number_matches[0]]
            if last_numeral != 0:
                anchors.extend(
                    [
                        page_number_matches[last_numeral],
                    ]
                )

        if last_numeral != len(page_number_matches) - 1:
            anchors.extend(
                [
                    page_number_matches[last_numeral + 1],
                    page_number_matches[-1],
                ]
            )

        # Create a template that can be filled out for all of the correct page numbers.

        final_page_numbers = [None] * len(pdf_page_matches)
        for i in range(0, len(pdf_page_matches) - 1):
            if (
                pdf_page_matches[i].end()
                < anchors[0].start()
                < pdf_page_matches[i + 1].start()
            ):
                final_page_numbers[i] = anchors[0]
                anchors.pop(0)
                if len(anchors) == 0:
                    break

            if (
                i == len(pdf_page_matches) - 2
                and pdf_page_matches[i + 1].end() < anchors[0].start()
            ):
                final_page_numbers[i + 1] = anchors[0]
                anchors.pop(0)
                break

        final_page_numbers = PageNumberSynchronizer.populate_final_page_numbers(
            final_page_numbers
        )

        return final_page_numbers

    @staticmethod
    def populate_final_page_numbers(final_page_numbers):
        """Populate final page numbers with roman numerals and integers.

        This will take a list of None, roman numerals and integers. It will add ''
        before the roman numerals, fill in the gaps in the roman numerals and then
        fill out the integers.

        Returns:
            list: A list with populated page numbers as strings and integers.
        """
        result = []
        current_roman_numeral = None
        current_int = None

        for i in range(0, len(final_page_numbers)):
            # Append the preceding empty string if needed
            if (
                (final_page_numbers[i] is None)
                and (not current_roman_numeral)
                and (not current_int)
            ):
                result.append("")
                continue

            # Found a none
            if final_page_numbers[i] is None:
                if current_int:
                    current_int += 1
                    result.append(current_int)
                elif current_roman_numeral:
                    current_roman_numeral += 1
                    result.append(roman.toRoman(current_roman_numeral).lower())

                continue

            # Found a roman numeral
            if not final_page_numbers[i].group(1).isdecimal():
                current_roman_numeral = roman.fromRoman(
                    final_page_numbers[i].group(1).upper()
                )
                if current_roman_numeral > 1:
                    fixing_index = i - 1
                    fixing_current_roman = current_roman_numeral - 1
                    while (
                        fixing_index >= 0
                        and fixing_current_roman > 0
                        and final_page_numbers[fixing_index] is None
                    ):
                        result[fixing_index] = roman.toRoman(
                            fixing_current_roman
                        ).lower()
                        fixing_current_roman -= 1
                        fixing_index -= 1
                result.append(roman.toRoman(current_roman_numeral).lower())
                continue

            # Found a digit
            if final_page_numbers[i].group(1).isdecimal():
                current_int = int(final_page_numbers[i].group(1))
                if current_int > 1 and i > 0 and isinstance(result[i - 1], str):
                    # The first number found is not the first page. This could be an error in the PDF or the page number was missed. Therefore we will replace the result until we reach the start or roman numerals
                    fixing_index = i - 1
                    fixing_current_int = current_int - 1
                    while (
                        fixing_index >= 0
                        and fixing_current_int > 0
                        and final_page_numbers[fixing_index] is None
                    ):
                        result[fixing_index] = fixing_current_int
                        fixing_current_int -= 1
                        fixing_index -= 1
                result.append(current_int)
                continue

        return result

    @staticmethod
    def reconcile_pdf_and_text_page_numbers(
        text: str, agency: Literal["ATSB", "TSB", "TAIC"]
    ) -> str:
        """Reconcile PDF page markers with document text page numbers.

        The text contains two sets of page numbers:
        1. PDF page markers (automatically generated): "--- Page n start ---"
        2. Text page numbers (from document headers): varies by agency

        These often don't align due to front matter with different numbering (roman numerals).
        This method synchronizes them by replacing PDF markers with consistent page markers
        in the format "<< Page n >>", using the text page numbers.

        Args:
            text (str): Extracted text with page markers from both PDF and document headers
            agency (str): Agency type for page number extraction (ATSB, TSB, or TAIC)

        Returns:
            str: Text with reconciled page numbers using "<< Page n >>" format
        """
        page_number_matches = PageNumberSynchronizer.get_page_numbers(text, agency)
        pdf_page_matches = list(_RegexPatterns.PDF_PAGE_MARKER.finditer(text))

        page_number_matches.sort(key=lambda x: x.span()[0])
        replacement_numbers = PageNumberSynchronizer.sync_page_numbers(
            page_number_matches, pdf_page_matches
        )

        # Perform the replacement of the PDF page numbers with the internal page numbers
        # This process leaves the original page numbers in the text
        results = []
        last_end = 0
        pdf_to_internal_page_numbers = list(zip(pdf_page_matches, replacement_numbers))
        for page_number_match, replacement_number in pdf_to_internal_page_numbers:
            start, end = page_number_match.span()
            results.append(text[last_end:start])
            if replacement_number != "":
                results.append(f"<< Page {replacement_number} >>")
            last_end = end

        results.append(text[last_end:])
        return "".join(results)


def convertPDFToText(
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
        parsed_reports_df = pd.DataFrame(columns=["report_id", "text", "valid"])

    new_parsed_reports = []

    logger.debug(
        f"Parsing {len(report_ids)} reports, there are currently {len(parsed_reports_df)} reports in the parsed reports dataframe"
    )

    for report_id in (pbar := tqdm(report_ids)):
        pbar.set_description(
            f"Extracting text from report PDFs, currently processing {report_id}"
        )

        # Skip if already processed
        if report_id in parsed_reports_df["report_id"].values:
            continue

        try:
            # Stream from storage
            temp_pdf_path = pdf_storage_manager.stream_pdf_to_temp_file(report_id)
            if temp_pdf_path is None:
                pbar.write(f"Failed to download {report_id} from storage")
                continue

            try:
                text = extractTextFromPDF(temp_pdf_path)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_pdf_path)
                except Exception:
                    pass

            cleaned_text = cleanText(text)

            cleaned_page_numbers = (
                PageNumberSynchronizer.reconcile_pdf_and_text_page_numbers(
                    cleaned_text, report_id.split("_")[0]
                )
            )

            new_parsed_reports.append(
                {
                    "report_id": report_id,
                    "text": cleaned_page_numbers,
                }
            )

            if len(new_parsed_reports) > 50:
                parsed_reports_df = pd.concat(
                    [parsed_reports_df, pd.DataFrame(new_parsed_reports)],
                    ignore_index=True,
                )
                parsed_reports_df.to_pickle(parsed_reports_df_file_name)
                logger.debug(
                    f"Saving {len(new_parsed_reports)} reports to {parsed_reports_df_file_name}. There are now {len(parsed_reports_df)} reports in the parsed dataframe."
                )
                new_parsed_reports = []

        except Exception as e:
            pbar.write(f"Error processing {report_id}: {e}")

    if len(new_parsed_reports) > 0:
        parsed_reports_df = pd.concat(
            [parsed_reports_df, pd.DataFrame(new_parsed_reports)], ignore_index=True
        )
        parsed_reports_df.to_pickle(parsed_reports_df_file_name)

    logger.info(f"Completed: {len(parsed_reports_df)} total reports in dataframe")


def extractTextFromPDF(pdf_path: str) -> str:
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
            lambda match: f"--- Page {int(match.group(1)) + 1} start ---",
            text_md,
        )

        # Add a start of page marker at the very beginning of the document
        text_with_complete_start_of_page_markers = (
            "--- Page 0 start ---\n" + text_with_start_of_page_markers
        )

        # Remove the last page marker as it is not a start of page marker anyomre.
        text_with_complete_start_of_page_markers = re.sub(
            r"--- Page \d+ start ---\s*$",
            "",
            text_with_complete_start_of_page_markers,
        )

        return text_with_complete_start_of_page_markers

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return ""


def cleanText(text: str) -> str:
    """Clean unusual characters from the extracted text.

    Replaces special Unicode characters with their ASCII equivalents for consistency.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text with standardized characters
    """
    characters_to_replace = [
        ("–", "-"),  # en dash
        ("'", "'"),  # left single quote
        ("'", "'"),  # right single quote
        (
            """, '"'),      # left double quote
        (""",
            '"',
        ),  # right double quote
        ("›", ">"),  # right angle
        ("‹", "<"),  # left angle
    ]

    for old_char, new_char in characters_to_replace:
        text = text.replace(old_char, new_char)

    return text
