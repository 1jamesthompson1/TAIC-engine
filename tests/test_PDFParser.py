"""PDF Parser tests with stable test data.

This module tests PDF parsing functionality using a dedicated Azure container
(test-stable-reportpdfs) that contains a consistent set of test PDFs.

Setting up the stable test container:
====================================

To populate the stable container with test PDFs, you can use this script:

```python
import logging
from pathlib import Path
from engine.utils.AzureStorage import PDFStorageManager

# Create connection to stable container
stable_manager = PDFStorageManager(
    os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
    os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
    "test-stable-reportpdfs"
)

# Upload test PDFs (replace with your actual test PDF files)
test_pdfs = [
    "ATSB_a_2007_030.pdf",
    "ATSB_a_2002_646.pdf",
    "TSB_m_2021_A0041.pdf",
    "TSB_a_2011_F0012.pdf",
    "TAIC_r_2004_121.pdf",
    "TAIC_r_2014_103.pdf",
    "TAIC_a_2019_006.pdf"
]

for pdf_file in test_pdfs:
    if os.path.exists(pdf_file):
        with open(pdf_file, 'rb') as f:
            pdf_data = f.read()
        report_id = pdf_file.replace('.pdf', '')
        stable_manager.upload_pdf(report_id, pdf_data, overwrite=True)
        print(f"Uploaded {report_id}")
```

The stable container is NOT subject to automatic cleanup and will retain
its contents between test runs for consistent testing.
"""

import logging

import pytest

from engine import PDFParsing, SavedDataFrames

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "report_id, expected_pages",
    [
        ("ATSB_r_2010_007", 13),
        ("ATSB_r_2021_004", 107),
        ("ATSB_a_2007_030", 30),
        ("TSB_m_2021_A0041", 57),
        ("TAIC_r_2004_121", 25),
        ("TAIC_r_2014_103", 53),
        ("TAIC_a_2019_006", 73),
    ],
)
def test_single_pdf_parsing(stable_pdf_storage_manager, report_id, expected_pages):
    """Test that the parsing of the PDF has the write number of pages.

    Args:
        stable_pdf_storage_manager: PDF storage manager for the stable Azure container
        report_id : ID of the report to test
        expected_pages: Expected number of pages in the PDF
    """
    extracted_text = PDFParsing.pdf_to_text(stable_pdf_storage_manager, report_id)

    assert extracted_text is not None, "Extracted text should not be None"

    # Check that the extracted text contains the expected number of pages
    page_numbers = PDFParsing.PDF_PAGE_MARKER_REGEX.findall(extracted_text)

    assert (
        len(page_numbers) == expected_pages
    ), f"Expected {expected_pages} pages, but found {len(page_numbers)} in {report_id}"


def test_handling_of_nonexistent_pdf(stable_pdf_storage_manager):
    """Test that the parser handles a non-existent PDF gracefully."""
    non_existent_report_id = "non_existent_report"
    extracted_text = PDFParsing.pdf_to_text(
        stable_pdf_storage_manager, non_existent_report_id
    )

    assert extracted_text is None, "Extracted text should be None for non-existent PDF"


def test_process_all_pdfs_into_text(tmp_path, stable_pdf_storage_manager):
    """Test PDF parsing processor using PDFs from the stable Azure container.

    This tests that it can find the reports. Download them, then process them into text and save the results in a dataframe.

    Also checks if it handle the case where the PDF can't be found.
    """
    parsed_reports_dc = SavedDataFrames.ParsedReports(tmp_path)
    # Check how many PDFs are in the stable container
    pdf_list = stable_pdf_storage_manager.list_pdfs()
    logger.info("Found %s PDFs in stable container: %s", len(pdf_list), pdf_list)

    # Use the stable PDF storage manager for consistent test data
    PDFParsing.process_all_pdfs_into_text(
        parsed_reports_dc=parsed_reports_dc,
        refresh=True,
        pdf_storage_manager=stable_pdf_storage_manager,
        max_workers=1,  # Sequential for testing
    )

    assert parsed_reports_dc.path.exists()

    parsed_reports_df = parsed_reports_dc.read()
    logger.info("Parsed %s reports", len(parsed_reports_df))
    logger.info("Parsed reports dataframe:\n%s", parsed_reports_df)

    if not parsed_reports_df.empty:
        # This number is based on the current contents of the stable container which is updated with `notebooks/admin/creating_test_data.ipynb`.
        expected_report_count = 11
        assert len(parsed_reports_df) == expected_report_count

        logger.info(
            "Successfully processed %s reports from stable container",
            len(parsed_reports_df),
        )
    else:
        pytest.skip(
            "No PDFs found in stable container - please populate test-stable-reportpdfs container"
        )
