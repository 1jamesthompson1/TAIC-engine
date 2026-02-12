"""PDF Parser tests with stable test data.

This module tests PDF parsing functionality using a dedicated Azure container
(test-stable-reportpdfs) that contains a consistent set of test PDFs.

Setting up the stable test container:
====================================

To populate the stable container with test PDFs, you can use this script:

```python
import os
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

import os
import re

import pandas as pd
import pytest
import roman

import engine.gather.PDFParser as PDFParser


def expectedPageNumbersBuilder(roman_numerals: int, int_range: tuple[int, int]):
    """Build a list of expected page numbers with roman numerals and integers.

    Args:
        roman_numerals (int): Number of roman numeral pages (starting from i).
        int_range (tuple[int, int]): Range of integer page numbers (inclusive start and end).

    Returns:
        list: List containing lowercase roman numerals followed by integer page numbers.

    Example:
        >>> expectedPageNumbersBuilder(3, (1, 5))
        ['i', 'ii', 'iii', 1, 2, 3, 4, 5]
    """
    romans = list(map(lambda n: roman.toRoman(n).lower(), range(1, roman_numerals + 1)))
    return romans + list(range(int_range[0], int_range[1] + 1))


@pytest.mark.parametrize(
    "report_id, expected",
    [
        pytest.param(
            "ATSB_a_2007_030",
            expectedPageNumbersBuilder(5, (1, 22)),
            id="ATSB_a_2007_030 (Incorrect roman numeral matches in text)",
        ),
        pytest.param(
            "ATSB_a_2002_646",
            expectedPageNumbersBuilder(2, (1, 26)),
            id="ATSB_a_2002_646 (Lenient regex match of roman numerals causing error)",
        ),
        pytest.param(
            "ATSB_r_2021_010",
            expectedPageNumbersBuilder(5, (6, 25)),
            id="ATSB_r_2021_010 (Uses the > num < pattern, and the ints don't start at 1)",
        ),
        pytest.param(
            "TSB_m_2021_A0041",
            expectedPageNumbersBuilder(0, (1, 56)),
            id="TSB_m_2021_A0041",
        ),
        pytest.param(
            "TSB_a_2011_F0012",
            expectedPageNumbersBuilder(0, (1, 19)),
            id="TSB_a_2011_F0012",
        ),
        pytest.param(
            "TAIC_r_2004_121",
            expectedPageNumbersBuilder(4, (1, 16)),
            id="TAIC_r_2004_121",
        ),
        pytest.param(
            "TAIC_r_2014_103",
            expectedPageNumbersBuilder(4, (1, 38)),
            id="TAIC_r_2014_103",
        ),
        pytest.param(
            "TAIC_a_2019_006",
            expectedPageNumbersBuilder(6, (1, 64)),
            id="TAIC_a_2019_006 (removing duplicate matches on the same page)",
        ),
    ],
)
def test_formatText(report_id, expected, stable_pdf_storage_manager):
    """Test PDF text formatting and page number extraction using PDFs from the stable Azure container."""
    # Download the PDF from the stable container to a temporary location
    pdf_data = stable_pdf_storage_manager.download_pdf(report_id)

    assert pdf_data is not None

    # Create a temporary file for the PDF
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_data)
        temp_pdf_path = tmp_file.name

    try:
        # Extract text from the temporary PDF file
        text = PDFParser.extractTextFromPDF(temp_pdf_path)

        # Format the text
        text = PDFParser.PageNumberSynchronizer.reconcile_pdf_and_text_page_numbers(
            PDFParser.cleanText(text), report_id.split("_")[0]
        )

        page_number_matches = list(
            re.finditer(
                r"^<< Page (\d+|[LXVI]{1,8}) >>$", text, re.MULTILINE + re.IGNORECASE
            )
        )

        matched = [
            int(match.group(1)) if match.group(1).isnumeric() else match.group(1)
            for match in page_number_matches
        ]
        print(f"found: {matched}")
        print(f"expected: {expected}")
        assert matched == expected

    finally:
        # Clean up the temporary file
        import os

        try:
            os.unlink(temp_pdf_path)
        except Exception:
            pass


def test_PDFParser(tmpdir, stable_pdf_storage_manager):
    """Test PDF parsing processor using PDFs from the stable Azure container."""
    parsed_reports_df_file_name = os.path.join(
        tmpdir.strpath,
        pytest.output_config["parsed_reports_df_file_name"],
    )

    # Check how many PDFs are in the stable container
    pdf_list = stable_pdf_storage_manager.list_pdfs()
    print(f"Found {len(pdf_list)} PDFs in stable container: {pdf_list}")

    # Use the stable PDF storage manager for consistent test data
    PDFParser.convertPDFToText(
        parsed_reports_df_file_name,
        refresh=True,
        pdf_storage_manager=stable_pdf_storage_manager,
    )

    assert os.path.exists(parsed_reports_df_file_name)

    parsed_reports_df = pd.read_pickle(parsed_reports_df_file_name)
    print(f"Parsed {len(parsed_reports_df)} reports")
    print(parsed_reports_df)

    if not parsed_reports_df.empty:
        # Assert that all processed PDFs are valid
        assert parsed_reports_df["valid"].all()

        # Assert we got the expected number of reports
        # (you can adjust this number based on what's in your stable container)
        expected_report_count = len(pdf_list)  # Should match the PDFs in the container
        assert len(parsed_reports_df) == expected_report_count

        print(
            f"Successfully processed {len(parsed_reports_df)} reports from stable container"
        )
    else:
        pytest.skip(
            "No PDFs found in stable container - please populate test-stable-reportpdfs container"
        )
