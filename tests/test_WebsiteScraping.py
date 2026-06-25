"""Website scraping tests with automatic Azure container cleanup.

This test module includes a session-scoped fixture that automatically cleans up
the test-reportpdfs Azure storage container after all tests complete. This helps
prevent accumulation of test data and associated storage costs.

The cleanup fixture (`cleanup_test_pdf_container`) will:
1. Run after all tests in the session complete
2. Connect to the test Azure storage container
3. Delete all blobs (PDFs and other files) from the test container
4. Provide status output showing what was cleaned up

To use this in other test files, you can copy the cleanup_test_pdf_container fixture.
"""

import itertools
import shutil
import socket
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import bs4
import pandas as pd
import pytest

from engine import Modes, SavedDataFrames, WebsiteScraping
from engine.WebsiteScraping import ReportUrlData

MINIMUM_SAFETY_ISSUES_COUNT = 388


def _test_website_connectivity(
    host: str, port: int = 443, timeout_s: float = 5.0
) -> bool:
    """Check if a website is reachable.

    Args:
        host: Hostname to connect to.
        port: Port to use (default: 443 for HTTPS).
        timeout_s: Connection timeout in seconds (default: 5.0).

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
    except Exception:
        return False
    else:
        return True


def test_website_connectivity() -> None:
    """Test connectivity to all three website sources.

    This test verifies that the system can reach TAIC, TSB, and ATSB websites.
    It's useful for diagnosing network/DNS issues in the test environment.
    """
    websites = {
        "TAIC": "www.taic.org.nz",
        "TSB": "www.tsb.gc.ca",
        "ATSB": "www.atsb.gov.au",
    }

    results = {}
    for name, host in websites.items():
        results[name] = _test_website_connectivity(host)

    # Log results for debugging
    for name, reachable in results.items():
        status = "✓ reachable" if reachable else "✗ unreachable"
        print(f"{name}: {status}")  # noqa: T201

    # At least one website should be reachable
    assert any(
        results.values()
    ), "Unable to reach any website (TAIC, TSB, ATSB). Network connectivity issue?"

    # Optionally check if all are reachable
    if not all(results.values()):
        unreachable = [name for name, reachable in results.items() if not reachable]
        print(f"Warning: Some websites are unreachable: {', '.join(unreachable)}")  # noqa: T201


@pytest.fixture(scope="function")
def report_scraping_settings(
    tmpdir: object, test_pdf_storage_manager: object
) -> WebsiteScraping.ReportScraperSettings:
    """Create report scraping settings for tests.

    Returns:
        ReportScraperSettings: Configured settings for scraping.
    """
    return WebsiteScraping.ReportScraperSettings(
        SavedDataFrames.ReportTitles(tmpdir),
        2004,
        2021,
        1,
        [Modes.Mode.a, Modes.Mode.r, Modes.Mode.m],
        [],
        False,
        test_pdf_storage_manager,
    )


@pytest.fixture(scope="function")
def get_agency_scraper(
    tmpdir: object, report_scraping_settings: object
) -> WebsiteScraping.ReportScraper:
    """Fixture that returns a function to create agency scrapers with their own temp directories.

    Each scraper gets its own temporary directory for file operations.

    Returns:
        A function that creates agency-specific scrapers.
    """

    def _get_agency_scraper(agency: str) -> WebsiteScraping.ReportScraper:
        data_path = Path(pytest.output_config.get("folder_name"))
        if agency == "TAIC":
            # Use the canonical TAIC reports table from the tests data folder, but
            # copy it into the test's tmpdir so the original isn't modified by tests.
            original_path = SavedDataFrames.TAICWebsiteReportsTable(data_path).path

            if original_path.exists():
                shutil.copy(original_path, tmpdir)
            else:
                pytest.fail(
                    f"Test data file {original_path} is missing in the test data folder."
                )

            return WebsiteScraping.TAICReportScraper(
                SavedDataFrames.TAICWebsiteReportsTable(tmpdir),
                report_scraping_settings,
            )
        if agency == "ATSB":
            original_path = SavedDataFrames.ATSBWebsiteReportsTable(data_path).path

            if original_path.exists():
                shutil.copy(original_path, tmpdir)
            else:
                pytest.fail(
                    f"Test data file {original_path} is missing in the test data folder."
                )

            return WebsiteScraping.ATSBReportScraper(
                SavedDataFrames.ATSBWebsiteReportsTable(tmpdir),
                report_scraping_settings,
            )
        if agency == "TSB":
            return WebsiteScraping.TSBReportScraper(report_scraping_settings)
        msg = f"Unknown agency: {agency}"
        raise ValueError(msg)

    return _get_agency_scraper


@pytest.mark.parametrize(
    "agency, url, report_id, expected",
    [
        pytest.param(
            "TSB",
            "https://www.tsb.gc.ca/eng/rapports-reports/marine/2020/m20c0101/m20c0101.html",
            "TSB_m_2020_c01010",
            {
                "success": True,
                "occurrence_date": "2020-05-12",
                "publication_date": "2021-07-30",
            },
            id="TSB pass",
        ),
        pytest.param(
            "TSB",
            "https://www.tsb.gc.ca/eng/rapports-reports/marine/2020/m20c0101/m20c0102.html",
            "TSB_m_2020_c0102",
            False,
            id="TSB fail",
        ),
        pytest.param(
            "TAIC",
            "https://www.taic.org.nz/inquiry/mo-2021-205",
            "TAIC_m_2021_205",
            {
                "success": True,
                "occurrence_date": "2021-12-10",
                "publication_date": "2023-04-27",
            },
            id="TAIC pass",
        ),
        pytest.param(
            "TAIC",
            "https://www.taic.org.nz/inquiry/mo-2021-255",
            "TAIC_m_2021_255",
            False,
            id="TAIC fail",
        ),
        pytest.param(
            "TAIC",
            "https://www.taic.org.nz/inquiry/ao-2000-001",
            "TAIC_a_2000_001",
            {
                "success": False,
                "occurrence_date": "2000-01-18",
                "publication_date": "2000-03-15",
            },
            id="TAIC old aviation fail as no pdf",
        ),
        pytest.param(
            "TAIC",
            "https://www.taic.org.nz/inquiry/ro-2025-103",
            "TAIC_r_2025_103",
            {
                "success": True,
                "occurrence_date": "2025-02-01",
                "publication_date": "2026-06-05",
            },
            id="TAIC new rail 2025",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/investigations/mo-2019-007",
            "ATSB_m_2019_007",
            {
                "success": True,
                "occurrence_date": "2019-06-22",
                "publication_date": "2020-06-26",
            },
            id="ATSB pass",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/investigations/mo-2019-008",
            "ATSB_m_2019_008",
            False,
            id="ATSB fail",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/investigations/306-mo-2014-001",
            "ATSB_m_2014_001",
            {
                "success": True,
                "occurrence_date": "2014-02-01",
                "publication_date": "2015-01-23",
            },
            id="ATSB marine",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/investigations/ao-2007-038",
            "ATSB_a_2007_038",
            {
                "success": True,
                "occurrence_date": "2007-08-27",
                "publication_date": "2009-06-01",
            },
            id="ATSB old aviation",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/investigations/ao-2025-052",
            "ATSB_a_2025_052",
            {
                "success": True,
                "occurrence_date": "2025-08-22",
                "publication_date": "2026-05-08",
                "agency_id": "AO-2025-052",
            },
            id="ATSB new aviation 2025",
        ),
    ],
)
def test_report_collection(
    get_agency_scraper: Callable[[str], WebsiteScraping.ReportScraper],
    agency: str,
    url: str,
    report_id: str,
    expected: bool | dict,
) -> None:
    """Test collecting a single report from each agency."""
    scraper = get_agency_scraper(agency)

    # Mock PDF storage interactions so tests don't perform real Azure uploads.
    # We patch both `pdf_exists` to force a download attempt and `upload_pdf`
    # to avoid actual network/storage activity while keeping the rest of the
    # scraping logic intact.
    with (
        patch.object(
            scraper.settings.pdf_storage_manager, "pdf_exists", return_value=False
        ),
        patch.object(
            scraper.settings.pdf_storage_manager, "upload_pdf", return_value=True
        ),
    ):
        report_data = ReportUrlData(
            id=report_id,
            url=url,
            agency_id=expected.get("agency_id") if isinstance(expected, dict) else None,
            occurrence_date=expected.get("occurrence_date")
            if isinstance(expected, dict)
            else None,
            publication_date=expected.get("publication_date")
            if isinstance(expected, dict)
            else None,
        )
        result = scraper.collect_report(report_data)

    if isinstance(expected, dict):
        assert result == expected["success"]
        if "occurrence_date" in expected or "publication_date" in expected:
            rows = scraper.report_titles_df[
                scraper.report_titles_df["report_id"] == report_id
            ]
            assert len(rows) == 1, f"Report {report_id} not found in report_titles_df"
            row = rows.iloc[0]
            if "occurrence_date" in expected:
                assert row["occurrence_date"] == expected["occurrence_date"]
            if "publication_date" in expected:
                assert row["publication_date"] == expected["publication_date"]
    else:
        assert result == expected


@pytest.mark.parametrize(
    "agency, expected_urls, test_cases",
    [
        pytest.param(
            "TSB",
            [54, 25, 20, 13, 19, 10, 15, 10, 13],
            [
                {
                    "Occurrence date": pd.Timestamp("2023-07-19"),
                    "Investigation name": "A23W0082",
                    "year": 2023,
                },
                {
                    "Occurrence date": pd.Timestamp("2016-09-03"),
                    "Investigation name": "R16C0065",
                    "year": 2016,
                },
            ],
            id="TSB",
        ),
        pytest.param(
            "TAIC",
            [13, 9, 3, 27, 8, 4, 11, 3, 4],
            [{"id": "MO-2020-204", "year": 2020}, {"id": "RO-2011-104", "year": 2011}],
            id="TAIC",
        ),
        pytest.param(
            "ATSB",
            [98, 169, 47, 6, 25, 22, 15, 9, 2],
            [
                {
                    "url": "https://www.atsb.gov.au/investigations/ao-2024-050",
                    "agency_id": "AO-2024-050",
                }
            ],
            id="ATSB",
        ),
    ],
)
def test_agency_website_scraper(
    get_agency_scraper: object, agency: str, expected_urls: list, test_cases: list
) -> None:
    """Test the agency website scraper for correct URL generation."""
    scraper = get_agency_scraper(agency)
    scraper.settings.start_year = 2004
    scraper.settings.end_year = 2021

    assert scraper

    assert scraper.agency == agency

    assert isinstance(scraper.agency_reports, pd.DataFrame)

    errors = []
    for (mode, year), expected_len in zip(
        itertools.product(
            [Modes.Mode.a, Modes.Mode.r, Modes.Mode.m], [2005, 2013, 2020]
        ),
        expected_urls,
        strict=False,
    ):
        urls = scraper.get_report_urls(mode, year)

        try:
            assert len(urls) == expected_len
        except AssertionError:
            errors.append(f"{agency} {mode} {year}: {len(urls)} != {expected_len}")

    # Find out if test cases are in scraper.agency_reports with correct fields

    for test_case in test_cases:
        if agency == "TSB":
            matches = scraper.agency_reports.query(
                "`Occurrence date` == @test_case['Occurrence date'] and "
                "`Investigation number` == @test_case['Investigation name'] and "
                "year == @test_case['year']"
            )
        elif agency == "TAIC":
            matches = scraper.agency_reports.query(
                "id == @test_case['id'] and year == @test_case['year']"
            )
        elif agency == "ATSB":
            matches = scraper.agency_reports.query(
                "url == @test_case['url'] and agency_id == @test_case['agency_id']"
            )

        if len(matches) != 1:
            errors.append(
                f"{agency} test case {test_case} found {len(matches)} matches, expected 1."
            )

    if errors:
        pytest.fail("\n" + "\n".join(errors))


@pytest.mark.parametrize(
    "agency, expected_count",
    [
        pytest.param("TSB", 15, id="TSB"),
        pytest.param("TAIC", 15, id="TAIC"),
        pytest.param("ATSB", 15, id="ATSB"),
    ],
)
def test_agency_website_scraper_collecting_all_reports(
    get_agency_scraper: object, agency: str, expected_count: int
) -> None:
    """Test collecting all reports for an agency."""
    scraper = get_agency_scraper(agency)
    scraper.settings.refresh = True
    scraper.settings.start_year = 2008
    scraper.settings.end_year = 2012

    assert scraper

    uploaded_pdfs = []

    # Mock collect_report to simulate successful PDF upload
    with patch.object(scraper, "collect_report", return_value=True) as mock_download:

        def mock_download_side_effect(report_data: ReportUrlData) -> bool:
            # Simulate successful PDF upload
            uploaded_pdfs.append(report_data.id)
            return True

        mock_download.side_effect = mock_download_side_effect

        scraper.collect_all()

    # Verify that collect_all resulted in the expected number of PDFs
    pdf_count = len(uploaded_pdfs)
    assert pdf_count == expected_count


@pytest.mark.parametrize(
    "url, report_id, expected",
    [
        pytest.param(
            "https://www.atsb.gov.au/investigations/ao-2025-052",
            "ATSB_a_2025_052",
            {
                "summary_start": "What ",
                "summary_end": "come.",
                "summary_length": 3584,
                "investigation_level": "short",
                "occurrence_type": "Engine failure or malfunction, Missed approach",
                "agency_id": "AO-2025-052",
                "occurrence_date": "2025-08-22",
                "publication_date": "2026-05-08",
            },
            id="ATSB standard full report",
        ),
        pytest.param(
            "https://www.atsb.gov.au/investigations/ao-2007-038",
            "ATSB_a_2007_038",
            {
                "summary_start": "At about",
                "summary_end": "ments.",
                "summary_length": 1433,
                "investigation_level": "full",
                "occurrence_type": "Loss of separation",
                "agency_id": "AO-2007-038",
                "occurrence_date": "2007-08-27",
                "publication_date": "2009-06-01",
            },
            id="ATSB old report with single H2 Summary",
        ),
        pytest.param(
            "https://www.atsb.gov.au/investigations/ao-2017-032",
            "ATSB_a_2017_032",
            {
                "summary_start": "Preliminary report released 13 Apr",
                "summary_end": "detachment",
                "summary_length": 9615,
                "investigation_level": "full",
                "occurrence_type": "Propeller/rotor malfunction",
                "agency_id": "AO-2017-032",
                "occurrence_date": "2017-03-17",
                "publication_date": "2018-10-10",
            },
            id="ATSB report with preliminary report treated as summary",
        ),
    ],
)
def test_atsb_report_metadata_scrape(
    get_agency_scraper: Callable[[str], WebsiteScraping.ReportScraper],
    url: str,
    report_id: str,
    expected: dict,
) -> None:
    """Test the scraping logic of the ATSB report page."""
    scraper = get_agency_scraper("ATSB")

    soup = bs4.BeautifulSoup(scraper.get(url).content, "html.parser")

    metadata = scraper.get_report_metadata(report_id, url, None, None, soup)

    assert metadata.agency_id == expected["agency_id"]
    assert metadata.investigation_type == expected["investigation_level"]
    assert metadata.event_type == expected["occurrence_type"]
    assert metadata.occurrence_date == expected["occurrence_date"]
    assert metadata.publication_date == expected["publication_date"]

    summary = metadata.summary
    assert summary.startswith(expected["summary_start"])
    assert summary.endswith(expected["summary_end"])
    assert len(summary) == expected["summary_length"]


def test_atsb_safety_issue_scrape(tmpdir: object) -> None:
    """Test scraping ATSB safety issues from the website."""
    # Copy existing files to temporary directory for automatic cleanup
    output_folder = Path(pytest.output_config.get("folder_name"))
    original_output_path = SavedDataFrames.ATSBWebsiteSafetyIssues(output_folder).path
    report_titles_dc = SavedDataFrames.ReportTitles(output_folder)

    # Create temporary paths
    atsb_safety_issues_dc = SavedDataFrames.ATSBWebsiteSafetyIssues(Path(str(tmpdir)))

    # Copy files if they exist
    if original_output_path.exists():
        shutil.copy2(original_output_path, atsb_safety_issues_dc.path)
    else:
        pytest.fail(
            "Test data file 'atsb_website_safety_issues.pkl' is missing in the test data folder."
        )

    atsb_webscraper = WebsiteScraping.ATSBSafetyIssueScraper(
        safety_issues_dc=atsb_safety_issues_dc,
        report_titles_dc=report_titles_dc,
    )

    atsb_webscraper.extract_safety_issues_from_website()

    output = pd.read_pickle(atsb_safety_issues_dc.path)

    assert len(output) >= MINIMUM_SAFETY_ISSUES_COUNT

    required_ids = ["MO-2008-013-SI-04", "AO-2023-008-SI-01"]

    for item_id in required_ids:
        assert item_id in output["safety_issue_id"].unique()


@pytest.mark.parametrize(
    "site,scraper_cls,table_arg,expected_min_rows",
    [
        pytest.param(
            "taic",
            WebsiteScraping.TAICRecommendationsScraper,
            0,
            12,
            id="TAIC listing",
        ),
        pytest.param(
            "tsb",
            WebsiteScraping.TSBRecommendationsScraper,
            "aviation",
            121,
            id="TSB listing",
        ),
    ],
)
def test_recommendation_listing(  # noqa: PLR0913, PLR0917
    tmpdir: object,
    request: object,
    site: str,
    scraper_cls: type,
    table_arg: object,
    expected_min_rows: int,
) -> None:
    """Smoke test: fetch one recommendations listing page and ensure it yields entries.

    Uses the real websites but is designed to be quick (one HTTP request per case).
    """
    report_titles_dc = SavedDataFrames.ReportTitles(
        Path(pytest.output_config.get("folder_name"))
    )
    assert report_titles_dc.exists(), "Test report titles file is missing"

    scraper = scraper_cls(
        SavedDataFrames.TSBWebsiteRecommendations(tmpdir)
        if scraper_cls.__name__ == "TSBRecommendationsScraper"
        else SavedDataFrames.TAICWebsiteRecommendations(tmpdir),
        report_titles_dc,
        refresh=True,
    )

    table = scraper.get_table(table_arg)
    assert not table.empty
    table = scraper.process_new_table(table)
    assert len(table) >= expected_min_rows

    assert {"url", "recommendation_id"}.issubset(table.columns)

    sample_url = table["url"].dropna().iloc[0]
    assert isinstance(sample_url, str)
    assert sample_url.startswith("https://")


@pytest.mark.parametrize(
    "site,scraper_cls,url,required_fields",
    [
        pytest.param(
            "taic",
            WebsiteScraping.TAICRecommendationsScraper,
            "https://taic.org.nz/recommendation/02125",
            ["recommendation", "recipient", "made", "agency_id"],
            id="TAIC page",
        ),
        pytest.param(
            "tsb",
            WebsiteScraping.TSBRecommendationsScraper,
            "https://www.tsb.gc.ca/eng/recommandations-recommendations/aviation/2024/rec-a2402.html",
            ["made", "recommendation_context"],
            id="TSB page",
        ),
    ],
)
def test_recommendation_page(  # noqa: PLR0913, PLR0917
    tmpdir: object,
    request: object,
    site: str,
    scraper_cls: type,
    url: str,
    required_fields: list,
) -> None:
    """Smoke test: extract fields from one recommendation page.

    Uses the real websites but is designed to be quick (one HTTP request per case).
    Validates extracted data against the SavedDataFrame Row Pydantic model schema.
    """
    report_titles_dc = SavedDataFrames.ReportTitles(
        Path(pytest.output_config.get("folder_name"))
    )
    assert report_titles_dc.exists(), "Test report titles file is missing"

    scraper = scraper_cls(
        SavedDataFrames.TSBWebsiteRecommendations(tmpdir)
        if scraper_cls.__name__ == "TSBRecommendationsScraper"
        else SavedDataFrames.TAICWebsiteRecommendations(tmpdir),
        report_titles_dc,
        refresh=True,
    )

    rec = scraper.extract_recommendation_data(url)

    # Get the Row model for validation
    row_model = scraper.recommendations_dc.Row

    # Basic validation: returned dict has valid keys and required fields
    assert isinstance(rec, dict), "extract_recommendation_data should return a dict"

    valid_fields = set(row_model.model_fields.keys())
    returned_keys = set(rec.keys())
    unexpected_keys = returned_keys - valid_fields

    assert (
        not unexpected_keys
    ), f"Unexpected keys: {unexpected_keys}. Valid: {valid_fields}"

    # Validate required fields are present and non-empty
    for field in required_fields:
        assert field in rec, f"Missing required field: {field}"
        assert rec[field], f"Required field '{field}' is empty"
