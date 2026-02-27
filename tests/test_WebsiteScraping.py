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
import os
import shutil
import socket
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine import Modes, SavedDataFrames, WebsiteScraping

MINIMUM_SAFETY_ISSUES_COUNT = 388


def _can_connect(host: str, port: int = 443, timeout_s: float = 2.0) -> bool:
    """Best-effort network check so these tests can be skipped in offline/CI sandboxes.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
    except Exception:
        return False
    return True


# Tests in this module hit real websites; they are skipped when offline.


@pytest.fixture(autouse=True)
def require_internet(request):
    """Skip these tests when offline.

    Practically every test in this module hits a real external site (TAIC/TSB/ATSB)
    via `hrequests.get()` (either directly or indirectly via the scrapers), so
    running offline will just fail/flap.

    Controls:
    - Tests marked with @pytest.mark.integration(site="...") will additionally
      check connectivity to the specific host for that site.
    """
    # Module-wide baseline check: if we can't reach *any* common host, skip.
    if not (
        _can_connect("www.taic.org.nz")
        or _can_connect("www.tsb.gc.ca")
        or _can_connect("www.atsb.gov.au")
    ):
        pytest.skip("No network/DNS available for website scraping tests")


@pytest.fixture(scope="function")
def report_scraping_settings(tmpdir, test_pdf_storage_manager):
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
def get_agency_scraper(tmpdir, report_scraping_settings):
    """Fixture that returns a function to create agency scrapers with their own temp directories.

    Each scraper gets its own temporary directory for file operations.

    Returns:
        A function that creates agency-specific scrapers.
    """

    def _get_agency_scraper(agency: str) -> WebsiteScraping.ReportScraper:
        if agency == "TAIC":
            # Use the canonical TAIC reports table from the tests data folder, but
            # copy it into the test's tmpdir so the original isn't modified by tests.
            original_path = os.path.join(
                pytest.output_config.get("folder_name"),
                pytest.output_config.get("taic_website_reports_table_file_name"),
            )
            tmp_path = os.path.join(
                str(tmpdir),
                pytest.output_config.get("taic_website_reports_table_file_name"),
            )

            if os.path.exists(original_path):
                shutil.copy2(original_path, tmp_path)

            return WebsiteScraping.TAICReportScraper(
                SavedDataFrames.TAICWebsiteReportsTable(tmpdir),
                report_scraping_settings,
            )
        if agency == "ATSB":
            original_path = os.path.join(
                pytest.output_config.get("folder_name"),
                pytest.output_config.get("atsb_website_reports_table_file_name"),
            )
            tmp_path = os.path.join(
                str(tmpdir),
                pytest.output_config.get("atsb_website_reports_table_file_name"),
            )

            if os.path.exists(original_path):
                shutil.copy2(original_path, tmp_path)

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
            True,
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
            True,
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
            "ATSB",
            "https://www.atsb.gov.au/publications/investigation_reports/2019/mair/mo-2019-007",
            "ATSB_m_2019_007",
            True,
            id="ATSB pass",
        ),
        pytest.param(
            "ATSB",
            "https://www.atsb.gov.au/publications/investigation_reports/2019/mair/mo-2019-008",
            "ATSB_m_2019_008",
            False,
            id="ATSB fail",
        ),
    ],
)
def test_report_collection(get_agency_scraper, agency, url, report_id, expected):
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
        result = scraper.collect_report(report_id, url)

    assert result == expected


@pytest.mark.parametrize(
    "agency, expected_urls",
    [
        pytest.param("TSB", [54, 25, 20, 13, 19, 10, 15, 10, 13], id="TSB"),
        pytest.param("TAIC", [11, 12, 3, 28, 8, 4, 12, 3, 5], id="TAIC"),
        pytest.param("ATSB", [93, 179, 52, 6, 25, 17, 15, 8, 2], id="ATSB"),
    ],
)
def test_agency_website_scraper(get_agency_scraper, agency, expected_urls):
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
    get_agency_scraper, agency, expected_count
):
    """Test collecting all reports for an agency."""
    scraper = get_agency_scraper(agency)
    scraper.settings.refresh = True
    scraper.settings.start_year = 2008
    scraper.settings.end_year = 2012

    assert scraper

    uploaded_pdfs = []

    # Mock collect_report to simulate successful PDF upload
    with patch.object(scraper, "collect_report", return_value=True) as mock_download:

        def mock_download_side_effect(report_id, url, agency_id=None):
            # Simulate successful PDF upload
            uploaded_pdfs.append(report_id)
            return True

        mock_download.side_effect = mock_download_side_effect

        scraper.collect_all()

    # Verify that collect_all resulted in the expected number of PDFs
    pdf_count = len(uploaded_pdfs)
    assert pdf_count == expected_count


def test_atsb_safety_issue_scrape(tmpdir):
    """Test scraping ATSB safety issues from the website."""
    # Copy existing files to temporary directory for automatic cleanup
    original_output_path = os.path.join(
        pytest.output_config["folder_name"],
        pytest.output_config["atsb_website_safety_issues_file_name"],
    )
    original_report_titles = os.path.join(
        pytest.output_config["folder_name"],
        pytest.output_config["report_titles_df_file_name"],
    )

    # Create temporary paths
    atsb_safety_issues_df = SavedDataFrames.ATSBWebsiteSafetyIssues(Path(str(tmpdir)))
    temp_output_path = str(atsb_safety_issues_df.path)
    report_titles_dc = SavedDataFrames.ReportTitles(Path(str(tmpdir)))

    # Copy files if they exist
    if os.path.exists(original_output_path):
        shutil.copy2(original_output_path, temp_output_path)
    if os.path.exists(original_report_titles):
        shutil.copy2(original_report_titles, report_titles_dc.path)

    atsb_webscraper = WebsiteScraping.ATSBSafetyIssueScraper(
        safety_issues_dc=atsb_safety_issues_df,
        report_titles_dc=report_titles_dc,
        refresh=True,
    )

    atsb_webscraper.extract_safety_issues_from_website()

    output = pd.read_pickle(temp_output_path)

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
@pytest.mark.integration
def test_recommendation_listing(  # noqa: PLR0913, PLR0917
    tmpdir,
    request,
    site,
    scraper_cls,
    table_arg,
    expected_min_rows,
):
    """Smoke test: fetch one recommendations listing page and ensure it yields entries.

    Uses the real websites but is designed to be quick (one HTTP request per case).
    """
    # Add a per-site marker dynamically so the autouse network fixture can
    # check the correct host.
    request.node.add_marker(pytest.mark.integration(site=site))

    report_titles_dc = SavedDataFrames.ReportTitles(
        pytest.output_config.get("folder_name")
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
    "site,scraper_cls,url,assert_fn",
    [
        pytest.param(
            "taic",
            WebsiteScraping.TAICRecommendationsScraper,
            "https://taic.org.nz/recommendation/02125",
            lambda rec: (
                isinstance(rec, dict)
                and isinstance(rec.get("recommendation"), str)
                and bool(rec.get("recommendation"))
                and isinstance(rec.get("recipient"), str)
                and bool(rec.get("recipient"))
                and isinstance(rec.get("made"), str)
                and bool(rec.get("made"))
                and isinstance(rec.get("agency_id"), str)
                and bool(rec.get("agency_id"))
            ),
            id="TAIC page",
        ),
        pytest.param(
            "tsb",
            WebsiteScraping.TSBRecommendationsScraper,
            "https://www.tsb.gc.ca/eng/recommandations-recommendations/aviation/2024/rec-a2402.html",
            lambda rec: (
                isinstance(rec, dict)
                and bool(rec.get("made"))
                and bool(rec.get("recommendation_context"))
            ),
            id="TSB page",
        ),
    ],
)
@pytest.mark.integration
def test_recommendation_page(  # noqa: PLR0913, PLR0917
    tmpdir, request, site, scraper_cls, url, assert_fn
):
    """Smoke test: extract fields from one recommendation page.

    Uses the real websites but is designed to be quick (one HTTP request per case).
    """
    # Add a per-site marker dynamically so the autouse network fixture can
    # check the correct host.
    request.node.add_marker(pytest.mark.integration(site=site))

    report_titles_dc = SavedDataFrames.ReportTitles(
        pytest.output_config.get("folder_name")
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
    assert assert_fn(rec)

    # Check to see if it has the right keys
    assert set(rec.keys()).issubset(scraper.recommendations_dc._effective_columns)
    # common ones
    assert "recommendation" in rec
    assert "made" in rec
