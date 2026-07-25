"""Website scraping module for transportation safety investigation reports.

This module provides classes for scraping reports, recommendations, and safety issues
from various transportation safety agencies (TAIC, ATSB, TSB). It handles HTTP requests,
It does the crawling of websitess and extracts report metadata and PDFs which is then uploaded to cloud storage.
"""

import io
import random
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from threading import Lock
from typing import ClassVar, Literal
from urllib.parse import urljoin, urlparse

import hrequests
import hrequests.exceptions
import pandas as pd
from bs4 import BeautifulSoup, Tag
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from engine import Modes
from engine.AzureStorage import PDFStorageManager
from engine.Logging import get_logger
from engine.SavedDataFrames import (
    ATSBWebsiteReportsTable,
    ATSBWebsiteSafetyIssues,
    ReportTitles,
    TAICWebsiteRecommendations,
    TAICWebsiteReportsTable,
    TSBWebsiteRecommendations,
)

logger = get_logger(__name__)


class ScraperRequestError(ValueError):
    """Raised when a remote resource cannot be fetched successfully."""

    def __init__(self, url: str, status_code: int | None = None) -> None:
        """Initialize ScraperRequestError with URL and optional status code.

        Args:
            url: The URL that failed to fetch.
            status_code: Optional HTTP status code from the failed request.
        """
        self.url = url
        self.status_code = status_code
        if status_code is None:
            message = f"Failed to fetch {url}"
        else:
            message = f"Failed to fetch {url}, status code: {status_code}"
        super().__init__(message)


@dataclass
class ReportUrlData:
    """Holds URL and date data scraped from the listing page for a single report."""

    id: str
    url: str
    agency_id: str | None
    occurrence_date: str | None = None
    publication_date: str | None = None


@dataclass
class ReportMetadata:
    """Small data class to hold the metadata so that it can be passed around easily between the functions."""

    report_id: str
    title: str
    event_type: str | None
    investigation_type: str
    summary: str | None
    misc: dict
    url: str
    agency_id: str | None
    occurrence_date: str | None = None
    publication_date: str | None = None

    def __repr__(self) -> str:
        """Return string representation of ReportMetadata.

        Returns:
            String representation with report_id and title.
        """
        return f"ReportMetadata({self.report_id}, {self.title})"

    def as_report_row(self) -> list:
        """Return values in the order defined by ReportTitles.Row.

        Returns:
            List of attribute values in column order.
        """
        return [
            getattr(self, field_name) for field_name in ReportTitles.Row.model_fields
        ]


@dataclass
class ReportScraperSettings:
    """Configuration settings for report scraping.

    Holds all configuration parameters needed for scraping reports from
    transportation safety agency websites.
    """

    report_titles_dc: ReportTitles
    start_year: int
    end_year: int
    max_per_year: int
    modes: list[Modes.Mode]
    ignored_report_ids: list[str]
    refresh: bool
    pdf_storage_manager: PDFStorageManager
    refresh_metadata: bool = False
    scraper_workers: int = 1

    def __post_init__(self) -> None:
        """Convert refresh_metadata to ignore_metadata for backward compatibility."""
        self.ignore_metadata = self.refresh_metadata


class WebsiteScraper:
    """Base class for scraping websites.

    Provides common functionality for HTTP requests and ID conversion.
    Subclasses handle agency-specific website structures.
    """

    def __init__(self, report_titles_dc: ReportTitles):
        """Initialize WebsiteScraper.

        Args:
            report_titles_dc: The data container for report titles.

        Raises:
            ValueError: If the report titles data container is empty.
        """
        # Keep the original headers as a fallback, but we'll use get_randomized_headers() instead
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Connection": "keep-alive",
        }

        try:
            report_titles_df = report_titles_dc.read()
        except FileNotFoundError as e:
            error_msg = f"Report titles df {report_titles_dc.path} does not exist"
            raise ValueError(error_msg) from e

        self.id_dict = {
            agency: {
                agency_id.upper(): report_id
                for report_id, agency_id in ids[["report_id", "agency_id"]].to_numpy()
            }
            for agency, ids in report_titles_df.assign(
                agency=report_titles_df["report_id"].map(lambda x: x.split("_")[0])
            ).groupby("agency")
        }

    @staticmethod
    def get_randomized_headers() -> dict[str, str]:
        """Generate randomized headers to avoid bot detection.

        Returns:
            Dictionary of randomized HTTP headers with shuffled order.
        """
        # Various User-Agent strings for different browsers and platforms
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]

        # Various Accept-Language options
        accept_languages = [
            "en-US,en;q=0.9",
            "en-US,en;q=0.8,fr;q=0.6",
            "en-GB,en;q=0.9,en-US;q=0.8",
            "en,en-US;q=0.9",
            "en-US,en;q=0.7,fr;q=0.3",
        ]

        # Various referer options
        referers = [
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
            "https://search.yahoo.com/",
            "https://www.google.co.uk/",
            "https://www.google.ca/",
        ]

        # Various Accept headers
        accept_headers = [
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        ]

        # Various Connection options
        connections = ["keep-alive", "close"]

        # Optionally include additional headers that browsers might send
        additional_headers = [
            ("Accept-Encoding", "gzip, deflate, br"),
            ("DNT", "1"),
            ("Upgrade-Insecure-Requests", "1"),
            ("Sec-Fetch-Dest", "document"),
            ("Sec-Fetch-Mode", "navigate"),
            ("Sec-Fetch-Site", "none"),
            ("Cache-Control", "max-age=0"),
        ]

        # 70% chance to include each additional header
        additional_header_probability = 0.7

        # Build the randomized headers dictionary
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": random.choice(accept_headers),
            "Accept-Language": random.choice(accept_languages),
            "Referer": random.choice(referers),
            "Connection": random.choice(connections),
        }

        # Randomly include some additional headers
        for header_name, header_value in additional_headers:
            if random.random() < additional_header_probability:
                headers[header_name] = header_value

        # Randomly shuffle the order of headers
        header_items = list(headers.items())
        random.shuffle(header_items)

        return dict(header_items)

    def get(
        self,
        url: str,
        *,
        headers: dict | None = None,
        attempts: int = 3,
        wait_min_s: float = 0.5,
        wait_max_s: float = 5.0,
        **kwargs: object,
    ) -> hrequests.Response:
        """HTTP GET with bounded retries and randomized headers.

        All website scrapers should use this instead of calling `hrequests.get` directly.

        Args:
            url: The URL to fetch.
            headers: Optional additional headers to merge with randomized headers.
            attempts: Maximum number of retry attempts.
            wait_min_s: Minimum wait time between retries in seconds.
            wait_max_s: Maximum wait time between retries in seconds.
            **kwargs: Additional arguments passed to hrequests.get.

        Returns:
            Response object from hrequests.get.
        """
        merged_headers = {**self.get_randomized_headers(), **(headers or {})}

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=wait_min_s, min=wait_min_s, max=wait_max_s
            ),
            retry=retry_if_exception_type(
                (hrequests.exceptions.ClientException, ScraperRequestError)
            ),
            reraise=True,
        )
        def _inner() -> hrequests.Response:
            response = hrequests.get(url, headers=merged_headers, **kwargs)
            if not response.ok:
                raise ScraperRequestError(url, response.status_code)
            return response

        return _inner()

    def id_converter(self, agency: str, agency_id: str) -> str | None:
        """Convert agency-specific ID to report ID using the COMPLETE report titles dataframe.

        The agency argument is needed as some of the agency IDs are not globally unique
        but instead just unique within an agency.

        Args:
            agency: The agency name (e.g., 'ATSB', 'TAIC', 'TSB').
            agency_id: The agency-specific identifier.

        Returns:
            The corresponding report_id or None if not found.

        Raises:
            ValueError: If the agency is not valid.
        """
        if agency not in self.id_dict:
            error_msg = f"{agency} is not a valid agency"
            raise ValueError(error_msg)
        return self.id_dict[agency].get(agency_id.upper())

    @staticmethod
    def html_table_to_dict(table: BeautifulSoup) -> dict[str, str]:
        """Take in a table that has two columns and turn it into a dictionary where the first column is the key and the second column is the value.

        Currently used by ATSB scrapers however could be useful elsehwere.

        Args:
            table: BeautifulSoup object of the HTML table.

        Returns:
            Dictionary representation of the table.
        """
        result = {}
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            number_of_cells = 2
            if len(cells) == number_of_cells:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                result[key] = value
        return result


class ReportScraper(WebsiteScraper, ABC):
    """Abstract base class for scraping reports from different transportation safety agencies.

    Subclasses must implement abstract methods to handle agency-specific website structures.
    """

    def __init__(self, settings: ReportScraperSettings, agency: str):
        """Initialize ReportScraper.

        Args:
            settings: Configuration settings for the scraper.
            agency: The agency name (e.g., 'TAIC', 'ATSB', 'TSB').
        """
        self.settings = settings
        self._metadata_file_lock = Lock()

        self.report_titles_df = self.settings.report_titles_dc.read_or_create()

        if self.report_titles_df.empty:
            self.settings.report_titles_dc.save(self.report_titles_df)

        self.agency = agency
        super().__init__(self.settings.report_titles_dc)

    def collect_all(self) -> None:
        """Collect reports for all configured modes."""
        logger.welcome(
            f"Downloading report PDFs for {self.agency}",
            {
                "PDF storage container": self.settings.pdf_storage_manager.container_name,
                "Report titles file path": str(self.settings.report_titles_dc.path),
                "Year range": f"{self.settings.start_year} - {self.settings.end_year}",
                "Max per year": self.settings.max_per_year,
                "Modes": ", ".join(mode.name for mode in self.settings.modes),
                "Ignored report IDs": len(self.settings.ignored_report_ids),
            },
        )
        # Loop through each mode
        for mode in self.settings.modes:
            self.collect_mode(mode)

    @abstractmethod
    def get_report_urls(self, mode: Modes.Mode, year: int) -> list[ReportUrlData]:
        """Retrieves all the potential report urls and ids for a given mode and year.

        This method must be implemented by subclasses to handle agency-specific
        website structures for finding report URLs.

        Args:
            mode: The mode/type of transportation (aviation, rail, marine).
            year: The year to search for reports.

        Returns:
            List of ReportUrlData objects.
        """

    def get_report_id(self, mode: Modes.Mode, year: int, report_id: str) -> str:
        """Generate a standardized report ID.

        Args:
            mode: The transportation mode.
            year: The year of the report.
            report_id: The agency-specific report ID.

        Returns:
            Standardized report ID in format '{agency}_{mode}_{year}_{id}'.
        """
        return f"{self.agency}_{mode.name}_{year}_{report_id}"

    def collect_mode(self, mode: Modes.Mode) -> None:
        """Collect all reports for a specific transportation mode.

        Args:
            mode: The transportation mode to collect reports for.
        """
        logger.info(f"Downloading reports for mode: {mode.name}")

        year_range = list(range(self.settings.start_year, self.settings.end_year + 1))

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(
            max_workers=min(len(year_range), self.settings.scraper_workers)
        ) as executor:
            # Submit all year collection tasks
            future_to_year = {
                executor.submit(self.collect_year, year, mode): year
                for year in year_range
            }

            completed_years = []

            # Process completed tasks as they finish
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_result = future.result()
                    completed_years.append(year_result)

                    logger.info(
                        f"Completed year {year_result['year']} for mode {mode.name}: "
                        f"{year_result['reports_collected']} reports collected out of {year_result['potential_reports']} in "
                        f"{year_result['duration']:.2f} seconds"
                    )

                except Exception:
                    logger.exception(f"Year {year} generated an exception")

        # Print final summary
        total_reports = sum(result["reports_collected"] for result in completed_years)
        total_time = sum(result["duration"] for result in completed_years)
        avg_time = total_time / len(completed_years) if completed_years else 0

        logger.info(f"Finished downloading reports for mode: {mode.name}")
        logger.info(f"Total reports collected: {total_reports}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per year: {avg_time:.2f} seconds")

    def collect_year(self, year: int, mode: Modes.Mode) -> dict:
        """Helper method to collect a year's reports with timing information.

        Args:
            year: The year to collect reports for.
            mode: The transportation mode.

        Returns:
            Dictionary with year, potential_reports, reports_collected, and duration.
        """
        start_time = datetime.now()

        number_for_year = 0
        report_urls = self.get_report_urls(mode, year)
        for report_url_data in report_urls:
            if report_url_data.id in self.settings.ignored_report_ids:
                continue

            outcome = self.collect_report(report_url_data)

            if outcome:
                number_for_year += 1

            if number_for_year >= self.settings.max_per_year:
                break

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "year": year,
            "potential_reports": len(report_urls),
            "reports_collected": number_for_year,
            "duration": duration,
        }

    def collect_report(self, report_data: ReportUrlData) -> bool:
        """Collect a single report.

        Args:
            report_data: ReportUrlData containing id, url, agency_id, and dates.

        Returns:
            True if report was successfully collected, False otherwise.
        """
        url = report_data.url
        report_id = report_data.id
        agency_id = report_data.agency_id
        occurrence_date = report_data.occurrence_date
        publication_date = report_data.publication_date

        if (
            not self.settings.ignore_metadata
            and self.report_titles_df.query(f"report_id == '{report_id}'").shape[0] > 0
        ):
            return True

        try:
            webpage = self.get(url)
        except hrequests.exceptions.ClientException as e:
            logger.warning(f"Timed out while trying to collect {url}: {e}")
            return False
        except hrequests.exceptions.BrowserTimeoutException:
            logger.warning(f"Failed to collect {url}, timeout error")
            return False
        soup = BeautifulSoup(webpage.content, "html.parser")

        if webpage.status_code not in {HTTPStatus.OK, HTTPStatus.NOT_FOUND}:
            logger.warning(
                f"Failed to collect {url}, status code: {webpage.status_code}"
            )
            return False
        if webpage.status_code == HTTPStatus.NOT_FOUND:
            logger.warning(f"Error 404: {url} not found")
            return False

        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        outcome = False
        # Download to cloud storage instead of local file
        outcome = self.download_report(report_id, soup, base_url, agency_id)

        if (
            self.report_titles_df.query("report_id == @report_id").empty
            or self.settings.ignore_metadata
            or self.settings.refresh
        ):
            metadata = self.get_report_metadata(
                report_id, url, occurrence_date, publication_date, soup
            )
            self.__add_report_metadata_to_df(metadata)
        elif outcome is None:
            outcome = True

        return outcome

    def download_report(
        self,
        report_id: str,
        soup: BeautifulSoup,
        base_url: str,
        agency_id: str | None = None,
    ) -> bool | None:
        """Download report directly to PDF storage container.

        Args:
            report_id: The standardized report identifier.
            soup: BeautifulSoup object of the report page.
            base_url: Base URL of the website.
            agency_id: Optional agency-specific ID for filtering PDF links.

        Returns:
            True if successful, False if failed, None if no action taken.
        """
        # Check if PDF already exists and we're not refreshing
        if not self.settings.refresh and self.settings.pdf_storage_manager.pdf_exists(
            report_id
        ):
            return True

        # Use provided agency_id for PDF link filtering
        # Find PDF links
        pdf_link = self._find_pdf_links(soup, report_id, agency_id)
        if pdf_link is False:
            return False

        link = urljoin(base_url, pdf_link)

        try:
            response = self.get(
                link,
                allow_redirects=True,
                timeout=30,
            )
            if response is None:
                logger.warning(f"{report_id}.pdf download failed: No response")
                return False

            # Upload to storage
            self.settings.pdf_storage_manager.upload_pdf(
                report_id, response.content, overwrite=self.settings.refresh
            )
        except Exception as e:
            logger.warning(f"{report_id}.pdf processing failed: {e}")
            return False
        else:
            return True

    @staticmethod
    def _find_pdf_links(
        soup: BeautifulSoup, report_id: str, agency_id: str | None = None
    ) -> str | Literal[False]:
        """Extract PDF links from BeautifulSoup object.

        Args:
            soup: BeautifulSoup object of the report page.
            report_id: The report identifier for logging.
            agency_id: Optional agency-specific ID for filtering.

        Returns:
            URL of the PDF link, or False if no suitable link found.
        """
        # Find all the links that end with .pdf and download them
        pdf_links = [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".pdf")
        ]

        # Remove duplicates
        pdf_links = list(dict.fromkeys(pdf_links))

        if len(pdf_links) == 0:
            logger.warning(
                f"Found no PDFs link for {report_id}. Will not download any."
            )
            return False
        if len(pdf_links) > 1:
            # Remove links that are simply subsets of other links
            suitable_pdf_links = [
                link
                for link in pdf_links
                if not any(link != other and link in other for other in pdf_links)
            ]

            # Remove duplicates that have the same filename
            suitable_pdf_links = [
                link
                for link in suitable_pdf_links
                if link.split("/")[-1]
                not in [
                    other.split("/")[-1]
                    for other in suitable_pdf_links
                    if link != other
                ]
            ]

            # If there are still multiple suitable links try and remove all the ones taht have "interim" or "prelim" in them
            if len(suitable_pdf_links) > 1:
                suitable_pdf_links = [
                    link
                    for link in suitable_pdf_links
                    if "interim" not in link.lower() and "prelim" not in link.lower()
                ]

            # If there are still multiple suitable links and we have an agency_id,
            # filter by links that contain the agency_id
            if len(suitable_pdf_links) > 1 and agency_id is not None:
                agency_filtered_links = [
                    link
                    for link in suitable_pdf_links
                    if agency_id.lower() in link.lower()
                ]
                if len(agency_filtered_links) > 0:
                    suitable_pdf_links = agency_filtered_links

            if len(suitable_pdf_links) > 1:
                links_str = "\n".join(suitable_pdf_links)
                logger.warning(
                    f"Found more than one PDF for {report_id}. Will not download any. Here are the links: \n{links_str}"
                )
                return False
            if len(suitable_pdf_links) == 0:
                links_str = "\n".join(pdf_links)
                logger.warning(
                    f"Found no suitable PDF link for {report_id}. Will not download any. Here are the original links:\n{links_str}"
                )
                return False
            pdf_links = suitable_pdf_links

        return pdf_links[0]

    @abstractmethod
    def get_report_metadata(
        self,
        report_id: str,
        url: str,
        occurrence_date: str | None,
        publication_date: str | None,
        soup: BeautifulSoup,
    ) -> ReportMetadata:
        """Gets the investigation webpage and scrapes extra information about the report.

        This method must be implemented by subclasses to handle agency-specific
        metadata extraction from report pages.

        Args:
            report_id: The identifier of the report.
            url: The URL of the report page.
            occurrence_date: The date of the occurrence.
            publication_date: The date of publication.
            soup: The BeautifulSoup object for the page.

        Returns:
            The report metadata object containing extracted information.
        """

    def __add_report_metadata_to_df(self, metadata: ReportMetadata) -> None:
        with self._metadata_file_lock:
            # Check if report_id exists
            existing_idx = self.report_titles_df.index[
                self.report_titles_df["report_id"] == metadata.report_id
            ].tolist()
            if existing_idx:
                # Replace the existing row
                self.report_titles_df.loc[existing_idx[0]] = metadata.as_report_row()
            else:
                # Add as new row
                self.report_titles_df.loc[len(self.report_titles_df)] = (
                    metadata.as_report_row()
                )

            self.settings.report_titles_dc.save(self.report_titles_df)


class TAICReportScraper(ReportScraper):
    """Report scraper for the Transport Accident Investigation Commission (TAIC) of New Zealand."""

    def __init__(
        self,
        website_reports_table_dc: TAICWebsiteReportsTable,
        settings: ReportScraperSettings,
    ):
        """Initialize TAICReportScraper.

        Args:
            website_reports_table_dc: The TAICWebsiteReportsTable dataframe class.
            settings: Configuration settings for the scraper.
        """
        super().__init__(
            settings,
            agency="TAIC",
        )

        self.website_reports_table_dc = website_reports_table_dc
        self.agency_reports = self.__get_taic_investigations()

    def __get_taic_investigations(self) -> pd.DataFrame:
        """TAICs websites provides an investigation table than can be easily read by pandas read_html.

        Returns:
            DataFrame of TAIC investigations indexed by mode.
        """
        investigations = self.website_reports_table_dc.read_or_create()
        page_num = 0
        pbar = tqdm()

        def extract_date(report: BeautifulSoup, label: str) -> str | None:
            date_type = report.find(
                "span", class_="date-type", string=lambda s: s and s.startswith(label)
            )
            if date_type:
                time_tag = date_type.find_next_sibling(
                    "span", class_="date-value"
                ).find("time")
                return time_tag["datetime"][:10] if time_tag else None
            return None

        while True:
            pbar.set_description(f"Scraping TAIC, page: {page_num}")
            try:
                new_content = self.get(
                    f"https://taic.org.nz/inquiries-recommendations?type=investigation&field_jurisdiction[11]=11&field_status[260]=260&sort_by=incident&page={page_num}"
                ).content
            except hrequests.exceptions.ClientException as e:
                logger.warning(f"Timeout while scraping TAIC investigations: {e}")
                logger.info(f"Retrying page {page_num}")
                continue

            soup = BeautifulSoup(new_content, "html.parser")

            results_list = soup.find_all("div", class_="search-results__list")

            if not results_list or len(results_list) == 0:
                tqdm.write(f"Reached end of pages at page {page_num}, stopping.")
                break

            all_reports_on_page = [
                {
                    "id": report.find("span", class_="card__incident").get_text(
                        strip=True
                    ),
                    "year": int(
                        report.find("span", class_="card__date")
                        .get_text(strip=True)
                        .split()[-1]
                    ),
                    "occurrence_date": extract_date(report, "Incident date"),
                    "publication_date": extract_date(report, "Publish date"),
                }
                for report in results_list[0].find_all("div", recursive=False)
            ]

            new_reports = pd.DataFrame(
                filter(
                    lambda r: r["id"] not in investigations["id"].to_numpy(),
                    all_reports_on_page,
                )
            )

            if new_reports.empty:
                tqdm.write(f"No new reports found on page {page_num}, stopping.")
                break

            investigations = pd.concat([investigations, new_reports], ignore_index=True)

            page_num += 1
            pbar.update(1)
        pbar.close()

        investigations_with_mode = investigations.copy()

        investigations_with_mode = investigations_with_mode.set_index(
            investigations_with_mode["id"].map(lambda x: Modes.Mode[x[0].lower()].name),
            inplace=False,
        )
        investigations_with_mode.index.name = None

        # save the investigations
        self.website_reports_table_dc.save(investigations_with_mode)

        return investigations_with_mode

    def get_report_urls(self, mode: Modes.Mode, year: int) -> list[ReportUrlData]:
        """Get report URLs for a specific mode and year.

        Args:
            mode: The transportation mode.
            year: The year to get reports for.

        Returns:
            List of ReportUrlData objects.
        """
        return [
            ReportUrlData(
                id=self.get_report_id(mode, year, row["id"][-3:]),
                url=f"https://www.taic.org.nz/inquiry/{row['id']}",
                agency_id=row["id"],
                occurrence_date=row["occurrence_date"],
                publication_date=row["publication_date"],
            )
            for row in self.agency_reports.loc[mode.name]
            .query(f"year == {year}")
            .to_dict("records")
        ]

    def get_report_metadata(  # noqa: PLR6301
        self,
        report_id: str,
        url: str,
        occurrence_date: str | None,
        publication_date: str | None,
        soup: BeautifulSoup,
    ) -> ReportMetadata:
        """Extract report metadata from TAIC website.

        Args:
            report_id: The standardized report identifier.
            url: The URL of the report page.
            occurrence_date: The date of the occurrence.
            publication_date: The date of publication.
            soup: BeautifulSoup object of the report page.

        Returns:
            ReportMetadata object with extracted information.
        """
        headers = soup.find("div", class_="screen-head__title")
        if headers is not None:
            title = headers.find("span", class_="heading--english").text.strip()
            agency_id = headers.find("span", class_="heading--accent").text.strip()
        else:
            logger.warning(f"Failed to get title for {report_id}")
            title = "Unknown Title"
            agency_id = None

        # Get the report text
        report_text_div = soup.find("div", class_="node--type-investigation")
        if report_text_div is None:
            logger.warning(f"Failed to get report text for {report_id}")
            summary = None
        else:
            sections = report_text_div.find_all("section", recursive=False)

            executive_summary_section = sections[0]

            summary = executive_summary_section.find("p").get_text().strip()

            if not summary:
                summary = None

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=title,
            event_type=None,
            investigation_type="full",
            agency_id=agency_id,
            summary=summary,
            occurrence_date=occurrence_date,
            publication_date=publication_date,
            misc={},
        )


class ATSBReportScraper(ReportScraper):
    """Report scraper for the Australian Transport Safety Bureau (ATSB)."""

    def __init__(
        self,
        website_report_table_dc: ATSBWebsiteReportsTable,
        settings: ReportScraperSettings,
    ):
        """Initialize ATSBReportScraper.

        Args:
            website_report_table_dc: The dataframe class for ATSB website reports.
            settings: Configuration settings for the scraper.
        """
        super().__init__(settings, agency="ATSB")
        self.report_table_dc = website_report_table_dc
        self.agency_reports = self.__get_atsb_investigations()

        logger.debug(
            f"ATSB investigations dataframe has {len(self.agency_reports)} rows after initial scrape."
        )

    @staticmethod
    def __parse_custom_divs_into_df(soup: BeautifulSoup) -> pd.DataFrame | None:
        """Parses the custom div based table from the investigation search and listing page.

        Args:
            soup: BeautifulSoup object of the ATSB investigations page.

        Returns:
            DataFrame of parsed investigation information.
        """
        table_container = soup.find("div", class_="ct-list__rows")
        if not table_container:
            logger.info(
                "Could not find the expected table container with class 'ct-list__rows' in the provided BeautifulSoup object."
            )
            return None

        table_cells = table_container.find_all("div", class_="col-xxs-12")

        investigations = []

        for cell in table_cells:
            title_link = cell.find("a", class_="ct-promo-card__title-link")
            agency_id = title_link.text.strip()
            url = title_link["href"]

            date_raw = cell.find("time", class_="ct-timestamp__start").text.strip()
            date_parsed = pd.to_datetime(date_raw, format="%d %b %Y", errors="coerce")
            date = date_parsed.strftime("%Y-%m-%d") if pd.notna(date_parsed) else None

            title = cell.find("div", class_="ct-promo-card__summary").text.strip()

            investigations.append(
                {
                    "title": title,
                    "url": url,
                    "agency_id": agency_id,
                    "occurrence_date": date,
                }
            )

        return pd.DataFrame(investigations)

    def __get_atsb_investigation_table_url(self, mode: str, page_num: int) -> str:
        """Constructs the URL for the ATSB investigation table based on the mode and page number.

        Args:
            mode: The transportation mode (aviation, rail, marine).
            page_num: The page number to access.

        Returns:
            The constructed URL for the ATSB investigation table.
        """
        dates = f"occurrence_date%5Bmin%5D={self.settings.start_year}-01-01&occurrence_date%5Bmax%5D={self.settings.end_year}-12-31"

        mode_to_number = {
            "aviation": 607,
            "rail": 610,
            "marine": 609,
        }

        return f"https://www.atsb.gov.au/investigations?atsb_sort=release_date_desc&transport_mode={mode_to_number[mode]}&investigation_number=&keywords=&location=&state=All&investigation_type=457&investigation_status=All&report_status=All&highest_injury_level=All&occurrence_class=All&{dates}&anticipated_completion%5Bmin%5D=&anticipated_completion%5Bmax%5D=&report_release_date%5Bmin%5D=&report_release_date%5Bmax%5D=&page={page_num}"

    def __get_atsb_investigations(self) -> pd.DataFrame:
        """ATSBs websites provides an investigation table than can be easily read by pandas read_html.

        The only catch is that the aviation goes all the way back to 1960s and so only the first few
        pages of the aviation table is scraped. It will then be combined with a complete scrape of
        the table to find the new ids.

        Returns:
            DataFrame of ATSB investigations indexed by mode.
        """
        investigations = self.report_table_dc.read_or_create()

        logger.debug(
            f"Existing investigations dataframe has {len(investigations)} rows before scraping."
        )

        dfs = []
        for mode in (
            pbar := tqdm(
                [Modes.Mode.as_string(mode).lower() for mode in self.settings.modes]
            )
        ):
            pages = []
            page_num = 0
            while True:
                pbar.set_description(f"Scraping mode: {mode}, page: {page_num}")

                page_url = self.__get_atsb_investigation_table_url(mode, page_num)

                try:
                    page = self.get(
                        page_url,
                    ).content

                except hrequests.exceptions.ClientException as e:
                    logger.warning(
                        f"Timeout while trying to scrape {mode} page {page_num}: {e}"
                    )
                    logger.info("Retrying...")
                    continue

                soup = BeautifulSoup(page, "html.parser")

                page_df = self.__parse_custom_divs_into_df(soup)

                if page_df is None or page_df.empty:
                    logger.warning(
                        f"No investigations found on page {page_num} for mode {mode}, stopping."
                    )
                    break

                page_df["url"] = page_df["url"].apply(
                    lambda x: urljoin("https://www.atsb.gov.au", x)
                )

                new_investigations = page_df[
                    ~page_df["agency_id"].isin(investigations["agency_id"])
                ].copy()

                new_investigations.loc[:, "year"] = pd.to_datetime(
                    new_investigations["occurrence_date"].to_list(),
                ).year

                new_investigations = new_investigations.query(
                    f"year >= {self.settings.start_year}"
                )

                new_investigations["report_id"] = new_investigations["agency_id"].map(
                    lambda x: re.search(r"(\d{3})$|(?:(?:\d{5})(\d{4}))$", str(x))
                )

                new_investigations = new_investigations.dropna(subset=["report_id"])
                new_investigations["report_id"] = new_investigations["report_id"].map(
                    lambda x: x.group(1) if x.group(1) is not None else x.group(2)
                )

                if new_investigations.empty:
                    logger.info(
                        f"Looking at page {page_num} for mode {mode}, found no new investigations, did find {len(page_df)} existing ones. Treating this as the end of the {mode} investigations and stopping."
                    )
                    break

                pages.append(new_investigations)

                page_num += 1

            if len(pages) == 0:
                logger.warning(f"No investigations found for mode: {mode}")
                continue

            mode_investigations = pd.concat(
                pages,
                ignore_index=True,
            )

            dfs.append(mode_investigations)

        if len(dfs) == 0:
            return investigations

        mode_keys = [m.name for m in self.settings.modes]
        new_investigations = pd.concat(dfs, axis=0, keys=mode_keys).reset_index(
            level=1, drop=True
        )

        updated_investigations = pd.concat(
            [investigations, new_investigations],
            axis=0,
        )

        self.report_table_dc.save(updated_investigations)

        return updated_investigations

    def get_report_urls(self, mode: Modes.Mode, year: int) -> list[ReportUrlData]:
        """Get report URLs for a specific mode and year.

        Args:
            mode: The transportation mode.
            year: The year to get reports for.

        Returns:
            List of ReportUrlData objects.
        """
        return [
            ReportUrlData(
                id=self.get_report_id(mode, year, str(row["agency_id"])[-3:]),
                url=row["url"],
                agency_id=row["agency_id"],
                occurrence_date=None,
                publication_date=None,
            )
            for row in self.agency_reports.loc[mode.name]
            .query(f"year == {year}")
            .dropna(subset=["url"])
            .to_dict("records")
        ]

    def get_report_metadata(
        self,
        report_id: str,
        url: str,
        occurrence_date: str | None,
        publication_date: str | None,
        soup: BeautifulSoup,
    ) -> ReportMetadata:
        """Extract report metadata from ATSB website.

        Args:
            report_id: The standardized report identifier.
            url: The URL of the report page.
            occurrence_date: The date of the occurrence.
            publication_date: The date of publication.
            soup: BeautifulSoup object of the report page.

        Returns:
            ReportMetadata object with extracted information.
        """
        report_mode = Modes.get_report_mode_from_id(report_id)

        title = soup.find("h1", class_="ct-banner__title").text.strip()

        table_div = soup.find(
            "div",
            class_="block-field-blocknodeinvestigation-reportfield-n-occurrence-date",
        )

        table_dict = self.html_table_to_dict(table_div)

        # Getting the safety summary
        summary = self.get_summary(soup)

        investigation_level = table_dict.get("Investigation level")
        investigation_type = "unknown"
        if investigation_level is None:
            investigation_type = "unknown"
        elif investigation_level in {"Defined", "Systemic"}:
            investigation_type = "full"
        else:
            investigation_type = "short"

        event_type = table_dict.get(
            f"{Modes.Mode.as_string(report_mode)} occurrence category"
        )

        agency_id = table_dict.get("Investigation number")

        raw_date = table_dict.get("Report release date")
        parsed_date = (
            pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
            if raw_date
            else pd.NaT
        )
        publication_date = (
            parsed_date.strftime("%Y-%m-%d") if pd.notna(parsed_date) else None
        )

        occurrence_date = table_dict.get("Occurrence date")
        parsed_occurrence_date = (
            pd.to_datetime(occurrence_date, dayfirst=True, errors="coerce")
            if occurrence_date
            else pd.NaT
        )
        occurrence_date = (
            parsed_occurrence_date.strftime("%Y-%m-%d")
            if pd.notna(parsed_occurrence_date)
            else None
        )

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=title,
            investigation_type=investigation_type,
            event_type=event_type,
            agency_id=agency_id,
            summary=summary,
            occurrence_date=occurrence_date,
            publication_date=publication_date,
            misc={"investigation_level": investigation_level},
        )

    @staticmethod
    def get_summary(soup: BeautifulSoup) -> str | None:
        """Gets the summary from the soup object.

        This is a placeholder as ATSB does not have a summary field in the report metadata.

        Args:
            soup: BeautifulSoup object of the report page.

        Returns:
            Summary text if found, None otherwise.
        """
        summary_div = soup.find("div", class_="atsb-investigation-section--first")
        if summary_div is None:
            return None

        summary_heading = None
        # Try to find h2 with "Executive summary" or "Investigation summary"
        for candidate in summary_div.find_all("h3"):
            heading = candidate.get_text(strip=True).lower()
            summary_headings = [
                "executive summary",
                "investigation summary",
                "safety summary",
            ]
            if any(h in heading for h in summary_headings):
                summary_heading = candidate
                break

        if summary_heading is None:
            h2_headings = list(summary_div.find_all("h2"))
            for candidate in h2_headings:
                if candidate.text.strip().lower() == "summary":
                    summary_heading = candidate
                    break

            if summary_heading is None:
                prelim_heading = [
                    h
                    for h in summary_div.find_all("h2")
                    if h.text.strip() == "Preliminary report"
                ]
                if len(prelim_heading) > 0:
                    summary_heading = prelim_heading[0]

            if summary_heading is None:
                # If we still can't find a summary heading, just return the whole text of the div as the summary
                logger.debug(
                    "Could not find specific summary heading, returning full text of the content div as summary."
                )
                return summary_div.get_text(" ", strip=True)

        logger.debug(f"Found summary heading: {summary_heading.text.strip()}")

        summary_parts = []
        for sibling in summary_heading.find_next_siblings():
            if sibling.name in {"h1", "h2", "h3"} or sibling.class_ == "show-more":
                break
            if (
                sibling.find("h2") is not None
                or sibling.get_text("", strip=True).lower() == "the occurrence"
            ):
                break
            logger.debug(
                f"Adding sibling to summary: {sibling.name}, text: {sibling.get_text(strip=True)[:100]}..."
            )
            summary_parts.append(sibling.get_text(" ", strip=True))

        return "\n".join([part for part in summary_parts if part.strip()])


class TSBReportScraper(ReportScraper):
    """Report scraper for the Transportation Safety Board (TSB) of Canada."""

    def __init__(self, settings: ReportScraperSettings):
        """Initialize TSBReportScraper.

        Args:
            settings: Configuration settings for the scraper.
        """
        super().__init__(settings, agency="TSB")

        self.agency_reports = self.__get_tsb_investigations()

    def __get_tsb_investigations(self) -> pd.DataFrame:
        """The TSB is very well setup and works friendly with the pandas read_html.

        Therefore I can just read all of the investigation tables from the TSB website
        and then have the exact IDs I need.

        Returns:
            DataFrame of TSB investigations indexed by mode.
        """
        modes = ["aviation", "rail", "marine"]

        def _read_table(mode: str) -> pd.DataFrame:
            url = f"https://www.tsb.gc.ca/eng/rapports-reports/{mode}/index.html"
            response = self.get(url)
            return pd.read_html(io.BytesIO(response.content), flavor="lxml")[0]

        modes_df = [_read_table(mode) for mode in tqdm(modes)]

        # Add dataframes togather with extra column identifier

        merged_modes_df = pd.concat(
            modes_df, keys=[Modes.Mode.a, Modes.Mode.r, Modes.Mode.m]
        )

        merged_modes_df["Occurrence date"] = pd.to_datetime(
            merged_modes_df["Occurrence date"], format="%Y-%m-%d", errors="coerce"
        )
        merged_modes_df["Report release date"] = pd.to_datetime(
            merged_modes_df["Report release date"], format="%Y-%m-%d", errors="coerce"
        )

        merged_modes_df["year"] = merged_modes_df["Occurrence date"].dt.year

        return merged_modes_df

    def get_report_urls(self, mode: Modes.Mode, year: int) -> list[ReportUrlData]:
        """Get report URLs for a specific mode and year.

        Args:
            mode: The transportation mode.
            year: The year to get reports for.

        Returns:
            List of ReportUrlData objects.
        """
        return [
            ReportUrlData(
                id=self.get_report_id(mode, year, row["Investigation number"][-5:]),
                url=f"https://www.tsb.gc.ca/eng/rapports-reports/{Modes.Mode.as_string(mode)}/{year}/{row['Investigation number']}/{row['Investigation number']}.html".lower(),
                agency_id=row["Investigation number"],
                occurrence_date=row["Occurrence date"].strftime("%Y-%m-%d")
                if pd.notna(row["Occurrence date"])
                else None,
                publication_date=row["Report release date"].strftime("%Y-%m-%d")
                if pd.notna(row["Report release date"])
                else None,
            )
            for row in self.agency_reports.loc[mode]
            .query(f"year == {year} & `Investigation status` == 'Completed'")
            .to_dict("records")
        ]

    @staticmethod
    def _extract_title_from_title_block(title_block: Tag) -> tuple[str, str | None]:
        """Extract title information from the title block element.

        Args:
            title_block: The BeautifulSoup element containing title block.

        Returns:
            Tuple of (title_text, event_type).
        """
        event_type = title_block.find("strong")
        if event_type:
            event_type = event_type.text

        legacy_text_div = title_block.find(
            "div", class_="field--name-field-occurrence-legacy-text"
        )
        paragraph_text = []
        if legacy_text_div:
            paragraph = legacy_text_div.find("p")
            if paragraph:
                paragraph_text = list(paragraph.stripped_strings)
            else:
                paragraph_text = list(legacy_text_div.stripped_strings)

        date_div = title_block.find("div", class_="field--name-field-occurrence-date")
        date_text = ""
        if date_div:
            time_tag = date_div.find("time")
            if time_tag:
                date_text = time_tag.text

        title = ", ".join([*paragraph_text, date_text])
        return title, event_type

    def _extract_investigation_level(self, soup: BeautifulSoup) -> str | None:  # noqa: PLR6301
        """Extract investigation level from soup.

        Args:
            soup: BeautifulSoup object of the report page.

        Returns:
            Investigation level string or None.
        """
        investigation_level = None
        h3_element = soup.find("h3", string="Class of investigation")
        if h3_element:
            text = h3_element.find_next_sibling("p").text
            match = re.match(r"This is a class (\d) investigation", text)
            if match:
                investigation_level = match.group(1)
        return investigation_level

    def _get_investigation_type(self, investigation_level: str | None) -> str:  # noqa: PLR6301
        """Get investigation type string from investigation level.

        Args:
            investigation_level: The investigation level string or None.

        Returns:
            Investigation type string.
        """
        if investigation_level is None:
            return "unknown"
        if investigation_level in {"1", "2", "3"}:
            return "full"
        return "short"

    def get_report_metadata(
        self,
        report_id: str,
        url: str,
        occurrence_date: str | None,
        publication_date: str | None,
        soup: BeautifulSoup,
    ) -> ReportMetadata:
        """Extract report metadata from TSB website.

        Args:
            report_id: The standardized report identifier.
            url: The URL of the report page.
            occurrence_date: The date of the occurrence.
            publication_date: The date of publication.
            soup: BeautifulSoup object of the report page.

        Returns:
            ReportMetadata object with extracted information.
        """
        # Due to TSB having the metadata on a page separate from the report pdf link, we need to get the new page
        split_id = report_id.split("_")
        tsb_id = f"{split_id[1]}{split_id[2][2:4]}{split_id[3]}"
        page = self.get(
            f"https://www.tsb.gc.ca/eng/enquetes-investigations/{Modes.Mode.as_string(Modes.get_report_mode_from_id(report_id))}/{split_id[2]}/{tsb_id}/{tsb_id}.html",
        )
        overview_page = BeautifulSoup(page.content, "html.parser")

        if (
            overview_page.find("h1", string="Page not found") is None
        ):  # Some of the older reports dont have the overview page
            soup = overview_page

        title_block = soup.find("div", class_="field--name-field-occurrence")
        if title_block is None:
            # Return a minimal ReportMetadata when title_block is not found
            return ReportMetadata(
                url=url,
                report_id=report_id,
                title="Unknown",
                event_type=None,
                investigation_type="unknown",
                summary=None,
                misc={},
                agency_id=None,
                occurrence_date=occurrence_date,
                publication_date=publication_date,
            )

        title, event_type = self._extract_title_from_title_block(title_block)
        investigation_level = self._extract_investigation_level(soup)
        investigation_type = self._get_investigation_type(investigation_level)

        agency_id = soup.find("h1", class_="page-header").text
        if agency_id is not None:
            agency_id = agency_id.strip().split(" ")[-1]

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=title,
            event_type=event_type,
            investigation_type=investigation_type,
            agency_id=agency_id,
            summary=None,  # TSB does not include summary text. However the press releases provide a summary of sorts.
            misc={"investigation_class": investigation_level},
            occurrence_date=occurrence_date,
            publication_date=publication_date,
        )


class ATSBSafetyIssueScraper(WebsiteScraper):
    """Scraper for extracting safety issues from the Australian Transport Safety Bureau (ATSB) website."""

    def __init__(
        self,
        safety_issues_dc: ATSBWebsiteSafetyIssues,
        report_titles_dc: ReportTitles,
        refresh: bool = False,
    ):
        """Initialize ATSBSafetyIssueScraper.

        Args:
            safety_issues_dc: ATSBWebsiteSafetyIssues dataframe manager.
            report_titles_dc: ReportTitles dataframe manager.
            refresh: Whether to refresh all safety issues from scratch.
        """
        super().__init__(report_titles_dc)
        self.safety_issues_dc = safety_issues_dc
        self.refresh = refresh

    @staticmethod
    def _process_safety_issues_page(
        html_content: str, mode: str, current_page: int, pbar: tqdm
    ) -> None:
        """Process a single page of safety issues.

        Args:
            html_content: The HTML content of the page.
            mode: The transportation mode (A, R, M).
            current_page: The current page number.
            pbar: The progress bar object.

        Returns:
            DataFrame of safety issues from the page, or None if page has no issues.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        table = soup.find("div", class_="ct-list__rows")

        if table is None:
            logger.info(
                f"Failed to scrape page {current_page} of mode {mode}. No table found, likely end of pages. Stopping."
            )
            return None

        safety_issues = []

        for row in table.select(".col-xxs-12"):
            if not row.contents:
                logger.debug(
                    f"Skipping empty row on page {current_page} of mode {mode}"
                )
                continue
            link = row.find("a", class_="ct-promo-card__title-link")
            if link is None:
                logger.warning(
                    f"Failed to find link for a safety issue on page {current_page} of mode {mode}. Skipping this issue. Likely due to it being the last page"
                )
                continue

            # IMPORTANT! This drops some safety issues from the scrape
            # This sometimes happens with older Safety issues where there is no date. For example on page: https://www.atsb.gov.au/safety-issues-and-actions?issue_owner=&issue_number=&date_issue_released%5Bmin%5D=&date_issue_released%5Bmax%5D=&issue_status=All&transport_mode=607&field_p_transport_function_target_id=All&page=48 you can see many don't have dates.
            date = row.find("time", class_="ct-timestamp__start")
            if date is None:
                logger.warning(
                    f"Failed to find date for a safety issue {link.text.strip()} on page {current_page} of mode {mode}. Skipping this issue."
                )
                continue

            safety_issues.append(
                {
                    "safety_issue_id": link.text.strip(),
                    "safety_issue_title": row.find(
                        "div", class_="ct-promo-card__summary"
                    ).text.strip(),
                    "safety_issue_link": link["href"].strip(),
                    "safety_issue_date": date.text.strip(),
                }
            )

        return pd.DataFrame(safety_issues)

    def extract_safety_issue_details(
        self, safety_issue_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract details for each safety issue by visiting their individual pages.

        Args:
            safety_issue_df: DataFrame containing safety issues with their links.

        Returns:
            DataFrame with extracted details for each safety issue.
        """
        # Only extract details for the new safety issue (i.e ones that have none as Safety Issue description).
        to_be_extracted = safety_issue_df[safety_issue_df["safety_issue"].isna()]

        already_extracted = safety_issue_df[
            ~safety_issue_df["report_id"].isin(to_be_extracted["report_id"])
        ]

        logger.info(
            f"Extracting details for {to_be_extracted.shape[0]} safety issues out of {safety_issue_df.shape[0]} total safety issues"
        )

        detailed_safety_issues = []

        for _, row in tqdm(
            to_be_extracted.iterrows(),
            total=to_be_extracted.shape[0],
            desc="Extracting safety issue details",
        ):
            url = "https://www.atsb.gov.au" + row["safety_issue_link"]
            try:
                response = self.get(url)
            except (hrequests.exceptions.ClientException, ScraperRequestError) as e:
                logger.warning(
                    f"Error while trying to scrape safety issue details for {row['safety_issue_id']} from {url}: {e}"
                )
                continue

            soup = BeautifulSoup(response.content, "html.parser")

            # Find the table that is a sibling of the h2 with text "Safety issue"
            h2 = soup.find("h2", string="Safety issue")
            if h2 is None:
                logger.warning(f"Failed to find safety issue details table in {url}")
                continue
            table = h2.find_next_sibling("table")

            if table is None:
                logger.warning(f"Failed to find safety issue details table in {url}")
                continue

            details_dict = self.html_table_to_dict(table)

            row["safety_issue"] = details_dict.pop("Safety issue description")

            detailed_safety_issues.append(row)

        new_si = pd.DataFrame(detailed_safety_issues)

        return pd.concat([already_extracted, new_si], ignore_index=True)

    def _format_and_save_safety_issues(self, safety_issue_df: pd.DataFrame) -> None:
        """Format safety issues and save them to disk.

        This methods creates the report id column from the safety issue id, and saves the dataframe to disk.

        Args:
            safety_issue_df: DataFrame containing all safety issues.
        """
        safety_issue_df = safety_issue_df.drop_duplicates(subset=["safety_issue_id"])

        safety_issue_df.loc[:, "report_id"] = (
            safety_issue_df["safety_issue_id"]
            .map(
                # This needed because the safety issue id is simply the agency_id plus some extrae identifiers on the end.
                lambda x: "-".join(x.split("-")[0:3])
            )
            .map(
                lambda x: (
                    self.id_converter("ATSB", x)
                    if self.id_converter("ATSB", x)
                    else f"Unmatched safety issue ({x})"
                )
            )
        )
        safety_issue_df.loc[:, "quality"] = "exact"
        safety_issue_df = safety_issue_df[
            ["report_id", "safety_issue_id", "safety_issue", "quality"]
        ]

        logger.info(f"""Now there are {safety_issue_df.shape[0]} safety issues
Spread across {safety_issue_df["report_id"].nunique()} reports""")

        self.safety_issues_dc.save(safety_issue_df)

    def extract_safety_issues_from_website(self) -> None:
        """Extract all safety issues from ATSB website."""
        if self.refresh:
            safety_issues_df = self.safety_issues_dc.create_empty()
        else:
            safety_issues_df = self.safety_issues_dc.read_or_create()

        base_url = "https://www.atsb.gov.au/safety-issues-and-actions?transport_mode={mode}&page={page}"

        starting_count = safety_issues_df.shape[0]

        logger.welcome(
            "Extracting safety issues from ATSB website",
            {
                "Output directory": self.safety_issues_dc.path,
                "Base URL": base_url,
                "Number of safety issues:": safety_issues_df.shape[0],
                "Number of reports with safety issues:": safety_issues_df[
                    "report_id"
                ].nunique(),
            },
        )
        mode_to_mode_num = {
            "A": "607",
            "R": "610",
            "M": "609",
        }

        for mode in (pbar := tqdm(["A", "R", "M"])):
            current_page = 0
            pbar.set_description(f"Scraping {mode} safety issues")

            max_failures = 5
            failed = 0
            while True:
                pbar.set_description(f"Scraping page {current_page} of mode {mode}")
                url = base_url.format(mode=mode_to_mode_num[mode], page=current_page)
                try:
                    response = self.get(url)
                except ScraperRequestError as e:
                    pbar.write(
                        f"Failed to scrape page {current_page} of mode {mode}: {e}"
                    )
                    failed += 1
                    if failed > max_failures:
                        failed = 0
                        current_page += 1
                    continue

                table = self._process_safety_issues_page(
                    response.content, mode, current_page, pbar
                )

                if table is None:
                    break

                new_safety_issues = table[
                    ~table["safety_issue_id"].isin(safety_issues_df["safety_issue_id"])
                ]

                if new_safety_issues.empty:
                    pbar.write(
                        f"No new safety issues found on page {current_page} of mode {mode}, moving onto next mode"
                    )
                    break

                safety_issues_df = pd.concat(
                    [safety_issues_df, new_safety_issues], ignore_index=True
                )

                current_page += 1

        detailed_safety_issues_df = self.extract_safety_issue_details(safety_issues_df)

        logger.info(
            f"Finished extracting safety issue details, now formatting and saving them, found {detailed_safety_issues_df.shape[0] - starting_count} new safety issues"
        )

        self._format_and_save_safety_issues(detailed_safety_issues_df)


class RecommendationScraper(WebsiteScraper, ABC):
    """Abstract base class for scraping recommendations from different transportation safety agencies.

    Subclasses must define these class variables:
    - BASE_URL: The base URL of the agency website
    - LOOP_ITER: Iterator for pagination or mode iteration
    - AGENCY: The agency name (e.g., 'TSB', 'TAIC')

    Subclasses must also implement abstract methods to handle agency-specific website structures.
    """

    # Class variables to be defined by subclasses
    BASE_URL: ClassVar[str]
    LOOP_ITER: ClassVar[list | range]
    AGENCY: ClassVar[str]
    BREAK_ON_NO_NEW_RECOMMENDATIONS: ClassVar[bool] = True

    def __init__(
        self,
        recommendations_dc: TSBWebsiteRecommendations | TAICWebsiteRecommendations,
        report_titles_dc: ReportTitles,
        refresh: bool = False,
    ):
        """Initialize RecommendationScraper.

        Args:
            recommendations_dc: The data class for recommendations.
            report_titles_dc: The data class for report titles.
            refresh: Whether to refresh all recommendations from scratch.
        """
        super().__init__(report_titles_dc)

        self.recommendations_dc = recommendations_dc
        self.refresh = refresh
        self.base_url = self.BASE_URL
        self.loop_iter = self.LOOP_ITER
        self.agency = self.AGENCY

    def extract_recommendations_from_website(self) -> None:
        """Extract all recommendations from the agency website.

        It does this through two passes. The first pass is to read through the recommendation tables. Then it will open up each recommendation page to extract the extra information.
        """
        if self.refresh:
            recommendations_df = self.recommendations_dc.create_empty()
        else:
            recommendations_df = self.recommendations_dc.read_or_create()

        logger.welcome(
            f"Scraping recommendations from {self.agency} website",
            {
                "Output directory": self.recommendations_dc.path,
                "Base URL": self.base_url,
                "Number of recommendations:": recommendations_df.shape[0],
                "Number of reports with recommendations:": recommendations_df[
                    "report_id"
                ].nunique(),
            },
        )

        logger.info("Reading recommendation tables to get recommendations webpages")

        all_recommendation_ids = recommendations_df["recommendation_id"]

        new_recommendations = self.recommendations_dc.create_empty()

        for element in (phbar := tqdm(self.loop_iter)):
            phbar.set_description(f"Scraping recommendations for {element}")

            table = self.get_table(element)

            if table.empty:
                break

            table = self.process_new_table(table)

            if len(recommendations_df) > 0:
                table = table[~table["recommendation_id"].isin(all_recommendation_ids)]

            if len(table) == 0 and self.BREAK_ON_NO_NEW_RECOMMENDATIONS:
                break
            new_recommendations = pd.concat(
                [new_recommendations, table], ignore_index=True
            )

        # Drop ones without a URL as we cannot extract the recommendation data without a URL
        new_recommendations = new_recommendations.dropna(subset=["url"])

        logger.info(
            f"Found {new_recommendations.shape[0]} new recommendations, reading each individual webpage now"
        )

        for i, row in (phbar := tqdm(list(new_recommendations.iterrows()))):
            phbar.set_description(
                f"Processing recommendation {row['recommendation_id']} from {row['agency_id']} with i:{i}"
            )
            recommendation_data = self.extract_recommendation_data(row["url"])
            if recommendation_data is None:
                new_recommendations = new_recommendations.drop(i)
                continue
            for key, value in recommendation_data.items():
                new_recommendations.loc[i, key] = value

        if len(new_recommendations) == 0:
            logger.info(
                "No new recommendations found, only updating report_id for existing recommendations"
            )
        else:
            recommendations_df = pd.concat(
                [recommendations_df, new_recommendations], ignore_index=True
            )

        # Always derive the complete report_id from the agency_id to ensure consistency and correctness.
        recommendations_df["report_id"] = recommendations_df["agency_id"].map(
            lambda x: (
                self.id_converter(self.agency, x)
                if self.id_converter(self.agency, x)
                else f"Unmatched {self.agency} ({x})"
            )
        )

        self.recommendations_dc.save(recommendations_df)

    @abstractmethod
    def extract_recommendation_data(self, url: str) -> dict | None:
        """Goes to the URL and extracts the needed data.

        This method must be implemented by subclasses to handle agency-specific
        recommendation data extraction from individual recommendation pages.

        Args:
            url: The URL of the recommendation page.

        Returns:
            Dictionary containing extracted recommendation data.
        """

    @abstractmethod
    def process_new_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Takes a recently read table and processes it according to agency-specific rules.

        This method must be implemented by subclasses to handle agency-specific
        table processing and column mapping.

        Args:
            table: The raw table data from the website.

        Returns:
            Processed table with standardized columns.
        """

    @abstractmethod
    def get_url(self, element: str | int) -> str:
        """Generates the URL for a given element (page number, mode, etc.).

        This method must be implemented by subclasses to handle agency-specific
        URL generation patterns.

        Parameters
        ----------
        element : str | int
            The element used to generate the URL (e.g., page number, mode)

        Returns:
        -------
        str
            The complete URL for the given element
        """

    @abstractmethod
    def get_table(self, element: str | int) -> pd.DataFrame:
        """Retrieves the recommendation table from the website for a given element.

        This method must be implemented by subclasses to handle agency-specific
        table retrieval logic.

        Parameters
        ----------
        element : Any
            The element used to retrieve the table (e.g., page number, mode)

        Returns:
        -------
        pd.DataFrame
            The recommendation table retrieved from the website
        """


class TSBRecommendationsScraper(RecommendationScraper):
    """Recommendations scraper for the Transportation Safety Board (TSB) of Canada."""

    BASE_URL: ClassVar[str] = "https://www.tsb.gc.ca"
    LOOP_ITER: ClassVar[list[str]] = ["rail", "marine", "aviation"]
    AGENCY: ClassVar[str] = "TSB"
    BREAK_ON_NO_NEW_RECOMMENDATIONS = False

    def get_url(self, element: str) -> str:
        """Get the URL for a recommendation element.

        Args:
            element: The element identifier.

        Returns:
            The URL string for the recommendation.
        """
        return (
            f"{self.base_url}/eng/recommandations-recommendations/{element}/index.html"
        )

    def get_table(self, element: str) -> pd.DataFrame:
        """Get the recommendations table for a specific element.

        Args:
            element: The element identifier.

        Returns:
            DataFrame containing the recommendations table.
        """
        url = self.get_url(element)
        try:
            response = self.get(url)
        except ScraperRequestError as e:
            logger.warning(f"Failed to scrape recommendations from {url}: {e}")
            return pd.DataFrame()

        tables = pd.read_html(
            io.BytesIO(response.content),
            flavor="lxml",
            extract_links="body",
        )

        if len(tables) == 0:
            logger.warning(f"No tables found on {url}")
            return pd.DataFrame()
        if len(tables) > 1:
            logger.info(f"Multiple tables found on {url}, using the first one")
        return tables[0]

    def extract_recommendation_data(self, url: str) -> dict | None:
        """Read the webpage and extract recommendation data.

        This will read the webpage and extract:
        - recommendation (This is because sometimes the recommendation inside the website table is not complete)
        - recommednation date
        - recommendation context
        ## TODO: Add in the recipient and reply text. This is not done at the moment as it is not needed
        - recipient
        - reply text.

        Args:
            url: The URL of the recommendation page.

        Returns:
            Dictionary containing extracted recommendation data with keys:
            - recommendation: The recommendation text
            - made: The date recommendation was made
            - recommendation_context: The context/rationale for the recommendation
        """
        try:
            response = self.get(url)
        except ScraperRequestError as e:
            logger.warning(str(e))
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        recommendation = soup.find(
            "div", class_="field--name-field-recommendation-well"
        )
        recommendation = recommendation.get_text() if recommendation else None

        recommendation_date = soup.find(
            "div", class_="field--name-field-recommendation-issued"
        )
        if recommendation_date is not None:
            recommendation_date = recommendation_date.find("time")["datetime"]
            recommendation_date = datetime.fromisoformat(
                recommendation_date.replace("Z", "+00:00")
            )

        context = soup.find("div", class_="field--name-field-recommendation-rationale")
        recommendation_context = None
        if context is not None:
            recommendation_context = "\n".join(
                [child.get_text() for child in context.children if child.name == "p"]
            )

        if (
            recommendation is None
            and recommendation_date is None
            and recommendation_context is None
        ):
            logger.warning(f"No recommendation data found at {url}")
            return None

        return {
            "recommendation": recommendation,
            "made": recommendation_date,
            "recommendation_context": recommendation_context,
        }

    def process_new_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Process a new recommendations table.

        Args:
            table: The raw recommendations table to process.

        Returns:
            Processed recommendations table with normalized columns.
        """
        # filter out older recommendations that are from the previous century
        table = table[
            table["Number"]
            .map(
                lambda x: int(
                    re.match(r"[amr](\d{2})-\d{2}", x[0], re.IGNORECASE).group(1)
                )
            )
            .between(0, 80)
        ].copy()

        # Add in the url

        table.loc[:, "url"] = table["Number"].map(
            lambda x: f"{self.base_url}{x[1]}" if x[1] else None
        )

        # Remove tuples

        table = table.map(lambda x: x[0] if isinstance(x, tuple) else x)

        # Give proper column names
        table.columns = [
            "recommendation_id",
            "recommendation",
            "agency_id",
            "current_assessment",
            "status",
            "watchlist",
            "url",
        ]

        # Remove rows with empty recommendation
        table = table[~table["recommendation"].isna()]

        return table.drop("recommendation", axis=1)


class TAICRecommendationsScraper(RecommendationScraper):
    """Recommendation scraper for the Transport Accident Investigation Commission (TAIC) of New Zealand.

    This scraper extracts recommendations from the TAIC recommendations webpage.
    """

    BASE_URL: ClassVar[str] = "https://www.taic.org.nz"
    LOOP_ITER: ClassVar[range] = range(300)
    AGENCY: ClassVar[str] = "TAIC"

    def get_url(self, element: int) -> str:
        """Get the URL for a recommendations page element.

        Args:
            element: The page number or element identifier.

        Returns:
            The URL string for the recommendations page.
        """
        return f"{self.base_url}/inquiries-recommendations?type=recommendation&field_recipient_name=All&sort_by=latest&page={element}"

    def get_table(self, element: int) -> pd.DataFrame:
        """Get the recommendations table for a specific page.

        Args:
            element: The page number or element identifier.

        Returns:
            DataFrame containing recommendations from the page.
        """
        url = self.get_url(element)

        try:
            response = self.get(url)
        except hrequests.exceptions.ClientException as e:
            logger.warning(f"Timeout while scraping TAIC recommendations listing: {e}")
            return pd.DataFrame(columns=["recommendation_id", "url"])

        soup = BeautifulSoup(response.content, "html.parser")

        results_list = soup.select_one("div.search-results__list")
        if not results_list:
            logger.info(
                f"Reached end of pages at page {element} (no results list), stopping."
            )
            return pd.DataFrame(columns=["recommendation_id", "url"])

        # In the current TAIC site, the cards are nested inside the list grid.
        # Example DOM (from browser):
        #   <div class="search-results__list ...">
        #       <div class="card card-type--recommendation" aria-label="044/25">...</div>
        #       ...
        #   </div>
        cards = results_list.select("div.card.card-type--recommendation")
        if not cards:
            logger.info(f"Reached end of pages at page {element} (no cards), stopping.")
            return pd.DataFrame(columns=["recommendation_id", "url"])

        rows: list[dict[str, str]] = []
        for card in cards:
            title = card.select_one(".card__title")
            rec_id = title.get_text(" ", strip=True).strip() if title else ""

            a = card.select_one("a[href]")
            href_attr = a.get("href") if a else None
            href = str(href_attr) if href_attr else None
            full_url = urljoin(self.base_url, href) if href else None

            if rec_id and full_url:
                rows.append({"recommendation_id": rec_id, "url": full_url})

        return pd.DataFrame(rows, columns=["recommendation_id", "url"])

    def process_new_table(self, table: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR6301
        """Process a new recommendations table.

        Args:
            table: The raw recommendations table to process.

        Returns:
            Processed recommendations table (unchanged in this implementation).
        """
        return table

    def extract_recommendation_data(self, url: str) -> dict | None:
        """Read the actual recommendation page and extract needed data.

        This will extract information that is not found in the table:
        - recommendation_text
        - reply_text (not currently extracted because new website does not support it)
        - recipient
        - made
        - agency_id.

        Args:
            url: The URL of the recommendation page.

        Returns:
            Dictionary containing extracted recommendation data with keys:
            - recommendation: The recommendation text
            - reply_text: Any reply text (may be None)
            - recipient: The recipient of the recommendation
            - made: The date the recommendation was made
            - agency_id: The agency identifier
        """
        try:
            response = self.get(url)
        except ScraperRequestError as e:
            logger.warning(str(e))
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        recommendation_text = soup.find(
            "div", class_="field--name-field-rich-content"
        ).get_text()

        header = soup.select_one("dl.screen-head__metadata.metadata")

        out: dict[str, object] = {}
        for item in header.select("div.metadata__item"):
            dt = item.find("dt", class_="metadata__label")
            dd = item.find("dd", class_="metadata__value")

            if not dt or not dd:
                continue

            label = dt.get_text(" ", strip=True)

            # "Related inquiries"
            links = dd.select("a")
            if links:
                out[label] = [
                    {"text": a.get_text(" ", strip=True), "href": a.get("href")}
                    for a in links
                    if a.get("href")
                ]
                continue

            # "Issue date"
            time_tag = dd.find("time")
            if time_tag:
                out[label] = time_tag.get_text(
                    " ", strip=True
                )  # if you want "19 Sep 2018"
                continue

            # FOr recipient
            out[label] = dd.get_text(" ", strip=True)

        agency_id = (
            out.get("Related inquiries")[0]["href"].split("/")[-1].upper()
            if "Related inquiries" in out
            else None
        )
        made_str = out.get("Issue date")
        made = (
            pd.to_datetime(made_str, format="%d %b %Y", errors="coerce")
            if made_str
            else None
        )

        return {
            "recommendation": recommendation_text,
            "reply_text": None,
            "recipient": out.get("Recipient"),
            "made": made,
            "agency_id": agency_id,
        }
