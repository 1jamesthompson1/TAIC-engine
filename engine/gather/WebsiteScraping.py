import os
import random
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Literal
from urllib.parse import urljoin, urlparse

import hrequests
import hrequests.exceptions
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from ..utils import Modes
from ..utils.AzureStorage import PDFStorageManager


class ReportMetadata:
    """
    Small data class to hold the metadata so that it can be passed around easily between the functions.
    """

    # Define the column order based on property names
    COLUMN_ORDER = [
        "report_id",
        "title",
        "event_type",
        "investigation_type",
        "summary",
        "misc",
        "url",
        "agency_id",
    ]

    def __init__(
        self,
        report_id,
        title,
        event_type,
        investigation_type,
        summary,
        misc,
        url,
        agency_id,
    ):
        self.report_id = report_id
        self.title = title
        self.event_type = event_type
        self.investigation_type = investigation_type
        self.summary = summary
        self.misc = misc
        self.url = url
        self.agency_id = agency_id

    def __repr__(self):
        return f"ReportMetadata({self.report_id}, {self.title})"

    def as_report_row(self):
        """Return values in the order defined by COLUMN_ORDER"""
        return [getattr(self, col) for col in self.COLUMN_ORDER]

    @classmethod
    def get_column_names(cls):
        """Return the column names in the correct order"""
        return cls.COLUMN_ORDER.copy()


class ReportScraperSettings:
    def __init__(
        self,
        report_titles_file_path,
        start_year,
        end_year,
        max_per_year,
        modes: list[Modes.Mode],
        ignored_report_ids: list[str],
        refresh,
        pdf_storage_manager: PDFStorageManager,
        refresh_metadata=False,
        scraper_workers=1,
    ):
        self.report_titles_file_path = report_titles_file_path
        self.start_year = start_year
        self.end_year = end_year
        self.max_per_year = max_per_year
        self.refresh = refresh
        self.modes = modes
        self.ignored_report_ids = ignored_report_ids
        self.ignore_metadata = refresh_metadata
        self.scraper_workers = scraper_workers
        self.pdf_storage_manager = pdf_storage_manager


class WebsiteScraper(ABC):
    """
    Abstract base class for scraping websites.
    Provides common functionality for HTTP requests and ID conversion.
    """

    def __init__(self, report_titles_file_path):
        # Keep the original headers as a fallback, but we'll use get_randomized_headers() instead
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Connection": "keep-alive",
        }

        if os.path.exists(report_titles_file_path):
            report_titles_df = pd.read_pickle(report_titles_file_path)
        else:
            raise ValueError(f"{report_titles_file_path} does not exist")
        self.id_dict = {
            agency: {
                agency_id: report_id
                for report_id, agency_id in ids[["report_id", "agency_id"]].values
            }
            for agency, ids in report_titles_df.assign(
                agency=report_titles_df["report_id"].map(lambda x: x.split("_")[0])
            ).groupby("agency")
        }

    def get_randomized_headers(self):
        """
        Generate randomized headers to avoid bot detection.
        Returns a random permutation of header options.
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
            if random.random() < 0.7:  # 70% chance to include each additional header
                headers[header_name] = header_value

        # Randomly shuffle the order of headers
        header_items = list(headers.items())
        random.shuffle(header_items)

        return dict(header_items)

    def id_converter(self, agency, agency_id):
        """
        This uses the COMPLETE report titles dataframe and creates a dictionary of agency_id:report_id.
        The agency argument is needed as some of the agency ids are not globally unique but instead just unique within an agency.
        """
        if agency not in self.id_dict:
            raise ValueError(f"{agency} is not a valid agency")
        else:
            return self.id_dict[agency].get(agency_id)


class ReportScraper(WebsiteScraper, ABC):
    """
    Abstract base class for scraping reports from different transportation safety agencies.
    Subclasses must implement abstract methods to handle agency-specific website structures.
    """

    def __init__(self, settings: ReportScraperSettings, agency: str):
        self.settings = settings
        self._metadata_file_lock = Lock()
        if os.path.exists(self.settings.report_titles_file_path):
            self.report_titles_df = pd.read_pickle(
                self.settings.report_titles_file_path
            )
            # Make sure index is from 0 to n-1
            self.report_titles_df.reset_index(drop=True, inplace=True)
        else:
            self.report_titles_df = pd.DataFrame(
                columns=ReportMetadata.get_column_names()
            )
            self.report_titles_df.to_pickle(self.settings.report_titles_file_path)
        self.agency = agency
        super().__init__(self.settings.report_titles_file_path)
        print(
            "=============================================================================================================================\n"
        )
        print(
            "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n"
        )
        print(
            f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - Downloading report PDFs for {self.agency} - - - - - - - - - - - - - - - - - - - - - -"
        )
        print(
            "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n"
        )
        print(
            "=============================================================================================================================\n"
        )
        print(
            f"  PDF storage container: {self.settings.pdf_storage_manager.container_name}"
        )
        print(f"  Report titles file path: {self.settings.report_titles_file_path}")
        print(
            f"  Start year: {self.settings.start_year},  End year: {self.settings.end_year}"
        )
        print(f"  Max reports per year: {self.settings.max_per_year}")
        print(f"  Modes: {self.settings.modes}")
        print(f"  Ignoring report ids: {self.settings.ignored_report_ids}")

    def collect_all(self):
        # Loop through each mode
        for mode in self.settings.modes:
            self.collect_mode(mode)

    @abstractmethod
    def get_report_urls(
        self, mode: Modes.Mode, year: int
    ) -> list[tuple[str, str, str | None]]:
        """
        Retrieves all the potential report urls and ids for a given mode and year.

        This method must be implemented by subclasses to handle agency-specific
        website structures for finding report URLs.

        Parameters
        ----------
        mode : Modes.Mode
            The mode/type of transportation (aviation, rail, marine)
        year : int
            The year to search for reports

        Returns
        -------
        list[tuple[str, str, str | None]]
            List of tuples containing (report_id, report_url, agency_id)
        """
        pass

    def get_report_id(self, mode: Modes.Mode, year: int, id: str) -> str:
        return f"{self.agency}_{mode.name}_{year}_{id}"

    def collect_mode(self, mode):
        print(f"======== Downloading reports for mode: {mode.name}==========")

        year_range = [
            year for year in range(self.settings.start_year, self.settings.end_year + 1)
        ]

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

                    print(
                        f"==Completed year {year_result['year']} for mode {mode.name}: "
                        f"{year_result['reports_collected']} reports collected out of {year_result['potential_reports']} in "
                        f"{year_result['duration']:.2f} seconds=="
                    )

                except Exception as exc:
                    print(f"==Year {year} generated an exception: {exc}==")

        # Print final summary
        total_reports = sum(result["reports_collected"] for result in completed_years)
        total_time = sum(result["duration"] for result in completed_years)
        avg_time = total_time / len(completed_years) if completed_years else 0

        print(f"==Finished downloading reports for mode: {mode.name}==")
        print(f"  Total reports collected: {total_reports}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per year: {avg_time:.2f} seconds")

    def collect_year(self, year, mode):
        """Helper method to collect a year's reports with timing information"""
        start_time = datetime.now()

        number_for_year = 0
        report_urls = self.get_report_urls(mode, year)
        for report_id, url, agency_id in report_urls:
            if report_id in self.settings.ignored_report_ids:
                continue

            outcome = self.collect_report(report_id, url, agency_id)

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

    def collect_report(self, report_id, url, agency_id=None):
        if (
            not self.settings.ignore_metadata
            and self.report_titles_df.query(f"report_id == '{report_id}'").shape[0] > 0
        ):
            return True

        try:
            webpage = hrequests.get(
                url, headers=self.get_randomized_headers(), timeout=5
            )
        except hrequests.exceptions.ClientException as e:
            print(f"  Timed out while trying to collect {url}, {e}, Retrying...")
            try:
                time.sleep(random.uniform(0.5, 2.0))
                webpage = hrequests.get(
                    url, headers=self.get_randomized_headers(), timeout=5
                )
            except hrequests.exceptions.ClientException as e:
                print(f"{e}")
                return False
        except hrequests.exceptions.BrowserTimeoutException:
            print(f"  Failed to collect {url}, timeout error")
            return False
        soup = BeautifulSoup(webpage.content, "html.parser")

        if webpage.status_code != 200 and webpage.status_code != 404:
            print(f"  Failed to collect {url}, status code: {webpage.status_code}")
            return False
        elif webpage.status_code == 404:
            print(f" Error 404: {url} not found")
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
            self.__add_report_metadata_to_df(
                self.get_report_metadata(report_id, url, soup)
            )
            if outcome is None:
                outcome = True

        return outcome

    def download_report(
        self,
        report_id: str,
        soup: BeautifulSoup,
        base_url: str,
        agency_id: str | None = None,
    ):
        """Download report directly to PDF storage container."""
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
            response = hrequests.get(
                link,
                allow_redirects=True,
                headers=self.get_randomized_headers(),
                timeout=30,
            )
            if response is None:
                print(f"  {report_id}.pdf download failed: No response")
                return False

            # Upload to storage
            self.settings.pdf_storage_manager.upload_pdf(
                report_id, response.content, overwrite=self.settings.refresh
            )

            return True

        except Exception as e:
            print(f"  {report_id}.pdf processing failed: {e}")
            return False

    def _find_pdf_links(
        self, soup: BeautifulSoup, report_id: str, agency_id: str | None = None
    ) -> str | Literal[False]:
        """Extract PDF links from BeautifulSoup object."""
        # Find all the links that end with .pdf and download them
        pdf_links = [
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].endswith(".pdf")
        ]

        # Remove duplicates
        pdf_links = list(dict.fromkeys(pdf_links))

        if len(pdf_links) == 0:
            print(
                f"WARNING: Found no PDFs link for {report_id}. Will not download any."
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
                print(
                    f"WARNING: Found more than one PDF for {report_id}. Will not download any.\n Here are the links: \n {links_str}"
                )
                return False
            if len(suitable_pdf_links) == 0:
                links_str = "\n".join(pdf_links)
                print(
                    f"WARNING: Found no suitable PDF link for {report_id}. Will not download any. Here are the original links:\n{links_str}"
                )
                return False
            pdf_links = suitable_pdf_links

        return pdf_links[0]

    @abstractmethod
    def get_report_metadata(
        self, report_id: str, url: str, soup: BeautifulSoup
    ) -> ReportMetadata:
        """
        Gets the investigation webpage and scrapes extra information about the report.

        This method must be implemented by subclasses to handle agency-specific
        metadata extraction from report pages.

        Parameters
        ----------
        report_id : str
            The identifier of the report
        url : str
            The URL of the report page
        soup : BeautifulSoup
            The BeautifulSoup object for the page
        pbar : tqdm, optional
            The progress bar to update, by default None

        Returns
        -------
        ReportMetadata
            The report metadata object containing extracted information
        """
        pass

    def __add_report_metadata_to_df(self, metadata: ReportMetadata):
        with self._metadata_file_lock:
            # Ensure DataFrame has correct column order
            expected_columns = ReportMetadata.get_column_names()
            if list(self.report_titles_df.columns) != expected_columns:
                # Reorder existing columns and add missing ones
                existing_data = self.report_titles_df.copy()
                self.report_titles_df = pd.DataFrame(columns=expected_columns)

                # Copy existing data to new DataFrame with correct column order
                for col in expected_columns:
                    if col in existing_data.columns:
                        self.report_titles_df[col] = existing_data[col]
                    # Missing columns will be filled with NaN by default

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

            self.report_titles_df.to_pickle(self.settings.report_titles_file_path)


class TAICReportScraper(ReportScraper):
    def __init__(self, reports_table_path, settings: ReportScraperSettings):
        super().__init__(
            settings,
            agency="TAIC",
        )

        self.agency_reports = self.__get_taic_investigations(reports_table_path)

    def __get_taic_investigations(self, reports_table_path):
        """TAICs websites provides an investigation table than can be easily read by pandas read_html"""
        if os.path.exists(reports_table_path):
            investigations = pd.read_pickle(reports_table_path)
        else:
            investigations = pd.DataFrame(columns=["id", "year"])
        page_num = 0
        while True:
            print(f"Processing page {page_num}")
            try:
                new_content = hrequests.get(
                    f"https://taic.org.nz/inquiries-recommendations?type=investigation&field_jurisdiction[11]=11&field_status[260]=260&sort_by=incident&page={page_num}"
                ).content
            except hrequests.exceptions.ClientException as e:
                print(f"Timeout while scraping TAIC investigations: {e}")
                print(f"Retrying page {page_num}")
                continue

            soup = BeautifulSoup(new_content, "html.parser")

            list = soup.find_all("div", class_="search-results__list")

            if not list or len(list) == 0:
                print(f"Reached end of pages at page {page_num}, stopping.")
                break

            all_reports_on_page = [
                {
                    "id": report.find("span", class_="card__incident").get_text(
                        strip=True
                    ),
                    "year": report.find("span", class_="card__date")
                    .get_text(strip=True)
                    .split()[-1],
                }
                for report in list[0].find_all("div", recursive=False)
            ]

            new_reports = pd.DataFrame(
                filter(
                    lambda r: r["id"] not in investigations["id"].values,
                    all_reports_on_page,
                )
            )

            if new_reports.empty:
                print(f"No new reports found on page {page_num}, stopping.")
                break

            investigations = pd.concat([investigations, new_reports], ignore_index=True)

            print(
                f"Found {len(all_reports_on_page)} reports, {len(investigations)} total"
            )

            page_num += 1

        investigations.set_index(
            investigations["id"].map(lambda x: Modes.Mode[x[0].lower()]),
            inplace=True,
        )
        investigations.index.name = None

        # Update the pickle file
        investigations.to_pickle(reports_table_path)

        return investigations

    def get_report_urls(self, mode, year):
        return [
            (
                self.get_report_id(mode, year, taic_id[-3:]),
                f"https://www.taic.org.nz/inquiry/{taic_id}",
                taic_id,  # This is the agency_id for TAIC
            )
            for taic_id in self.agency_reports.loc[mode]
            .query(f"year == {year}")["id"]
            .to_list()
        ]

    def get_report_metadata(self, report_id: str, url: str, soup: BeautifulSoup):
        headers = soup.find("div", class_="screen-head__title")
        if headers is not None:
            title = headers.find("span", class_="heading--english").text.strip()
            agency_id = headers.find("span", class_="heading--accent").text.strip()
        else:
            print(f"Failed to get title for {report_id}")
            title = "Unknown Title"
            agency_id = None

        # Get the report text
        report_text_div = soup.find("div", class_="node--type-investigation")
        if report_text_div is None:
            print(f"Failed to get report text for {report_id}")
            summary = None
        else:
            sections = report_text_div.find_all("section", recursive=False)

            executive_summary_section = sections[0]

            summary = executive_summary_section.find("p").get_text().strip()

            if summary is None or summary == "":
                summary = None

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=title,
            event_type=None,
            investigation_type="full",
            agency_id=agency_id,
            summary=summary,
            misc={},
        )


class ATSBReportScraper(ReportScraper):
    def __init__(
        self,
        website_reports_file_name,
        settings: ReportScraperSettings,
    ):
        super().__init__(settings, agency="ATSB")
        self.agency_reports = self.__get_atsb_investigations(website_reports_file_name)

    def __get_atsb_investigations(self, website_reports_file_name=None):
        """ATSBs websites provides an investigation table than can be easily read by pandas read_html. The only catch is that the aviation goes all the way back to 1960s and so only the first few pages of the aviation table is scraped. It will then be combined with a complete scrape of the table to find the new ids."""
        if website_reports_file_name is None or not os.path.exists(
            website_reports_file_name
        ):
            investigations = pd.DataFrame(
                columns=[
                    "Investigation title",
                    "Investigation number",
                    "Investigation webpage",
                    "Occurrence date",
                    "Report status",
                    "Report release",
                ]
            )
        else:
            investigations = pd.read_pickle(website_reports_file_name)

        url_start_date = (
            f"field_occurence_date_value%5Bmin%5D={self.settings.start_year}-01-01"
        )
        url_end_date = (
            f"field_occurence_date_value%5Bmax%5D={self.settings.end_year}-12-31"
        )

        url = "https://www.atsb.gov.au/{mode}-investigation-reports?order=field_occurence_date&sort=desc&page={page_num}&field_investigation_type_target_id=162&field_report_status_target_id=93&{url_start_date}&{url_end_date}"

        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

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
                try:
                    page = hrequests.get(
                        url.format(
                            mode=mode,
                            page_num=page_num,
                            url_start_date=url_start_date,
                            url_end_date=url_end_date,
                        ),
                        headers=self.get_randomized_headers(),
                    ).content

                    page_df = pd.read_html(
                        page,
                        flavor="lxml",
                        extract_links="body",
                    )[0]

                    page_df["Investigation webpage"] = page_df[
                        "Investigation title"
                    ].apply(lambda x: urljoin(base_url, x[1]))

                    page_df = page_df.map(lambda x: x[0] if isinstance(x, tuple) else x)

                    page_df.rename(
                        {
                            "Occurrence date  Sort ascending": "Occurrence date",
                            "Occurrence date  Sort descending": "Occurrence date",
                        },
                        axis=1,
                        inplace=True,
                    )

                    new_investigations = page_df[
                        ~page_df["Investigation number"].isin(
                            investigations["Investigation number"]
                        )
                    ]

                    new_investigations.loc[:, "year"] = pd.to_datetime(
                        new_investigations["Occurrence date"].to_list(),
                        format="%d/%m/%Y",
                        errors="coerce",
                    ).year

                    new_investigations = new_investigations.query(
                        f"year >= {self.settings.start_year}"
                    )

                    new_investigations["id"] = new_investigations[
                        "Investigation number"
                    ].map(
                        lambda x: re.search(r"(\d{3})$|(?:(?:\d{5})(\d{4}))$", str(x))
                    )

                    new_investigations = new_investigations.dropna(subset=["id"])
                    new_investigations["id"] = new_investigations["id"].map(
                        lambda x: x.group(1) if x.group(1) is not None else x.group(2)
                    )

                    if new_investigations.empty:
                        break

                    pages.append(new_investigations)

                    page_num += 1
                except hrequests.exceptions.ClientException as e:
                    print(f"Timeout while trying to scrape {mode} page {page_num}: {e}")
                    print("Retrying...")
                    continue
                except (ValueError, TypeError) as e:
                    print(f"Failed to scrape {mode} page {page_num}: {e}")
                    print(f"Assuming end of {mode} reports")
                    break

            if len(pages) == 0:
                print(f"No investigations found for mode: {mode}")
                continue

            mode_investigations = pd.concat(
                pages,
                ignore_index=True,
            )

            dfs.append(mode_investigations)

        if len(dfs) == 0:
            return investigations

        new_investigations = pd.concat(
            dfs, axis=0, keys=self.settings.modes
        ).reset_index(level=1, drop=True)

        investigations = investigations = pd.concat(
            [investigations, new_investigations],
            axis=0,
        )
        if website_reports_file_name is not None:
            investigations.to_pickle(website_reports_file_name)

        return investigations

    def get_report_urls(self, mode, year):
        return [
            (self.get_report_id(mode, year, str(atsb_id)[-3:]), url, str(atsb_id))
            for atsb_id, url in self.agency_reports.loc[mode]
            .query(f"year == {year} & `Report status` == 'Final'")
            .dropna(subset=["Investigation webpage"])[
                [
                    "Investigation number",
                    "Investigation webpage",
                ]
            ]
            .to_records(index=False)
        ]

    def get_report_metadata(
        self, report_id: str, url: str, soup: BeautifulSoup
    ) -> ReportMetadata:
        report_mode = Modes.get_report_mode_from_id(report_id)
        event_type = None
        if report_mode is Modes.Mode.a:
            event_type_div = soup.find(
                "div", class_="field--name-field-aviation-occurrence-type"
            )
            if event_type_div is not None:
                event_type = event_type_div.find("div", class_="field__item").text

        investigation_level = soup.find(
            "div", class_="field--name-field-investigation-level"
        )
        if investigation_level is not None:
            investigation_level = investigation_level.find(
                "div", class_="field__item"
            ).text

        title_div = soup.find("div", class_="field--name-title")

        title = title_div.text

        agency_id = soup.find("div", class_="field--name-field-report-id")
        if agency_id is not None:
            agency_id = agency_id.find("div", class_="field__item").text.strip()
            agency_id = re.sub(r"^\d{3}-M", "M", agency_id, flags=re.IGNORECASE)

        # Getting the safety summary
        summary = self.get_summary(soup)

        investigation_type = "unknown"
        if investigation_level is None:
            investigation_type = "unknown"
        elif investigation_level in ["Defined", "Systemic"]:
            investigation_type = "full"
        else:
            investigation_type = "short"

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=title,
            investigation_type=investigation_type,
            event_type=event_type,
            agency_id=agency_id,
            summary=summary,
            misc={"investigation_level": investigation_level},
        )

    def get_summary(self, soup):
        """
        Gets the summary from the soup object.
        This is a placeholder as ATSB does not have a summary field in the report metadata.
        """
        summary_div = soup.find("div", class_="field--type-text-with-summary")
        if summary_div is None:
            return None

        # Try to find h2 with "Executive summary" or "Investigation summary"
        h2 = None
        for candidate in summary_div.find_all("h2"):
            heading = candidate.get_text(strip=True).lower()
            summary_headings = [
                "executive summary",
                "investigation summary",
                "safety summary",
            ]
            if any(h in heading for h in summary_headings):
                h2 = candidate
                break

        if h2 is None:  # Return the full text is no summary is found.
            full_text = summary_div.get_text(" ", strip=True)
            # Could do check to see if it is smale than som uppper bound but for now will just return the full scrap which may be very large.
            return full_text

        summary_parts = []
        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":
                break
            if (
                sibling.find("h2") is not None
                or sibling.get_text("", strip=True).lower() == "the occurrence"
            ):
                break
            summary_parts.append(sibling.get_text(" ", strip=True))
        summary = "\n".join([part for part in summary_parts if part.strip()])
        return summary


class TSBReportScraper(ReportScraper):
    def __init__(self, settings: ReportScraperSettings):
        super().__init__(settings, agency="TSB")

        self.agency_reports = self.__get_tsb_investigations()

    def __get_tsb_investigations(self):
        """
        The TSB is very well setup and works friendly with the pandas read_html. Therefore I can just read all of the investigation tables from the TSB website and then have the exact IDs I need.
        """
        modes = ["aviation", "rail", "marine"]

        modes_df = [
            pd.read_html(
                hrequests.get(
                    f"https://www.tsb.gc.ca/eng/rapports-reports/{mode}/index.html",
                    headers=self.get_randomized_headers(),
                ).content,
                flavor="lxml",
            )[0]
            for mode in tqdm(modes)
        ]

        # Add dataframes togather with extra column identifier

        merged_modes_df = pd.concat(
            modes_df, keys=[Modes.Mode.a, Modes.Mode.r, Modes.Mode.m]
        )

        merged_modes_df["Occurrence date"] = pd.to_datetime(
            merged_modes_df["Occurrence date"], format="%Y-%m-%d", errors="coerce"
        )

        merged_modes_df["year"] = merged_modes_df["Occurrence date"].dt.year

        return merged_modes_df

    def get_report_urls(self, mode, year):
        return [
            (
                self.get_report_id(mode, year, tsb_id[-5:]),
                f"https://www.tsb.gc.ca/eng/rapports-reports/{Modes.Mode.as_string(mode)}/{year}/{tsb_id}/{tsb_id}.html".lower(),
                tsb_id,  # This is the agency_id for TSB
            )
            for tsb_id in self.agency_reports.loc[mode]
            .query(f"year == {year} & `Investigation status` == 'Completed'")[
                "Investigation number"
            ]
            .to_list()
        ]

    def get_report_metadata(
        self, report_id: str, url: str, soup: BeautifulSoup
    ) -> ReportMetadata:
        # Due to TSB having the metadata on a page separate from the report pdf link, we need to get the new page
        split_id = report_id.split("_")
        tsb_id = f"{split_id[1]}{split_id[2][2:4]}{split_id[3]}"
        page = hrequests.get(
            f"https://www.tsb.gc.ca/eng/enquetes-investigations/{Modes.Mode.as_string(Modes.get_report_mode_from_id(report_id))}/{split_id[2]}/{tsb_id}/{tsb_id}.html",
            headers=self.get_randomized_headers(),
            timeout=30,
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
            )
        else:
            event_type = title_block.find("strong")
            if event_type:
                event_type = event_type.text
        legacy_text_div = title_block.find(
            "div", class_="field--name-field-occurrence-legacy-text"
        )
        paragraph_text = []
        if legacy_text_div:
            # Handle case where <p> tag is present
            paragraph = legacy_text_div.find("p")
            if paragraph:
                paragraph_text = [text for text in paragraph.stripped_strings]
            else:
                # Handle case where <p> tag is not present
                paragraph_text = [text for text in legacy_text_div.stripped_strings]

        date_div = title_block.find("div", class_="field--name-field-occurrence-date")
        date_text = ""
        if date_div:
            time_tag = date_div.find("time")
            if time_tag:
                date_text = time_tag.text

        all_text = ", ".join(paragraph_text + [date_text])

        # Get investigation level
        investigation_level = None

        h3_element = soup.find("h3", string="Class of investigation")
        if h3_element:
            text = h3_element.find_next_sibling("p").text
            match = re.match(r"This is a class (\d) investigation", text)
            if match:
                investigation_level = match.group(1)

        agency_id = soup.find("h1", class_="page-header").text
        if agency_id is not None:
            agency_id = agency_id.strip().split(" ")[-1]

        return ReportMetadata(
            url=url,
            report_id=report_id,
            title=all_text,
            event_type=event_type,
            investigation_type="unknown"
            if investigation_level is None
            else "full"
            if investigation_level in ["1", "2", "3"]
            else "short",
            agency_id=agency_id,
            summary=None,  # TSB does not include summary text. However the press releases provide a summary of sorts.
            misc={"investigation_class": investigation_level},
        )


class ATSBSafetyIssueScraper(WebsiteScraper):
    def __init__(
        self, output_file_path: str, report_titles_file_path, refresh: bool = False
    ):
        super().__init__(report_titles_file_path)
        self.output_file_path = output_file_path
        self.refresh = refresh

    def extract_safety_issues_from_website(self):
        if os.path.exists(self.output_file_path) and not self.refresh:
            safety_issues_df = pd.read_pickle(self.output_file_path)
        else:
            safety_issues_df = pd.DataFrame(columns=["report_id", "safety_issues"])

        base_url = "https://www.atsb.gov.au/safety-issues-and-actions?field_issue_number_value={mode}O&page={page}"
        if len(safety_issues_df["safety_issues"]) > 0:
            widened_safety_issues_df = pd.concat(
                safety_issues_df["safety_issues"].tolist()
            )
        else:
            widened_safety_issues_df = pd.DataFrame(
                columns=["safety_issue_id", "safety_issue", "quality"]
            )

        print(
            "-------------------------- Scraping safety issues from ATSB ----------------------"
        )
        print(f"     Output file: {self.output_file_path}")
        print(f"     Currently have {widened_safety_issues_df.shape[0]} safety issues")
        print(f"     Spread across {len(safety_issues_df)} reports")
        print(
            "----------------------------------------------------------------------------------"
        )

        for mode in (pbar := tqdm(["A", "R", "M"])):
            current_page = 0
            pbar.set_description(f"Scraping {mode} safety issues")

            failed = 0
            while True:
                pbar.set_description(f"Scraping page {current_page} of mode {mode}")
                url = base_url.format(mode=mode, page=current_page)
                response = hrequests.get(url, headers=self.get_randomized_headers())

                if response.status_code != 200:
                    pbar.write(
                        f"Failed to scrape page {current_page} of mode {mode}\nWith error {response.status_code}"
                    )
                    failed += 1
                    if failed > 5:
                        failed = 0
                        current_page += 1
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                if not soup.find("div", class_="view-content"):
                    pbar.write(
                        f"Failed to scrape page {current_page} of mode {mode}. No table found"
                    )
                    break

                safety_issues = [
                    {
                        field.find(class_="field__label").get_text(
                            strip=True
                        ): field.find(class_="field__item").get_text(strip=True)
                        for field in row.find_all(class_="field--label-inline")
                    }
                    for row in soup.find("div", class_="view-content").children
                    if not isinstance(row, str)
                ]
                table = pd.DataFrame(safety_issues)

                if "Safety issue title" not in table.columns:
                    pbar.write(
                        f"Failed to scrape page {current_page} of mode {mode}. Page contains no safety issues"
                    )
                    break

                table["safety_issue"] = table.apply(
                    lambda row: f"{row['Safety issue title']}\n{row['Safety Issue Description']}",
                    axis=1,
                )

                table["safety_issue_id"] = table["Issue number"].astype(str)

                new_safety_issues = table[
                    ~table["safety_issue_id"].isin(
                        widened_safety_issues_df["safety_issue_id"]
                    )
                ]

                if new_safety_issues.empty:
                    pbar.write(
                        f"No new safety issues found on page {current_page} of mode {mode}, moving onto next mode"
                    )
                    break

                widened_safety_issues_df = pd.concat(
                    [widened_safety_issues_df, new_safety_issues], ignore_index=True
                )

                current_page += 1

        widened_safety_issues_df = widened_safety_issues_df.drop_duplicates(
            subset=["safety_issue_id"]
        )

        widened_safety_issues_df["report_id"] = (
            widened_safety_issues_df["safety_issue_id"]
            .map(
                # This needed because the safety issue id is simply the agency_id plus some extrae identifiers on the end.
                lambda x: "-".join(x.split("-")[0:3])
            )
            .map(
                lambda x: self.id_converter("ATSB", x)
                if self.id_converter("ATSB", x)
                else f"Unmatched ({x})"
            )
        )
        widened_safety_issues_df["quality"] = "exact"
        widened_safety_issues_df = widened_safety_issues_df[
            ["report_id", "safety_issue_id", "safety_issue", "quality"]
        ]

        print(f"  Now there are {widened_safety_issues_df.shape[0]} safety issues")

        grouped = widened_safety_issues_df.groupby("report_id")
        formatted_df = pd.DataFrame(
            {
                "report_id": grouped.groups.keys(),
                "safety_issues": [group.reset_index(drop=True) for _, group in grouped],
            }
        )

        print(f"  Spread across {formatted_df.shape[0]} reports")

        formatted_df.to_pickle(self.output_file_path)


class RecommendationScraper(WebsiteScraper, ABC):
    """
    Abstract base class for scraping recommendations from different transportation safety agencies.
    Subclasses must implement abstract methods to handle agency-specific website structures.
    """

    def __init__(
        self,
        output_file_path: str,
        report_titles_file_path: str,
        columns: list[str],
        base_url: str,
        loop_iter: list | range,
        agency: str,
        refresh: bool = False,
    ):
        super().__init__(report_titles_file_path)

        self.output_file_path = output_file_path
        self.refresh = refresh
        self.columns = columns
        self.base_url = base_url
        self.loop_iter = loop_iter
        self.agency = agency

    def extract_recommendations_from_website(self):
        if not self.refresh and os.path.exists(self.output_file_path):
            recommendations_df = pd.read_pickle(self.output_file_path)
        else:
            recommendations_df = pd.DataFrame(columns=["report_id", "recommendations"])

        new_recommendations = pd.DataFrame(columns=self.columns)

        print(
            "------------------------ Scraping recommendations ------------------------"
        )
        print(f"    Output directory: {self.output_file_path}")
        print(f"    Scraping from base url {self.base_url}")
        print(
            f"    Currently have {recommendations_df.shape[0]} reports with recommendations"
        )
        print(
            f"    With {recommendations_df['recommendations'].apply(len).sum()} recommendations"
        )
        print(
            "----------------------------------------------------------------------------------"
        )

        print("  Reading recommendation tables to get recommendations webpages")

        for element in (phbar := tqdm(self.loop_iter)):
            phbar.set_description(f"Scraping recommendations for {element}")

            url = self.get_url(element)

            response = hrequests.get(url, headers=self.get_randomized_headers())

            if response.status_code != 200:
                raise ValueError(
                    f"Failed to scrape recommendations for {element}. Error code {response.status_code}"
                )

            table = pd.read_html(response.content, flavor="lxml", extract_links="body")

            if len(table) == 0 or table[0].empty:
                break
            table = table[0]

            table = self.process_new_table(table)

            if len(recommendations_df) > 0:
                table = table[
                    ~table["recommendation_id"].isin(
                        pd.concat(recommendations_df["recommendations"].tolist())[
                            "recommendation_id"
                        ]
                    )
                ]

            if len(table) == 0:
                break
            new_recommendations = pd.concat(
                [new_recommendations, table], ignore_index=True
            )

        print(
            f"  Found {new_recommendations.shape[0]} new recommendations, reading each individual webpage now"
        )

        for i, row in (phbar := tqdm(list(new_recommendations.iterrows()))):
            phbar.set_description(
                f"Processing recommendation {row['recommendation_id']} from {row['agency_id']} with i:{i}"
            )
            recommendation_data = self.extract_recommendation_data(row["url"])
            for key, value in recommendation_data.items():
                new_recommendations.at[i, key] = value

        new_recommendations["report_id"] = new_recommendations["agency_id"].map(
            lambda x: self.id_converter(self.agency, x)
            if self.id_converter(self.agency, x)
            else f"Unmatched {self.agency} ({x})"
        )

        recommendations_df = pd.concat(
            [
                recommendations_df,
                pd.DataFrame(
                    {
                        "report_id": new_recommendations.groupby(
                            "report_id"
                        ).groups.keys(),
                        "recommendations": [
                            group.reset_index(drop=True).drop("report_id", axis=1)
                            for _, group in new_recommendations.groupby("report_id")
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )

        recommendations_df.to_pickle(self.output_file_path)

    @abstractmethod
    def extract_recommendation_data(self, url) -> dict:
        """
        Goes to the URL and extracts the needed data.

        This method must be implemented by subclasses to handle agency-specific
        recommendation data extraction from individual recommendation pages.

        Parameters
        ----------
        url : str
            The URL of the recommendation page

        Returns
        -------
        dict
            Dictionary containing extracted recommendation data
        """
        pass

    @abstractmethod
    def process_new_table(self, table) -> pd.DataFrame:
        """
        Takes a recently read table and processes it according to agency-specific rules.

        This method must be implemented by subclasses to handle agency-specific
        table processing and column mapping.

        Parameters
        ----------
        table : pd.DataFrame
            The raw table data from the website

        Returns
        -------
        pd.DataFrame
            Processed table with standardized columns
        """
        pass

    @abstractmethod
    def get_url(self, element) -> str:
        """
        Generates the URL for a given element (page number, mode, etc.).

        This method must be implemented by subclasses to handle agency-specific
        URL generation patterns.

        Parameters
        ----------
        element : Any
            The element used to generate the URL (e.g., page number, mode)

        Returns
        -------
        str
            The complete URL for the given element
        """
        pass


class TSBRecommendationsScraper(RecommendationScraper):
    def __init__(self, output_file_path, report_titles_file_path, refresh=False):
        columns = [
            "recommendation_id",
            "recommendation",
            "agency_id",
            "current_assessment",
            "status",
            "watchlist",
            "url",
            "made",
            "recommendation_context",
        ]
        base_url = "https://www.tsb.gc.ca"
        loop_iter = ["rail", "marine", "aviation"]
        agency = "TSB"

        super().__init__(
            output_file_path,
            report_titles_file_path,
            columns,
            base_url,
            loop_iter,
            agency,
            refresh,
        )

    def get_url(self, element):
        return (
            f"{self.base_url}/eng/recommandations-recommendations/{element}/index.html"
        )

    def extract_recommendation_data(self, url, retry=3):
        """
        This will read the webpage and extract:
        - recommendation (This is because sometimes the recommendation inside the website table is not complete)
        - recommednation date
        - recommendation context
        ## TODO: Add in the recipient and reply text. This is not done at the moment as it is not needed
        - recipient
        - reply text
        """
        if url is None:
            return {
                "recommendation": None,
                "made": None,
                "recommendation_context": None,
            }

        response = hrequests.get(url, headers=self.get_randomized_headers())

        if response.status_code != 200:
            if retry > 0:
                time.sleep(1)
                return self.extract_recommendation_data(url, retry - 1)
            raise ValueError(
                f"Failed to scrape recommendations from {url}. Error code {response.status_code}"
            )

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

        return {
            "recommendation": recommendation,
            "made": recommendation_date,
            "recommendation_context": recommendation_context,
        }

    def process_new_table(self, table):
        # filter out older recommendations
        table = table[
            table["Number"]
            .map(
                lambda x: int(
                    re.match(r"[amr](\d{2})-\d{2}", x[0], re.IGNORECASE).group(1)
                )
            )
            .between(0, 80)
        ]

        table["url"] = table["Number"].map(
            lambda x: f"{self.base_url}{x[1]}" if x[1] else None
        )

        table = table.map(lambda x: x[0] if isinstance(x, tuple) else x)

        table.columns = self.columns[:7]
        table.drop("recommendation", axis=1, inplace=True)
        return table


class TAICRecommendationsScraper(RecommendationScraper):
    def __init__(self, output_file_path, report_titles_file_path, refresh=False):
        columns = [
            "recommendation_id",
            "made",
            "agency_id",
            "recipient",
            "recommendation",
            "reply_text",
        ]
        base_url = "https://www.taic.org.nz"
        loop_iter = range(300)
        agency = "TAIC"

        super().__init__(
            output_file_path,
            report_titles_file_path,
            columns,
            base_url,
            loop_iter,
            agency,
            refresh,
        )

    def get_url(self, element):
        return f"{self.base_url}/recommendations?page={element}"

    def process_new_table(self, table):
        table = table.iloc[:, :4]

        table.columns = self.columns[:4]
        table["url"] = table["recommendation_id"].map(lambda x: self.base_url + x[1])

        table = table.map(lambda x: x[0] if isinstance(x, tuple) else x)

        table["recommendation_id"] = table["recommendation_id"].map(
            lambda x: re.sub(" (Aviation)|(Rail)|(Marine)", "", x)
        )
        table = table[
            table["recommendation_id"]
            .map(
                lambda x: int(re.match(r"\d{3}\w?/(\d{2})", x, re.IGNORECASE).group(1))
            )
            .between(0, 80)
        ]

        return table

    def extract_recommendation_data(self, url) -> dict:
        """
        This will extract information that is not found in the table
        - recommendation_text
        - reply_text
        """

        if url is None:
            return {
                "recommendation": None,
                "reply_text": None,
            }

        response = hrequests.get(url, headers=self.get_randomized_headers())

        if response.status_code != 200:
            retry = 3
            if retry > 0:
                time.sleep(1)
                return self.extract_recommendation_data(url)
            raise ValueError(
                f"Failed to scrape recommendations from {url}. Error code {response.status_code}"
            )

        soup = BeautifulSoup(response.content, "html.parser")

        text = soup.find("div", class_="field--name-field-sr-text")

        recommendation_text = (
            text.find("div", class_="field__item").get_text() if text else None
        )

        reply_text = soup.find("div", class_="field--name-field-sr-replytext")
        reply_text = (
            reply_text.find("div", class_="field__item").get_text()
            if reply_text
            else None
        )

        return {
            "recommendation": recommendation_text,
            "reply_text": reply_text,
        }
