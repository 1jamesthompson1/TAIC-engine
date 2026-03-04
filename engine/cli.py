"""CLI entry points for downloading, extracting, analyzing, and uploading report data."""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from . import (
    Config,
    DataGetting,
    Embedding,
    Modes,
    PDFParsing,
    ReportExtracting,
    ReportTypeAssignment,
    SavedDataFrames,
    WebsiteScraping,
)
from .AzureStorage import (
    EngineOutputDownloader,
    EngineOutputUploader,
    PDFStorageManager,
)
from .Logging import configure_logging, get_logger

logger = get_logger(__name__)


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600


def format_duration(seconds):
    """Format duration to show appropriate time units.

    Returns:
        str: A human-readable duration string.
    """
    if seconds < SECONDS_IN_MINUTE:
        return f"{seconds:.2f} seconds"
    if seconds < SECONDS_IN_HOUR:
        minutes = int(seconds // SECONDS_IN_MINUTE)
        remaining_seconds = seconds % SECONDS_IN_MINUTE
        return f"{minutes}m {remaining_seconds:.1f}s ({seconds:.2f} seconds)"
    hours = int(seconds // SECONDS_IN_HOUR)
    minutes = int((seconds % SECONDS_IN_HOUR) // SECONDS_IN_MINUTE)
    remaining_seconds = seconds % SECONDS_IN_MINUTE
    return f"{hours}h {minutes}m {remaining_seconds:.1f}s ({seconds:.2f} seconds)"


def log_timing_summary(timing_results, total_time):
    """Log the timing summary for the executed steps."""
    logger.info("%s", "=" * 60)
    logger.info("TIMING SUMMARY")
    logger.info("%s", "=" * 60)

    for step, duration in timing_results.items():
        formatted_time = format_duration(duration)
        logger.info("%s: %s", step.upper().rjust(10), formatted_time)

    if len(timing_results) > 1:
        logger.info("%s", "-" * 60)
        formatted_total = format_duration(total_time)
        logger.info("%s: %s", "TOTAL".rjust(10), formatted_total)

    logger.info("%s", "=" * 60)


def run_step(step_name, func, timing_results, *args, **kwargs):
    """Run a step function and record its timing."""
    start_time = time.time()
    func(*args, **kwargs)
    timing_results[step_name] = time.time() - start_time


def download(container: str, output_dir: Path, refresh: bool):
    """Download the latest engine output from Azure Storage and get generic data.

    Args:
        container (str): The name of the Azure Storage container to download from.
        output_dir (Path): The local directory to save the downloaded files to.
        refresh (bool): Whether to refresh cached data.
    """
    downloader = EngineOutputDownloader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        container,
        output_dir,
    )

    downloader.download_latest_output()

    # Get generic data
    config = Config.config_reader.get_config()["engine"]
    data_getter = DataGetting.DataGetter(
        Path(config.get("data").get("data_local_folder_location")),
        config.get("data").get("data_remote_folder_location"),
        refresh,
    )
    data_getter.get_generic_data(
        config.get("data").get("event_types_file_name"),
        output_dir / config.get("output").get("all_event_types_df_file_name"),
    )
    logger.info("Got event types")


def scrape(output_dir: Path, config: dict, refresh: bool):
    """Scrape reports from websites and extract additional data.

    Args:
        output_dir (Path): Directory where output artifacts are written.
        config (dict): Engine configuration settings.
        refresh (bool): Whether to refresh cached data and re-download sources.
    """
    output_config = config.get("output")
    download_config = config.get("download")

    logger.info("Scraping reports from websites")

    logger.info("Setting up PDF storage manager...")
    pdf_storage_manager = PDFStorageManager(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        output_config["pdf_container_name"],
    )
    logger.info(
        "PDF storage container: %s",
        output_config["pdf_container_name"],
    )

    # Download the PDFs
    report_scraping_settings = WebsiteScraping.ReportScraperSettings(
        SavedDataFrames.ReportTitles(output_dir),
        download_config.get("start_year"),
        download_config.get("end_year"),
        download_config.get("max_per_year"),
        [Modes.Mode[mode] for mode in download_config.get("modes")],
        download_config.get("ignored_reports"),
        refresh,
        pdf_storage_manager,
    )

    for agency in download_config.get("agencies"):
        match agency:
            case "TSB":
                WebsiteScraping.TSBReportScraper(report_scraping_settings).collect_all()
            case "TAIC":
                WebsiteScraping.TAICReportScraper(
                    SavedDataFrames.TAICWebsiteReportsTable(output_dir),
                    report_scraping_settings,
                ).collect_all()
            case "ATSB":
                WebsiteScraping.ATSBReportScraper(
                    SavedDataFrames.ATSBWebsiteReportsTable(output_dir),
                    report_scraping_settings,
                ).collect_all()
            case _:
                logger.warning("Unknown agency '%s', skipping", agency)

    atsb_safety_issues_df = SavedDataFrames.ATSBWebsiteSafetyIssues(output_dir)
    atsb_si_scraper = WebsiteScraping.ATSBSafetyIssueScraper(
        atsb_safety_issues_df,
        SavedDataFrames.ReportTitles(output_dir),
        refresh,
    )

    atsb_si_scraper.extract_safety_issues_from_website()

    tsb_recs_scraper = WebsiteScraping.TSBRecommendationsScraper(
        SavedDataFrames.TSBWebsiteRecommendations(output_dir),
        SavedDataFrames.ReportTitles(output_dir),
        refresh,
    )
    tsb_recs_scraper.extract_recommendations_from_website()

    taic_recs_scraper = WebsiteScraping.TAICRecommendationsScraper(
        SavedDataFrames.TAICWebsiteRecommendations(output_dir),
        SavedDataFrames.ReportTitles(output_dir),
        refresh,
    )
    taic_recs_scraper.extract_recommendations_from_website()


def create_extracted_reports_df(output_dir: Path, output_config: dict):
    """Create the combined extracted reports dataframe and persist it to disk.

    Args:
        output_dir (Path): Directory where output artifacts are written.
        output_config (dict): Output configuration containing expected file names.
    """
    raise NotImplementedError(
        "This needs to be updated to reflect the new simpler processing pipeline. It will take in all of the different dataframes and combine them into a single long dataframe"
    )

    dataframes = [
        pd.read_pickle(output_dir / file_name).set_index("report_id")
        for file_name in [
            output_config.get("parsed_reports_df_file_name"),
            output_config.get("toc_df_file_name"),
            output_config.get("report_sections_df_file_name"),
            output_config.get("report_event_types_df_file_name"),
            output_config.get("recommendations_df_file_name"),
            output_config.get("safety_issues_df_file_name"),
        ]
    ]

    dataframes[-2] = dataframes[-2].rename(
        columns={
            "important_text": "important_text_recommendation",
            "pages_read": "pages_read_recommendation",
        }
    )
    dataframes[-1] = dataframes[-1].rename(
        columns={
            "important_text": "important_text_safety_issue",
            "pages_read": "pages_read_safety_issue",
        }
    )

    combined_df = dataframes[0].join(dataframes[1:], how="outer")

    # Adding agency_id and url
    report_titles = pd.read_pickle(
        output_dir / output_config.get("report_titles_df_file_name")
    )
    combined_df = combined_df.merge(
        report_titles[["report_id", "agency_id", "url", "summary"]],
        how="left",
        on="report_id",
    )

    # Add metadata columns
    problematic_ids = []
    for report_id in combined_df["report_id"]:
        # Check to see if they follow the correct format defined by f"{self.agency}_{mode.name}_{year}_{id}"
        parts = str(report_id).split("_")
        report_id_parts = 4
        if len(parts) != report_id_parts:
            problematic_ids.append((report_id, len(parts), parts))

    if problematic_ids:
        logger.warning("Found %s problematic report IDs", len(problematic_ids))

    # Drop all problematic ids
    combined_df = combined_df[
        ~combined_df["report_id"].isin([x[0] for x in problematic_ids])
    ]

    combined_df["year"] = [
        int(x.split("_")[2]) if "_" in x else None for x in combined_df["report_id"]
    ]
    combined_df["mode"] = combined_df["report_id"].map(
        lambda x: str(Modes.get_report_mode_from_id(x).value) if "_" in x else None
    )

    combined_df["agency"] = [
        (x.split("_")[0] if "_" in x else None) for x in combined_df["report_id"]
    ]

    combined_df.to_pickle(
        output_dir / output_config.get("extracted_reports_df_file_name")
    )


def extract(output_dir: Path, config: dict, refresh: bool):
    """Extract report artifacts from PDFs.

    Args:
        output_dir (Path): Directory where output artifacts are written.
        config (dict): Engine configuration settings.
        refresh (bool): Whether to refresh cached data and reprocess sources.
    """
    output_config = config.get("output")

    logger.welcome("Extracting Report Artifacts", {"Output directory": str(output_dir)})

    logger.info("Setting up PDF storage manager...")
    pdf_storage_manager = PDFStorageManager(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        output_config["pdf_container_name"],
    )
    logger.info(
        "PDF storage container: %s",
        output_config["pdf_container_name"],
    )

    # Parse PDFs into text
    PDFParsing.process_all_pdfs_into_text(
        SavedDataFrames.ParsedReports(output_dir),
        refresh,
        pdf_storage_manager,
    )

    # Extract reports into structured data
    extraction_config = config.get("extraction").get("ai_extraction_config")
    ReportExtracting.process_reports_parallel(
        SavedDataFrames.ParsedReports(output_dir),
        SavedDataFrames.ExtractedReports(output_dir),
        ai_extraction_config=extraction_config,
    )


def analyze(output_dir: Path, config: dict, refresh: bool):
    """Analyze extracted reports by assigning types, linking recommendations, and classifying responses.

    Args:
        output_dir (Path): Directory where output artifacts are written.
        config (dict): Engine configuration settings.
        refresh (bool): Whether to refresh cached data and reprocess sources.
    """
    logger.welcome("Analyzing Extracted Reports", {"Output directory": str(output_dir)})

    # Assign report types
    ReportTypeAssignment.ReportTypeAssigner(
        SavedDataFrames.ReportEventTypes(output_dir),
        SavedDataFrames.ReportTitles(output_dir),
        SavedDataFrames.ParsedReports(output_dir),
        SavedDataFrames.AllEventTypes(output_dir),
    ).assign_report_types()


def embed(output_dir: Path, config: dict, refresh: bool):
    """Embed extracted reports into the vector database.

    Args:
        output_dir (Path): Directory where output artifacts are written.
        config (dict): Engine configuration settings.
        refresh (bool): Whether to refresh cached data and reprocess sources.
    """
    vector_config = config.get("vector")

    logger.welcome(
        "Embedding Reports",
        {
            "Output directory": str(output_dir),
            "Vector DB URI": os.environ.get("VECTORDB_URI", "Not set"),
            "Table name": vector_config["table_name"],
        },
    )

    vector_db = Embedding.VectorDB(
        SavedDataFrames.VectorDBDocumentIDs(output_dir),
        os.environ["VECTORDB_URI"],
        vector_config["model"]["name"],
        vector_config["model"]["context_limit"],
        vector_config["table_name"],
    )
    vector_db.process_extracted_reports(
        SavedDataFrames.ExtractedReports(output_dir),
        [
            (
                "safety_issues",
                "safety_issue",
            ),
            (
                "recommendations",
                "recommendation",
            ),
            (
                "sections",
                "section",
            ),
            (
                "summary",
                "summary",
            ),
        ],
    )


def upload(container_name: str, output_dir: Path, output_config: dict):
    """Upload the latest engine output artifacts to Azure Storage.

    Args:
        container_name (str): The name of the Azure Storage container to upload to.
        output_dir (Path): The local directory containing output artifacts.
        output_config (dict): Output configuration settings.
    """
    uploader = EngineOutputUploader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        container_name,
        output_dir,
    )

    uploader.upload_latest_output()


def cli():
    """Main CLI entry point for the engine."""
    configure_logging(log_level="INFO")

    parser = argparse.ArgumentParser(
        description="A engine that will download, extract, and summarize PDFs from the marine accident investigation reports. More information can be found here: https://github.com/1jamesthompson1/TAIC-engine/"
    )
    parser.add_argument(
        "-r",
        "--refresh",
        help="Clears the output directory, otherwise functions will be run with what is already there.",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--calculate_cost",
        help="Calculate the API cost of doing a summarize. Note this action itself will use some API token, however it should be a negligible amount. Currently not going to give an accurate response",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--run_type",
        choices=["download", "scrape", "extract", "analyze", "embed", "upload", "all"],
        required=True,
        help="This is function that you want to run.",
    )

    args = parser.parse_args()

    # Initialize timing tracker
    timing_results = {}
    total_start_time = time.time()

    # Get the config settings for the engine.
    engine_settings = Config.config_reader.get_config()["engine"]

    # Set working directory to output folder
    output_path = Path(engine_settings.get("output").get("folder_name"))

    output_path.mkdir(parents=True, exist_ok=True)

    # Define step configurations
    step_configs = {
        "download": (
            "download",
            download,
            (
                engine_settings.get("output").get("container_name"),
                output_path,
                args.refresh,
            ),
        ),
        "scrape": ("scrape", scrape, (output_path, engine_settings, args.refresh)),
        "extract": ("extract", extract, (output_path, engine_settings, args.refresh)),
        "analyze": ("analyze", analyze, (output_path, engine_settings, args.refresh)),
        "embed": ("embed", embed, (output_path, engine_settings, args.refresh)),
        "upload": (
            "upload",
            upload,
            (
                engine_settings.get("output").get("container_name"),
                output_path,
                engine_settings.get("output"),
            ),
        ),
    }

    if args.run_type == "all":
        for step_name, func, step_args in step_configs.values():
            run_step(step_name, func, timing_results, *step_args)
    else:
        step_name, func, step_args = step_configs[args.run_type]
        run_step(step_name, func, timing_results, *step_args)

    # Calculate total time
    total_time = time.time() - total_start_time

    # Print timing summary
    log_timing_summary(timing_results, total_time)


if __name__ == "__main__":
    cli()
