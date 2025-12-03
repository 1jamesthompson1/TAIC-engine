import argparse
import os
import time

import pandas as pd

from ..analyze import (
    Embedding,
    RecommendationResponseClassification,
    RecommendationSafetyIssueLinking,
)
from ..extract import ReportExtracting, ReportTypeAssignment
from ..gather import DataGetting, PDFParser, WebsiteScraping
from . import Config, Modes
from .AzureStorage import (
    EngineOutputDownloader,
    EngineOutputUploader,
    PDFStorageManager,
)


def download(container, output_dir):
    downloader = EngineOutputDownloader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        container,
        output_dir,
    )

    downloader.download_latest_output()


def gather(output_dir, config, refresh):
    output_config = config.get("output")
    download_config = config.get("download")

    print("Getting all data needed for engine")

    dataGetter = DataGetting.DataGetter(
        config.get("data").get("data_local_folder_location"),
        config.get("data").get("data_remote_folder_location"),
        refresh,
    )

    dataGetter.get_generic_data(
        config.get("data").get("event_types_file_name"),
        os.path.join(output_dir, output_config.get("all_event_types_df_file_name")),
    )
    print("Got event types")

    print("Setting up PDF storage manager...")
    pdf_storage_manager = PDFStorageManager(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        output_config["pdf_container_name"],
    )
    print(f"PDF storage container: {output_config['pdf_container_name']}")

    # Download the PDFs
    report_scraping_settings = WebsiteScraping.ReportScraperSettings(
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
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
                    os.path.join(
                        output_dir,
                        output_config.get("taic_website_reports_table_file_name"),
                    ),
                    report_scraping_settings,
                ).collect_all()
            case "ATSB":
                WebsiteScraping.ATSBReportScraper(
                    os.path.join(
                        output_dir,
                        output_config.get("atsb_website_reports_table_file_name"),
                    ),
                    report_scraping_settings,
                ).collect_all()

    # Extract the text from the PDFs
    PDFParser.convertPDFToText(
        os.path.join(output_dir, output_config.get("parsed_reports_df_file_name")),
        refresh,
        pdf_storage_manager,
    )

    ATSB_si_scraper = WebsiteScraping.ATSBSafetyIssueScraper(
        os.path.join(
            output_dir, output_config.get("atsb_website_safety_issues_file_name")
        ),
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
        refresh,
    )

    ATSB_si_scraper.extract_safety_issues_from_website()

    TSB_recs_scraper = WebsiteScraping.TSBRecommendationsScraper(
        os.path.join(
            output_dir, output_config.get("tsb_website_recommendations_file_name")
        ),
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
        refresh,
    )
    TSB_recs_scraper.extract_recommendations_from_website()

    TAIC_recs_scraper = WebsiteScraping.TAICRecommendationsScraper(
        os.path.join(
            output_dir, output_config.get("taic_website_recommendations_file_name")
        ),
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
        refresh,
    )
    TAIC_recs_scraper.extract_recommendations_from_website()


def create_extracted_reports_df(output_dir, output_config):
    dataframes = [
        pd.read_pickle(os.path.join(output_dir, file_name)).set_index("report_id")
        for file_name in [
            output_config.get("parsed_reports_df_file_name"),
            output_config.get("toc_df_file_name"),
            output_config.get("report_sections_df_file_name"),
            output_config.get("report_event_types_df_file_name"),
            output_config.get("recommendations_df_file_name"),
            output_config.get("safety_issues_df_file_name"),
        ]
    ]

    dataframes[-2].rename(
        columns={
            "important_text": "important_text_recommendation",
            "pages_read": "pages_read_recommendation",
        },
        inplace=True,
    )
    dataframes[-1].rename(
        columns={
            "important_text": "important_text_safety_issue",
            "pages_read": "pages_read_safety_issue",
        },
        inplace=True,
    )

    combined_df = dataframes[0].join(dataframes[1:], how="outer")

    # Adding agency_id and url
    report_titles = pd.read_pickle(
        os.path.join(output_dir, output_config.get("report_titles_df_file_name"))
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
        if len(parts) != 4:
            problematic_ids.append((report_id, len(parts), parts))

    if problematic_ids:
        print(f"Found {len(problematic_ids)} problematic report IDs:")
        # for report_id, num_parts, parts in problematic_ids:
        #     print(f"  Report ID: '{report_id}' has {num_parts} parts: {parts}")
        # raise ValueError(f"Report IDs need to follow a particular format, instead found {len(problematic_ids)} problematic IDs. Here are some examples: {problematic_ids[:5]}...")

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
        os.path.join(output_dir, output_config.get("extracted_reports_df_file_name"))
    )


def extract(output_dir, config, refresh):
    output_config = config.get("output")

    report_extractor = ReportExtracting.ReportExtractingProcessor(
        os.path.join(output_dir, output_config.get("parsed_reports_df_file_name")),
        refresh,
    )

    report_extractor.extract_table_of_contents_from_reports(
        os.path.join(output_dir, output_config.get("toc_df_file_name"))
    )

    report_extractor.extract_safety_issues_from_reports(
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
        os.path.join(output_dir, output_config.get("toc_df_file_name")),
        os.path.join(
            output_dir, output_config.get("atsb_website_safety_issues_file_name")
        ),
        os.path.join(output_dir, output_config.get("safety_issues_df_file_name")),
    )

    report_extractor.extract_recommendations(
        os.path.join(output_dir, output_config.get("recommendations_df_file_name")),
        os.path.join(
            output_dir, output_config.get("tsb_website_recommendations_file_name")
        ),
        os.path.join(
            output_dir, output_config.get("taic_website_recommendations_file_name")
        ),
        os.path.join(output_dir, output_config.get("toc_df_file_name")),
    )

    report_extractor.extract_sections_from_text(
        15, os.path.join(output_dir, output_config.get("report_sections_df_file_name"))
    )

    ReportTypeAssignment.ReportTypeAssigner(
        os.path.join(output_dir, output_config.get("all_event_types_df_file_name")),
        os.path.join(output_dir, output_config.get("report_titles_df_file_name")),
        os.path.join(output_dir, output_config.get("parsed_reports_df_file_name")),
        os.path.join(output_dir, output_config.get("report_event_types_df_file_name")),
    ).assign_report_types()

    print(
        f"Merging all dataframes into {output_config.get('extracted_reports_df_file_name')}"
    )

    create_extracted_reports_df(output_dir, output_config)


def analyze(output_dir, config, refresh):
    output_config = config.get("output")

    RecommendationSafetyIssueLinking.RecommendationSafetyIssueLinker().evaluate_links_for_report(
        os.path.join(output_dir, output_config.get("extracted_reports_df_file_name")),
        os.path.join(
            output_dir,
            output_config.get("recommendation_safety_issue_links_df_file_name"),
        ),
    )

    RecommendationResponseClassification.RecommendationResponseClassificationProcessor().process(
        os.path.join(output_dir, output_config.get("recommendations_df_file_name")),
        os.path.join(
            output_dir,
            output_config.get("recommendation_response_classification_df_file_name"),
        ),
    )
    vector_config = config.get("vector")

    vectordb = Embedding.VectorDB(
        os.path.join(output_dir, output_config.get("vector_db_document_ids_file_name")),
        os.environ["VECTORDB_URI"],
        vector_config["model"]["name"],
        vector_config["model"]["context_limit"],
        vector_config["table_name"],
    )
    vectordb.process_extracted_reports(
        os.path.join(output_dir, output_config.get("extracted_reports_df_file_name")),
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


def upload(container_name, output_dir, output_config):
    uploader = EngineOutputUploader(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        container_name,
        output_dir,
    )

    uploader.upload_latest_output()


def cli():
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
        choices=["download", "gather", "extract", "analyze", "upload", "all"],
        required=True,
        help="This is function that you want to run.",
    )

    args = parser.parse_args()

    # Initialize timing tracker
    timing_results = {}
    total_start_time = time.time()

    # Get the config settings for the engine.
    engine_settings = Config.configReader.get_config()["engine"]

    # Set working directory to output folder
    output_path = engine_settings.get("output").get("folder_name")

    if not os.path.exists(output_path):
        # Create the directory
        os.makedirs(output_path)

    match args.run_type:
        case "download":
            start_time = time.time()
            download(engine_settings.get("output").get("container_name"), output_path)
            timing_results["download"] = time.time() - start_time
        case "gather":
            start_time = time.time()
            gather(output_path, engine_settings, args.refresh)
            timing_results["gather"] = time.time() - start_time
        case "extract":
            start_time = time.time()
            extract(output_path, engine_settings, args.refresh)
            timing_results["extract"] = time.time() - start_time
        case "analyze":
            start_time = time.time()
            analyze(output_path, engine_settings, args.refresh)
            timing_results["analyze"] = time.time() - start_time
        case "upload":
            start_time = time.time()
            upload(
                engine_settings.get("output").get("container_name"),
                output_path,
                engine_settings.get("output"),
            )
            timing_results["upload"] = time.time() - start_time
        case "all":
            # Download step
            start_time = time.time()
            download(engine_settings.get("output").get("container_name"), output_path)
            timing_results["download"] = time.time() - start_time

            # Gather step
            start_time = time.time()
            gather(output_path, engine_settings, args.refresh)
            timing_results["gather"] = time.time() - start_time

            # Extract step
            start_time = time.time()
            extract(output_path, engine_settings, args.refresh)
            timing_results["extract"] = time.time() - start_time

            # Analyze step
            start_time = time.time()
            analyze(output_path, engine_settings, args.refresh)
            timing_results["analyze"] = time.time() - start_time

            # Upload step
            start_time = time.time()
            upload(
                engine_settings.get("output").get("container_name"),
                output_path,
                engine_settings.get("output"),
            )
            timing_results["upload"] = time.time() - start_time

    # Calculate total time
    total_time = time.time() - total_start_time

    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)

    def format_duration(seconds):
        """Format duration to show appropriate time units"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s ({seconds:.2f} seconds)"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return (
                f"{hours}h {minutes}m {remaining_seconds:.1f}s ({seconds:.2f} seconds)"
            )

    for step, duration in timing_results.items():
        formatted_time = format_duration(duration)
        print(f"{step.upper():>10}: {formatted_time}")

    if len(timing_results) > 1:
        print("-" * 60)
        formatted_total = format_duration(total_time)
        print(f"{'TOTAL':>10}: {formatted_total}")

    print("=" * 60)


if __name__ == "__main__":
    cli()
