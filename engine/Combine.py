"""A module that takes in all the individual data outputs of the engine and formats them into a single long dataframe."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from engine import Modes
from engine.Logging import get_logger
from engine.SavedDataFrames import (
    ATSBWebsiteSafetyIssues,
    DataForVectorDB,
    ExtractedReports,
    ParsedReports,
    ReportTitles,
    TAICWebsiteRecommendations,
    TSBWebsiteRecommendations,
)

logger = get_logger(__name__)

MISSING_REPORTS_MAX_DISPLAY = 20


@dataclass(frozen=True)
class LongDataFormatDCs:
    """Group file paths required for creating long format report data.

    This structure encapsulates all engine output paths required to combine
    reports into a single long dataframe for downstream embedding and analysis.
    """

    parsed_reports_dc: ParsedReports
    report_titles_dc: ReportTitles
    extracted_reports_dc: ExtractedReports
    atsb_safety_issues_dc: ATSBWebsiteSafetyIssues
    tsb_recommendations_dc: TSBWebsiteRecommendations
    taic_recommendations_dc: TAICWebsiteRecommendations


def expand_extracted_report_metadata(extracted_report: pd.DataFrame) -> pd.DataFrame:
    """Expand the extracted report to add the occurence metadata of each report to the rows.

    This is used later to build the long format where each row is information rich.

    Args:
        extracted_report: The dataframe containing the extracted report information, including the occurence metadata.

    Returns:
        A dataframe where the occurence metadata has been expanded to be columns of the dataframe, so that each row is information rich.
    """
    extracted_report = extracted_report.copy()

    extracted_report = extracted_report.drop(
        columns=["safety_issues", "recommendations", "sections"]
    )

    def pull_metadata(metadata):
        occurrence_metadata = metadata["occurrence"]
        return {
            "location": occurrence_metadata["location"]["standardized_location"],
            "occurrence_date": occurrence_metadata["occurrence_datetime"][
                "local_datetime"
            ],
            "occurrence_type": occurrence_metadata["occurrence_type"],
            "fatalities": occurrence_metadata["fatalities"],
            "injuries": occurrence_metadata["injuries"],
            "damage": occurrence_metadata["damage_description"],
            "who_may_benefit": occurrence_metadata["who_may_benefit"],
        }

    pulled_metadata = pd.DataFrame(
        extracted_report["metadata"].map(pull_metadata).tolist()
    )
    pulled_metadata["occurrence_date"] = pd.to_datetime(
        pulled_metadata["occurrence_date"], errors="coerce"
    )

    return pd.concat(
        [extracted_report.drop(columns=["metadata"]), pulled_metadata], axis=1
    )


def create_complete_report_metadata(
    extracted_metadata: pd.DataFrame, scraped_metadata: pd.DataFrame
) -> pd.DataFrame:
    """Combine the extracted report metadata with the scraped report metadata to create a complete report metadata dataframe.

    This is used later to build the long format where each row is information rich. It combines the extracted metadata and scraped metadata togather. Only reports with both extracted and scraped metadata will be included.

    Args:
        extracted_metadata: The dataframe containing the extracted report metadata.
        scraped_metadata: The dataframe containing the scraped report metadata.

    Returns:
        A dataframe where the extracted report metadata and scraped report metadata have been combined, so that each row is information rich.
    """
    combined_metadata = extracted_metadata.merge(
        scraped_metadata[["report_id", "url", "agency_id"]], on="report_id", how="inner"
    )

    missing_scraped_metadata_report_ids = set(
        extracted_metadata["report_id"].unique()
    ) - set(scraped_metadata["report_id"].unique())
    if missing_scraped_metadata_report_ids:
        logger.warning(
            f"There are {len(missing_scraped_metadata_report_ids)} reports in the extracted metadata that do not have scraped metadata and so are skipped. These reports are:\n{combined_metadata[combined_metadata['report_id'].isin(missing_scraped_metadata_report_ids)][['report_id']].to_csv(index=False) if len(missing_scraped_metadata_report_ids) < MISSING_REPORTS_MAX_DISPLAY else 'too many to display (>20)'}"
        )

    # Take the scraped event type if it exists, otherwise take the extracted occurrence type.
    combined_metadata["occurrence_type"] = scraped_metadata["event_type"].combine_first(
        extracted_metadata["occurrence_type"]
    )

    # Add in mode
    combined_metadata["mode"] = combined_metadata["report_id"].apply(
        lambda x: str(Modes.get_report_mode_from_id(x).value)
    )

    # Add in year (from report ID which is the year of investigation launch)
    combined_metadata["year"] = combined_metadata["report_id"].apply(
        lambda x: int(x.split("_")[2])
    )

    combined_metadata["agency"] = combined_metadata["report_id"].apply(
        lambda x: x.split("_")[0]
    )

    return combined_metadata


def create_document_id(
    report_id: str, document_type: Literal["sec", "sum", "rec", "si"], item_id: int
) -> str:
    """Create a unique document id for each document that we want to embed, which includes the report id, the document type and an item id.

    Args:
        report_id: The id of the report that the document is associated with.
        document_type: The type of the document (e.g. section, safety issue, recommendation, etc).
        item_id: A unique identifier for the document within the report (e.g. section number, safety issue number, recommendation number, etc).

    Returns:
        A string that is a unique identifier for the document, which includes the report id, the document type and the item id.
    """
    return f"{report_id}_{document_type}_{item_id}"


def combine_recommendations(atsb_extracted, tsb_scraped, taic_scraped) -> pd.DataFrame:
    """Combine the extracted recommendations and scraped recommendations.

    Turns into standard format to be enriched with metadata.

    Args:
        atsb_extracted: The dataframe containing the extracted recs from reports.
        tsb_scraped: The dataframe containing the scraped recs from the TSB website.
        taic_scraped: The dataframe containing the scraped recs from the TAIC website.

    Returns:
        A dataframe contains the standardised 4 columns of report_id, document_id, document, and document_type along with url to use as document url.
    """
    taic_scraped["recommendation_context"] = taic_scraped["reply_text"]

    tsb_scraped["recipient"] = None

    atsb_extracted = atsb_extracted.dropna(subset=["recommendations"]).join(
        atsb_extracted.explode("recommendations")["recommendations"].apply(pd.Series)
    )
    atsb_extracted["url"] = None

    columns_to_merge = [
        "report_id",
        "recommendation_id",
        "made",
        "recipient",
        "recommendation",
        "recommendation_context",
        "url",
    ]

    combined_recs = pd.concat(
        [
            atsb_extracted[columns_to_merge],
            tsb_scraped[columns_to_merge],
            taic_scraped[columns_to_merge],
        ],
        ignore_index=True,
        axis=0,
    )

    combined_recs["document_id"] = combined_recs.apply(
        lambda row: create_document_id(
            row["report_id"], "rec", row["recommendation_id"]
        ),
        axis=1,
    )

    final_recs = combined_recs[["report_id", "document_id", "url"]].copy()

    def construct_recommendation_document(row):
        document = row["recommendation"]
        if row["recommendation_context"]:
            document += f"\n\n---\n**context**\n{row['recommendation_context']}"
        if row["recipient"]:
            document += f"\n**recipient:** {row['recipient']}"
        if row["made"]:
            document += f"\n**made:** {row['made']}"
        return document

    final_recs["document"] = combined_recs.apply(
        construct_recommendation_document, axis=1
    )

    final_recs["document_type"] = "recommendation"

    return final_recs


def combine_safety_issues(extracted_si, atsb_scraped_si) -> pd.DataFrame:
    """Combine the extracted safety issues and scraped safety issues.

    Turns into standard format to be enriched with metadata.

    Args:
        extracted_si: The dataframe containing the extracted safety issues from reports.
        atsb_scraped_si: The dataframe containing the scraped safety issues from the TSB website.

    Returns:
        A dataframe contains the standardised 4 columns of report_id, document_id, document, and document_type along with url to use as document url.
    """
    extracted_si = (
        extracted_si.dropna(subset=["safety_issues"])
        .join(extracted_si.explode("safety_issues")["safety_issues"].apply(pd.Series))
        .reset_index(drop=True)
    )

    extracted_si["safety_issue_id"] = extracted_si["safety_issue_id"].astype("string")

    # For extracted si that have ID none simply label them as 1,2,3 etc. within each report, so that we can create a unique document id for each safety issue.
    generated_ids = extracted_si.groupby("report_id").cumcount() + 1

    extracted_si["safety_issue_id"] = extracted_si["safety_issue_id"].fillna(
        generated_ids.astype(str)
    )

    columns_to_merge = ["report_id", "safety_issue_id", "safety_issue", "quality"]

    combined_si = pd.concat(
        [
            extracted_si[columns_to_merge],
            atsb_scraped_si[columns_to_merge],
        ],
        ignore_index=True,
        axis=0,
    )

    combined_si["document_id"] = combined_si.apply(
        lambda row: create_document_id(row["report_id"], "si", row["safety_issue_id"]),
        axis=1,
    )

    final_si = combined_si[["report_id", "document_id"]].copy()

    final_si.loc[:, "document"] = combined_si.apply(
        lambda row: f"{row['safety_issue']}\n\n(quality: {row['quality']})", axis=1
    )

    final_si.loc[:, "document_type"] = "safety_issue"

    return final_si


def create_long_data_format(
    dcs: LongDataFormatDCs, long_data_format_dc: DataForVectorDB
) -> None:
    """Collate all engine output data into a long format for easier embedding and analysis.

    The output will be a dataframe that is in a long format, where each row corresponds to a single docuemnt (which could be of any time, e.g section, safety issues, recommendation, summary, etc). Along with the document its id it will also contain metadata for the report it is associated with.

    Args:
        dcs: DataClass containing data connectors (parsed reports, report titles,
            extracted reports, safety issues, and recommendations).
        long_data_format_dc: Data connector where the resulting long-format
            dataframe will be saved.
    """
    extracted_reports = dcs.extracted_reports_dc.read()

    report_metadata = create_complete_report_metadata(
        expand_extracted_report_metadata(dcs.extracted_reports_dc.read()),
        dcs.report_titles_dc.read(),
    )

    combined_recs = combine_recommendations(
        extracted_reports,
        dcs.tsb_recommendations_dc.read(),
        dcs.taic_recommendations_dc.read(),
    )
    combined_si = combine_safety_issues(
        extracted_reports, dcs.atsb_safety_issues_dc.read()
    )

    sections = (
        extracted_reports[["report_id", "sections"]]
        .join(
            extracted_reports.explode("sections", ignore_index=True)["sections"].apply(
                pd.Series
            )
        )
        .rename(columns={"text": "document", "section": "document_id"})
        .drop(columns=["sections"])
    )

    sections["document_id"] = sections.apply(
        lambda row: create_document_id(row["report_id"], "sec", row["document_id"]),
        axis=1,
    )
    sections["document_type"] = "section"

    summaries = (
        dcs.report_titles_dc.read()[["report_id", "summary"]]
        .dropna()
        .rename(columns={"summary": "document"})
    )
    summaries["document_id"] = summaries.apply(
        lambda row: create_document_id(row["report_id"], "sum", 0), axis=1
    )
    summaries["document_type"] = "summary"

    report_text = dcs.parsed_reports_dc.read()[["report_id", "text"]].rename(
        columns={"text": "document"}
    )
    report_text["document_id"] = report_text["report_id"]
    report_text["document_type"] = "report_text"

    long_format = pd.concat(
        [
            sections,
            summaries,
            report_text,
            combined_recs,
            combined_si,
        ],
        ignore_index=True,
        axis=0,
    )

    missing_metadata_report_ids = set(long_format["report_id"].unique()) - set(
        report_metadata["report_id"].unique()
    )

    if missing_metadata_report_ids:
        logger.warning(
            f"There are {len(missing_metadata_report_ids)} reports in the long format that do not have metadata and will be dropped when we merge with the metadata."
        )
        logger.info(
            f"Documents with missing metadata:\n{long_format[long_format['report_id'].isin(missing_metadata_report_ids)][['report_id', 'document_type', 'document_id']].to_csv(index=False) if len(missing_metadata_report_ids) < MISSING_REPORTS_MAX_DISPLAY else 'too many to display (>20)'}"
        )

    long_format = long_format.merge(
        report_metadata,
        on="report_id",
        how="inner",
        suffixes=("", "_metadata"),
    )

    # FInd rows with missing document
    missing_document_rows = long_format[long_format["document"].isna()]
    if not missing_document_rows.empty:
        logger.warning(
            f"There are {len(missing_document_rows)} rows in the long format that have missing document text. These rows will be dropped when we save the long format.\n{missing_document_rows[['report_id', 'document_type', 'document_id']].to_csv(index=False) if len(missing_document_rows) < MISSING_REPORTS_MAX_DISPLAY else 'too many to display (>20)'}"
        )

    long_format = long_format.dropna(subset=["document"])

    # Allow specific URLs for particular documents (i.e recommendations) to be used over the generic report URL from the metadata.
    long_format["url"] = long_format["url"].combine_first(long_format["url_metadata"])
    long_format = long_format.drop(columns=["url_metadata"])

    long_data_format_dc.save(long_format)
