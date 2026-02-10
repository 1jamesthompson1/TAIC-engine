"""A module for extracting structured informatin from accident investigation reports.

This includes extracting safety issues, recommendations, and chunking the report into sections based on page numbers. The safety issue and recommendation extracting is done by AI. While chunking is a simple recursive text splitter.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd
import regex as re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

from engine.utils.AICaller import ai_caller


class SafetyIssueItem(BaseModel):
    """Represents a safety issue extracted from the report."""

    safety_issue: str = Field(
        ...,
        description="The text of the actual safety issue (e.g ignore 'safety issue -').",
    )
    quality: Literal["exact", "inferred"] = Field(
        ...,
        description=(
            "Whether the safety issue is an exact safety issue "
            "(i.e a verbatim safety issue) or an inferred safety issue "
            "(i.e implied in the report)."
        ),
    )


class RecommendationItem(BaseModel):
    """Represents a recommendation extracted from the report."""

    recommendation: str = Field(
        ...,
        description="The text of the recommendation made by the agency in the report. Copy the recommendation verbatim.",
    )
    recommendation_id: str | None = Field(
        default=None,
        description="The unique identifier for the recommendation if it exists in the report (sometimes called 'id', 'number'). If none is given then return None.",
    )
    recipient: str | None = Field(
        default=None,
        description="The recipient of the recommendation. I.e who the recommendation was addressed to.",
    )
    recommendation_context: str | None = Field(
        default=None,
        description="The context or background information related to the recommendation, if available. This is not always present. It is normally a paragraph or two that states the reasoning behind the recommendation (typically can end with something along the lines of 'therefore we recommend...').",
    )
    made: str | None = Field(
        default=None,
        description="The date or time when the recommendation was made, if available.",
    )


def ai_read_report(
    agency_name: str, report_text: str, safety_issues: bool, recommendations: bool
) -> BaseModel:
    """Use AI to read the report and extract safety issues and recommendations as needed.

    Args:
        agency_name: The name of the investigation agency (e.g., 'TAIC', 'TSB', 'ATSB').
        report_text: The full text of the report.
        safety_issues: Whether to extract safety issues using AI.
        recommendations: Whether to extract recommendations using AI.

    Returns:
        A BaseModel containing the extracted safety issues and/or recommendations, depending on the input flags.

    Raises:
        ValueError: If neither safety_issues nor recommendations is True, or if the AI response is None.

    """
    if not any([safety_issues, recommendations]):
        raise ValueError(
            "At least one of safety_issues or recommendations must be True"
        )

    system_prompt = """
You are a highly skilled AI specialized in extracting structured information from safety investigation reports. Your task is to read the provided report text and extract specific information based on the given instructions.    

There are techincal definitions you should understand:
Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would increase the likelihood of an occurrence, and/or the severity of any adverse consequences associated with the occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the potential to adversely affect the safety of future operations, and
• is characteristic of an organisation, a system, or an operational environment at a specific point in time.

Safety Issues are derived from safety factors classified either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may cover a single safety issue, or two or more related safety issues.

Recommendations - Formal suggestions made by the investigation agency to address identified safety issues. Recommendations are directed towards specific entities, such as regulatory bodies, industry organizations, or operators, with the aim of improving safety and preventing future occurrences.
    """

    taic_specific = """
An exact safety issue will start with something like 'safety issue: ...' and will generaly go until the end of the "paragraph" (i.e until it reaches a line that breaks earlier).
"""
    tsb_specific = """
An inferred safety issue will generally be found in the "findings" section of the report. Look for phrases and issuees that imply a safety issue (a problem that could have a safety impact in the future) even if it is not explicitly stated as a safety issue. Exact Safety issues are not generally stated in TSB reports, the exception being when there is a format like "Safety issue: ...".
"""
    safety_issue_prompt = f"""
Safety issue extraction instructions:
Please only respond with safety issues that are quite clearly stated ("exact" safety issues) or implied ("inferred" safety issues) in the report. Each report will only contain one type of safety issue. If exact safety issues are stated then only respond with those. If no exact safety issues are stated then respond with inferred safety issues.

If the report conatins a phrase like "No safety issues were identified" or "No safety issues were found" then respond with an empty list of safety issues.

{taic_specific if agency_name == "TAIC" else tsb_specific if agency_name == "TSB" else ""}
"""

    recommendation_prompt = f"""
Recommendation extraction instructions:
Please extract all recommendations made in the report. Copy the recommendation verbatim. If there is a unique identifier for the recommendation (e.g Recommendation 1, Rec-01, etc) then include that as the recommendation_id. If there is any context or background information related to the recommendation, include that as recommendation_context (this is sometimes present like a paragraph just before or just after the stated recommendations).
I only want recommendations that are formally made by {agency_name} in the report and which are specific to this particular accident (i.e not previous recommendations they have made). Do not include any recommendations made by other agencies or entities.
"""

    prompt = f"""
You are provided with the following report text:
'''
{report_text}
'''

Based on the provided report text, please extract the following information:

{safety_issue_prompt if safety_issues else ""}

{recommendation_prompt if recommendations else ""}
"""

    # Dynamically create the output structure based on what needs to be extracted
    fields = {}

    if safety_issues:
        fields["safety_issues"] = (
            list[SafetyIssueItem],
            Field(
                default_factory=list,
                description="A list of all safety issues identified in the report.",
            ),
        )

    if recommendations:
        fields["recommendations"] = (
            list[RecommendationItem],
            Field(
                default_factory=list,
                description="A list of all recommendations made in the report.",
            ),
        )

    ExtractedReport = create_model("ExtractedReport", **fields)

    response = ai_caller.query(
        model="gpt-5-mini",
        reasoning="high",
        system=system_prompt,
        user=prompt,
        output_structure=ExtractedReport,
    )

    if response is None:
        raise ValueError("AI extract response is None")

    return response


def chunk_report_into_sections(report_text: str) -> dict[str, str]:
    """Chunks the report into sections based on headings, with each chunk labeled by its starting page.

    Args:
        report_text (str): The full text of the report.

    Returns:
        dict[str, str]: A dictionary mapping page numbers (as strings) to their corresponding chunk text.
                       Keys are formatted as "page_X" where X is the page number.

    """
    # Find all page markers and their positions
    page_regex = re.compile(r"<< Page (\d+|[xvi]+) >>")
    page_matches = list(page_regex.finditer(report_text))

    # Create a mapping of character position to page number
    position_to_page = {}
    for match in page_matches:
        position_to_page[match.start()] = match.group(1)

    # Split the report into chunks
    reporter_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,  # Use character length for splitting
    )
    sections = reporter_splitter.split_text(report_text)

    # Assign each chunk to a page number based on where it starts in the original text
    chunks_with_pages = {}
    page_for_sections = []
    search_from = 0

    # Find out which page each section belongs to by looking at where it starts in the original text and finding the most recent page marker before that position
    for section in sections:
        section_start = report_text.find(section, search_from)
        if section_start == -1:
            section_start = report_text.find(section)
        else:
            search_from = section_start + len(section)

        # Find the most recent page marker before this position
        current_page = "pre"
        for pos in sorted(position_to_page.keys()):
            if pos <= section_start:
                current_page = position_to_page[pos]
            else:
                break

        page_for_sections.append((current_page, section))

    # Count how many sections belong to each page
    page_totals = {}
    for page, _section in page_for_sections:
        page_totals[page] = page_totals.get(page, 0) + 1

    # Create the final mapping of page keys to section text, adding suffixes for multiple sections on the same page
    page_counts = {}
    for page, section in page_for_sections:
        page_counts[page] = page_counts.get(page, 0) + 1
        suffix = page_counts[page]
        if page_totals[page] > 1:
            page_key = f"page_{page}.{suffix}"
        else:
            page_key = f"page_{page}"
        chunks_with_pages[page_key] = section

    return chunks_with_pages


def extract_report(
    report_row: pd.Series | dict,
    ai_extraction_config: dict = {
        "ATSB": {
            "safety_issues": False,
            "recommendations": True,
        },
        "TSB": {
            "safety_issues": True,
            "recommendations": False,
        },
        "TAIC": {
            "safety_issues": True,
            "recommendations": False,
        },
    },
) -> dict:
    """Extract safety issues, recommendations, and sections from a report row.

    Args:
        report_row: Row containing 'report_id' and 'report_text'.
        ai_extraction_config: Configuration dictionary specifying which extraction tasks (safety_issues, recommendations) to perform for each agency. Defaults to extracting safety issues for TAIC and TSB, and recommendations for ATSB.

    Returns:
        dict: Extracted fields with keys 'report_id', 'safety_issues',
              'recommendations', and 'sections'.

    """
    report_id = report_row["report_id"]
    report_text = report_row["text"]
    agency = report_id.split("_")[0]

    extraction_config = ai_extraction_config[agency].copy()

    # Add ATSB reports that are older than 2008
    if agency == "ATSB":
        report_year = int(report_id.split("_")[2])
        if report_year < 2008:
            extraction_config["safety_issues"] = True

    # Read report using AI to extract safety issues and/or recommendations
    extracted_data = ai_read_report(
        agency_name=agency,
        report_text=report_text,
        **extraction_config,
    )

    safety_issues = (
        extracted_data.safety_issues if extraction_config["safety_issues"] else []
    )
    recommendations = (
        extracted_data.recommendations if extraction_config["recommendations"] else []
    )

    # Chunk the report into sections
    sections = chunk_report_into_sections(report_text)

    return {
        "report_id": report_id,
        "safety_issues": safety_issues,
        "recommendations": recommendations,
        "sections": sections,
    }


def process_reports_parallel(
    reports_df: pd.DataFrame,
    current_extracted_df: pd.DataFrame | None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Process reports in parallel, skipping those already extracted.

    This function identifies which reports need to be processed by comparing the input
    reports_df with current_extracted_df. It then processes new reports in parallel
    using ThreadPoolExecutor for improved performance.

    Args:
        reports_df: DataFrame with columns ['report_id', 'report_text'] containing the reports to process.
        current_extracted_df: DataFrame with columns ['report_id', 'safety_issues', 'recommendations', 'sections'] containing already processed reports. Pass an empty DataFrame to process all reports.
        max_workers: Maximum number of parallel workers (default: 4). Adjust based on your API rate limits and system resources.

    Returns:
        pd.DataFrame: Updated DataFrame containing both the input current_extracted_df reports and newly extracted reports. Columns include:
                     - report_id: Unique identifier for the report
                     - safety_issues: List of extracted safety issues
                     - recommendations: List of extracted recommendations
                     - sections: Dictionary of report sections

    Example:
        >>> reports = pd.DataFrame({
        ...     'report_id': ['TAIC_001', 'TAIC_002'],
        ...     'report_text': ['...', '...']
        ... })
        >>> current = pd.DataFrame({
        ...     'report_id': ['TAIC_001'],
        ...     'safety_issues': [[...]],
        ...     'recommendations': [[...]],
        ...     'sections': [{...}]
        ... })
        >>> result = process_reports_parallel(reports, current, max_workers=8)
        >>> # Only TAIC_002 will be processed, TAIC_001 data preserved

    """
    # Identify reports that need processing
    if current_extracted_df is not None and len(current_extracted_df) > 0:
        already_processed = set(current_extracted_df["report_id"])
        reports_to_process = reports_df[
            ~reports_df["report_id"].isin(already_processed)
        ]
    else:
        current_extracted_df = pd.DataFrame(
            columns=["report_id", "safety_issues", "recommendations", "sections"]
        )
        reports_to_process = reports_df

    print(
        f"Processing {len(reports_to_process)} reports "
        f"(skipping {len(reports_df) - len(reports_to_process)} already processed)"
    )

    if len(reports_to_process) == 0:
        return current_extracted_df

    # Process reports in parallel
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_report = {
            executor.submit(extract_report, row): row["report_id"]
            for _, row in reports_to_process.iterrows()
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_report),
            total=len(future_to_report),
            desc="Processing",
        ):
            report_id = future_to_report[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Error processing {report_id}: {exc}")
                continue
            if result is not None:
                results.append(result)

    # Create DataFrame from results
    new_extracted_df = pd.DataFrame(results)

    # Combine with existing data
    if len(current_extracted_df) > 0:
        completed_df = pd.concat(
            [current_extracted_df, new_extracted_df], ignore_index=True
        )
    else:
        completed_df = new_extracted_df

    return completed_df
