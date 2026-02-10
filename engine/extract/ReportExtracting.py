from enum import Enum

import regex as re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, create_model

from engine.utils.AICaller import ai_caller


class SafetyIssueType(str, Enum):
    EXACT = "exact"
    INFERRED = "inferred"


class SafetyIssueItem(BaseModel):
    safety_issue: str = Field(
        ...,
        description="The text of the actual safety issue (e.g ignore 'safety issue -').",
    )
    quality: SafetyIssueType = Field(
        ...,
        description="Whether the safety issue is an exact safety issue (i.e a verbatim safety issue) or an inferred safety issue (i.e implied in the report).",
    )


class RecommendationItem(BaseModel):
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


class CompleteExtractedReport(BaseModel):
    safety_issues: list[SafetyIssueItem] = Field(
        ...,
        description="A list of all safety issues identified in the report.",
    )
    recommendations: list[RecommendationItem] = Field(
        ...,
        description="A list of all recommendations made in the report.",
    )
    sections: dict[str, str] = Field(
        ...,
        description="A dictionary mapping section titles to their corresponding text in the report. This is the chunking of the report into sections.",
    )


# A config setup of which agnecies need ai extraction of the items

ai_extraction_needed = {
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
}


def extract_report(report_id: str, report_text: str) -> CompleteExtractedReport:
    """
    Extracts safety issues, recommendations, and sections from the report.

    This is the main function that orchestrates the extraction process.
    """

    agency = report_id.split("_")[0]

    extraction_config = ai_extraction_needed[agency]

    # Read report using AI to extract safety issues and/or recommendations
    extracted_data = ai_read_report(
        agency_name=agency,
        report_text=report_text,
        **extraction_config,
    )

    if extraction_config["safety_issues"]:
        safety_issues = extracted_data.safety_issues
    else:
        safety_issues = []

    if extraction_config["recommendations"]:
        recommendations = extracted_data.recommendations
    else:
        recommendations = []

    # Chunk the report into sections
    sections = chunk_report_into_sections(report_text)

    return CompleteExtractedReport(
        safety_issues=safety_issues,
        recommendations=recommendations,
        sections=sections,
    )


def ai_read_report(
    agency_name: str, report_text: str, safety_issues: bool, recommendations: bool
):
    """
    Uses AI to read the report and extract safety issues and recommendations as needed.

    Args:
        report_text (str): The full text of the report.
        safety_issues (bool): Whether to extract safety issues using AI.
        recommendations (bool): Whether to extract recommendations using AI.
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
    """
    Chunks the report into sections based on headings, with each chunk labeled by its starting page.

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
