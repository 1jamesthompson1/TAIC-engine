"""A module for extracting structured informatin from accident investigation reports.

This includes extracting safety issues, recommendations, and chunking the report into sections based on page numbers. The safety issue and recommendation extracting is done by AI. While chunking is a simple recursive text splitter.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import regex as re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, create_model, field_validator
from tqdm import tqdm

from engine import Modes
from engine.AICaller import ai_caller
from engine.Logging import get_logger
from engine.SavedDataFrames import ExtractedReports, ParsedReports

logger = get_logger(__name__)


# ISO 6709 basic coordinate representation: +DD.DDDD+DDD.DDDD[/]
ISO6709_PATTERN = re.compile(
    r"^[+-]\d{2}(?:\.\d+)?[+-]\d{3}(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)?/$"
)


class InvalidExtractionConfigError(ValueError):
    """Raised when none of the extraction flags (safety_issues, recommendations, metadata) are True."""

    def __init__(self):
        """Initialize the exception with an appropriate error message."""
        super().__init__(
            "At least one of safety_issues, recommendations, or metadata must be True"
        )


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
        description="The context or background information related to the recommendation, if available. This is not always present. It is normally a paragraph or two that states the reasoning behind the recommendation (typically can end with something along the lines of 'therefore we recommend...'). In some reports recommendation are made only if the safety actions were not sufficient. Then for these situations the context should bethe safety actions taken and why the investigation agency deemed these actions insufficient (i.e the reasoning behind the recommendation). If recommendation context is not present in the recommendaion section then it should be None (do not look for context in other sections of the report, only the section that is talking abuot safety issues/actions and recommendations).",
    )
    made: str | None = Field(
        default=None,
        description="The date when the recommendation was made, if available in ISO 8601 format. If not available, set to None.",
    )


class OccurrenceDateTime(BaseModel):
    """Represents occurrence datetime with timezone as separate fields."""

    local_datetime: datetime | str = Field(
        description=(
            "Local occurrence datetime as ISO 8601 without timezone "
            "(YYYY-MM-DDTHH:MM:SS). Time is required. Return as string and the model will auto format it into a python datetime"
        ),
    )
    time_zone: str = Field(
        description=(
            "Timezone identifier or abbreviation for the occurrence in UTC offset (e.g 'UTC+13')"
        ),
    )
    time_zone_source: Literal["explicit_in_report", "inferred"] | None = Field(
        default=None,
        description=(
            "Whether the timezone is explicitly stated in the report or inferred."
        ),
    )

    @field_validator("local_datetime", mode="before")
    @classmethod
    def ensure_specific_time(cls, value):
        """Require datetime values to include a specific time and no timezone offset.

        Returns:
            datetime | None: A timezone-naive datetime value, or None if input is None.

        Raises:
            TypeError: If ``value`` is not a ``datetime`` or ISO 8601 string.
            ValueError: If ``value`` does not include a specific time, is not valid ISO 8601,
                or includes timezone information.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            cleaned = value.strip()
            if "T" not in cleaned and " " not in cleaned:
                msg = (
                    "local_datetime must include a specific time "
                    "(YYYY-MM-DDTHH:MM[:SS])"
                )
                raise ValueError(msg)
            if "T" not in cleaned and " " in cleaned:
                cleaned = cleaned.replace(" ", "T", 1)
            if len(cleaned) == 16:  # noqa: PLR2004
                cleaned = f"{cleaned}:00"

            try:
                parsed = datetime.fromisoformat(cleaned)
            except ValueError as e:
                msg = (
                    "local_datetime must be ISO 8601 with date and time "
                    "(YYYY-MM-DDTHH:MM[:SS]) and no timezone"
                )
                raise ValueError(msg) from e
        else:
            msg = "local_datetime must be a datetime or ISO 8601 string"
            raise TypeError(msg)

        if parsed.tzinfo is not None:
            msg = "local_datetime must not include timezone information"
            raise ValueError(msg)

        return parsed


class OccurrenceLocation(BaseModel):
    """Represents occurrence location in both descriptive and parseable formats."""

    description: str | None = Field(
        default=None,
        description="Human-readable location description from the report text (e.g., '2 NM north of Ardmore Aerodrome').",
    )
    standardized_location: str | None = Field(
        default=None,
        description="Standardized location in ISO 6709 format (e.g., '-37.02972+174.97333/').",
    )

    @field_validator("standardized_location", mode="before")
    @classmethod
    def ensure_iso6709_standardized_location(cls, value):
        """Validate standardized_location is ISO 6709 coordinate format.

        Returns:
            str | None: The stripped coordinate string if valid, otherwise None.

        Raises:
            TypeError: If value is not a string or None.
            ValueError: If value is not valid ISO 6709 format.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            msg = "standardized_location must be a string in ISO 6709 format"
            raise TypeError(msg)

        cleaned = value.strip()
        if ISO6709_PATTERN.fullmatch(cleaned):
            return cleaned

        msg = (
            "standardized_location must be ISO 6709, for example '-37.02972+174.97333/'"
        )
        raise ValueError(msg)


class OccurrenceMetadata(BaseModel):
    """Represents common metadata extracted from an accident occurrence report (all modes)."""

    occurrence_datetime: OccurrenceDateTime = Field(
        description="Structured occurrence datetime with local datetime, timezone, and timezone source."
    )
    location: OccurrenceLocation = Field(
        description="Structured occurrence location with descriptive and standardized values.",
    )

    # Not sure if i wll move remove this in favor of the other report type extraction. Or I could remove the other report type exrtaction and get it to work from this one here. Might be better as it is reading the entire report so more likley to have it correct
    occurrence_type: str | None = Field(
        default=None,
        description="The type or classification of the occurrence (e.g., collision, derailment, ditching, etc.). This may match existing event type classifications.",
    )
    total_persons_involved: int | None = Field(
        default=None,
        description="The total number of persons involved in the occurrence.",
    )
    fatalities: int | None = Field(
        default=None,
        description="The number of fatalities resulting from the occurrence.",
    )
    injuries: int | None = Field(
        default=None,
        description="The number of persons injured in the occurrence.",
    )

    # Unaware of if this would be useful to have as a category or just a string. It cuold be a category of the worst damage.
    damage_description: str | None = Field(
        default=None,
        description="Description of damage to equipment, property, or environment resulting from the occurrence. It should be a brief summary of the damage as described in the report (e.g 'the aircraft was destroyed', 'the train derailed with minor damage', 'the vessel sustained substantial damage but remained afloat', etc.).",
    )


class AircraftMetadata(BaseModel):
    """Represents aircraft-specific metadata from an air accident report."""

    aircraft_type: (
        Literal["Fixed-wing", "Rotary-wing", "ballon", "glider", "other"] | None
    ) = Field(
        default=None,
        description="Type of aircraft",
    )
    registration: str | None = Field(
        default=None,
        description="Aircraft registration/tail number.",
    )
    make: str | None = Field(
        default=None,
        description="Aircraft manufacturer/make.",
    )
    model: str | None = Field(
        default=None,
        description="Aircraft model.",
    )
    number_of_engines: int | None = Field(
        default=None,
        description="Number of engines on the aircraft.",
    )
    type_of_engines: (
        Literal[
            "piston",
            "turboprop",
            "turbojet",
            "turbofan",
            "turboshaft",
            "electric",
            "hybrid",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description="Type of engines (e.g., turbofan, piston, electric).",
    )
    year_manufactured: int | None = Field(
        default=None,
        description="Year the aircraft was manufactured.",
    )
    operator: str | None = Field(
        default=None,
        description="Operating airline or organization.",
    )

    # Is this the correct breakdown?
    flight_type: (
        Literal["scheduled service", "charter", "private", "training"] | None
    ) = Field(
        default=None,
        description="Type of flight",
    )
    persons_on_board_total: int | None = Field(
        default=None,
        description="Total number of persons on board the aircraft.",
    )
    persons_on_board_crew: int | None = Field(
        default=None,
        description="Number of crew members on boar that is cabin crew plus flight crew (pilots)",
    )
    persons_on_board_passengers: int | None = Field(
        default=None,
        description="Number of passengers on board.",
    )

    # What damage is the correct breakdown? Is there some official taxionomy to use.
    damage: Literal["destroyed", "substantial damage", "minor damage", "nil"] | None = (
        Field(
            default=None,
            description="Level of damage to the aircraft as classified in the report (e.g., 'destroyed', 'substantial damage', 'minor damage').",
        )
    )


class PilotMetadata(BaseModel):
    """Represents pilot-specific metadata from an air accident report."""

    role: str | None = Field(
        default=None,
        description="Pilot role (e.g., captain, first officer, second officer) for multi aircraft incidents write with format 'captain of yyyyyy'.",
    )
    licence: str | None = Field(
        default=None,
        description="Pilot licence type and number if available.",
    )
    age: int | None = Field(
        default=None,
        description="Pilot's age at time of occurrence.",
    )
    total_flying_experience: int | None = Field(
        default=None,
        description="Total flying experience (e.g., 10000, 15) as hours",
    )
    experience_on_type: int | None = Field(
        default=None,
        description="Flying experience on the specific aircraft type as hours",
    )


class TrainMetadata(BaseModel):
    """Represents train-specific metadata from a rail accident report."""

    # What is the correct list of categories
    train_type: Literal["passenger", "freight", "tanker"] | None = Field(
        default=None,
        description="Type of train",
    )
    train_number: str | None = Field(
        default=None,
        description="Train number or identifier.",
    )
    length: str | None = Field(
        default=None,
        description="Length of the train.",
    )
    weight: str | None = Field(
        default=None,
        description="Weight/gross tonnage of the train.",
    )
    classification: (
        Literal[
            "commuter",
            "metro",
            "regional",
            "intercity",
            "high-speed",
            "heavy-haul",
            "shunting",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description="Train classification.",
    )
    year_manufactured: int | None = Field(
        default=None,
        description="Year the train was manufactured.",
    )
    operator: str | None = Field(
        default=None,
        description="Railway operator.",
    )
    operating_crew: int | None = Field(
        default=None,
        description="Number of operating crew members.",
    )


class VesselMetadata(BaseModel):
    """Represents vessel-specific metadata from a marine accident report."""

    vessel_name: str | None = Field(
        default=None,
        description="Name of the vessel.",
    )
    vessel_type: (
        Literal[
            "cargo",
            "container",
            "bulk carrier",
            "tanker",
            "passenger",
            "ferry",
            "fishing",
            "tug",
            "barge",
            "recreational",
            "offshore support",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description="Type of vessel",
    )

    # Could this be a category?
    classification: str | None = Field(
        default=None,
        description="Classification or class of the vessel.",
    )

    # Is there some structure this could be
    classification_limits: str | None = Field(
        default=None,
        description="Classification limits for the vessel.",
    )
    length: float | None = Field(
        default=None,
        description="Length of the vessel in meters",
    )
    breadth: float | None = Field(
        default=None,
        description="Breadth (beam width) of the vessel in meters",
    )
    gross_tonnage: float | None = Field(
        default=None,
        description="Gross tonnage of the vessel.",
    )
    manufacturer: str | None = Field(
        default=None,
        description="Manufacturer/shipbuilder of the vessel.",
    )
    manufacturer_model: str | None = Field(
        default=None,
        description="Manufacturer model if applicable.",
    )
    year_built: int | None = Field(
        default=None,
        description="Year the vessel was built.",
    )

    # Is this the correct breakdown of propulsion types?
    propulsion: (
        Literal[
            "diesel",
            "diesel-electric",
            "gas turbine",
            "steam turbine",
            "battery-electric",
            "wind",
            "human",
            "jet",
        ]
        | None
    ) = Field(
        default=None,
        description="Type of propulsion",
    )
    total_power: float | None = Field(
        default=None,
        description="Total power of the vessel's propulsion system in kW",
    )
    service_speed: int | None = Field(
        default=None,
        description="Service speed of the vessel (in knots).",
    )
    owner_operator: str | None = Field(
        default=None,
        description="Owner or operator of the vessel.",
    )
    port_of_registry: str | None = Field(
        default=None,
        description="Port where the vessel is registered.",
    )
    primary_port: str | None = Field(
        default=None,
        description="Primary port of operation.",
    )
    minimum_crew: int | None = Field(
        default=None,
        description="Minimum required crew size.",
    )


class ReportMetadata(BaseModel):
    """Represents complete metadata extracted from an accident investigation report."""

    occurrence: OccurrenceMetadata = Field(
        description="Common occurrence metadata (all modes).",
    )
    aircraft: list[AircraftMetadata] = Field(
        default_factory=list,
        description="List of aircraft involved (for air accidents). May be empty for non-air reports.",
    )
    pilots: list[PilotMetadata] = Field(
        default_factory=list,
        description="List of pilots involved (for air accidents). May be empty for non-air reports.",
    )
    trains: list[TrainMetadata] = Field(
        default_factory=list,
        description="List of trains involved (for rail accidents). May be empty for non-rail reports.",
    )
    vessels: list[VesselMetadata] = Field(
        default_factory=list,
        description="List of vessels involved (for marine accidents). May be empty for non-marine reports.",
    )


def load_event_type_taxonomy(
    event_types_csv_path: Path,
) -> dict[Modes.Mode, list[str]]:
    """Load and group allowed event types by mode from taxonomy CSV.

    Args:
        event_types_csv_path: Path to the taxonomy CSV (for example, data/event_types.csv).

    Returns:
        dict[Modes.Mode, list[str]]: Allowed event type values by report mode.
    """
    if not event_types_csv_path.exists():
        logger.warning("Event taxonomy file not found at %s", event_types_csv_path)
        return {}

    event_types_df = pd.read_csv(event_types_csv_path)
    required_columns = {"Value", "mode"}
    if not required_columns.issubset(event_types_df.columns):
        logger.warning(
            "Event taxonomy is missing required columns %s in %s",
            sorted(required_columns),
            event_types_csv_path,
        )
        return {}

    mode_name_map = {
        "aviation": Modes.Mode.a,
        "rail": Modes.Mode.r,
        "marine": Modes.Mode.m,
        "a": Modes.Mode.a,
        "r": Modes.Mode.r,
        "m": Modes.Mode.m,
    }

    grouped: dict[Modes.Mode, list[str]] = {}
    for mode_name, rows in event_types_df.groupby(event_types_df["mode"].str.lower()):
        mode = mode_name_map.get(mode_name)
        if mode is None:
            continue

        # Keep first-seen order while removing duplicates.
        values = list(dict.fromkeys(rows["Value"].dropna().astype(str).tolist()))
        if values:
            grouped[mode] = values

    return grouped


def _build_metadata_model_for_mode(
    report_mode: Modes.Mode | None,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[str]] | None,
) -> type[BaseModel]:
    """Build a mode-specific metadata model with constrained occurrence_type values.

    Returns:
        type[BaseModel]: A metadata model for the supplied mode, or ReportMetadata
        when mode-specific constraints are unavailable.
    """
    if report_mode is None or event_type_taxonomy_by_mode is None:
        return ReportMetadata

    allowed_event_types = event_type_taxonomy_by_mode.get(report_mode)
    if not allowed_event_types:
        return ReportMetadata

    allowed_type = Literal[tuple(allowed_event_types)]
    occurrence_metadata_model = create_model(
        f"OccurrenceMetadata_{report_mode.name}",
        __base__=OccurrenceMetadata,
        occurrence_type=(
            allowed_type | None,
            Field(
                default=None,
                description=(
                    "Occurrence type constrained to the taxonomy values "
                    "for this report mode."
                ),
            ),
        ),
    )

    return create_model(
        f"ReportMetadata_{report_mode.name}",
        __base__=ReportMetadata,
        occurrence=(
            occurrence_metadata_model,
            Field(description="Common occurrence metadata (all modes)."),
        ),
    )


def ai_read_report(  # noqa: PLR0913, PLR0917
    agency_name: str,
    report_text: str,
    safety_issues: bool,
    recommendations: bool,
    metadata: bool = True,
    report_mode: Modes.Mode | None = None,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[str]] | None = None,
) -> BaseModel:
    """Use AI to read the report and extract safety issues, recommendations, and metadata as needed.

    Args:
        agency_name: The name of the investigation agency (e.g., 'TAIC', 'TSB', 'ATSB').
        report_text: The full text of the report.
        safety_issues: Whether to extract safety issues using AI.
        recommendations: Whether to extract recommendations using AI.
        metadata: Whether to extract occurrence metadata using AI (default: True).
        report_mode: Optional mode parsed from report ID to constrain taxonomy.
        event_type_taxonomy_by_mode: Optional taxonomy values keyed by mode.

    Returns:
        A BaseModel containing the extracted safety issues, recommendations, and/or metadata,
        depending on the input flags.

    Raises:
        InvalidExtractionConfigError: If none of safety_issues, recommendations, or metadata is True.
        RuntimeError: If the AI query fails.

    """
    if not any([safety_issues, recommendations, metadata]):
        raise InvalidExtractionConfigError()

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
An inferred safety issue will generally be found in the "findings" section of the report. You are to treat all "findings as to risk" as 'inferred' safety issues, this is due to a slight terminomology difference between TSB and other agencies. Exact Safety issues are not generally stated in TSB reports, the exception being when there is a format like "Safety issue: ...".
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
I only want recommendations that are formally made by {agency_name} in the report and which are specific to this particular accident (i.e not previous recommendations they have made). Do not include any recommendations made by other agencies or entities. I want all recommendations regardless of their response status.
"""

    metadata_prompt = """
Metadata extraction instructions:
Please extract all available metadata about the occurrence and any vehicles/vessels/people involved. This may be found in a 'Data Summary', 'Factual Information', or similar section of the report. If multiple aircraft, trains, or vessels are involved, include all of them. However you are only to include the main people and vehicles involved in the occurrence, do not include minor or peripheral people/vehicles that are only mentioned in passing.
For occurrence metadata:
- Extract occurrence_datetime.local_datetime as ISO 8601 local datetime (YYYY-MM-DDTHH:MM:SS). Time is required and must not include timezone offset.
- Extract occurrence_datetime.time_zone separately (e.g., Pacific/Auckland, NZDT, UTC+13) when available.
- Set occurrence_datetime.time_zone_source to 'explicit_in_report' when the timezone appears in the report; otherwise use 'inferred' if timezone can be inferred.
- Extract location.description as the report's plain-language geographic location.
- Extract location.standardized_location in ISO 6709 format (e.g., '-37.02972+174.97333/').
- Identify the occurrence type (e.g., collision, derailment, ditching)
- Count or identify the total number of persons involved
- Extract number of fatalities and injuries
- Describe any damage

For air accidents:
- Extract aircraft details (type, registration, make, model, engines, manufacture year, operator, flight type, persons on board totals)
- Extract pilot details (role, licence, age, experience, ratings) for each pilot if available

For rail accidents:
- Extract train details (type, number, length, weight, classification, year, operator, crew, persons involved)

For marine accidents:
- Extract vessel details (name, type, classification, dimensions, tonnage, builder, propulsion, speed, owner, registry port, crew, persons involved)

If information is not available in the report, leave the field as None/null. Do not infer or guess values.
"""

    allowed_event_types = []
    if report_mode and event_type_taxonomy_by_mode:
        allowed_event_types = event_type_taxonomy_by_mode.get(report_mode, [])

    taxonomy_constrained_occurrence_type_prompt = ""
    if metadata and allowed_event_types:
        allowed_event_types_list = "\n".join(
            [f"- {event_type}" for event_type in allowed_event_types]
        )
        taxonomy_constrained_occurrence_type_prompt = f"""

Occurrence type assignment instructions:
- Assign metadata.occurrence.occurrence_type using occurrence details in the report text.
- The value MUST be one of the allowed event types for this mode:
{allowed_event_types_list}
- Choose the most specific matching type.
"""

    prompt = f"""
You are provided with the following report text:
'''
{report_text}
'''

Based on the provided report text, please extract the following information:

{safety_issue_prompt if safety_issues else ""}

{recommendation_prompt if recommendations else ""}

{metadata_prompt if metadata else ""}

{taxonomy_constrained_occurrence_type_prompt}
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

    if metadata:
        metadata_model = _build_metadata_model_for_mode(
            report_mode=report_mode,
            event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        )
        fields["metadata"] = (
            metadata_model,
            Field(
                default_factory=ReportMetadata,
                description="Metadata extracted from the report including occurrence details and vehicle/vessel/personnel information.",
            ),
        )

    extracted_report = create_model("ExtractedReport", **fields)

    try:
        response = ai_caller.query(
            model="gpt-5-mini",
            reasoning="high",
            system=system_prompt,
            user=prompt,
            output_structure=extracted_report,
        )
    except Exception as e:
        msg = f"AI extraction failed for agency '{agency_name}'"
        raise RuntimeError(msg) from e

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
        page_key = f"page_{page}.{suffix}" if page_totals[page] > 1 else f"page_{page}"
        chunks_with_pages[page_key] = section

    return chunks_with_pages


def extract_report(
    report_row: pd.Series | dict,
    ai_extraction_config: dict,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[str]] | None = None,
) -> dict:
    """Extract safety issues, recommendations, metadata, and sections from a report row.

    Args:
        report_row: Row containing 'report_id' and 'report_text'.
        ai_extraction_config: Configuration dictionary specifying which extraction tasks
            (safety_issues, recommendations, metadata) to perform for each agency.
        event_type_taxonomy_by_mode: Allowed occurrence types per mode.

    Returns:
        dict: Extracted fields with keys 'report_id', 'safety_issues',
              'recommendations', 'metadata', and 'sections'.

    """
    report_id = report_row["report_id"]
    report_text = report_row["text"]
    agency = report_id.split("_")[0]
    report_mode = Modes.get_report_mode_from_id(report_id)

    extraction_config = ai_extraction_config[agency].copy()

    # Add ATSB reports that are older than 2008. This is because the website only tracks safety issues back to 2008
    add_atsb_safety_issues = 2008

    if agency == "ATSB":
        report_year = int(report_id.split("_")[2])
        if report_year < add_atsb_safety_issues:
            extraction_config["safety_issues"] = True

    # Read report using AI to extract safety issues, recommendations, and metadata
    extracted_data = ai_read_report(
        agency_name=agency,
        report_text=report_text,
        report_mode=report_mode,
        event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        **extraction_config,
    )

    safety_issues = getattr(extracted_data, "safety_issues", [])
    recommendations = getattr(extracted_data, "recommendations", [])
    metadata = getattr(extracted_data, "metadata", None)

    safety_issues = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in safety_issues
    ]
    recommendations = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in recommendations
    ]

    # Convert metadata to dict if available
    if metadata and hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    elif metadata and not isinstance(metadata, dict):
        metadata = None
    else:
        metadata = metadata or {}

    # Chunk the report into sections
    sections = chunk_report_into_sections(report_text)

    return {
        "report_id": report_id,
        "safety_issues": safety_issues,
        "recommendations": recommendations,
        "metadata": metadata,
        "sections": sections,
    }


def process_reports_parallel(
    parsed_reports_dc: ParsedReports,
    extracted_reports_dc: ExtractedReports,
    ai_extraction_config: dict,
    event_types_csv_path: Path = Path("data/event_types.csv"),
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Process reports in parallel, skipping those already extracted.

    This function loads input data from disk, identifies reports that need
    processing, processes new reports in parallel using ThreadPoolExecutor,
    and writes the updated extraction results back to disk.

    Args:
        parsed_reports_dc: ParsedReports instance for reading parsed reports.
        extracted_reports_dc: ExtractedReports instance for reading/writing
            extracted reports.
        ai_extraction_config: Configuration dictionary specifying which extraction tasks
            (safety_issues, recommendations, metadata) to perform for each agency.
        event_types_csv_path: Path to taxonomy CSV used for occurrence_type values.
        max_workers: Maximum number of parallel workers. Adjust
            based on your API rate limits and system resources.

    Returns:
        pd.DataFrame: Updated DataFrame containing both previously extracted
        and newly extracted reports. Columns include:
            - report_id: Unique identifier for the report
            - safety_issues: List of extracted safety issues
            - recommendations: List of extracted recommendations
            - metadata: Dictionary of extracted occurrence and mode-specific metadata
            - sections: Dictionary of report sections

    Example:
        >>> result = process_reports_parallel(
        ...     ParsedReports(Path("output")),
        ...     ExtractedReports(Path("output")),
        ...     max_workers=8,
        ... )
        >>> # Only new reports are processed, existing data preserved

    """
    reports_df = parsed_reports_dc.read()

    event_type_taxonomy_by_mode = load_event_type_taxonomy(event_types_csv_path)

    current_extracted_df = extracted_reports_dc.read_or_create()

    # Identify reports that need processing
    if len(current_extracted_df) > 0:
        already_processed = set(current_extracted_df["report_id"])
        reports_to_process = reports_df[
            ~reports_df["report_id"].isin(already_processed)
        ]
    else:
        reports_to_process = reports_df

    logger.welcome(
        "Extracting Report Data",
        {
            "Output file": str(extracted_reports_dc.path),
            "Reports to process": str(len(reports_to_process)),
            "Already processed": str(len(reports_df) - len(reports_to_process)),
        },
    )

    if len(reports_to_process) == 0:
        return current_extracted_df

    # Process reports in parallel
    results = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_report = {
            executor.submit(
                extract_report,
                row,
                ai_extraction_config,
                event_type_taxonomy_by_mode,
            ): row["report_id"]
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
                logger.warning(f"Error processing {report_id}: {exc}", exc_info=True)
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

    extracted_reports_dc.save(completed_df)

    return completed_df
