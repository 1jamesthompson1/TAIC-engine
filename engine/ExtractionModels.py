"""Extraction models for accident investigation reports.

This module contains Pydantic models for extracting structured information from
accident investigation reports, including safety issues, recommendations, and metadata.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from engine import Modes


class SafetyIssueItem(BaseModel):
    """Represents a safety issue extracted from the report."""

    safety_issue: str = Field(
        ...,
        description="The text of the actual safety issue (e.g ignore 'safety issue -'). This should be copied verbatim from the report text. If multiple wording of the safety issue is found, then the version that is found in a 'safety issue' like section of the report is considered the most authoritative version of the safety issue.",
    )
    safety_issue_id: str | None = Field(
        default=None,
        description="The unique identifier for the safety issue if it exists in the report (sometimes called 'id', 'number'). This is usually an id that will be used in some external database. The section number is not a safety issue id. If none is given then return None.",
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
        description="The text of the recommendation made by the agency in the report. It usually starts with something like 'On ...', which should be included. Copy the recommendation verbatim.",
    )
    recommendation_id: str | None = Field(
        default=None,
        description="The unique identifier for the recommendation if it exists in the report (sometimes called 'id', 'number'). If none is given then return None.",
    )
    recipient: str | None = Field(
        default=None,
        description="The recipient of the recommendation. I.e who the recommendation was addressed to. Avoid using acronyms or abbreviations and prefer to use the fully qualified name (i.e the complete name of company that would be most unambiguous outside the context of this particular report, this might require you using information from other parts of the report). If no recipient is given then return None.",
    )
    recommendation_context: str | None = Field(
        default=None,
        description="The context or background information related to the recommendation, if available. This should include the safety issue or issues that the recommendation is addressing, the entire safety action/response of the recommendation recipient and the final justification from the investiagtion agency as to why their safety action was inadequate. For each of these three elements only include what is actually present in the report. If no context is given around the recommendation then return None. It is important that whatever is copied is copied verbatim and not invented or summarised although removing markdown formatting and cleaing up PDF text extraction artefacts is acceptable. Information on the current status of the recommendation is not to be included.",
    )
    made: str | None = Field(
        default=None,
        description="The date when the recommendation was made, if available in ISO 8601 format. If not available, set to None.",
    )


class OccurrenceDateTime(BaseModel):
    """Represents occurrence datetime with timezone as separate fields."""

    local_datetime: str = Field(
        description="Local occurrence datetime as ISO 8601 without timezone (YYYY-MM-DDTHH:MM). Time is required and should come from the report text, not be invented. It should be the time that the accident sequence begins (i.e when boat starts to sink). Use the most accurate time found in the report which is normally in the history/narrative/factual information section.",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$",
    )
    time_zone: str = Field(
        description=(
            "Canonical UTC offset string for the occurrence local time "
            "in the form UTC+HH:MM or UTC-HH:MM "
            "(e.g. 'UTC+13:00', 'UTC-06:00', 'UTC+09:30'). "
            "This should be the time zone that is used throughout the report."
        ),
        pattern=r"^UTC[+-]\d{2}:\d{2}$",
    )
    time_zone_source: Literal["explicit_in_report", "inferred"] | None = Field(
        default=None,
        description="Set to 'explicit_in_report' when timezone wording appears in the report. Set to 'inferred' only when timezone is derived from context (for example location or standard local time references).",
    )


class OccurrenceLocation(BaseModel):
    """Represents occurrence location in both descriptive and parseable formats."""

    description: str = Field(
        description="Raw human-readable location extracted from the report wording (do not normalize this field). Keep it faithful to how the report describes the location. Best to use full names where possible (avoid airport codes, abbreviations etc).",
    )
    standardized_location: str | None = Field(
        default=None,
        description="""Normalized location string in exactly 4 comma-separated items: 'exact location, city/town/port, region/state, country' (country is stored separately in the country field). Use this only when a location can be stated from report context; if a part is unknown, use 'unknown' in that part. If report provides a location description that can be coerced into a 4 part strucutre then do so (i.e by adding in country). Avoid quantifiers like "region", "town" etc unless they are explicitly stated in the report.
        If dealing with boats it should be the port than the location of the port etc etc. If a train is on a rail ferry than the location should be "On board the [name of ferry]" then where the ferry is etc (at terminal/in harbour/in strait etc).
        For airports use the name of the airport and do not include the code. For aircraft accidents if a plane is mid flight then the location should be 'en route' (not 'above and near' or 'in flight' etc) followed by the next three parts which corresepond to the location at which they are above.""",
        pattern=r"^[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+$",
    )


class OccurrenceMetadata(BaseModel):
    """Represents common metadata extracted from an accident occurrence report (all modes)."""

    occurrence_datetime: OccurrenceDateTime = Field(
        description="Structured occurrence datetime with local time, UTC offset, and timezone source as stated in the report."
    )
    location: OccurrenceLocation = Field(
        description="Structured occurrence location shared across all modes: raw extracted description, normalized 4-part standardized location, and separate country code.",
    )

    occurrence_type: str | None = Field(
        default=None,
        # Intentionally blank description to be overridden in mode-specific metadata models with the specific taxonomy for that mode.
        description="",
    )
    total_persons_involved: int | None = Field(
        default=None,
        description="The total number of persons involved in the occurrence. This should be the total number of people that are involved and/or could of been harmed in the occurrence regardless of if they are on board a vehicle or not. Only use None in the situation where the report does not provide any information at all.",
    )
    fatalities: int = Field(
        description="The number of fatalities resulting from the occurrence. Use 0 to indicate no fatalities. People who are considered missing (i.e not found by time of publication) should be counted as fatalities.",
    )
    injuries: int = Field(
        description="The number of persons injured (non-fatal) in the occurrence. Use 0 to indicate no injuries. Do not count fatalities here - they go in the separate 'fatalities' field.",
    )

    damage_description: Literal["nil"] | str = Field(
        description="Brief summary of damage to equipment, property, or environment from the report text. The summary should include a few word overview of the damage (e.g. 'lost', 'destroyed', 'substantial damage', 'minor damage' ) followed by a colon and then details separated by semicolons. For example, 'substantial damage: left wing damaged; engine detached.'. If there are multiple items of damage, separate them with semicolons. The overall description should be concise and informative and optimised so that future retrieval is efficient (e.g., using keywords or tags). If there is no damage, use literal 'nil'.",
        pattern=r"((.+):(.+;?)\.?)|(nil)",
    )
    who_may_benefit: str | None = Field(
        default=None,
        description="Some reports include a section, sentence or phrase that explicitly states who may benefit from the lessons of the report. If this is explicitly stated in the report then include it here verbatim. If there is no explicit section/sentence of the sort 'who may benefit' then leave it as None. This should be a text field and not a list of categories (i.e do not attempt to categorise this information into predefined buckets, just copy the text as is from the report if it is present).",
    )


class PilotMetadata(BaseModel):
    """Represents pilot-specific metadata from an air accident report."""

    role: (
        Literal[
            "Captain",
            "First Officer",
            "Second Officer",
            "Student Pilot",
            "Instructor",
            "Sole Pilot",
            "Other",
        ]
        | None
    ) = Field(
        default=None,
        description="Pilot's role or position (e.g., captain, instructor, second officer etc). Use Sole pilot for situations where there is no explicit rank yet there is only a single pilot (common in light aircraft accidents). Student pilot is for situations where there is an isntructor and other pilot who is flying 'under instruction' (even if both are pilots, any situation involving an instructor has a student). Use 'other' when the report states a role that does not fit the allowed literals.",
    )
    responsibility: Literal["Pilot flying", "Pilot monitoring"] | None = Field(
        default=None,
        description="Pilot responsibility for the occurrence phase when reported. You are allowed to infer this as it is not always explicitly stated.",
    )
    licence: (
        Literal[
            "Air Transport Pilot Licence (Aeroplane)",
            "Air Transport Pilot Licence (Helicopter)",
            "Commercial Pilot Licence (Aeroplane)",
            "Commercial Pilot Licence (Helicopter)",
            "Private Pilot Licence (Aeroplane)",
            "Private Pilot Licence (Helicopter)",
            "Student Pilot",
            "Balloon Pilot Licence",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description="Pilot licence category (if they hold multiple use the license that is the most relevant to the occurrence). Use 'other' if a licence is provided but does not match listed categories.",
    )
    age: int | None = Field(
        default=None,
        description="Pilot's age at time of occurrence.",
    )
    total_flying_experience: int | None = Field(
        default=None,
        description="Total flying experience (e.g., 10000, 15) as hours. Round to the nearest whole number (using standard rounding rules) and do not include any text (e.g. 'hours'). IF an approximate is given then just use that.",
    )
    experience_on_type: int | None = Field(
        default=None,
        description="Total of all time Flying experience on the specific aircraft type as hours. Round to the nearest whole number (using standard rounding rules) and do not include any text (e.g. 'hours'). IF an approximate is given then just use that. Avoid using the flight hours in just the last n days as this is not the total of all time experience on the type.",
    )


class AircraftMetadata(BaseModel):
    """Represents aircraft-specific metadata from an air accident report."""

    aircraft_type: (
        Literal[
            "Aeroplane",
            "Helicopter",
            "Glider",
            "Balloon",
            "Gyroplane",
            "Drone/UAV/RPAS",
            "Other",
        ]
        | None
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
        description="Aircraft manufacturer/make. Just the name of the manufacturer, not the model (e.g., Boeing, Airbus, Cessna). Remove the words after the company name from the manufacturer name if it is present (e.g. 'Bombardier' not 'Bombardier Company').",
    )
    model: str | None = Field(
        default=None,
        description="Aircraft model. Just the model of the aircraft, not the manufacturer (e.g., 737-800, A320, 172). If the model commonly includs a name (e.g 'P-51 Mustang') then include the name as part of the model, but do not include the manufacturer (e.g. 'Q400' not 'Bombardier Q400').",
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

    flight_type: (
        Literal[
            "scheduled service",
            "charter",
            "cargo",
            "private",
            "training",
            "aerial work",
            "emergency services",
            "ferry/positioning",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Type of flight. All medevac/air ambulance/rescue flights should be mapped to 'emergency services'. Map agricultural/survey/patrol/sling-load/firefighting to 'aerial work'. Ferry/positioning used to rrepresent any situation where the aircraft is not presently carrrying paying passengers or goods or conducting operations (i.e flying to airport to pick up passengers). All air taxi flights should be mapped to 'charter'."
        ),
    )
    persons_on_board_total: int = Field(
        description="Total number of persons on board the aircraft.",
    )
    persons_on_board_crew: int = Field(
        description="Number of operational crew members on board (cabin crew plus flight crew/pilots). Other 'crew' such as other pilots that are not part of the operating crew should be counted as passengers (e.g. a second crew that is being transported but not operating the flight).",
    )
    persons_on_board_passengers: int = Field(
        description="Number of passengers on board.",
    )

    damage: Literal["destroyed", "substantial damage", "minor damage", "nil"] | None = (
        Field(
            default=None,
            description="Damage severity classification for the aircraft from the report.",
        )
    )

    pilots: list[PilotMetadata] = Field(
        default_factory=list,
        description="List of pilots associated with the aircraft. Always filled out with pilot metadata even if the information is not available in the report (in which case the fields will be None).",
    )


class TrainMetadata(BaseModel):
    """Represents train-specific metadata from a rail accident report."""

    train_type: (
        Literal[
            "passenger",
            "freight",
            "work train",
            "maintenance vehicle",
            "hi-rail vehicle",
            "shunt",
            "locomotive only",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Broad type of rail movement or rail vehicle. Use 'maintenance "
            "vehicle' for tampers/regulators/ballast cleaners and similar track "
            "machines. Use 'shunt' for yard/depot/ferry-terminal shunting "
            "movements. Use 'work train' for engineering/construction trains "
            "hauling work materials or equipment."
        ),
    )
    train_number: str | None = Field(
        default=None,
        description="Train number or identifier. Should be a report-specific identifier for the train involved in the occurrence as stated in the report (e.g., 'Train 1234'). If both a id and a name is known then use the format 'ID - Name'. This should not be invented if not explicitly stated in the report.",
    )
    length: float | None = Field(
        default=None,
        description=(
            "Length of the train in meters as a numeric value only "
            "(for example, 130.0). Do not include unit text. Carefully search the report for any information about the length of the train. If the report does not provide any information about the length of the train then leave this field as None."
        ),
    )
    weight: float | None = Field(
        default=None,
        description=(
            "Weight of the train in metric tons as a numeric value only "
            "(for example, 311.0). Do not include unit text. Most tons listed will be metric tons. Treat all TSB reports weights as US short tons and convert them to metric tons (1 short ton = 0.907184 metric tons), unless explicitly stated otherwise. If the report does not specify the type of ton then assume it is metric tons and do not convert."
        ),
    )
    classification: (
        Literal[
            "commuter",
            "metro",
            "regional",
            "intercity",
            "high-speed",
            "heavy-haul",
            "unit",
            "manifest",
            "intermodal",
            "yard/shunting",
            "inspection",
            "maintenance",
            "test/commissioning",
            "other",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Service/operational class for the movement. Use the closest "
            "canonical value and avoid introducing new labels. Use 'manifest' "
            "for mixed/general freight consists and 'intermodal' for container "
            "or trailer-on-flatcar operations."
        ),
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
        description="Number of operating crew members (all forms of crew).",
    )


class VesselMetadata(BaseModel):
    """Represents vessel-specific metadata from a marine accident report."""

    vessel_name: str | None = Field(
        default=None,
        description="Name of the vessel.",
    )
    vessel_type: (
        Literal[
            "container",
            "bulk carrier",
            "tanker",
            "passenger",
            "ferry",
            "taxi",
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
        description="Type of vessel. This should be what the vessel is built for, not what it is being used for at the time of the occurrence. Passenger is a ship which is designed to accomodate passengers (e.g cruise ship, yacht, ferry etc). Taxi is for vessels that are designed to transport people but only over short distances (e.g water taxi, harbour shuttle etc). Recreational is for vessels that are designed for leisure and not for transportation (e.g fishing boats, sailing boats, motorboats etc).",
    )

    classification: (
        Literal[
            "DNV",
            "Lloyd's Register",
            "ABS",
            "Bureau Veritas",
            "ClassNK",
            "RINA",
            "CCS",
            "KR",
            "IRS",
            "unclassed",
            "other",
        ]
        | None
    ) = Field(
        description=(
            "Marine classification society. Use one of the allowed literals  or 'unclassed' when it is not classed (i.e local vessels), or 'other' if a named society is not listed. Use None only in the situation where one would expect a class yet it is not listed (i.e a large internationally operating vessel that one would expect to be classed but the report does not provide any information about the class)."
        ),
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
    year_built: int | None = Field(
        default=None,
        description="Year the vessel was built.",
    )

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
        description="Primary propulsion type. Wind is to be used for sailing vessels, human is to be used for manually rowed vessels and jet is to be used for jet boats. For all vessels with sails use 'wind' as the propulsion regardless of if they have an auxiliary engine or not. For vessels with multiple propulsion types use the primary propulsion type (e.g a sail boat with a small auxiliary engine should be classified as 'wind').",
    )
    total_power: float | None = Field(
        default=None,
        description="Total power of the vessel's propulsion system in kW. This may require some simple calculations based on the information in the report and standard conversion factors.",
    )
    service_speed: float | None = Field(
        default=None,
        description="Service speed of the vessel (in knots).",
    )
    owner_operator: str | None = Field(
        default=None,
        description="Owner or operator of the vessel.",
    )
    port_of_registry: str | None = Field(
        default=None,
        description="Port where the vessel is registered. This can just be the country that the boat is from if the port is not specified (i.e in case of local vessels).",
    )


def _build_metadata_model_for_mode(
    report_mode: Modes.Mode,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[dict[str, str]]],
) -> type[BaseModel]:
    """Build a mode-specific metadata model with mode-relevant vehicle lists and occurrence_type.

    Args:
        report_mode: The report mode (air, rail, or marine).
        event_type_taxonomy_by_mode: Taxonomy of allowed occurrence types by mode.

    Returns:
        type[BaseModel]: A metadata model for the supplied mode, with:
        - Constrained occurrence_type to mode-allowed values only.
        - Only relevant vehicle lists (aircraft for air, trains for rail, vessels for marine).

    Raises:
        ValueError: When report_mode is None, unknown, or taxonomy is unavailable.
    """
    allowed_event_type_entries = event_type_taxonomy_by_mode.get(report_mode)
    if not allowed_event_type_entries:
        msg = f"Unknown report mode: {report_mode}. Allowed modes: {list(event_type_taxonomy_by_mode.keys())}"
        raise ValueError(msg)

    allowed_event_types = [
        entry["event_type"]
        for entry in allowed_event_type_entries
        if entry.get("event_type")
    ]
    if not allowed_event_types:
        msg = f"No event types configured for mode: {report_mode}"
        raise ValueError(msg)

    allowed_type: Any = Literal[tuple(allowed_event_types)]
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

    mode_fields: dict[str, tuple[Any, Any]] = {
        "occurrence": (
            occurrence_metadata_model,
            Field(description="Common occurrence metadata (all modes)."),
        ),
    }

    if report_mode == Modes.Mode.a:
        mode_fields["aircraft"] = (
            list[AircraftMetadata],
            Field(
                default_factory=list,
                description="List of aircraft involved in the accident. For reports that are about ATC occurrences (i.e where ATC agent made mistakes and the report is investigating why), none of the aircraft should be included.",
            ),
        )
    elif report_mode == Modes.Mode.r:
        mode_fields["trains"] = (
            list[TrainMetadata],
            Field(
                default_factory=list,
                description="List of trains involved in the accident. If multiple rolling stock are attached togather than they are a single train. Particularly for shunting accidents usually these should be treated as a single train with multiple vehicles rather than multiple trains.",
            ),
        )
    elif report_mode == Modes.Mode.m:
        mode_fields["vessels"] = (
            list[VesselMetadata],
            Field(
                default_factory=list,
                description="List of vessels involved in the accident.",
            ),
        )

    mode_fields_for_create: Any = mode_fields
    return create_model(
        f"ReportMetadata_{report_mode.name}",
        **mode_fields_for_create,
    )


def build_extraction_output_model(
    safety_issues_enabled: bool,
    recommendations_enabled: bool,
    metadata_enabled: bool,
    report_mode: Modes.Mode | None = None,
    event_type_taxonomy_by_mode: dict[Modes.Mode, list[dict[str, str]]] | None = None,
) -> type[BaseModel]:
    """Build the output Pydantic model based on enabled extractions.

    Returns:
        type[BaseModel]: The dynamically constructed Pydantic model for the extracted report.

    Raises:
        ValueError: If metadata extraction is enabled but report_mode or event_type_taxonomy_by_mode is not provided.
    """
    fields: dict[str, tuple[Any, Any]] = {}

    if safety_issues_enabled:
        fields["safety_issues"] = (
            list[SafetyIssueItem],
            Field(
                default_factory=list,
                description="A list of all safety issues identified in the report.",
            ),
        )

    if recommendations_enabled:
        fields["recommendations"] = (
            list[RecommendationItem],
            Field(
                default_factory=list,
                description="A list of all recommendations made in the report.",
            ),
        )

    if metadata_enabled:
        if report_mode is None or event_type_taxonomy_by_mode is None:
            msg = "Metadata extraction requires report_mode and event_type_taxonomy_by_mode to be provided."
            raise ValueError(msg)
        fields["metadata"] = (
            _build_metadata_model_for_mode(report_mode, event_type_taxonomy_by_mode),
            Field(
                description="Type or classification of the occurrence (e.g., collision, derailment, ditching). You should use the most precise category available in the supplied taxonomy.",
            ),
        )

    return create_model("ExtractedReport", **fields)
