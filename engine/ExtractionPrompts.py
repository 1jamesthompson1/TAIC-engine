"""Prompt builder for structured extraction from accident investigation reports."""

import warnings

from engine import Modes


class PromptBuilder:
    """Builder for constructing extraction prompts with agency-specific instructions."""

    def __init__(  # noqa: PLR0913
        self,
        agency_name: str,
        report_id: str | None = None,
        agency_id: str | None = None,
        *,
        safety_issues: bool = False,
        recommendations: bool = False,
        metadata: bool = False,
    ):
        """Initialize the PromptBuilder.

        Args:
            agency_name: The name of the investigation agency (e.g., 'TAIC', 'TSB', 'ATSB').
            report_id: The identifier of the report being processed.
            agency_id: Agency-native occurrence identifier (for example, ATSB short ID).
            safety_issues: Whether to include safety issue extraction instructions.
            recommendations: Whether to include recommendation extraction instructions.
            metadata: Whether to include metadata extraction instructions.
        """
        self.agency_name = agency_name
        self.report_id = report_id
        self.agency_id = agency_id
        self._safety_issues_enabled = safety_issues
        self._recommendations_enabled = recommendations
        self._metadata_enabled = metadata

        if self.agency_id is None:
            warning_msg = "The Agency ID is not provided. This may make it difficult to determine which information in the report text is relevant to the specific occurrence, especially for agencies like ATSB where the report text may contain information about multiple occurrences. It is recommended to provide the agency ID if possible."
            warnings.warn(warning_msg, stacklevel=2)

    def build_system_prompt(self) -> str:
        """Build the system prompt with agency-specific instructions.

        Returns:
            The system prompt string.
        """
        base = """You are a highly skilled AI specialized in extracting structured information from safety investigation reports. You are helping the New Zealand Transport Accident Investigation Commission build a structured database. Your task is to read the provided report text and extract specific information based on the given instructions. These safety investigation reports are all published publically on their websites by government agencies after they have completed a no-blame investigation into a transport accident. The goal of these investigations are to identify problems that reduce safety (safety issues) and to make recommendations to address those safety issues.Your goal is to help extract the data from these reports in a structured way to help safety researchers and practitioners learn from past accidents and improve safety in the future.

There are techincal definitions you should understand:
Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would increase the likelihood of an occurrence, and/or the severity of any adverse consequences associated with the occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the potential to adversely affect the safety of future operations, and
• is characteristic of an organisation, a system, or an operational environment at a specific point in time.

Safety Issues are derived from safety factors classified either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may cover a single safety issue, or two or more related safety issues.

Recommendations - Formal suggestions made by the investigation agency to address identified safety issues. Recommendations are directed towards specific entities, such as regulatory bodies, industry organizations, or operators, with the aim of improving safety and preventing future occurrences."""

        if self.agency_name == "ATSB":
            base += """

Note that some reports will actually have text that is from a short report bulletin that contains the information for many differnt short investigations. You should take great care in only extracting information that is relevant to the specific occurrence in question. You should use the report id and occurrence details to determine which information is relevant to the specific occurrence. If you are unsure about which information is relevant to the specific occurrence then it is better to not extract that information."""

        return (
            base
            + """We really appreciate you helping us with the best of your abilities."""
        )

    def build_user_prompt(
        self,
        report_mode: Modes.Mode | None,
        report_text: str,
        event_type_taxonomy: list[dict[str, str]] | None = None,
    ) -> str:
        """Build the user prompt with extraction instructions.

        Args:
            report_mode: The transport mode classification for the report.
            report_text: The full text of the report.
            event_type_taxonomy: Optional taxonomy entries for occurrence types.

        Returns:
            The user prompt string.
        """
        mode_label = report_mode.value if report_mode else "unknown"
        parts = [
            f"""You are processing report ID: {self.report_id or "Unknown"}
which is a report of mode {mode_label}
Agency occurrence ID: {self.agency_id or "Unknown"}
From investigation agency: {self.agency_name}
Based on the provided report text, please extract the following information:
"""
        ]

        if self._safety_issues_enabled:
            parts.append(self._safety_issues_prompt())

        if self._recommendations_enabled:
            parts.append(self._recommendations_prompt())

        if self._metadata_enabled:
            parts.append(self._metadata_prompt(event_type_taxonomy))

        parts.append(
            f"""You are provided with the following report text:\n"
'''
{report_text}
'''

Thank you for your help in extracting this information.
"""
        )

        return "\n\n".join(parts)

    def _safety_issues_prompt(self) -> str:
        """Builds the safety issues extraction instructions based on the agency name.

        For TAIC it is trying to extract exact safety issues.
        For TSB it is going to treat all findings as to risk as safety issues
        For ATSB it is going to infer safety issues from the report test.

        Returns:
            The safety issues extraction instructions string.
        """
        prompt = """Safety issue extraction instructions:
Please only respond with safety issues that are quite clearly stated ("exact" safety issues) or implied ("inferred" safety issues) in the report. Each report will only contain one type of safety issue. If exact safety issues are stated then only respond with those. If no exact safety issues are stated then respond with inferred safety issues. Exact safety issues are ones that explicitly state safety issues, for example "Safety issue: ...". For exact safety issues if there are multippoe that are simliar yet have different ids then you need to include them all.

If the report conatins a phrase like "No safety issues were identified" or "No safety issues were found" then respond with an empty list of safety issues.

For reference here are some examples of some safety issues.

"The maintenance provider did not have a robust quality assurance process to ensure maintenance activities were being completed and documented in accordance with their approved maintenance procedures. This increased the risk of maintenance errors not being identified and rectified"

"The safe navigation bridge procedures described in the operator's safety management system were not adequately implemented by the bridge team. As a result, the risk mitigations inherent to those safe navigation procedures were absent, which allowed minor hazards to escalate and have serious consequences"

For inferred safety issues, only include distinct, high-level safety issues that identify a systemic safety factor with the potential to adversely affect future operations. Do not list ordinary factual findings, observations, or individual events as separate safety issues. Group closely related findings into one safety issue. For ATSB reports, base inferred safety issues primarily on the report's findings or safety issues sections, not on the narrative or factual information sections.
"""

        if self.agency_name == "TAIC":
            prompt += """

An exact safety issue will start with something like 'safety issue: ...' and will generaly go until the end of the "paragraph" (i.e until it reaches a line that breaks earlier)."""
        elif self.agency_name == "TSB":
            prompt += """

For TSB reports, only extract the numbered items in the "Findings as to risk" section as inferred safety issues. Do not extract items from "Findings as to causes and contributing factors", factual information, analysis, or safety action sections. The number of inferred safety issues should equal the number of numbered findings as to risk. Exact safety issues are not generally stated in TSB reports, the exception being when there is a format like "Safety issue: ..."."""

        return prompt

    def _recommendations_prompt(self) -> str:
        return f"""Recommendation extraction instructions:
Please extract only formal safety recommendations made in the report. Copy the recommendation verbatim. If there is a unique identifier for the recommendation (e.g Recommendation 1, Rec-01, etc) then include that as the recommendation_id. If there is any context or background information related to the recommendation, include that as recommendation_context (this is sometimes present like a paragraph just before or just after the stated recommendations).
Do not include safety advisory notices, safety issue descriptions, observations, or other advisory text. For ATSB reports, a safety recommendation is usually labelled "Safety recommendation" (for example SR-xxx); a safety advisory notice (SAN-xxx) is not a recommendation.
For the recipient, preserve directorates, branches, and parenthetical qualifiers. If the report uses a shortened name or role title, expand it using the full name or organisation defined elsewhere in the report. For example, expand "RailCorp" to "Rail Corporation New South Wales" when that is the defined name, and include the organisation for role titles such as "Director of Civil Aviation, Civil Aviation Authority of New Zealand" or "Chief Executive Officer of Airways Corporation of New Zealand Limited". Do not add country prefixes that are not part of the defined name, and do not substitute a different parent organisation.
I only want recommendations that are formally made by {self.agency_name} in the report and which are specific to this particular accident (i.e not previous recommendations they have made in relation to other accidents. However recommendations specific to this accident that were made prior to publication should be included). Do not include any recommendations made by other agencies or entities. I want all recommendations regardless of their response status."""

    @staticmethod
    def _metadata_prompt(event_type_taxonomy: list[dict[str, str]] | None) -> str:
        prompt = """Metadata extraction instructions:
Extract metadata according to the schema and each field description.
Include only the main occurrence participants (not minor/peripheral mentions). For marine occurrences, include the vessel(s) involved in the occurrence and any vessel directly participating in the operation at the time, including tugs released before the accident, barges, towed vessels, and support vessels, even if fewer details are available. Do not include vessels that only responded after the occurrence or that are mentioned as background traffic. For each vessel, use only details explicitly associated with that vessel in the report. Do not copy a detail from another vessel (for example, a ship's classification society) to a tug or barge. For vessel classification, only use information explicitly stated for that vessel; otherwise return None. For owner_operator, use an operator explicitly named in the report, including a company described as operating the vessel or fleet; do not infer it from general knowledge. If the report states that one company operates a group of vessels (for example, a tug fleet), use that operator for each vessel in the group unless the report assigns a different operator.
For occurrence_datetime.time_zone, use the time zone explicitly stated in the report. If the occurrence narrative or summary gives a specific named time (for example, "Atlantic Daylight Time") and a generic footnote says "all times are..." with a different zone, prefer the specific named time in the occurrence narrative or summary. If a named time zone is given, convert it to the correct UTC offset for the occurrence date, including daylight saving. For example, Atlantic Daylight Time is UTC-03:00 and Eastern Daylight Time is UTC-04:00. Do not substitute a nearby province or state time zone.
IMPORTANT:
If a data summary is provided in the report that is to be used to decide on which vehicles/vessels should be included or not included for metadata extraction. Not all data may be found in the table and so you should use the table as a guide but also use the rest of the report to find any missing information.
You are free to make unit conversions as needed. Some fields require more inference than others and will be stated in descriptions. For most of the fields reasonable inference is allowed.

Vessel propulsion extraction instructions:
- For vessels with sails (sail ships, sailing vessels, etc.), ALWAYS use 'wind' as the propulsion type, regardless of whether they have auxiliary engines (diesel, etc.)
- For vessels with multiple propulsion types, extract only the PRIMARY propulsion type
- Example: A sail training ship with auxiliary diesel engines should be classified as propulsion='wind', not 'diesel'
- The primary propulsion for a sailing vessel is wind/sail, even if auxiliary engines are mentioned in the report

Occurrence type hierarchy rule:
- If the occurrence ends in a consequence/outcome event (e.g., collision with terrain, forced landing) caused by a contributing factor (e.g., fuel starvation, loss of control), assign the consequence/outcome event type, NOT the contributing factor.
- Do NOT treat a recovery or response action (e.g., go-around, missed approach, emergency/precautionary descent) as the occurrence type when it was a response to an earlier event. Use the earlier event type instead.
- Examples:
  * Fuel starvation that causes a forced landing → "Forced / Precautionary landing", NOT "Fuel related"
  * Loss of control that causes a collision with terrain → "Terrain collisions", NOT "Aircraft control"
  * Loss of separation followed by a go-around → "Aircraft separation", NOT "Missed approach / Go-around"
  * Cabin depressurisation followed by an emergency descent → "Systems", NOT "Emergency / Precautionary descent"
"""
        if event_type_taxonomy:
            event_types_list = "\n".join(
                f"- {entry['event_type']}: {entry['description']}"
                if entry.get("description")
                else f"- {entry['event_type']}"
                for entry in event_type_taxonomy
                if entry.get("event_type")
            )
            prompt += f"""

Occurrence type assignment instructions:
- Assign metadata.occurrence.occurrence_type using occurrence details in the report text.
- The value MUST be one of the allowed event types for this mode:
{event_types_list}
- Choose the most specific matching type.

Aviation ATC reports:
For reports that are occurrences involving air traffic control giving bad/incorrect instructions to pilots, you should NOT include aircraft metadata for the aircraft involved in the occurrence, as the main occurrence participants are really the air traffic controllers and the ATC system, not the aircraft.

"""

        return prompt
