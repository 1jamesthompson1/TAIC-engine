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
        base = """You are a highly skilled AI specialized in extracting structured information from safety investigation reports. Your task is to read the provided report text and extract specific information based on the given instructions. These safety investigation reports are all published publically by government agencies after they have completed a no-blame investigation into a transport accident.

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

        return base

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
            f"You are processing report ID: {self.report_id or 'Unknown'} which is a report of mode {mode_label}\n"
            f"Agency occurrence ID: {self.agency_id or 'Unknown'}\n"
            f"From investigation agency: {self.agency_name}\n\n"
            f"You are provided with the following report text:\n"
            f"'''\n{report_text}\n'''\n\n"
            f"Based on the provided report text, please extract the following information:\n"
        ]

        if self._safety_issues_enabled:
            parts.append(self._safety_issues_prompt())

        if self._recommendations_enabled:
            parts.append(self._recommendations_prompt())

        if self._metadata_enabled:
            parts.append(self._metadata_prompt(event_type_taxonomy))

        return "\n\n".join(parts)

    def _safety_issues_prompt(self) -> str:
        prompt = """Safety issue extraction instructions:
Please only respond with safety issues that are quite clearly stated ("exact" safety issues) or implied ("inferred" safety issues) in the report. Each report will only contain one type of safety issue. If exact safety issues are stated then only respond with those. If no exact safety issues are stated then respond with inferred safety issues.

If the report conatins a phrase like "No safety issues were identified" or "No safety issues were found" then respond with an empty list of safety issues."""

        if self.agency_name == "TAIC":
            prompt += """

An exact safety issue will start with something like 'safety issue: ...' and will generaly go until the end of the "paragraph" (i.e until it reaches a line that breaks earlier)."""
        elif self.agency_name == "TSB":
            prompt += """

An inferred safety issue will generally be found in the "findings" section of the report. You are to treat all "findings as to risk" as 'inferred' safety issues, this is due to a slight terminomology difference between TSB and other agencies. Exact Safety issues are not generally stated in TSB reports, the exception being when there is a format like "Safety issue: ..."."""

        return prompt

    def _recommendations_prompt(self) -> str:
        return f"""Recommendation extraction instructions:
Please extract all recommendations made in the report. Copy the recommendation verbatim. If there is a unique identifier for the recommendation (e.g Recommendation 1, Rec-01, etc) then include that as the recommendation_id. If there is any context or background information related to the recommendation, include that as recommendation_context (this is sometimes present like a paragraph just before or just after the stated recommendations).
I only want recommendations that are formally made by {self.agency_name} in the report and which are specific to this particular accident (i.e not previous recommendations they have made). Do not include any recommendations made by other agencies or entities. I want all recommendations regardless of their response status."""

    @staticmethod
    def _metadata_prompt(event_type_taxonomy: list[dict[str, str]] | None) -> str:
        prompt = """Metadata extraction instructions:
Extract metadata according to the schema and each field description.
Include only the main occurrence participants (not minor/peripheral mentions).
IMPORTANT:
If a data summary is provided in the report that is to be used to decide on which vehicles/vessels should be included or not included for metadata extraction. Not all data may be found in the table and so you should use the table as a guide but also use the rest of the report to find any missing information.
You are free to make unit conversions as needed. Some fields require more inference than others and will be stated in descriptions. For most of the fields reasonable inference is allowed.

Vessel propulsion extraction instructions:
- For vessels with sails (sail ships, sailing vessels, etc.), ALWAYS use 'wind' as the propulsion type, regardless of whether they have auxiliary engines (diesel, etc.)
- For vessels with multiple propulsion types, extract only the PRIMARY propulsion type
- Example: A sail training ship with auxiliary diesel engines should be classified as propulsion='wind', not 'diesel'
- The primary propulsion for a sailing vessel is wind/sail, even if auxiliary engines are mentioned in the report"""

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
- Choose the most specific matching type."""

        return prompt
