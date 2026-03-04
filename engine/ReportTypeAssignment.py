"""Report type assignment module for extracting event types from report titles.

This module provides functionality to assign event types to reports using AI,
based on the report title and available event type options.
"""

import concurrent.futures
from typing import Literal

import pandas as pd
from pydantic import Field, create_model
from tqdm import tqdm

from engine import Modes
from engine.AICaller import ai_caller
from engine.Logging import get_logger
from engine.SavedDataFrames import (
    AllEventTypes,
    ParsedReports,
    ReportEventTypes,
    ReportTitles,
)

logger = get_logger(__name__)

tqdm.pandas()


class ReportTypeAssigner:
    """Assigns event types to reports using AI analysis of report titles.

    This class manages event type assignment by reading available event types,
    processing unassigned reports in parallel, and using AI to classify reports.
    """

    def __init__(
        self,
        report_event_types_dc: ReportEventTypes,
        report_titles_dc: ReportTitles,
        parsed_reports_dc: ParsedReports,
        all_event_types_dc: AllEventTypes,
    ):
        """Initialize the ReportTypeAssigner with required dataframes and managers.

        Args:
            report_event_types_dc: ReportEventTypes instance for storing assigned types.
            report_titles_dc: ReportTitles instance containing report metadata.
            parsed_reports_dc: ParsedReports instance containing parsed reports.
            all_event_types_dc: AllEventTypes instance containing allowed event types.

        Raises:
            ValueError: If any required dataframe is missing.
        """
        self.report_event_types_dc = report_event_types_dc

        self.report_titles_df = report_titles_dc.read()

        self.parsed_reports_df = parsed_reports_dc.read()

        self.all_event_types = all_event_types_dc.read()
        self.all_event_types["mode"] = self.all_event_types["mode"].map(
            lambda x: Modes.Mode[x[0]]
        )
        self.all_event_types = self.all_event_types.set_index("mode", drop=True)

        # Build the structured output models per mode
        self._event_type_output_model_by_mode: dict[Modes.Mode, tuple[type, str]] = {}
        for mode in self.all_event_types.index.unique():
            allowed_event_types = self.all_event_types.loc[mode]["Value"].to_list()
            if not allowed_event_types:
                msg = f"No allowed event types for mode {mode}"
                raise ValueError(msg)

            allowed_type = Literal[tuple(allowed_event_types)]
            allowed_types_model = create_model(
                f"EventTypeResponse_{mode.name}",
                type=(
                    allowed_type,
                    Field(
                        description=(
                            "The event type extracted from the report title. "
                            "Must be one of the possible event types provided."
                        )
                    ),
                ),
            )
            allowed_event_types_str = "\n".join(
                [f"- {event_type}" for event_type in allowed_event_types]
            )

            self._event_type_output_model_by_mode[mode] = (
                allowed_types_model,
                allowed_event_types_str,
            )

    def assign_report_types(self):
        """Assign event types to all unassigned reports using AI analysis.

        This method processes unassigned reports in parallel, using AI to
        determine the correct event type based on report titles. Results are
        saved to disk.
        """
        if self.report_event_types_dc.exists():
            report_types_df = self.report_event_types_dc.read()
        else:
            report_types_df = self.report_event_types_dc.create_empty()

        # Get all unassigned report_types
        merged_df = report_types_df.merge(
            self.parsed_reports_df.merge(self.report_titles_df, on="report_id"),
            on=["report_id", "title"],
            how="outer",
        )
        # Remove duplicates based on report_id and title
        # TODO: Have a more robust way of handling duplicates something that uses the title from the report_titles_df
        merged_df = merged_df.drop_duplicates(subset=["report_id"])

        unassigned_df = merged_df[merged_df["type"].isna()]
        assigned_df = merged_df[~merged_df["type"].isna()]

        logger.welcome(
            "Assigning Report Event Types",
            {
                "Output file": str(self.report_event_types_dc.path),
                "Total event types": str(len(self.all_event_types)),
                "Reports to assign": str(len(unassigned_df)),
                "Already assigned": str(len(assigned_df)),
            },
        )
        if len(unassigned_df) == 0:
            return
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.process_report, index, report_id, report_title, event_type
                ): index
                for index, report_id, report_title, event_type in unassigned_df[
                    ["report_id", "title", "event_type"]
                ].itertuples()
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing Reports",
            ):
                index, assigned_event_type = future.result()
                unassigned_df.loc[index, "type"] = assigned_event_type

        combined_df = pd.concat([assigned_df, unassigned_df], ignore_index=True)
        combined_df_final = combined_df[["report_id", "type", "title"]]
        self.report_event_types_dc.save(combined_df_final)

    def process_report(self, index, report_id, report_title, event_type):
        """Process a single report and assign an event type.

        Args:
            index: Row index of the report in the dataframe.
            report_id: Unique identifier for the report.
            report_title: Title of the report.
            event_type: Suggested event type for the report.

        Returns:
            tuple: (index, assigned_event_type) where assigned_event_type
                is the determined event type or None if assignment failed.
        """
        report_mode = Modes.get_report_mode_from_id(report_id)
        if report_mode is None:
            # If we can't infer mode, fall back to the suggested event_type (or None)
            return index, event_type
        if event_type in self.all_event_types.loc[report_mode]["Value"].to_list():
            return index, event_type
        assigned_event_type = self.assign_report_type(
            report_title, report_mode, event_type
        )
        return index, assigned_event_type

    def assign_report_type(
        self, report_title: str, mode: Modes.Mode, suggested_event_type: str
    ):
        """Assign an event type to a report using AI.

        Args:
            report_title: The title of the report.
            mode: The mode/agency of the report.
            suggested_event_type: A suggested event type to aid classification.

        Returns:
            str or None: The assigned event type, or None if assignment failed.
        """
        event_type_response, allowed_event_types_str = (
            self._event_type_output_model_by_mode.get(mode)
        )  # type: ignore

        system_message = f"""
You are helping me extract and assign event types to reports based off their titles.

Can you please extract the accident event type from the report title.

Here is a list of the possible event types:
{allowed_event_types_str}

Some events types overlap so make sure to read the entire list and choose the most specific one.

Your response will be a single event type without any other words.
"""

        user_message = f"""
Here are examples of what the classification should look like:

Extract event category from "Hawker Beechcraft Corporation 1900D, ZK-EAQ cargo door opening in flight, Auckland International Airport, 9 April 2010":
Aircraft Loading

Extract event category from "Chokyo Maru No.68, ran aground, Hauraki Gulf, New Zealand, 16 April 2024":
Grounding

Extract event category from "Cessna 152 ZK-ETY and Robinson R22 ZK-HGV, mid-air collision, Paraparaumu, 17 February 2008":
Collision

Extract event category from "f.v. Pacific Challenger, crewmember missing, off Waimarama coast, 1 April 2024":
Missing assumed lost

Extract event category from "Stern trawler Pantas No.1, fatality while working cargo, No.5 berth, Island Harbour, Bluff, 22 April 2009":
Fatality

Extract event category from "{f"{suggested_event_type} - " if suggested_event_type else ""}{report_title}":
"""

        result = ai_caller.query(
            system=system_message,
            user=user_message,
            model="gpt-4",
            temp=0,
            output_structure=event_type_response,
        )

        if result is None:
            return None

        return result.type
