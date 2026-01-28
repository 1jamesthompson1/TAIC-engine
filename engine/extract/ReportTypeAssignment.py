import concurrent.futures
import os
from typing import Literal, Tuple

import pandas as pd
from pydantic import Field, create_model
from tqdm import tqdm

import engine.utils.Modes as Modes
from engine.utils.AICaller import ai_caller

tqdm.pandas()


class ReportTypeAssigner:
    def __init__(
        self,
        report_event_type_df_path,
        report_titles_df_path,
        parsed_reports_df_path,
        report_types_df_path,
    ):
        self.report_types_df_path = report_types_df_path

        if os.path.exists(report_titles_df_path):
            self.report_titles_df = pd.read_pickle(report_titles_df_path)
        else:
            raise ValueError(f"{report_titles_df_path} does not exist")

        if os.path.exists(parsed_reports_df_path):
            self.parsed_reports_df = pd.read_pickle(parsed_reports_df_path)
        else:
            raise ValueError(f"{parsed_reports_df_path} does not exist")

        if os.path.exists(report_event_type_df_path):
            self.all_event_types = pd.read_pickle(report_event_type_df_path)
            self.all_event_types["mode"] = self.all_event_types["mode"].map(
                lambda x: Modes.Mode[x[0]]
            )
            self.all_event_types.set_index("mode", inplace=True, drop=True)
        else:
            raise ValueError(f"{report_event_type_df_path} does not exist")

        # Build the structured output models per mode
        self._event_type_output_model_by_mode: dict[Modes.Mode, Tuple[type, str]] = {}
        for mode in self.all_event_types.index.unique():
            allowed_event_types = self.all_event_types.loc[mode]["Value"].to_list()
            if not allowed_event_types:
                raise ValueError(f"No allowed event types for mode {mode}")

            AllowedType = Literal[tuple(allowed_event_types)]
            AllowedTypesModel = create_model(
                f"EventTypeResponse_{mode.name}",
                type=(
                    AllowedType,
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
                AllowedTypesModel,
                allowed_event_types_str,
            )

    def assign_report_types(self):
        print("==================================================" * 2)
        print("       Assigning report event types")
        print(f"       There are {len(self.all_event_types)} possible event types")
        print(f"        output: {self.report_types_df_path}")
        if os.path.exists(self.report_types_df_path):
            report_types_df = pd.read_pickle(self.report_types_df_path)
        else:
            report_types_df = pd.DataFrame(columns=["report_id", "type", "title"])

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

        print(
            f"  There are {len(unassigned_df)} reports that need to be assigned types out of {len(merged_df)} total reports"
        )
        print("==================================================" * 2)
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
                unassigned_df.at[index, "type"] = assigned_event_type

        combined_df = pd.concat([assigned_df, unassigned_df], ignore_index=True)
        combined_df[["report_id", "type", "title"]].to_pickle(self.report_types_df_path)

    def process_report(self, index, report_id, report_title, event_type):
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
        EventTypeResponse, allowed_event_types_str = (
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
            output_structure=EventTypeResponse,
        )

        if result is None:
            return None

        return result.type
