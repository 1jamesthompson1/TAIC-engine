import concurrent.futures
import os
from enum import Enum
from functools import lru_cache
from typing import Optional

import pandas as pd
import regex as re
from pydantic import BaseModel, Field
from tqdm import tqdm

from engine.utils.AICaller import ai_caller


@lru_cache(maxsize=20000)
def get_regex(regex_string, flags=0):
    return re.compile(regex_string, flags)


# Structured output classes


class roman_numerals(str, Enum):
    i = "i"
    ii = "ii"
    iii = "iii"
    iv = "iv"
    v = "v"
    vi = "vi"
    vii = "vii"
    viii = "viii"
    ix = "ix"
    x = "x"


class PageToRead(BaseModel):
    start_page: int | roman_numerals | None = Field(
        default=None, description="The starting page number to read (Inclusive)."
    )
    end_page: int | roman_numerals | None = Field(
        default=None, description="The ending page number to read (Exclusive)."
    )


class PagesToReadOutput(BaseModel):
    pages: list[PageToRead] | None = Field(
        default_factory=list,
        description="List of page ranges to read. If no relevant pages found, return an empty list.",
    )

    def to_tuples(self):
        result = []
        for page in self.pages:
            if page.start_page is not None and page.end_page is not None:
                result.append((page.start_page, page.end_page))
        return result


class ReportExtractor:
    def __init__(self, report_text, report_id, headers="Empty"):
        self.report_text = report_text
        self.report_id = report_id
        self.headers = (
            self.create_hierarchy(headers.assign(Page=headers["Page"].replace("", "0")))
            if isinstance(headers, pd.DataFrame)
            else headers
        )

        self.max_important_text_len = (
            128_000 * 3
        )  # This account for the fact that a token is about 4 characters and at the moment the LLMs have limits of about 128,000 tokens

    def extract_important_text(self, table_of_contents):
        """
        This function extracts the important text for a particular extractor

        What is determined as important text is decided by the extractors `extract_pages_to_read()` method
        """
        # In cases where the content section doesn't exist or is not adequate we will try to read the entire report.
        pages = list(
            re.finditer(
                r"<< Page (\d+|[xvi]+) >>",
                self.report_text,
                re.MULTILINE | re.IGNORECASE,
            )
        )
        default_response = (
            (self.report_text, f"{pages[0].group(1)},{pages[-1].group(1)}")
            if len(self.report_text) < self.max_important_text_len
            else (None, None)
        )

        pages_to_read = self.extract_pages_to_read(table_of_contents)

        if pages_to_read is None:
            return default_response

        text = "\n".join(
            map(
                lambda x: x if x is not None else "",
                [
                    self.extract_text_between_page_numbers(
                        page_to_read.start_page, page_to_read.end_page
                    )
                    if (page_to_read.end_page != page_to_read.start_page)
                    else self.extract_page(page_to_read.start_page)
                    for page_to_read in pages_to_read.pages
                ],
            )
        )

        # Remove duplicate page numbers in the text, assuming that roman numerals wont be duped.
        # This is because the LLM can have issues with reading the section numbers as page numbers and end up with overlapping pages to read.
        page_numbers = list(re.finditer(r"<< Page (\d+) >>", text))
        if len(page_numbers) == 0:
            return text, pages_to_read
        first_page_number = int(page_numbers[0].group(1))
        last_page_number = int(page_numbers[-1].group(1))

        for page_number in range(first_page_number, last_page_number + 1):
            if text.count(f"<< Page {page_number} >>") > 1:
                # Delete all but first version of this page
                text = re.sub(
                    rf"(?<=<< Page {page_number} >>[\s\S]+)(<< Page {page_number} >>[\s\S]+?)(<< Page \d+ >>)",
                    r"\2",
                    text,
                    flags=re.MULTILINE | re.IGNORECASE,
                )

        return text, pages_to_read

    def extract_page(self, page_to_read):
        if page_to_read in ["0", 0]:
            first_page = get_regex(
                r"<< Page \d+|[xvi]+ >>", re.MULTILINE | re.IGNORECASE
            ).search(self.report_text)
            return self.report_text[: first_page.start()]

        page = get_regex(
            rf"<< Page {page_to_read} >>[\s\S]+<< Page (\d+|[xvi]+) >>",
            re.MULTILINE | re.IGNORECASE,
        ).search(self.report_text)
        if page is None:
            final_page = get_regex(
                rf"<< Page {page_to_read} >>[\s\S]+", re.MULTILINE | re.IGNORECASE
            ).search(self.report_text)
            if final_page is None:
                return None
            return final_page.group(0)
        else:
            return page.group(0)

    def extract_text_between_page_numbers(self, page_number_1, page_number_2) -> str:
        # Create a regular expression pattern to match the page numbers and the text between them
        if page_number_1 == page_number_2:
            return self.extract_page(page_number_1)
        if page_number_1 != 0:
            starting_page_match = get_regex(
                rf"<< Page {page_number_1} >>", re.MULTILINE | re.IGNORECASE
            ).search(self.report_text)
            if starting_page_match is None:
                print(
                    f" {self.report_id} No starting page number for text between pages {page_number_1} and {page_number_2}"
                )
                return None

            starting_index = starting_page_match.start()
        else:
            starting_index = 0

        ending_page_match = get_regex(
            rf"<< Page {page_number_2} >>",
            re.MULTILINE | re.IGNORECASE,
        ).search(self.report_text)

        if ending_page_match is None:
            print(
                f"  {self.report_id} No ending page number for text between pages {page_number_1} and {page_number_2}"
            )
            return None

        return self.report_text[starting_index : ending_page_match.end()]

    def create_hierarchy(self, df):
        hierarchy = []
        df["Level_indent"] = df["Level"].apply(lambda x: "- - " * (x - 1))
        for _, row in df.iterrows():
            title = f"{row['Level_indent']}{row['Title']} {row['Page']}"
            hierarchy.append(title)
        return "\n".join(hierarchy)

    def extract_table_of_contents(self):
        max_content_section_length = 40_000
        startRegex = r"(contents?)([ \w]{0,30}.+)([\n\w\d\sāēīōūĀĒĪŌŪ]*)(.*[ \.]{5,})"
        endRegex = (
            r"^(.*(\.{5,}|(\. ){5,}).*[\dxvi]+.{0,5}?)|((\d+\.){1,3}\d+\.?.* \d+)$"
        )

        if not (isinstance(self.headers, str) or self.headers is None):
            raise ValueError("headers cannot be left to default value")

        endOfContentSection = len(self.report_text) / 3.5

        # Get the entire string between the start and end regex
        startMatch = get_regex(startRegex, re.IGNORECASE).search(self.report_text)
        if startMatch:
            if startMatch.end() > endOfContentSection:
                startMatch = None
        endMatches = list(
            get_regex(endRegex, re.MULTILINE | re.IGNORECASE).finditer(self.report_text)
        )
        if endMatches:
            endMatches = [
                endMatch
                for endMatch in endMatches
                if endMatch.start() < endOfContentSection
            ]

        if startMatch:
            if len(endMatches) == 0:
                print(f"Found a start {self.report_id} but no end: {startMatch}")
                return self.headers, self.headers
            endMatches = [
                endMatch
                for endMatch in endMatches
                if endMatch.start() - startMatch.end() < max_content_section_length
            ]
            if len(endMatches) == 0:
                print(
                    f"Found a start {self.report_id} but no end that isn't too far away: {startMatch}"
                )
                return self.headers, self.headers
            endMatch = endMatches[-1]
        elif len(endMatches) > 1:
            endMatches = [
                endMatch
                for endMatch in endMatches
                if endMatch.start() - endMatches[0].end() < max_content_section_length
            ]

            startMatch = endMatches[0]
            endMatch = endMatches[-1]
        else:
            if len(endMatches) > 0:
                print(f"Found an end {self.report_id} but no start: {endMatches[-1]}")
            return self.headers, self.headers

        raw_content_section = self.report_text[startMatch.start() : endMatch.end()]

        class ContentSectionItem(BaseModel):
            section_number: Optional[int] = None
            section_title: Optional[str] = None
            page_number: Optional[str | roman_numerals] = None
            subsections: Optional[list["ContentSectionItem"]] = None

        class StructuredContentSection(BaseModel):
            items: list[ContentSectionItem] = None

        structured_content_section = ai_caller.query(
            system="""
You are a helpful assistant. You will just respond with the answer no need to explain.
Can you please format this table of contents? Please include in the format the section number (if it has one) the section title and section page number. Make sure to include all of the pages the the table of contents has even it they are roman numerals.
Figures and table lists should be ommitted. Yet appendices if present should be included

""",
            user=f"""
{raw_content_section}
""",
            temp=0,
            model="gpt-4",
            output_structure=StructuredContentSection,
        )

        # Convert the structured content section back to a string representation

        def to_string(item, indent=""):
            if item.section_number:
                return f"{indent}{item.section_number} - {item.section_title} {item.page_number}\n"
            else:
                return f"{indent}- {item.section_title} {item.page_number}\n"

        def process_item(item, indent=""):
            result = to_string(item, indent)
            if item.subsections:
                for subsection in item.subsections:
                    result += process_item(subsection, indent + "  ")
                return result
            return result

        cleaned_content_section = ""
        for item in structured_content_section.items:
            cleaned_content_section += process_item(item)

        return cleaned_content_section, raw_content_section

    def extract_pages_to_read(self, content_section) -> list:
        """
        This takes a content section, reads it and then returns the pages that need to be read. It is then used to extract the needed text.
        """
        raise NotImplementedError


class SafetyIssueExtractor(ReportExtractor):
    def __init__(
        self, report_text, report_id, table_of_contents, investigation_type, agency
    ):
        super().__init__(report_text, report_id)

        if table_of_contents is None:
            raise ValueError("table of contents cannot be None")
        self.table_of_contents = table_of_contents
        self.investigation_type = investigation_type
        self.agency = agency

    def extract_pages_to_read(self, content_section) -> list:
        model_response = ai_caller.query(
            system="""
You are helping me read the content section of a report from a transport accident investigation.
The content section is either a text extraction from the pdf or a parsing of the pdf header links. Note that the content section may be malformed.
I am looking to find the section of the reports that will help me identify safety issues. I need to the page ranges I need to read.

The sections I want you to find:
- Analysis
- Findings (any section that mentions findings)
- Executive Summary / Summary / Safety summary (normally at the start of the report, but it does not always exist)
- Safety issues (any section the explicitly mentions safety issues) 

I want to know the page ranges of the sections you found in the report. Include the start page and end page, where the end page is the page number that the next section starts on. For sections you can't find just omit them from your response. If no sections were found just return "None".
""",
            user=content_section,
            model="gpt-4",
            temp=0,
            output_structure=PagesToReadOutput,
        )

        if (
            model_response is None
            or model_response.pages is None
            or len(model_response.pages) == 0
        ):
            return None

        return model_response

    def extract_safety_issues(self):
        """
        Extract safety issues from a report.
        Return a tuple
        (safety_issues, text_read, pages_read)
        """
        # This abstraction allows the development of various extraction techniques whether it be regex or inferences.

        important_text, pages_read = self.extract_important_text(self.table_of_contents)

        if important_text is None:
            print("  No important text found for report", self.report_id)
            return None, None, None

        if len(important_text) < 100:
            print("  Important text too short for report", self.report_id)
            return None, important_text, pages_read

        important_text_len = len(important_text)
        # Confirm that this report should be included and have its safety issues extracted
        match self.agency:
            case "ATSB":
                raise NotImplementedError(
                    f"ATSB not implemented yet, tried to extract safety issues from {self.report_id}"
                )
                ## TODO: Figure out a better way to include pre 2008 atsb reports.
                # if year >= 2008 or (
                #     investigation_type == "unknown"
                #     and (
                #         important_text_len < 40_000 and isinstance(pages_read, str)
                #     )
                # ):
                #     continue
            case "TSB":
                if (
                    self.investigation_type == "unknown"
                    and (important_text_len < 40_000 and isinstance(pages_read, str))
                ):  # Trying to remove unknown investigation type which might be small ones or ones with foreign influence.
                    return None, important_text, pages_read
            case "TAIC":
                pass  # All TAIC reports should be extracted from
            case _:
                raise ValueError(
                    f"Unknown agency: {self.agency} for report {self.report_id}"
                )

        safety_issues = self._extract_safety_issues_with_inference(important_text)

        return safety_issues, important_text, pages_read

    def _extract_safety_issues_with_regex(self, important_text=None):
        """
        This function will use regex and search the text for any safety issues.
        It will not be used in the main engine pipeline but it useful for development purposes while we dont have a reliable inference extraction.
        """
        if important_text is None:
            raise Exception(
                "  No important text provided to extract safety issues from"
            )

        def safety_regex(x):
            return f"s ?a ?f ?e ?t ?y ? ?i ?s ?s ?u ?e ?s? {{0,3}}{x} {{0,3}}"

        end_regex = r"([\s\S]+?)(?=(?:\d+\.(?:\d+\.)?(?:\d+)?)|(?:s ?a ?f ?e ?t ?y ? ?i ?s ?s ?u ?e ?s?))"

        uncompiled_regexes = [
            "(" + safety_regex(sep) + end_regex + ")" for sep in ["-", ":"]
        ]

        safety_issue_regexes = [
            re.compile(regex, re.MULTILINE | re.IGNORECASE)
            for regex in uncompiled_regexes
        ]

        safety_issues_from_report = []

        matches = [regex.findall(important_text) for regex in safety_issue_regexes]

        # Choose one of the matches that has the most matches
        matches = max(matches, key=lambda x: len(x))

        for full_match, safety_issue_match in matches:
            safety_issues_from_report.append(safety_issue_match)

        return safety_issues_from_report

    def _extract_safety_issues_with_inference(self, important_text):
        """
        Search for safety issues using inference from GPT 4 turbo.
        """

        definitions = """
Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would increase the likelihood of an occurrence, and/or the severity of any adverse consequences associated with the occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the potential to adversely affect the safety of future operations, and
• is characteristic of an organisation, a system, or an operational environment at a specific point in time.

Safety Issues are derived from safety factors classified either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may cover a single safety issue, or two or more related safety issues.
"""

        system_message = f"""
You are going help me read a transport accident investigation report.

I want you to please read the report and respond with the safety issues identified in the report.

Please only respond with safety issues that are quite clearly stated ("exact" safety issues) or implied ("inferred" safety issues) in the report. Each report will only contain one type of safety issue. If exact safety issues are stated then only respond with those. If no exact safety issues are stated then respond with inferred safety issues.

An exact safety issue will start with something like 'safety issue: ...' and will generaly go until the end of the "paragraph" (i.e until it reaches a line that breaks earlier)

Remember the definitions given
{definitions}
"""

        def message(text):
            agency = self.report_id.split("_")[0]
            match agency:
                case "TSB":
                    instruction_core = """
I want to know the safety issues which this investigation has found. If the safety issues are not explicitly stated you will need to infer them. You need to respond with what safety issues this report has identified. Note that sometimes the report will not have any relevant safety issues. In this case you can respond with an empty list.

If no safety issues are stated explicitly, then you need to inferred them. These inferred safety issues are "inferred" safety issues."""
                case "TAIC":
                    instruction_core = """
I want to know the safety issues which this investigation has found.

For each safety issue you find I need to know what is the quality of this safety issue.
Some reports will have safety issues explicitly stated with something like "safety issue - ..." or "safety issue: ...", these are "exact" safety issues. Note that the text may have extra spaces or characters in it. Furthermore findings do not count as safety issues.

If no safety issues are stated explicitly, then you need to inferred them. These inferred safety issues are "inferred" safety issues.
"""
                case _:
                    raise ValueError(
                        f"{agency} is not a supported agency for safety issue extraction"
                    )

            return f"""
{text}

=Instructions=

{instruction_core}

Remember the definitions given
{definitions}
"""

        class SafetyIssueQuality(str, Enum):
            exact = "exact"
            inferred = "inferred"

        class SafetyIssueItem(BaseModel):
            safety_issue: str = Field(
                ...,
                description="The text of the actual safety issue (e.g ignore 'safety issue -').",
            )
            quality: SafetyIssueQuality

        class SafetyIssueListOutput(BaseModel):
            items: list[SafetyIssueItem] = Field(
                default_factory=list,
                description="List of safety issues found in the report, if none found then return an empty list. Only include each safety issue once.",
            )

        parsed = ai_caller.query(
            system_message,
            message(important_text),
            model="gpt-5-mini",
            output_structure=SafetyIssueListOutput,
        )

        if len(parsed.items) == 0:
            print("  Could not get safety issues from the report.")
            return None

        safety_issues = parsed.model_dump()["items"]

        return safety_issues


@lru_cache(maxsize=15000)
def get_section_start_end_regexs(section_str: str):
    """
    This function will get the start and end regex for a section
    You need to give it a string like 5, 5.1, 5.1.1, etc.
    It will return a tuple of (start_regex, [end_regexs])

    note that the end_regexs will have extras incrase the next section is missing.
    """

    def base_regex_template(section):
        return f"((( {section}(?! ?(m )|(metre))) {{1,3}}(?![\\s\\S]*^{section} ))|((^{section}) {{1,3}}))(?![\\S\\s()]{{1,100}}\\.{{2,}})"

    split_section = section_str.split(".")
    section = split_section[0]
    endRegex_nextSection = base_regex_template(rf"{int(section) + 1}\.1\.?")
    startRegex = base_regex_template(rf"{int(section)}\.1\.?")
    endRegexs = [endRegex_nextSection]
    if len(split_section) > 1:
        paragraph = split_section[1]
        # Added to prevent single unfindable section ruining search
        endRegex_nextnextSubSection = base_regex_template(
            rf"{section}\.{int(paragraph) + 2}\.?"
        )
        endRegexs.insert(0, endRegex_nextnextSubSection)
        endRegex_nextSubSection = base_regex_template(
            rf"{section}\.{int(paragraph) + 1}\.?"
        )
        endRegexs.insert(0, endRegex_nextSubSection)
        startRegex = base_regex_template(rf"{section}\.{int(paragraph)}\.?")

    if len(split_section) > 2:
        sub_paragraph = split_section[2]
        endRegex_nextnextParagraph = base_regex_template(
            rf"{section}\.{paragraph}\.{int(sub_paragraph) + 2}\.?"
        )
        endRegexs.insert(0, endRegex_nextnextParagraph)
        endRegex_nextParagraph = base_regex_template(
            rf"{section}\.{paragraph}\.{int(sub_paragraph) + 1}\.?"
        )
        endRegexs.insert(0, endRegex_nextParagraph)
        startRegex = base_regex_template(
            rf"{section}\.{paragraph}\.{int(sub_paragraph)}\.?"
        )

    return (
        get_regex(startRegex, re.MULTILINE | re.IGNORECASE),
        [get_regex(endRegex, re.MULTILINE | re.IGNORECASE) for endRegex in endRegexs],
    )


class ReportSectionExtractor(ReportExtractor):
    def __init__(self, report_text, report_id, headers="Empty"):
        super().__init__(report_text, report_id, headers)

    def _get_previous_section(self, section_str: str):
        """
        This function will get the previous section name from a section name.
        You need to give it a string like 5, 5.1, 5.1.1, etc.
        5.3.2 -> 5.3.1, 3.1 -> 2 etc

        """
        split_section = section_str.split(".")

        if len(split_section) == 3:
            if int(split_section[2]) == 1:
                split_section = [split_section[0], split_section[1]]
            else:
                return (
                    split_section[0]
                    + "."
                    + split_section[1]
                    + "."
                    + str(int(split_section[2]) - 1)
                )

        if len(split_section) == 2:
            if int(split_section[1]) == 1:
                split_section = [split_section[0]]
            else:
                return split_section[0] + "." + str(int(split_section[1]) - 1)

        if len(split_section) == 1:
            if int(split_section[0]) == 1:
                return split_section[0]
            return str(int(split_section[0]) - 1)

    def _get_section_search_bounds(self, section_str: str, endRegexs):
        ##
        # Figure out when to start the search from.
        ##

        # At the start of the rpoert there can be factual information that can has numbers formatted like sections
        page_regex = r"<< Page 1 >>"
        page_regex_match = get_regex(page_regex).search(self.report_text)
        if page_regex_match:
            first_page_pos = page_regex_match.end()
        else:
            first_page_pos = 0

        # As there can be other random numbers in the report or references to latter sections this rules out anything before hand.
        # Note that it relies on the assumption that the prevous few sections exists and are findable.
        previous_section_pos = 0
        attempt = 3
        previous_section_str = section_str
        while previous_section_pos == 0 and attempt > 0:
            previous_section_str = self._get_previous_section(previous_section_str)
            previous_section_regex, _ = get_section_start_end_regexs(
                previous_section_str
            )
            previous_section_match = previous_section_regex.search(
                self.report_text,
                first_page_pos - 1,
            )
            if previous_section_match:
                previous_section_pos = previous_section_match.end()
            else:
                attempt -= 1

        # Adding the 10 so that it still captures the first seciton of a report.
        start_pos = max(max(previous_section_pos, first_page_pos) - 10, 0)

        ##
        # Figure out when to end the search
        ##
        # This wil be done by looking for the next big section as that will give an upper bound

        next_section_match = endRegexs[-1].search(self.report_text, start_pos)

        end_pos = (
            next_section_match.end() + 10
            if next_section_match
            else len(self.report_text)
        )

        return (start_pos, end_pos)

    def extract_section(self, section_str: str, useLLM=True):
        """
        This function extract a numbered section from the report.
        You need to give it a string like 5, 5.1, 5.1.1, etc. It can struggle with the last or second to last section in the report. In this case it utilses AI
        """

        startRegex, endRegexs = get_section_start_end_regexs(section_str)

        start_pos, end_pos = self._get_section_search_bounds(section_str, endRegexs)

        ##
        # Search the report for start and end of the section
        ##

        startMatch = startRegex.search(
            self.report_text,
            start_pos,
            end_pos,
        )

        endMatch = None

        if startMatch:
            endRegexMatches = [
                endRegex.search(
                    self.report_text,
                    start_pos,
                    end_pos,
                )
                for endRegex in endRegexs
            ]

            endMatch = min(
                endRegexMatches, key=lambda x: x.start() if x else len(self.report_text)
            )

        if startMatch is None or endMatch is None:
            # print("Warning: could not find section")
            # print(f"  startMatch: {startMatch} with regex {startRegex} \n  endMatch: {endMatch} with regex {endRegex}")

            if useLLM:
                print("  Attempting to extract section using page numbers and LLMs")
                return self.__extract_section_using_LLM(section_str)
            else:
                return None

        if endMatch.end() < startMatch.end():
            # print(f"Error: endMatch is before startMatch")
            # print(f"  startMatch: {startMatch[0]} \n  endMatch: {endMatch[0]}")
            # print(f"  Regexs: {startRegex} \n  {endRegex}")
            return None

        # if endMatch.end() - startMatch.end() > 16_000:
        #     # print(f"Error: section is too long")
        #     return None

        if startMatch and endMatch:
            section_text = self.report_text[startMatch.start() : endMatch.start()]

            return section_text.strip()

        # print(f"Error: could not find section")
        return None

    def __extract_section_using_LLM(self, section):
        """
        A helper function to extract_section that will read the content section and find the page numbers then extract it from there.
        """

        content_section = self.extract_table_of_contents()

        pages = ai_caller.query(
            """
You are helping me read a content section.

I will send you a content section and a section and you will return the pages with which that section will cover. Include all of the pages that section covers up until the next section starts.
If not found just return None for the pages
""",
            f"""
'''
{content_section}
'''

The section number I am looking for is {section}
            """,
            model="gpt-4",
            temp=0,
            output_structure=PageToRead,
        )

        if pages.start_page is None or pages.end_page is None:
            print(
                f"  Failed to find the section using the LLM, it responded with '{pages}'. The search was for section '{section}'"
            )
            return None

        print(" Found the section using the LLM" + pages)

        # Retrieve that actual text for the page numbers.
        section_text = self.extract_text_between_page_numbers(
            pages.start_page, pages.end_page
        )

        return section_text

    def split_into_paragraphs(self):
        raw_splits = [
            paragraph.strip()
            for paragraph in get_regex(r"\n *\n").split(self.report_text)
            if len(paragraph.strip()) > 0
        ]

        splits_df = pd.DataFrame(raw_splits, columns=["section_text"])

        page_regex = get_regex(r"<< Page (\d+|[xvi]+) >>")

        splits_df["page"] = splits_df["section_text"].map(
            lambda x: page_regex.match(x).group(1) if page_regex.match(x) else None
        )
        splits_df.ffill(inplace=True)
        splits_df.replace({pd.NA: "0", None: "0"}, inplace=True)
        splits_df["paragraph_num"] = splits_df.groupby(["page"]).cumcount()
        splits_df["section"] = (
            "p" + splits_df["page"] + "." + splits_df["paragraph_num"].astype(str)
        )

        splits_df = splits_df[
            splits_df["section_text"].map(lambda x: len(page_regex.sub("", x).strip()))
            > 8
        ]

        return splits_df[["section", "section_text"]]


class RecommendationsExtractor(ReportExtractor):
    def __init__(self, report_text, report_id, table_of_contents):
        super().__init__(report_text, report_id)

        self.table_of_contents = table_of_contents

    def extract_recommendations(self):
        """
        Extract recommendations from a report.
        """
        if self.report_id.split("_")[0] != "ATSB":
            raise NotImplementedError(
                f"{self.report_id.split('_')[0]} is not currently supported yet for recommendation extraction."
            )

        recommendation_section, pages_read = self.extract_important_text(
            self.table_of_contents
        )

        if recommendation_section is None or recommendation_section.strip() == "":
            print(
                "  Could not get recommendations as there was no recommendation section"
            )
            return None, None, None

        extracted_recommendations = self._extract_recommendations_from_text(
            recommendation_section
        )
        recommendation_df = pd.DataFrame(
            extracted_recommendations,
            columns=[
                "recommendation",
                "recommendation_id",
                "recommendation_context",
                "recipient",
                "made",
            ],
        )
        if recommendation_df is not None:
            # This is an error as all modern recommendations have a recommendation_id, and if some recommendations have ids then all should
            if (int(self.report_id.split("_")[2]) > 2010) or (
                recommendation_df["recommendation_id"].isna().sum()
                < len(recommendation_df)
            ):
                recommendation_df = recommendation_df[
                    ~(
                        (recommendation_df["recommendation_id"].isna())
                        | (recommendation_df["recommendation_id"] == "")
                    )
                ]

            # If none of them have IDs then we create them
            if len(recommendation_df) > 0 and all(
                recommendation_df["recommendation_id"].isin(["", None, pd.NA])
            ):
                recommendation_df["recommendation_id"] = [
                    f"{self.report_id}_rec_{i}"
                    for i in range(len(extracted_recommendations))
                ]

        return recommendation_df, recommendation_section, pages_read

    def _extract_recommendations_from_text(self, text):
        """
        This will look for the recommendations that are present in the given text.
        Returns a list of dicts with the following keys:
        - recommendation
        - recommendation_id
        - recipient
        - recommendation_context
        - made
        """

        agency = self.report_id.split("_")[0]
        if agency == "ATSB":
            agency_text = "ATSB (Australia Transport Safety Bureau)"
        else:
            raise NotImplementedError(
                f"{agency} is not currently supported yet for recommendation extraction."
            )

        class RecommendationItem(BaseModel):
            recommendation: str = Field(
                ...,
                description="The text of the recommendation made by the agency in the report. Copy the recommendation verbatim.",
            )
            recommendation_id: str | None = Field(
                default=None,
                description="The unique identifier for the recommendation if it exists in the report. If none is given then return None.",
            )
            recipient: str | None = Field(
                default=None,
                description="The recipient of the recommendation. I.e who the recommendation was addressed to.",
            )
            recommendation_context: str | None = Field(
                default=None,
                description="The context or background information related to the recommendation, if available. Sometimes this is given in the report as a paragraph before the recommendation itself.",
            )
            made: str | None = Field(
                default=None,
                description="The date or time when the recommendation was made, if available.",
            )

        class RecommendationListOutput(BaseModel):
            items: list[RecommendationItem] = Field(
                default_factory=list,
                description="List of recommendations made by the agency in the report. Copy the recommendations verbatim. If none were made then return an empty list.",
            )

            def as_dicts(self) -> list[dict]:
                return [i.model_dump(exclude_none=True) for i in self.items]

        print(f"Text to look at is {len(text)} characters long\n{text}")

        response = ai_caller.query(
            f"""
You are going help me read and parse a transport accident investigation report.
This is the section of a report that may or may not contain recommendations. I want to have a list of all of the distinct recommendations that were made. It is important that the recommendations are copied verbatim.

Recommendations are made to those who can make the changes needed to address safety issues identified during an inquiry.  I only want recommendations that were made by the {agency_text}.

If no appropriate recommendations were made then return "None".
""",
            text,
            model="gpt-4",
            temp=0,
            output_structure=RecommendationListOutput,
        )

        if response is None or len(response.items) == 0:
            return None

        return response.model_dump().get("items")

    def extract_pages_to_read(self, content_section):
        """
        This will get the pages that need to be read to get the recommendations section
        """

        pages_to_read = ai_caller.query(
            system="""
You are helping me read the content sections of a report.

Can you please find the starting and end sections of the recommendations/safety actions section. Start page is inclusive and end page is exclusive. The end of it is the same as the start of the next section. Note that generally the page number will be on the right side of the content page. If it is the final section in the report then just return the last page number of the report as the end page.
A single page should look like "start_page": 25, "end_page": 25
If recommendation or safety action section is not found just return an empty list.
""",
            user=content_section,
            model="gpt-4",
            temp=0,
            output_structure=PagesToReadOutput,
        )

        print(f"Read {content_section}")

        return pages_to_read


class ReportExtractingProcessor:
    """
    This is the class that interacts with the report extracting classes. It however is the ones that actually connects to the folder structure.
    """

    def __init__(self, report_text_df_path: str, refresh=False):
        if not os.path.exists(report_text_df_path):
            raise ValueError(f"{report_text_df_path} does not exist")
        self.report_text_df = pd.read_pickle(report_text_df_path)

        self.refresh = refresh

        self.important_text_df = None

    def extract_safety_issues_from_reports(
        self,
        report_titles_df_path,
        toc_df_path,
        atsb_safety_issues_df_path,
        output_file,
    ):
        print(
            "-----------------------------------------------------------------------------"
        )
        print("                        Extracting safety issues")
        print(f"    Output file: {output_file}")
        print(f"    Report titles: {report_titles_df_path}")
        print(f"    Table of contents: {toc_df_path}")
        print(f"    ATSB safety issues: {atsb_safety_issues_df_path}")

        ## -- Safety issue datasets -- ##
        # Get previously extracted safety issues
        if os.path.exists(output_file) and not self.refresh:
            all_safety_issues_df = pd.read_pickle(output_file)
        else:
            all_safety_issues_df = pd.DataFrame(
                columns=["report_id", "safety_issues", "important_text", "pages_read"]
            )

        # Get atsb safety_issue_dataset
        if os.path.exists(atsb_safety_issues_df_path):
            atsb_safety_issues_df = pd.read_pickle(atsb_safety_issues_df_path)
        else:
            raise ValueError(f"{atsb_safety_issues_df_path} does not exist")

        all_safety_issues_df = pd.concat(
            [all_safety_issues_df, atsb_safety_issues_df]
        ).drop_duplicates("report_id", keep="first")

        ## -- metadata datasets -- ##
        # Get the important text
        # Get the report titles
        if os.path.exists(report_titles_df_path):
            report_titles_df = pd.read_pickle(report_titles_df_path)
        else:
            raise ValueError(f"{report_titles_df_path} does not exist")
        if not os.path.exists(toc_df_path):
            raise ValueError(f"{toc_df_path} does not exist")
        toc_df = pd.read_pickle(toc_df_path)

        ## -- Merging datasets together -- ##
        merged_df = (
            self.report_text_df.merge(
                toc_df[["report_id", "toc"]], on="report_id", how="outer"
            )
            .merge(report_titles_df, on="report_id", how="left")
            .reset_index(drop=True)
        )

        print(
            f"    There are {len(merged_df)} total reports. There are {len(merged_df[merged_df['toc'].isna()])} reports without a content section and {len(all_safety_issues_df)} reports with safety issues. "
        )
        print(
            f"    Removing {len(merged_df[merged_df['investigation_type'] == 'short'])} short reports."
        )
        merged_df = merged_df[merged_df["investigation_type"] != "short"]

        print(
            f"    There are {len(merged_df[merged_df['report_id'].isin(all_safety_issues_df['report_id'])])} reports that already have safety issues"
        )
        print(
            "-----------------------------------------------------------------------------"
        )
        new_safety_issues = []
        for (
            _,
            report_id,
            report_text,
            toc,
            investigation_type,
        ) in (
            pbar := tqdm(
                list(
                    merged_df[
                        [
                            "report_id",
                            "text",
                            "toc",
                            "investigation_type",
                        ]
                    ].itertuples()
                )
            )
        ):
            pbar.set_description(f"Extracting safety issues from {report_id}")
            agency = report_id.split("_")[0]

            if agency == "ATSB":
                # For now ATSB should just be excluded as their safety issues are webscraped back until 2008
                continue

            if report_id in all_safety_issues_df["report_id"].tolist():
                continue

            if pd.isna(toc):
                pbar.write(f"  No table of contents found for {report_id}")
                continue

            safety_issues_results = SafetyIssueExtractor(
                report_text, report_id, toc, investigation_type, agency
            ).extract_safety_issues()

            if safety_issues_results is None:
                pbar.write("Could not extract safety issues")
                continue

            safety_issues_list, important_text, pages_read = safety_issues_results

            safety_issues_df = pd.DataFrame(
                safety_issues_list,
                columns=["safety_issue_id", "safety_issue", "quality"],
            )
            safety_issues_df["safety_issue_id"] = [
                str(i) for i in safety_issues_df.index
            ]

            new_safety_issues.append(
                {
                    "report_id": report_id,
                    "safety_issues": safety_issues_df,
                    "important_text": important_text,
                    "pages_read": pages_read,
                }
            )
            if len(new_safety_issues) > 50:
                all_safety_issues_df = pd.concat(
                    [all_safety_issues_df, pd.DataFrame(new_safety_issues)]
                ).reset_index(drop=True)
                all_safety_issues_df.to_pickle(output_file)
                pbar.write(
                    f" Saving {len(new_safety_issues)} safety issues to bring it to a total of {len(all_safety_issues_df)} of safety issues."
                )
                new_safety_issues = []

        all_safety_issues_df = pd.concat(
            [all_safety_issues_df, pd.DataFrame(new_safety_issues)]
        ).reset_index(drop=True)
        all_safety_issues_df.to_pickle(output_file)

    def extract_table_of_contents_from_reports(self, output_file):
        if os.path.exists(output_file):
            toc_df = pd.read_pickle(output_file)
        else:
            toc_df = pd.DataFrame(columns=["report_id", "toc", "raw_toc"])

        print(
            "-----------------------------------------------------------------------------"
        )
        print(
            f"    Extracting table of contents from {len(self.report_text_df)} reports."
        )
        print(f"    Output file: {output_file}")
        print(
            f"    There are {len(toc_df)} reports with table of contents already extracted."
        )
        print(
            "-----------------------------------------------------------------------------"
        )

        def process_report(report_id, report_text, toc_df, header):
            if report_id in toc_df["report_id"].values:
                return None
            table_of_contents, raw_table_of_contents = ReportExtractor(
                report_text, report_id, header
            ).extract_table_of_contents()
            if table_of_contents is None:
                return None
            return report_id, table_of_contents, raw_table_of_contents

        def save_result(result, toc_df, output_file):
            report_id, toc, raw_toc = result
            if toc is None:
                print(f"  Could not extract toc from {report_id}")
            else:
                toc_df.loc[len(toc_df)] = [report_id, toc, raw_toc]
                toc_df.to_pickle(output_file)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _, report_id, report_text, _, header in (
                pbar := tqdm(list(self.report_text_df.itertuples()))
            ):
                pbar.set_description(f"Extracting toc from {report_id}")
                futures.append(
                    executor.submit(
                        process_report,
                        report_id,
                        report_text,
                        toc_df,
                        header,
                    )
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                result = future.result()
                if result is not None:
                    save_result(result, toc_df, output_file)

        toc_df.to_pickle(output_file)

    def __extract_sections(
        num_sections,
        all_potential_sections,
        report_text,
        report_id,
        debug=False,
        headers=None,
    ):
        get_parts_regex = r"(((\d{1,2}).\d{1,2}).\d{1,2})"
        get_parts_pattern = get_regex(get_parts_regex)

        extractor = ReportSectionExtractor(report_text, num_sections, headers)

        sections = []

        for section in all_potential_sections:
            if debug:
                print(
                    f"Looking at section {get_parts_pattern.search(section[0][0]).group(3)}"
                )
            subsection_empty = 0
            subsection_missing_count = 0
            for sub_section in section:
                sub_section_str = get_parts_pattern.search(sub_section[0]).group(2)
                if debug:
                    print(f" Looking at subsection {sub_section_str}")

                paragraph_missing_count = 0

                paragraphs = []
                for paragraph in sub_section:
                    if debug:
                        print(f"  Looking for paragraph {paragraph}")

                    paragraph_text = extractor.extract_section(paragraph, useLLM=False)

                    if paragraph_text is None and (
                        paragraph_missing_count > 0 or paragraph[-1] == "1"
                    ):
                        break
                    elif paragraph_text is None:
                        paragraph_missing_count += 1
                        continue

                    paragraphs.append(
                        {"section": paragraph, "section_text": paragraph_text}
                    )

                if len(paragraphs) == 0:
                    if debug:
                        print(" No paragraphs found ")
                    sub_section_text = extractor.extract_section(
                        sub_section_str, useLLM=False
                    )

                    if sub_section_text is None and subsection_missing_count > 0:
                        if debug:
                            print(" No subsection found")
                        subsection_empty += 1
                        break
                    elif sub_section_text is None:
                        subsection_missing_count += 1
                        continue

                    subsection_empty = 0

                    sections.append(
                        {"section": sub_section_str, "section_text": sub_section_text}
                    )
                else:
                    sections.extend(paragraphs)

            if subsection_empty > 1:
                # Stop looking at more main sections if two empty subsections in a row happen.
                break

        df = pd.DataFrame(sections, columns=["section", "section_text"])

        # Check if this worked. Otherwise extract with paragraph splitting.

        report_sections = None

        report_sections = None

        if len(df) > 4 and df["section_text"].map(len).mean() < 2_000:
            report_sections = df.copy()

        else:
            report_sections = extractor.split_into_paragraphs()

        # Filter out sections that are irrelevant

        report_sections["section_text_length"] = report_sections["section_text"].map(
            len
        )

        source_line = report_sections["section_text"].str.match(r"^Source: [\w\s]+$")

        copyright = report_sections["section_text"].str.match(r"^© [\w\s]+$")

        blank_page = (
            report_sections["section_text"].str.match(
                r"^(<< Page \d+ >>)|([A-Z ]+)$", flags=re.MULTILINE
            )
        ) & (report_sections["section_text_length"] < 250)
        to_remove = source_line | copyright | blank_page

        filtered_report_sections = (
            report_sections[~to_remove]
            .reset_index(drop=True)
            .drop(columns=["section_text_length"])
        )

        # Merge small sections togather

        merged_sections = []

        current_text = ""
        current_sections = []
        first_section_id = None

        for idx, row in filtered_report_sections.iterrows():
            if not current_text:
                current_text = row["section_text"]
                current_sections = [row["section"]]
                first_section_id = row["section"]
            else:
                if len(current_text) < 1000:
                    current_text += "\n" + row["section_text"]
                    current_sections.append(row["section"])
                else:
                    merged_sections.append(
                        {
                            "section": f"{first_section_id}-{current_sections[-1]}"
                            if len(current_sections) > 1
                            else current_sections[0],
                            "section_text": current_text,
                            "report_id": report_id,
                        }
                    )
                    current_text = row["section_text"]
                    current_sections = [row["section"]]
                    first_section_id = row["section"]

        # Add any remaining buffer
        if current_text:
            merged_sections.append(
                {
                    "section": f"{first_section_id}-{current_sections[-1]}"
                    if len(current_sections) > 1
                    else current_sections[0],
                    "section_text": current_text,
                    "report_id": report_id,
                }
            )

        merged_sections = pd.DataFrame(
            merged_sections, columns=["section", "section_text", "report_id"]
        )

        return merged_sections

    def extract_sections_from_text(self, num_sections, output_file_path):
        print(
            "-----------------------------------------------------------------------------"
        )
        print(f"    Extracting sections from {len(self.report_text_df)} reports.")
        sections = list(map(str, range(1, 15)))

        subsections = [
            [section + "." + str(subsection) for subsection in range(1, 100)]
            for section in sections
        ]

        paragraphs = [
            [
                [subsection + "." + str(paragraph) for paragraph in range(1, 100)]
                for subsection in section
            ]
            for section in subsections
        ]

        if os.path.exists(output_file_path):
            report_sections_df = pd.read_pickle(output_file_path)
        else:
            report_sections_df = pd.DataFrame(columns=["report_id", "sections"])

        new_reports = []

        print(f"    Output file: {output_file_path}")
        print(f"    Number of sections: {num_sections}")
        print(f"    Already extracted reports: {len(report_sections_df)}")
        print(
            "-----------------------------------------------------------------------------"
        )
        for _, report_id, report_text, _, headers in (
            pbar := tqdm(list(self.report_text_df.itertuples()))
        ):
            pbar.set_description(f"Extracting sections from {report_id}")
            if report_id in report_sections_df["report_id"].values:
                continue

            sections_df = ReportExtractingProcessor.__extract_sections(
                num_sections,
                paragraphs,
                report_text,
                report_id,
                debug=False,
                headers=headers,
            )
            sections_df["report_id"] = report_id

            new_reports.append({"report_id": report_id, "sections": sections_df})
            if len(new_reports) > 50:
                report_sections_df = pd.concat(
                    [report_sections_df, pd.DataFrame(new_reports)], ignore_index=True
                )
                report_sections_df.to_pickle(output_file_path)
                pbar.write(
                    f" Saving {len(new_reports)} new extracted reports to bring it to a total of {len(report_sections_df)} of extracted reports."
                )
                new_reports = []

        report_sections_df = pd.concat(
            [report_sections_df, pd.DataFrame(new_reports)], ignore_index=True
        )
        report_sections_df.to_pickle(output_file_path)

        get_section_start_end_regexs.cache_clear()

    def extract_recommendations(
        self,
        output_path,
        tsb_recommendations_path,
        taic_recommendations_path,
        toc_df_path,
    ):
        print(
            "-----------------------------------------------------------------------------"
        )
        print("                        Extracting recommendations")
        print(f"    Output file: {output_path}")
        print(f"    Table of contents: {toc_df_path}")
        print(f"    ATSB recommendations: {tsb_recommendations_path}")
        print(f"    TAIC recommendations: {taic_recommendations_path}")

        if os.path.exists(output_path) and not self.refresh:
            recommendations_df = pd.read_pickle(output_path)
        else:
            recommendations_df = pd.DataFrame(
                columns=["report_id", "recommendations", "important_text", "pages_read"]
            )
        if os.path.exists(tsb_recommendations_path):
            tsb_recommendations_df = pd.read_pickle(tsb_recommendations_path)
        else:
            tsb_recommendations_df = pd.DataFrame(
                columns=["report_id", "recommendations"]
            )

        if os.path.exists(taic_recommendations_path):
            taic_recommendations_df = pd.read_pickle(taic_recommendations_path)
        else:
            taic_recommendations_df = pd.DataFrame(
                columns=["report_id", "recommendations"]
            )

        recommendations_df = pd.concat(
            [recommendations_df, tsb_recommendations_df, taic_recommendations_df],
            ignore_index=True,
        ).drop_duplicates("report_id")

        if not os.path.exists(toc_df_path):
            raise ValueError(f"{toc_df_path} does not exist")
        toc_df = pd.read_pickle(toc_df_path)

        atsb_reports = self.report_text_df[
            self.report_text_df["report_id"].map(lambda x: x.split("_")[0]) == "ATSB"
        ]
        new_reports = recommendations_df.merge(
            atsb_reports, how="right", on="report_id"
        )

        new_reports = new_reports.merge(
            toc_df[["report_id", "toc"]], how="left", on="report_id"
        )
        new_reports = new_reports[
            (new_reports["recommendations"].isna()) & (~new_reports["toc"].isna())
        ]

        print(f"    There are {len(new_reports)} reports with no recommendations.")
        print(f"    Current recommendations: {len(recommendations_df)}")
        print(
            "-----------------------------------------------------------------------------"
        )

        for _, report_id, report_text, toc in (
            pbar := tqdm(list(new_reports[["report_id", "text", "toc"]].itertuples()))
        ):
            pbar.set_description(f"Extracting recommendations from {report_id}")
            if report_id in recommendations_df["report_id"].values:
                continue

            recommendations, important_text, pages_read = RecommendationsExtractor(
                report_text, report_id, toc
            ).extract_recommendations()

            recommendations_df.loc[len(recommendations_df)] = [
                report_id,
                recommendations,
                important_text,
                pages_read,
            ]
            recommendations_df.to_pickle(output_path)

        recommendations_df.to_pickle(output_path)
