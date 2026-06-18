"""Tests for report extraction functionality.

This module contains tests for the ReportExtracting module,
including tests for safety issue extraction, recommendation extraction,
report chunking, and parallel processing of multiple reports.
"""

import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from engine import Modes
from engine.ExtractionModels import (
    RecommendationItem,
    SafetyIssueItem,
)
from engine.ReportExtracting import (
    ai_read_report,
    chunk_report_into_sections,
    load_event_type_taxonomy,
    process_reports_parallel,
)
from engine.SavedDataFrames import ExtractedReports, ParsedReports, ReportTitles


def _output_dir_from_pytest() -> Path:
    """Get the configured test output directory from pytest globals.

    Returns:
        Path: Output folder path.
    """
    return Path(pytest.output_config["folder_name"])  # type: ignore[attr-defined]


def _ai_extraction_config_from_pytest() -> dict:
    """Get AI extraction config from pytest globals.

    Returns:
        dict: AI extraction configuration by agency.
    """
    return pytest.config["engine"]["extraction"]["ai_extraction_config"]  # type: ignore[attr-defined]


# Test data loading functions
def load_json_test_data(filename: str) -> list | dict:
    """Load test data from a JSON file in the tests/data directory.

    Returns:
        list | dict: Parsed JSON data from the specified file.
    """
    test_data_path = Path(__file__).parent / "data" / filename
    with open(test_data_path, encoding="utf-8") as f:
        return json.load(f)


def load_safety_issue_test_cases() -> list:
    """Load safety issue test cases and convert to parametrize format.

    Returns:
        list: List of pytest.param objects with proper test IDs.
    """
    test_cases = load_json_test_data("safety_issue_test_cases.json")
    params = []
    for case in test_cases:
        report_id = case["report_id"]
        expected = [SafetyIssueItem(**item) for item in case["expected"]]
        params.append(pytest.param(report_id, expected, id=f"{report_id}_{case['id']}"))
    return params


def load_recommendation_test_cases() -> list:
    """Load recommendation test cases and convert to parametrize format.

    Returns:
        list: List of pytest.param objects with proper test IDs.
    """
    test_cases = load_json_test_data("recommendation_test_cases.json")
    params = []
    for case in test_cases:
        report_id = case["report_id"]
        expected = [RecommendationItem(**item) for item in case["expected"]]
        params.append(pytest.param(report_id, expected, id=f"{report_id}_{case['id']}"))
    return params


def get_report_text(report_id: str) -> str:
    """Fetch the text of a report.

    Args:
        report_id (str): The unique identifier for the report.

    Raises:
        ValueError: If the report ID is not found in the extracted reports.

    Returns:
        str: The text content of the report.

    """
    parsed_reports_dc = ParsedReports(_output_dir_from_pytest())
    extracted_reports = parsed_reports_dc.read()
    try:
        return extracted_reports.query("report_id == @report_id")["text"].iloc[0]
    except IndexError as e:
        msg = f"Report ID {report_id} not found in extracted reports."
        raise ValueError(msg) from e


class TestAIExtraction:
    """Tests AI extraction for safety issues and recommendations.

    This class is the key class that provides some rigor to the extraction methods. It should be expanded yet currently given AI capabilities it seems reasonable that with these few tests we can expect fair generalization.
    """

    @pytest.mark.parametrize(
        "report_id, expected",
        load_safety_issue_test_cases(),
    )
    def test_safety_issue_extraction(  # noqa: PLR6301
        self,
        report_id: str,
        expected: list[SafetyIssueItem],
        agency_id_lookup: dict[str, str],
    ) -> None:
        """Test the extraction of safety issues from a report.

        Test both TSB and TAIC as well as pre 2008 ATSb reports.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[SafetyIssueItem]): The expected safety issues.
            agency_id_lookup (dict[str, str]): Report ID to agency ID mapping.

        """
        report_text = get_report_text(report_id)
        agency_id = agency_id_lookup.get(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_", maxsplit=1)[0],
            report_text=report_text,
            safety_issues=True,
            recommendations=False,
            metadata=False,
            report_id=report_id,
            agency_id=agency_id,
        )

        extracted = getattr(extracted_data, "safety_issues", [])
        failures = []

        # Is it inferred or not
        expecting_inferred = any(item.quality == "inferred" for item in expected)

        # Check count matches
        if expecting_inferred:
            max_allowed = math.ceil(len(expected) * 1.2)
            if len(extracted) < len(expected) or len(extracted) > max_allowed:
                failures.append(
                    f"Count mismatch (inferred): expected between {len(expected)} and {max_allowed} safety issues, got {len(extracted)}"
                )
        elif len(extracted) != len(expected):
            failures.append(
                f"Count mismatch: expected {len(expected)} safety issues, got {len(extracted)}"
            )

        # Make sure all inferred or exact
        if not (
            (
                not expecting_inferred
                and all(item.quality == "exact" for item in extracted)
            )
            or (
                expecting_inferred
                and all(item.quality == "inferred" for item in extracted)
            )
        ):
            failures.append(
                f"Quality mismatch: expected all safety issues to be {expected[0].quality}, but got a mix of qualities in extracted data\n{[item.quality for item in extracted]}"
            )

        # Compare extracted and expected items using best-match alignment
        threshold = 0.4 if expecting_inferred else 0.95

        for expected_item in expected:
            current_best = (None, 0, None)  # (extracted_item, similarity, index)
            for i, extracted_item in enumerate(extracted):
                similarity = SequenceMatcher(
                    None,
                    extracted_item.safety_issue.lower(),
                    expected_item.safety_issue.lower(),
                ).ratio()

                if similarity > current_best[1]:
                    current_best = (extracted_item, similarity, i)

            if current_best[1] < threshold:
                failures.append(
                    f"Expected: {expected_item.safety_issue!r}\nInstead got: {current_best[0].safety_issue!r}\nSimilarity: {current_best[1]:.2f} < {threshold}"
                )

        # Report all failures at once
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("report_id, expected", load_recommendation_test_cases())
    def test_recommendation_extraction(  # noqa: PLR6301
        self,
        report_id: str,
        expected: list[RecommendationItem],
        agency_id_lookup: dict[str, str],
        recommendation_similarity_threshold: float = 0.95,
        context_similarity_threshold: float = 0.3,  # Only weak match is needed
    ) -> None:
        """Test the recommendation extraction of ATSB.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[RecommendationItem]): The expected recommendations.
            agency_id_lookup (dict[str, str]): Report ID to agency ID mapping.
            recommendation_similarity_threshold (float): Minimum similarity for recommendation text.
            context_similarity_threshold (float): Minimum similarity for recommendation context.

        """
        report_text = get_report_text(report_id)
        agency_id = agency_id_lookup.get(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_", maxsplit=1)[0],
            report_text=report_text,
            safety_issues=False,
            recommendations=True,
            metadata=False,
            report_id=report_id,
            agency_id=agency_id,
        )

        extracted = getattr(extracted_data, "recommendations", [])
        failures = []

        expected_ids = {item.recommendation_id for item in expected}
        extracted_ids = {item.recommendation_id for item in extracted}

        # 1. Check all expected recommendation IDs are present and no unexpected IDs are present
        missing_ids = expected_ids - extracted_ids
        unexpected_ids = extracted_ids - expected_ids
        if missing_ids:
            failures.append(f"Missing recommendation IDs: {missing_ids}")
        if unexpected_ids:
            failures.append(f"Unexpected recommendation IDs: {unexpected_ids}")

        # 2. Check each item (count permitting)
        for idx, expected_item in enumerate(expected):
            extracted_item = next(
                (
                    item
                    for item in extracted
                    if item.recommendation_id == expected_item.recommendation_id
                ),
                None,
            )
            if not extracted_item:
                continue  # Already reported missing ID, skip further checks for this item

            # Check recipient
            if extracted_item.recipient != expected_item.recipient:
                failures.append(f"""Item {idx} ({extracted_item.recommendation_id}): Recipient mismatch
    Expected:
{expected_item.recipient}
    Got:
{extracted_item.recipient}""")

            # Check recommendation text similarity
            similarity = SequenceMatcher(
                None,
                extracted_item.recommendation.lower(),
                expected_item.recommendation.lower(),
            ).ratio()

            if similarity < recommendation_similarity_threshold:
                failures.append(f"""Item {idx} ({extracted_item.recommendation_id}): Recommendation text similarity {similarity:.2f} < {recommendation_similarity_threshold}
    Expected:
{expected_item.recommendation}
    Got:
{extracted_item.recommendation}""")

            # Check context similarity
            context_similarity = SequenceMatcher(
                lambda x: re.match(r"\s+", x)
                is not None,  # Ignore whitespace differences
                (extracted_item.recommendation_context or "").lower(),
                (expected_item.recommendation_context or "").lower(),
            ).ratio()

            if context_similarity < context_similarity_threshold:
                failures.append(f"""Item {idx} ({extracted_item.recommendation_id}): Context similarity {context_similarity:.2f} < {context_similarity_threshold}
    Expected:
{expected_item.recommendation_context}
    Got:
{extracted_item.recommendation_context}""")

        # Report all failures at once
        assert not failures, "\n".join(failures)


_full_extraction_reports = [
    "TAIC_a_2015_009",
    "TAIC_r_2021_101",
    "ATSB_m_2022_007",
    "TSB_a_2023_W0096",
]


def _load_test_case_for_report(filename: str, report_id: str) -> dict | list | None:
    """Load a test case from a JSON file for a specific report_id.

    Args:
        filename: JSON filename in tests/data directory.
        report_id: The report ID to find.

    Returns:
        The 'expected' value from the matching test case, or None if not found.
    """
    test_cases = load_json_test_data(filename)
    for case in test_cases:
        if case["report_id"] == report_id:
            return case["expected"]
    return None


@pytest.mark.parametrize("report_id", _full_extraction_reports)
def test_full_extraction(  # noqa: PLR0912, PLR0914, PLR0915
    report_id: str,
    agency_id_lookup: dict[str, str],
) -> None:
    """Test full extraction (safety issues + recommendations + metadata) all at once.

    Loads expected values from the saved extracted_reports.pkl and compares
    the results of ai_read_report with all extraction flags enabled.

    Args:
        report_id: The unique identifier for the report.
        agency_id_lookup: Report ID to agency ID mapping.
    """
    # Hear to prevent circular import issues.
    from test_MetadataExtraction import (  # noqa: PLC0415
        _compare_metadata_values,  # noqa: PLC2701
    )

    report_text = get_report_text(report_id)
    agency_id = agency_id_lookup.get(report_id)
    report_mode = Modes.get_report_mode_from_id(report_id)
    event_type_taxonomy_by_mode = load_event_type_taxonomy(Path("data/event_types.csv"))

    # Load expected data from JSON test case files
    si_expected = _load_test_case_for_report("safety_issue_test_cases.json", report_id)
    recs_expected = _load_test_case_for_report(
        "recommendation_test_cases.json", report_id
    )
    meta_expected = _load_test_case_for_report("metadata_test_cases.json", report_id)
    expected_safety_issues = (
        [SafetyIssueItem(**item) for item in si_expected] if si_expected else []
    )
    expected_recommendations = (
        [RecommendationItem(**item) for item in recs_expected] if recs_expected else []
    )
    expected_metadata = meta_expected or {}

    # Run full extraction
    extracted_data = ai_read_report(
        agency_name=report_id.split("_", maxsplit=1)[0],
        report_text=report_text,
        safety_issues=True,
        recommendations=True,
        metadata=True,
        report_mode=report_mode,
        event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        report_id=report_id,
        agency_id=agency_id,
    )

    failures = []

    # ---- Compare safety issues ----
    extracted_safety_issues = getattr(extracted_data, "safety_issues", [])

    if expected_safety_issues:
        expecting_inferred = any(
            item.quality == "inferred" for item in expected_safety_issues
        )
        if expecting_inferred:
            max_allowed = math.ceil(len(expected_safety_issues) * 1.2)
            if (
                len(extracted_safety_issues) < len(expected_safety_issues)
                or len(extracted_safety_issues) > max_allowed
            ):
                failures.append(
                    f"safety_issues count mismatch (inferred): expected between {len(expected_safety_issues)} and {max_allowed}, got {len(extracted_safety_issues)}"
                )
        elif len(extracted_safety_issues) != len(expected_safety_issues):
            failures.append(
                f"safety_issues count mismatch: expected {len(expected_safety_issues)}, got {len(extracted_safety_issues)}"
            )

        threshold = 0.4 if expecting_inferred else 0.95
        if not (
            expecting_inferred
            and all(item.quality == "inferred" for item in extracted_safety_issues)
        ) and not (
            not expecting_inferred
            and all(item.quality == "exact" for item in extracted_safety_issues)
        ):
            failures.append(
                f"safety_issues quality mismatch: expected all {expected_safety_issues[0].quality}, got qualities {[item.quality for item in extracted_safety_issues]}"
            )

        for expected_item in expected_safety_issues:
            current_best = (None, 0)
            for extracted_item in extracted_safety_issues:
                similarity = SequenceMatcher(
                    None,
                    extracted_item.safety_issue.lower(),
                    expected_item.safety_issue.lower(),
                ).ratio()
                if similarity > current_best[1]:
                    current_best = (extracted_item, similarity)
            if current_best[1] < threshold:
                failures.append(
                    f"safety_issue mismatch: expected {expected_item.safety_issue!r}, best match similarity {current_best[1]:.2f} < {threshold}"
                )

    # ---- Compare recommendations ----
    extracted_recommendations = getattr(extracted_data, "recommendations", [])

    if expected_recommendations:
        expected_ids = {r.recommendation_id for r in expected_recommendations}
        extracted_ids = {r.recommendation_id for r in extracted_recommendations}
        missing_ids = expected_ids - extracted_ids
        unexpected_ids = extracted_ids - expected_ids
        if missing_ids:
            failures.append(f"Missing recommendation IDs: {missing_ids}")
        if unexpected_ids:
            failures.append(f"Unexpected recommendation IDs: {unexpected_ids}")

        for idx, expected_item in enumerate(expected_recommendations):
            extracted_item = next(
                (
                    r
                    for r in extracted_recommendations
                    if r.recommendation_id == expected_item.recommendation_id
                ),
                None,
            )
            if not extracted_item:
                continue
            if extracted_item.recipient != expected_item.recipient:
                failures.append(
                    f"recommendation {idx} ({expected_item.recommendation_id}): recipient mismatch"
                )
            similarity = SequenceMatcher(
                None,
                extracted_item.recommendation.lower(),
                expected_item.recommendation.lower(),
            ).ratio()
            if similarity < 0.95:  # noqa: PLR2004
                failures.append(
                    f"recommendation {idx} ({expected_item.recommendation_id}): text similarity {similarity:.2f} < 0.95"
                )

    # ---- Compare metadata ----
    extracted_metadata = getattr(extracted_data, "metadata", None)
    if extracted_metadata is None:
        failures.append("metadata extraction returned None")
    elif expected_metadata:
        extracted_payload = extracted_metadata.model_dump(mode="json")
        _compare_metadata_values(
            "metadata", extracted_payload, expected_metadata, failures
        )

    assert not failures, "\n".join(failures)


_chunking_params = [
    ("TAIC_m_2004_203", 29),
    ("TSB_a_2023_W0096", 34),
    ("TAIC_a_2020_003", 55),
    ("ATSB_r_2010_007", 18),
]


@pytest.mark.parametrize(
    "report_id, num_sections",
    _chunking_params,
)
def test_chunking_into_section(report_id: str, num_sections: int) -> None:
    """Test that does a basic sanity check on the chunking of a report into sections."""
    report_text = get_report_text(report_id)

    sections = chunk_report_into_sections(report_text)

    assert (
        len(sections) == num_sections
    ), f"Expected {num_sections} sections but got {len(sections)}"
    assert all(isinstance(section, dict) for section in sections)
    assert all({"text", "section"}.issubset(section) for section in sections)


def test_parallel_extraction(tmp_path: object) -> None:
    """Test that we can extract from multiple reports in parallel without issues."""
    ids = {
        "ATSB_a_2000_157": {"si": 7, "recs": 8, "sections": 193},
        "ATSB_r_2014_001": {"si": None, "recs": 2, "sections": 107},
        "ATSB_a_2007_018": {"si": 4, "recs": 0, "sections": 51},
        "ATSB_m_2007_241": {"si": 4, "recs": 1, "sections": 48},
        "ATSB_m_2022_007": {"si": None, "recs": 3, "sections": 113},
        "ATSB_r_2010_007": {"si": None, "recs": 2, "sections": 18},
        "TAIC_a_2020_003": {"si": 0, "recs": None, "sections": 55},
        "TSB_a_2023_W0096": {"si": 2, "recs": None, "sections": 34},
        "ATSB_a_2005_912": {"si": 1, "recs": 0, "sections": 28},
    }

    # These extra ids will be used to make sure that it can ignore reports that have already been processed

    test_report_texts = [get_report_text(report_id) for report_id in list(ids.keys())]

    report_texts_df = pd.DataFrame(
        {"report_id": list(ids.keys()), "text": test_report_texts}
    )

    # Save to ParsedReports using SavedDataFrames
    parsed_reports_dc = ParsedReports(tmp_path)
    parsed_reports_dc.save(report_texts_df)

    extracted_reports_dc = ExtractedReports(tmp_path)
    report_titles_dc = ReportTitles(tmp_path)
    source_report_titles_dc = ReportTitles(_output_dir_from_pytest())
    report_titles_dc.save(source_report_titles_dc.read())

    # Extract from all reports in parallel
    results = process_reports_parallel(
        parsed_reports_dc=parsed_reports_dc,
        extracted_reports_dc=extracted_reports_dc,
        report_titles_dc=report_titles_dc,
        ai_extraction_config=_ai_extraction_config_from_pytest(),
    )

    # Compare results to expected values
    failures = []
    for report_id, expected in ids.items():
        extracted = results[results["report_id"] == report_id]
        if extracted.empty:
            failures.append(f"{report_id}: No extraction result found")
            continue
        extracted = extracted.iloc[0]

        if expected["si"] and len(extracted.safety_issues) != expected["si"]:
            failures.append(
                f"{report_id}: safety_issues expected {expected['si']}, got {len(extracted.safety_issues)}"
            )
        if expected["recs"] and len(extracted.recommendations) != expected["recs"]:
            failures.append(
                f"{report_id}: recommendations expected {expected['recs']}, got {len(extracted.recommendations)}"
            )
        if len(extracted.sections) != expected["sections"]:
            failures.append(
                f"{report_id}: sections expected {expected['sections']}, got {len(extracted.sections)}"
            )

    assert not failures, "Extraction mismatches:\n  " + "\n  ".join(failures)


def test_process_handle_already_processed(
    tmp_path: object,
    expected_new_report_count: int = 2,
) -> None:
    """Test that process_reports_parallel only processes new reports and skips already processed ones.

    Args:
        tmp_path: Pytest temporary directory.
        expected_new_report_count: Expected number of new reports to process.
    """
    already_processed_ids = [
        "ATSB_r_2014_001",
        "ATSB_a_2007_018",
        "ATSB_m_2007_241",
        "ATSB_m_2022_007",
    ]

    # IDs to process: one is a duplicate (already processed), two are new
    to_process_ids = [
        "ATSB_a_2007_018",  # Duplicate - should NOT be re-processed
        "ATSB_m_2020_007",  # New - should be processed
        "TAIC_m_2004_203",  # New - should be processed
    ]

    # Expected new report IDs (duplicates removed)
    expected_new_ids = ["ATSB_m_2020_007", "TAIC_m_2004_203"]

    # Mock up a current extracted df with some already processed reports
    current_extracted_df = pd.DataFrame(
        {
            "report_id": already_processed_ids,
            "safety_issues": [[] for _ in already_processed_ids],
            "recommendations": [[] for _ in already_processed_ids],
            "metadata": [{} for _ in already_processed_ids],
            "sections": [[] for _ in already_processed_ids],
        }
    )

    # Create a reports df with both already processed and new reports
    report_texts_df = pd.DataFrame(
        {
            "report_id": already_processed_ids + to_process_ids,
            "text": ["dummy_text"] * (len(already_processed_ids) + len(to_process_ids)),
        }
    )

    # Save using SavedDataFrames
    parsed_reports_dc = ParsedReports(tmp_path)
    extracted_reports_dc = ExtractedReports(tmp_path)
    report_titles_dc = ReportTitles(tmp_path)

    parsed_reports_dc.save(report_texts_df)
    extracted_reports_dc.save(current_extracted_df)
    source_report_titles_dc = ReportTitles(_output_dir_from_pytest())
    report_titles_dc.save(source_report_titles_dc.read())

    # Mock the extract_report function to track calls
    with patch("engine.ReportExtracting.extract_report") as mock_extract:
        # Configure mock to return a dict with expected structure
        mock_extract.side_effect = lambda row, config, *_args: {
            "report_id": row["report_id"],
            "safety_issues": [],
            "recommendations": [],
            "metadata": {},
            "sections": [],
        }

        # Process reports
        results_df = process_reports_parallel(
            parsed_reports_dc=parsed_reports_dc,
            extracted_reports_dc=extracted_reports_dc,
            report_titles_dc=report_titles_dc,
            ai_extraction_config=_ai_extraction_config_from_pytest(),
        )

    # Assertions: Verify that extract_report was called only for new reports
    assert (
        mock_extract.call_count == expected_new_report_count
    ), f"extract_report should be called {expected_new_report_count} times (for new reports), but was called {mock_extract.call_count} times"

    # Verify the correct reports were processed by checking the call arguments
    called_report_ids = [
        call_arg[0][0]["report_id"] for call_arg in mock_extract.call_args_list
    ]
    assert (
        set(called_report_ids) == set(expected_new_ids)
    ), f"extract_report should be called for {expected_new_ids}, but was called for {called_report_ids}"

    # Assertions: Verify results contain all reports (already processed + new)
    result_ids = set(results_df["report_id"].tolist())
    expected_all_ids = set(already_processed_ids + expected_new_ids)
    assert (
        result_ids == expected_all_ids
    ), f"Results should contain {expected_all_ids}, but contains {result_ids}"

    # Assertions: Verify that all required columns are present
    assert (
        set(results_df.columns)
        == {
            "report_id",
            "safety_issues",
            "recommendations",
            "metadata",
            "sections",
        }
    ), f"Results should have columns ['report_id', 'safety_issues', 'recommendations', 'metadata', 'sections'], but has {list(results_df.columns)}"

    # Assertions: No duplicate report_ids in results
    assert (
        len(results_df) == len(result_ids)
    ), f"Results should have {len(expected_all_ids)} unique reports, but has {len(results_df)} rows"
