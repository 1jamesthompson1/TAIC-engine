"""Tests for report extraction functionality.

This module contains comprehensive tests for the ReportExtracting module,
including tests for safety issue extraction, recommendation extraction,
report chunking, and parallel processing of multiple reports.

This is the key test module for validating that the AI extraction works.
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from engine import Modes
from engine.ReportExtracting import (
    AircraftMetadata,
    OccurrenceMetadata,
    RecommendationItem,
    SafetyIssueItem,
    TrainMetadata,
    VesselMetadata,
    _build_metadata_model_for_mode,  # noqa: PLC2701
    ai_read_report,
    chunk_report_into_sections,
    load_event_type_taxonomy,
    process_reports_parallel,
)
from engine.SavedDataFrames import ExtractedReports, ParsedReports

RECOMMENDATION_SIMILARITY_THRESHOLD = 0.95
CONTEXT_SIMILARITY_THRESHOLD = 0.9
EXPECTED_NEW_REPORT_COUNT = 2

# Use a weak threshold for non-literal free-text string comparisons.
METADATA_WEAK_STRING_SIMILARITY_THRESHOLD = 0.5

EXACT_MATCH_PATHS = {
    "occurrence.occurrence_datetime.local_datetime",
    "occurrence.occurrence_datetime.time_zone",
    "occurrence.occurrence_type",
    "occurrence.location.standardized_location",
}


def _canonical_sort_key(value):
    """Build a stable sort key for list item comparison.

    Returns:
        str: Canonical JSON string representation used for sorting.
    """
    return json.dumps(value, sort_keys=True, default=str)


def _list_item_sort_key(path: str, value):
    """Choose stable item keys so list comparison is less order-sensitive.

    Returns:
        str: A deterministic sort key.
    """
    if not isinstance(value, dict):
        return _canonical_sort_key(value)

    if path == "aircraft":
        return str(value.get("registration") or value.get("model") or "")
    if path.endswith(".pilots"):
        return str(value.get("role") or value.get("rank") or "")
    if path == "trains":
        return str(value.get("train_number") or value.get("operator") or "")
    if path == "vessels":
        return str(value.get("vessel_name") or value.get("port_of_registry") or "")

    return _canonical_sort_key(value)


def _normalize_metadata_path(path: str) -> str:
    """Normalize list indices in metadata paths.

    Returns:
        str: Path with list indices replaced by [].
    """
    return re.sub(r"\[\d+\]", "[]", path)


def _unwrap_optional_annotation(annotation):
    """Remove None from optional annotations when present.

    Returns:
        object: Annotation without the optional None wrapper when possible.
    """
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return annotation

    non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(non_none_args) == 1:
        return non_none_args[0]
    return annotation


def _collect_literal_string_paths(
    model_cls: type[BaseModel], prefix: str = ""
) -> set[str]:
    """Walk a Pydantic model and collect paths for Literal-backed fields.

    Returns:
        set[str]: Normalized metadata paths for literal-backed fields.
    """
    literal_paths = set()

    for field_name, field_info in model_cls.model_fields.items():
        path = f"{prefix}.{field_name}" if prefix else field_name
        annotation = _unwrap_optional_annotation(field_info.annotation)
        origin = get_origin(annotation)

        if origin is Literal:
            literal_values = [
                value for value in get_args(annotation) if value is not None
            ]
            if literal_values:
                literal_paths.add(path)
            continue

        if origin is list:
            item_annotation = _unwrap_optional_annotation(get_args(annotation)[0])
            if isinstance(item_annotation, type) and issubclass(
                item_annotation, BaseModel
            ):
                literal_paths.update(
                    _collect_literal_string_paths(item_annotation, f"{path}[]")
                )
            continue

        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            literal_paths.update(_collect_literal_string_paths(annotation, path))

    return literal_paths


def _get_all_literal_paths() -> set[str]:
    """Collect all literal paths from all vehicle types (aircraft, trains, vessels).

    Returns:
        set[str]: All literal paths across all three vehicle type hierarchies.
    """
    all_paths = set()
    all_paths.update(_collect_literal_string_paths(AircraftMetadata, "aircraft[]"))
    all_paths.update(_collect_literal_string_paths(TrainMetadata, "trains[]"))
    all_paths.update(_collect_literal_string_paths(VesselMetadata, "vessels[]"))
    all_paths.update(_collect_literal_string_paths(OccurrenceMetadata, "occurrence"))
    return all_paths


LITERAL_PATH_PATTERNS = _get_all_literal_paths()


def _should_use_exact_match(path: str) -> bool:
    """Determine whether a metadata path should use exact comparison.

    Returns:
        bool: True when the path maps to a literal-backed or strict field.
    """
    normalized_path = _normalize_metadata_path(path)
    return normalized_path in LITERAL_PATH_PATTERNS or path in EXACT_MATCH_PATHS


def _compare_metadata_values(path: str, actual, expected, failures: list[str]):  # noqa: PLR0911, PLR0912
    """Recursively compare metadata with simple string thresholds."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            failures.append(
                f"{path} type mismatch: expected dict, got {type(actual).__name__}"
            )
            return

        expected_keys = set(expected)
        actual_keys = set(actual)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)

        for key in missing:
            failures.append(f"{path}.{key} missing in extracted metadata")
        for key in extra:
            failures.append(f"{path}.{key} unexpected in extracted metadata")

        for key in sorted(expected_keys & actual_keys):
            next_path = f"{path}.{key}" if path else key
            _compare_metadata_values(next_path, actual[key], expected[key], failures)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            failures.append(
                f"{path} type mismatch: expected list, got {type(actual).__name__}"
            )
            return

        if len(actual) != len(expected):
            failures.append(
                f"{path} length mismatch: expected {len(expected)}, got {len(actual)}"
            )

        actual_sorted = sorted(actual, key=lambda item: _list_item_sort_key(path, item))
        expected_sorted = sorted(
            expected, key=lambda item: _list_item_sort_key(path, item)
        )
        for idx, (actual_item, expected_item) in enumerate(
            zip(actual_sorted, expected_sorted, strict=False)
        ):
            _compare_metadata_values(
                f"{path}[{idx}]", actual_item, expected_item, failures
            )
        return

    if isinstance(expected, str):
        if _should_use_exact_match(path):
            if actual.lower() != expected.lower():
                failures.append(
                    f"{path} mismatch: expected {expected!r}, got {actual!r}"
                )
            return

        if not isinstance(actual, str):
            failures.append(
                f"{path} type mismatch: expected str, got {type(actual).__name__}"
            )
            return

        similarity = SequenceMatcher(
            None,
            actual,
            expected,
        ).ratio()
        if similarity < METADATA_WEAK_STRING_SIMILARITY_THRESHOLD:
            failures.append(
                f"{path} similarity {similarity:.2f} < {METADATA_WEAK_STRING_SIMILARITY_THRESHOLD}: "
                f"expected {expected!r}, got {actual!r}"
            )
        return

    if isinstance(expected, bool):
        if actual is not expected:
            failures.append(f"{path} mismatch: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, int) and not isinstance(expected, bool):
        if actual != expected:
            failures.append(f"{path} mismatch: expected {expected!r}, got {actual!r}")
        return

    if isinstance(expected, float):
        if actual != expected:
            failures.append(f"{path} mismatch: expected {expected!r}, got {actual!r}")
        return

    if actual != expected:
        failures.append(f"{path} mismatch: expected {expected!r}, got {actual!r}")


# Test data loading functions
def load_json_test_data(filename: str) -> list | dict:
    """Load test data from a JSON file in the tests/data directory.

    Returns:
        list | dict: Parsed JSON data from the specified file.
    """
    test_data_path = Path(__file__).parent / "data" / filename
    with open(test_data_path, encoding="utf-8") as f:
        return json.load(f)


def load_safety_issue_test_cases():
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


def load_recommendation_test_cases():
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


def load_metadata_test_cases():
    """Load metadata test cases and convert to parametrize format.

    Returns:
        list: List of pytest.param objects with proper test IDs.

    Raises:
        ValueError: If a metadata test case has an unknown report mode.
    """
    test_cases = load_json_test_data("metadata_test_cases.json")
    taxonomy_dict = load_event_type_taxonomy(Path("data/event_types.csv"))

    params = []
    for case in test_cases:
        report_id = case["report_id"]
        mode = Modes.get_report_mode_from_id(report_id)
        if mode is None:
            msg = f"Unknown mode for report_id '{report_id}' in metadata test data"
            raise ValueError(msg)
        metadata_model = _build_metadata_model_for_mode(mode, taxonomy_dict)

        # Construct expected metadata using mode-specific model
        # Pass raw expected data dict; Pydantic will construct the correct type
        expected = metadata_model.model_validate(case["expected"])
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
    output_path = Path(pytest.output_config["folder_name"])
    parsed_reports_dc = ParsedReports(output_path)
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
        self, report_id: str, expected: list[SafetyIssueItem]
    ):
        """Test the extraction of safety issues from a report.

        Test both TSB and TAIC as well as pre 2008 ATSb reports.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[SafetyIssueItem]): The expected safety issues.

        """
        report_text = get_report_text(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_", maxsplit=1)[0],
            report_text=report_text,
            safety_issues=True,
            recommendations=False,
            metadata=False,
        )

        extracted = getattr(extracted_data, "safety_issues", [])
        failures = []

        # 1. Check count matches
        if len(extracted) != len(expected):
            failures.append(
                f"Count mismatch: expected {len(expected)} safety issues, got {len(extracted)}"
            )

        # 2. Check each item (count permitting)
        for idx, (extracted_item, expected_item) in enumerate(
            zip(extracted, expected, strict=False)
        ):
            # Check similarity
            similarity = SequenceMatcher(
                None,
                extracted_item.safety_issue.lower(),
                expected_item.safety_issue.lower(),
            ).ratio()

            threshold = 0.95 if expected_item.quality == "exact" else 0.7
            quality_type = expected_item.quality

            if similarity < threshold:
                failures.append(
                    f"Item {idx} ({quality_type}): Text similarity {similarity:.2f} < {threshold}\n"
                    f"  Expected: {expected_item.safety_issue}\n"
                    f"  Got:      {extracted_item.safety_issue}"
                )

            # Check quality matches
            if extracted_item.quality != expected_item.quality:
                failures.append(
                    f"Item {idx}: Quality mismatch\n"
                    f"  Expected: {expected_item.quality}\n"
                    f"  Got:      {extracted_item.quality}"
                )

        # Report all failures at once
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize("report_id, expected", load_recommendation_test_cases())
    def test_recommendation_extraction(  # noqa: PLR6301
        self, report_id: str, expected: list[RecommendationItem]
    ):
        """Test the recommendation extraction of ATSB.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[RecommendationItem]): The expected recommendations.

        """
        report_text = get_report_text(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_", maxsplit=1)[0],
            report_text=report_text,
            safety_issues=False,
            recommendations=True,
            metadata=False,
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
                failures.append(
                    f"Item {idx} ({extracted_item.recommendation_id}): Recipient mismatch\n"
                    f"  Expected: {expected_item.recipient}\n"
                    f"  Got:      {extracted_item.recipient}"
                )

            # Check recommendation text similarity
            similarity = SequenceMatcher(
                None,
                extracted_item.recommendation.lower(),
                expected_item.recommendation.lower(),
            ).ratio()

            if similarity < RECOMMENDATION_SIMILARITY_THRESHOLD:
                failures.append(
                    f"Item {idx} ({extracted_item.recommendation_id}): Recommendation text similarity {similarity:.2f} < {RECOMMENDATION_SIMILARITY_THRESHOLD}\n"
                    f"  Expected: {expected_item.recommendation}\n"
                    f"  Got:      {extracted_item.recommendation}"
                )

            # Check context similarity
            context_similarity = SequenceMatcher(
                None,
                (extracted_item.recommendation_context or "").lower(),
                (expected_item.recommendation_context or "").lower(),
            ).ratio()

            if context_similarity < CONTEXT_SIMILARITY_THRESHOLD:
                failures.append(
                    f"Item {idx} ({extracted_item.recommendation_id}): Context similarity {context_similarity:.2f} < 0.9\n"
                    f"  Expected: {expected_item.recommendation_context}\n"
                    f"  Got:      {extracted_item.recommendation_context}"
                )

        # Report all failures at once
        assert not failures, "\n".join(failures)

    @pytest.mark.parametrize(
        "report_id, expected",
        load_metadata_test_cases(),
    )
    def test_metadata_extraction(  # noqa: PLR6301
        self, report_id: str, expected: BaseModel
    ):
        """Test the extraction of metadata from a report.

        Tests occurrence metadata, aircraft details, pilot information,
        train details, and vessel information as applicable.

        Args:
            report_id (str): The unique identifier for the report.
            expected (BaseModel): The expected metadata structure (mode-specific model).

        """
        report_text = get_report_text(report_id)
        report_mode = Modes.get_report_mode_from_id(report_id)
        event_type_taxonomy_by_mode = load_event_type_taxonomy(
            Path("data/event_types.csv")
        )

        extracted_data = ai_read_report(
            agency_name=report_id.split("_", maxsplit=1)[0],
            report_text=report_text,
            safety_issues=False,
            recommendations=False,
            metadata=True,
            report_mode=report_mode,
            event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        )

        extracted = getattr(extracted_data, "metadata", None)
        if extracted is None:
            pytest.fail("Metadata extraction failed - metadata is None")

        failures = []

        extracted_payload = extracted.model_dump(mode="json")
        expected_payload = expected.model_dump(mode="json")
        _compare_metadata_values("", extracted_payload, expected_payload, failures)

        # Verify that at least one mode-specific item is extracted for the report type
        is_air = "a_" in report_id
        is_rail = "r_" in report_id
        is_marine = "m_" in report_id

        if is_air and len(extracted.aircraft) == 0:
            failures.append("Air accident should have aircraft metadata")
        if is_rail and len(extracted.trains) == 0:
            failures.append("Rail accident should have train metadata")
        if is_marine and len(extracted.vessels) == 0:
            failures.append("Marine accident should have vessel metadata")

        # Report all failures at once
        assert not failures, "\n".join(failures)


_chunking_params = [
    ("TAIC_m_2004_203", 29),
    ("TSB_a_2023_W0096", 55),
    ("TAIC_a_2020_003", 34),
    ("ATSB_r_2010_007", 18),
]


@pytest.mark.parametrize(
    "report_id, num_sections",
    _chunking_params,
)
def test_chunking_into_section(report_id, num_sections):
    """Test that does a basic sanity check on the chunking of a report into sections."""
    report_text = get_report_text(report_id)

    sections = chunk_report_into_sections(report_text)

    assert (
        len(sections) == num_sections
    ), f"Expected {num_sections} sections but got {len(sections)}"


def test_parallel_extraction(tmp_path):
    """Test that we can extract from multiple reports in parallel without issues."""
    ids = {
        "ATSB_a_2000_157": {"si": 7, "recs": 8, "sections": 193},
        "ATSB_r_2014_001": {"si": 0, "recs": 2, "sections": 107},
        "ATSB_a_2007_018": {"si": 4, "recs": 0, "sections": 51},
        "ATSB_m_2007_241": {"si": 4, "recs": 1, "sections": 48},
        "ATSB_m_2022_007": {"si": 0, "recs": 3, "sections": 113},
        "ATSB_r_2010_007": {"si": 0, "recs": 2, "sections": 18},
        "TAIC_a_2020_003": {"si": 0, "recs": 0, "sections": 55},
        "TSB_a_2023_W0096": {"si": 2, "recs": 0, "sections": 34},
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

    # Extract from all reports in parallel
    results = process_reports_parallel(
        parsed_reports_dc=parsed_reports_dc,
        extracted_reports_dc=extracted_reports_dc,
        ai_extraction_config=pytest.config["engine"]["extraction"][
            "ai_extraction_config"
        ],
    )

    # Compare results to expected values
    failures = []
    for report_id, expected in ids.items():
        extracted = results[results["report_id"] == report_id]
        if extracted.empty:
            failures.append(f"{report_id}: No extraction result found")
            continue
        extracted = extracted.iloc[0]

        if len(extracted.safety_issues) != expected["si"]:
            failures.append(
                f"{report_id}: safety_issues expected {expected['si']}, got {len(extracted.safety_issues)}"
            )
        if len(extracted.recommendations) != expected["recs"]:
            failures.append(
                f"{report_id}: recommendations expected {expected['recs']}, got {len(extracted.recommendations)}"
            )
        if len(extracted.sections) != expected["sections"]:
            failures.append(
                f"{report_id}: sections expected {expected['sections']}, got {len(extracted.sections)}"
            )

    assert not failures, "Extraction mismatches:\n  " + "\n  ".join(failures)


def test_process_handle_already_processed(tmp_path):
    """Test that process_reports_parallel only processes new reports and skips already processed ones."""
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
            "sections": [{} for _ in already_processed_ids],
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

    parsed_reports_dc.save(report_texts_df)
    extracted_reports_dc.save(current_extracted_df)

    # Mock the extract_report function to track calls
    with patch("engine.ReportExtracting.extract_report") as mock_extract:
        # Configure mock to return a dict with expected structure
        mock_extract.side_effect = lambda row, config, *_args: {
            "report_id": row["report_id"],
            "safety_issues": [],
            "recommendations": [],
            "metadata": {},
            "sections": {},
        }

        # Process reports
        results_df = process_reports_parallel(
            parsed_reports_dc=parsed_reports_dc,
            extracted_reports_dc=extracted_reports_dc,
            ai_extraction_config=pytest.config["engine"]["extraction"][
                "ai_extraction_config"
            ],
        )

    # Assertions: Verify that extract_report was called only for new reports
    assert (
        mock_extract.call_count == EXPECTED_NEW_REPORT_COUNT
    ), f"extract_report should be called {EXPECTED_NEW_REPORT_COUNT} times (for new reports), but was called {mock_extract.call_count} times"

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
