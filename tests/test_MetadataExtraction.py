"""Tests for metadata extraction from accident reports.

This module tests the extraction of occurrence metadata, vehicle details,
and personnel information from accident investigation reports.
"""

import json
import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel
from test_ReportExtracting import get_report_text, load_json_test_data

from engine import Logging, Modes
from engine.ExtractionModels import (
    AircraftMetadata,
    OccurrenceMetadata,
    TrainMetadata,
    VesselMetadata,
    _build_metadata_model_for_mode,  # noqa: PLC2701
)
from engine.ReportExtracting import ai_read_report, load_event_type_taxonomy

logging = Logging.get_logger(__name__)


def _parse_occurrence_local_datetime(value: str) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def _canonical_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _list_item_sort_key(path: str, value: object) -> tuple | str:
    if not isinstance(value, dict):
        return _canonical_sort_key(value)

    if path == "aircraft":
        return str(value.get("registration") or value.get("model") or "")
    if path.endswith(".pilots"):
        return (
            int(value.get("age") or -1),
            str(value.get("responsibility") or ""),
            int(value.get("total_flying_experience") or -1),
            str(value.get("role") or value.get("rank") or ""),
        )
    if path == "trains":
        return str(value.get("train_number") or value.get("operator") or "")
    if path == "vessels":
        return str(value.get("vessel_name") or value.get("port_of_registry") or "")

    return _canonical_sort_key(value)


def _normalize_metadata_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def _unwrap_optional_annotation(annotation: object) -> object:
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
    all_paths = set()
    all_paths.update(_collect_literal_string_paths(AircraftMetadata, "aircraft[]"))
    all_paths.update(_collect_literal_string_paths(TrainMetadata, "trains[]"))
    all_paths.update(_collect_literal_string_paths(VesselMetadata, "vessels[]"))
    all_paths.update(_collect_literal_string_paths(OccurrenceMetadata, "occurrence"))
    return all_paths


LITERAL_PATH_PATTERNS = _get_all_literal_paths()


def _should_use_exact_match(
    path: str,
    exact_match_paths: set[str] | None = None,
) -> bool:
    if exact_match_paths is None:
        exact_match_paths = {
            "occurrence.occurrence_datetime.time_zone",
            "occurrence.occurrence_type",
        }
    normalized_path = _normalize_metadata_path(path)
    return normalized_path in LITERAL_PATH_PATTERNS or path in exact_match_paths


def _compare_metadata_values(  # noqa: PLR0911, PLR0912, PLR0913
    path: str,
    actual: object,
    expected: object,
    failures: list[str],
    *,
    occurrence_local_datetime_path: str = "occurrence.occurrence_datetime.local_datetime",
    datetime_tolerance_minutes: int = 30,
    metadata_weak_string_similarity_threshold: float = 0.25,
    float_relative_tolerance: float = 0.03,
) -> None:
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
        if not isinstance(actual, str):
            failures.append(
                f"{path} type mismatch: expected str, got {type(actual).__name__}"
            )
            return

        if path == occurrence_local_datetime_path:
            actual_dt = _parse_occurrence_local_datetime(actual)
            expected_dt = _parse_occurrence_local_datetime(expected)

            if actual_dt is None or expected_dt is None:
                if actual != expected:
                    failures.append(
                        f"{path} mismatch: expected {expected!r}, got {actual!r}"
                    )
                return

            delta_minutes = abs((actual_dt - expected_dt).total_seconds()) / 60
            if delta_minutes > datetime_tolerance_minutes:
                failures.append(
                    f"{path} mismatch: delta {delta_minutes:.1f} minutes exceeds "
                    f"{datetime_tolerance_minutes} minutes (expected {expected!r}, got {actual!r})"
                )
            return

        if _should_use_exact_match(path):
            if actual.lower() != expected.lower():
                failures.append(
                    f"{path} mismatch: expected {expected!r}, got {actual!r}"
                )
            return

        similarity = SequenceMatcher(
            None,
            actual,
            expected,
        ).ratio()
        if similarity < metadata_weak_string_similarity_threshold:
            failures.append(
                f"{path} similarity {similarity:.2f} < {metadata_weak_string_similarity_threshold}: "
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
        if not isinstance(actual, int | float):
            failures.append(
                f"{path} type mismatch: expected float, got {type(actual).__name__}"
            )
            return

        if not math.isclose(
            float(actual),
            expected,
            rel_tol=float_relative_tolerance,
            abs_tol=0.0,
        ):
            failures.append(
                f"{path} mismatch: expected {expected!r}, got {actual!r} "
                f"(tol: +/- {float_relative_tolerance * 100:.1f}%)"
            )
        return

    if actual != expected:
        failures.append(f"{path} mismatch: expected {expected!r}, got {actual!r}")


def load_metadata_test_cases() -> list:
    """Load metadata test cases from JSON file and prepare parameters for testing.

    Returns:
        A list of pytest parameters: (report_id, expected_model) prepared for
        parametrized tests.

    Raises:
        ValueError: If an unknown mode is encountered for a report_id in the
            metadata test data.
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

        expected = metadata_model.model_validate(case["expected"])
        params.append(pytest.param(report_id, expected, id=f"{report_id}_{case['id']}"))
    return params


@pytest.mark.parametrize(
    "report_id, expected",
    load_metadata_test_cases(),
)
def test_metadata_extraction(
    report_id: str,
    expected: BaseModel,
    agency_id_lookup: dict[str, str],
) -> None:
    """Test the extraction of metadata from a report.

    Tests occurrence metadata, aircraft details, pilot information,
    train details, and vessel information as applicable.

    Args:
        report_id (str): The unique identifier for the report.
        expected (BaseModel): The expected metadata structure (mode-specific model).
        agency_id_lookup (dict[str, str]): Report ID to agency ID mapping.

    """
    report_text = get_report_text(report_id)
    agency_id = agency_id_lookup.get(report_id)
    report_mode = Modes.get_report_mode_from_id(report_id)
    event_type_taxonomy_by_mode = load_event_type_taxonomy(Path("data/event_types.csv"))

    extracted_data = ai_read_report(
        agency_name=report_id.split("_", maxsplit=1)[0],
        report_text=report_text,
        safety_issues=False,
        recommendations=False,
        metadata=True,
        report_mode=report_mode,
        event_type_taxonomy_by_mode=event_type_taxonomy_by_mode,
        report_id=report_id,
        agency_id=agency_id,
    )

    extracted = getattr(extracted_data, "metadata", None)
    if extracted is None:
        pytest.fail("Metadata extraction failed - metadata is None")

    failures = []

    extracted_payload = extracted.model_dump(mode="json")
    expected_payload = expected.model_dump(mode="json")
    _compare_metadata_values("", extracted_payload, expected_payload, failures)

    # Verify that at least one mode-specific item is extracted for the report type
    # Air accidents are omitted due to the presence of ATC only reports.
    is_rail = "r_" in report_id
    is_marine = "m_" in report_id

    if is_rail and len(extracted.trains) == 0:
        failures.append("Rail accident should have train metadata")
    if is_marine and len(extracted.vessels) == 0:
        failures.append("Marine accident should have vessel metadata")

    # Separate failures into critical (datetime, occurence_type, fatalities, injuries)
    # and non-critical (everything else, allowed up to 2 mismatches)
    critical_paths = {
        "occurrence.occurrence_datetime",
        "occurrence.occurrence_type",
        "occurrence.fatalities",
        "occurrence.injuries",
    }
    critical_failures = [
        f for f in failures if any(f.startswith(p) for p in critical_paths)
    ]
    non_critical_failures = [
        f for f in failures if not any(f.startswith(p) for p in critical_paths)
    ]

    if critical_failures:
        assert not critical_failures, (
            "Critical metadata mismatches for report "
            f"{report_id}:\n" + "\n".join(critical_failures)
        )

    if len(non_critical_failures) > 2:  # noqa: PLR2004
        assert not non_critical_failures, (
            f"Too many non-critical metadata mismatches "
            f"({len(non_critical_failures)} > 2) for report "
            f"{report_id}:\n" + "\n".join(non_critical_failures)
        )
    elif non_critical_failures:
        logging.warning(
            f"{len(non_critical_failures)} non-critical metadata mismatches for "
            f"report {report_id}:\n" + "\n".join(non_critical_failures)
        )
