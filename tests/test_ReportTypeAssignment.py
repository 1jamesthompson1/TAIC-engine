"""Tests for ReportTypeAssignment module."""

from pathlib import Path

import pytest

from engine import Modes, ReportTypeAssignment, SavedDataFrames


def test_report_type_assignment(tmp_path):
    """Test the full report type assignment process."""
    output_path = Path(pytest.output_config["folder_name"])

    report_type_assigner = ReportTypeAssignment.ReportTypeAssigner(
        SavedDataFrames.ReportEventTypes(output_path),
        SavedDataFrames.ReportTitles(output_path),
        SavedDataFrames.ParsedReports(output_path),
        SavedDataFrames.AllEventTypes(output_path),
    )
    report_type_assigner.assign_report_types()

    report_event_types_dc = SavedDataFrames.ReportEventTypes(output_path)
    assert report_event_types_dc.exists()

    report_types_df = report_event_types_dc.read()

    assert report_types_df["type"].isna().sum() == 0


@pytest.mark.parametrize(
    "report_title, mode, expected_type",
    [
        pytest.param(
            "Hawker Beechcraft Corporation 1900D, ZK-EAQ cargo door opening in flight, Auckland International Airport, 9 April 2010",
            0,
            "Aircraft loading",
            id="aircraft_loading",
        ),
        pytest.param(
            "Passenger freight ferry 'Aratere,' steering malfunctions, Wellington Harbour and Queen Charlotte Sound, 9 February and 20 February 2005",
            2,
            "Machinery failure",
            id="machinery_failure",
        ),
        pytest.param(
            "Track warrant control irregularities, Woodville and Otane, 18 January 2005",
            1,
            "Safeworking Rule or Procedure Breach",
            id="safeworking_rule_or_procedure_breach",
        ),
        pytest.param(
            "Cessna 185A, ZK-CBY and Tecnam P2002, ZK-WAK Mid-air collision, near Masterton, 16 June 2019",
            0,
            "Aircraft separation",
            id="aircraft_separation",
        ),
    ],
)
def test_single_report_type_assignment(report_title, mode, expected_type, tmp_path):
    """Test report type assignment for a single report."""
    output_path = Path(pytest.output_config["folder_name"])

    report_type_assigner = ReportTypeAssignment.ReportTypeAssigner(
        SavedDataFrames.ReportEventTypes(tmp_path),
        SavedDataFrames.ReportTitles(output_path),
        SavedDataFrames.ParsedReports(output_path),
        SavedDataFrames.AllEventTypes(output_path),
    )

    assert (
        report_type_assigner.assign_report_type(
            report_title, Modes.Mode(mode), False
        ).lower()
        == expected_type.lower()
    )
