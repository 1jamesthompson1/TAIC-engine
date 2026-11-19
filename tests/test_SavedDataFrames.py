"""Tests for typed dataframe storage and schema evolution."""

from pathlib import Path

import pandas as pd

from engine.SavedDataFrames import ExtractedReports


def test_read_adds_missing_optional_extraction_info_column(tmp_path: Path) -> None:
    """Older pickles without extraction_info remain readable."""
    old_df = pd.DataFrame(
        {
            "report_id": ["TAIC_m_2024_001"],
            "safety_issues": [[]],
            "recommendations": [[]],
            "metadata": [{}],
            "sections": [[]],
        }
    )
    old_df.to_pickle(tmp_path / "extracted_reports.pkl")

    loaded = ExtractedReports(tmp_path).read()

    assert list(loaded.columns) == list(ExtractedReports.Row.model_fields)
    assert loaded["extraction_info"].iloc[0] == {}
