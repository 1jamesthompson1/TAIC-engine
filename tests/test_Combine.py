"""Tests for the Combine module, which is responsible for combing all of the scraped and extracted dataframes togather and making it in a long format that is suitable for indexing in a vectordb."""

from pathlib import Path

import pandas as pd
import pytest

from engine import Combine, SavedDataFrames


def test_metadata_expansion() -> None:
    """Test that the metadata expansion function correctly expands occurrence metadata into columns."""
    extracted_reports = SavedDataFrames.ExtractedReports(
        Path(pytest.output_config["folder_name"])
    ).read()

    expanded_df = Combine.expand_extracted_report_metadata(extracted_reports)

    assert isinstance(expanded_df, pd.DataFrame), "Output should be a DataFrame"

    # Schema to compare against
    expected_schema = {
        "report_id": str,
        "location": str | None,
        "occurrence_date": pd.Timestamp | None,
        "occurrence_type": str,
        "fatalities": int,
        "injuries": int,
        "damage": str,
        "who_may_benefit": (str, type(None)),
    }

    # Check data types row by row to get more informative error messages which includes the entire row id.
    for _, row in expanded_df.iterrows():
        for col, expected_type in expected_schema.items():
            assert (
                col in expanded_df.columns
            ), f"Column '{col}' is missing from the output DataFrame"
            value = row[col]
            if isinstance(expected_type, tuple):
                assert isinstance(
                    value, expected_type
                ), f"Row {row.report_id}: Value '{value}' in column '{col}' should be one of types {expected_type}"
            else:
                assert isinstance(
                    value, expected_type
                ), f"Row {row.report_id}: Value '{value}' in column '{col}' should be of type {expected_type}"


def test_metadata_combination() -> None:
    """Test that the metadata combination function correctly combines the extracted and scraped metadata, prioritizing scraped event type over extracted accident type."""
    output_path = Path(pytest.output_config["folder_name"])

    extracted_metadata = SavedDataFrames.ExtractedReports(output_path).read()

    report_titles = SavedDataFrames.ReportTitles(output_path).read()

    expanded_extracted_metadata = Combine.expand_extracted_report_metadata(
        extracted_metadata
    )

    combined_metadata = Combine.create_complete_report_metadata(
        expanded_extracted_metadata, report_titles
    )

    assert isinstance(combined_metadata, pd.DataFrame), "Output should be a DataFrame"

    # Check that the combined metadata has all the columns from the extracted metadata plus the url and agency_id from the scraped metadata, and that the occurrence_type column is present.
    expected_columns = set(expanded_extracted_metadata.columns) | {
        "url",
        "agency_id",
        "occurrence_type",
        "mode",
        "agency",
        "year",
    }
    assert (
        set(combined_metadata.columns) == expected_columns
    ), f"Combined metadata should have columns {expected_columns}, but got {set(combined_metadata.columns)}"

    # Number of rows should be the same as the inner product of the two sets of report ids
    combined_report_ids = set(expanded_extracted_metadata["report_id"].dropna()) & set(
        report_titles["report_id"].dropna()
    )
    assert (
        len(combined_metadata) == len(combined_report_ids)
    ), f"Combined metadata should have {len(combined_report_ids)} rows, but got {len(combined_metadata)}"


def test_recommendation_combination() -> None:
    """Test that the recommendation combination function correctly combines the extracted and scraped recommendations, and that the resulting dataframe has the expected columns."""
    atsb_extracted = SavedDataFrames.ExtractedReports(
        Path(pytest.output_config["folder_name"])
    ).read()

    tsb_scraped = SavedDataFrames.TSBWebsiteRecommendations(
        Path(pytest.output_config["folder_name"])
    ).read()

    taic_scraped = SavedDataFrames.TAICWebsiteRecommendations(
        Path(pytest.output_config["folder_name"])
    ).read()

    combined_recs = Combine.combine_recommendations(
        atsb_extracted, tsb_scraped, taic_scraped
    )

    assert isinstance(combined_recs, pd.DataFrame), "Output should be a DataFrame"

    expected_columns = {"report_id", "document_id", "url", "document", "document_type"}
    assert (
        set(combined_recs.columns) == expected_columns
    ), f"Combined recommendations should have columns {expected_columns}, but got {set(combined_recs.columns)}"

    expected_length = (
        len(
            atsb_extracted.explode("recommendations").dropna(subset=["recommendations"])
        )
        + len(tsb_scraped)
        + len(taic_scraped)
    )
    assert (
        len(combined_recs) == expected_length
    ), f"Combined recommendations should have {expected_length} rows, but got {len(combined_recs)}"


def test_safety_issue_combination() -> None:
    """Test that the safety issue combination function correctly combines the extracted and scraped safety issues, and that the resulting dataframe has the expected columns."""
    atsb_scraped_si = SavedDataFrames.ATSBWebsiteSafetyIssues(
        Path(pytest.output_config["folder_name"])
    ).read()

    extracted_si = SavedDataFrames.ExtractedReports(
        Path(pytest.output_config["folder_name"])
    ).read()

    combined_si = Combine.combine_safety_issues(extracted_si, atsb_scraped_si)

    assert isinstance(combined_si, pd.DataFrame), "Output should be a DataFrame"

    expected_columns = {"report_id", "document_id", "document", "document_type"}
    assert (
        set(combined_si.columns) == expected_columns
    ), f"Combined safety issues should have columns {expected_columns}, but got {set(combined_si.columns)}"

    expected_length = len(
        extracted_si.explode("safety_issues").dropna(subset=["safety_issues"])
    ) + len(atsb_scraped_si)

    assert (
        len(combined_si) == expected_length
    ), f"Combined safety issues should have {expected_length} rows, but got {len(combined_si)}"

    test_report = combined_si[combined_si["report_id"] == "TAIC_m_2004_205"]

    assert test_report["document_id"].tolist() == [
        "TAIC_m_2004_205_si_1",
        "TAIC_m_2004_205_si_2",
    ], "Safety issues from the same report should have unique document IDs"


def test_creation_of_long_data_format(tmp_path: object) -> None:
    """Test that the function to create the long data format correctly combines all the different dataframes into a single long-format dataframe with the expected columns."""
    # For this test we will just check that the function runs and produces a dataframe with the expected columns, since the actual content of the dataframe will depend on the input data and the combination logic which is tested in other tests.

    pytest_output_path = Path(pytest.output_config["folder_name"])

    output_dc = SavedDataFrames.DataForVectorDB(tmp_path)

    Combine.create_long_data_format(
        Combine.LongDataFormatDCs(
            parsed_reports_dc=SavedDataFrames.ParsedReports(pytest_output_path),
            extracted_reports_dc=SavedDataFrames.ExtractedReports(pytest_output_path),
            report_titles_dc=SavedDataFrames.ReportTitles(pytest_output_path),
            atsb_safety_issues_dc=SavedDataFrames.ATSBWebsiteSafetyIssues(
                pytest_output_path
            ),
            tsb_recommendations_dc=SavedDataFrames.TSBWebsiteRecommendations(
                pytest_output_path
            ),
            taic_recommendations_dc=SavedDataFrames.TAICWebsiteRecommendations(
                pytest_output_path
            ),
        ),
        output_dc,
    )

    output = output_dc.read()

    assert isinstance(output, pd.DataFrame), "Output should be a DataFrame"

    # Makes sure that important columns are non null
    important_columns = ["report_id", "document_id", "document"]
    for col in important_columns:
        assert (
            col in output.columns
        ), f"Column '{col}' is missing from the output DataFrame"
        assert (
            output[col].notna().all()
        ), f"Column '{col}' should not have any null values"
