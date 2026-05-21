"""Typed DataFrame storage classes with built-in validation and I/O.

This module provides classes for managing engine output artifacts. Each class
encapsulates schema validation, file path management, and read/save operations
for a specific type of data produced by the engine pipeline.

No manual read/write of pickle files should be done outside of these classes to ensure consistent validation and error handling. Each class defines the expected columns, and row validation using Pydantic models.
"""

from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from engine.Logging import get_logger

logger = get_logger(__name__)


class OutputDirectoryNotFoundError(FileNotFoundError):
    """Raised when the output directory does not exist."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the error with the missing output directory.

        Args:
            output_dir: Base directory where output artifacts are expected.
        """
        super().__init__(f"Output directory does not exist: {output_dir}")


class ValidationError(ValueError):
    """Raised when dataframe validation fails."""

    def __init__(self, message: str) -> None:
        """Initialize the error with a validation failure message.

        Args:
            message: Details describing the validation failure.
        """
        super().__init__(f"DataFrame validation error: {message}")


class SavedDataFrame(ABC):
    """Base class for all saved dataframe artifacts.

    Each subclass must define:
    - filename: The pickle filename (class variable)
    - Row: Nested Pydantic model for row validation (class)

    Expected columns are automatically derived from Row model field names in order.
    The base class handles all read/save/validate logic automatically.
    All rows are validated (typically fast even for 100k rows).
    """

    filename: ClassVar[str]
    Row: type[BaseModel]  # Nested class defined in each subclass

    def __init__(self, output_dir: Path):
        """Initialize with output directory.

        Args:
            output_dir: Base directory where pickle file will be stored.
        """
        self.output_dir = output_dir
        self.path = output_dir / self.filename

    @property
    def _effective_columns(self) -> list[str]:
        """Get expected columns from Row model field names in order.

        Returns:
            List of column names in order.
        """
        return list(type(self).Row.model_fields.keys())

    def validate(self, df: pd.DataFrame) -> None:
        """Validate dataframe schema and all row values.

        Checks:
        1. Column names match expected_columns exactly (order matters)
        2. All row values conform to row_model (if defined, very fast for 100k rows)

        Args:
            df: DataFrame to validate.

        Raises:
            ValidationError: If validation fails.
        """
        if df.empty:
            logger.warning(f"Validating empty dataframe for {self.filename}")
            return

        # Validate column names and order
        if list(df.columns) != self._effective_columns:
            msg = f"Expected columns {self._effective_columns}, got {list(df.columns)}"
            raise ValidationError(msg)

        # Validate all row values using Row model
        for idx, row in df.iterrows():
            try:
                row_dict = {}
                for key, value in row.to_dict().items():
                    is_missing = pd.isna(value)
                    if isinstance(is_missing, bool):
                        row_dict[key] = None if is_missing else value
                    else:
                        row_dict[key] = value
                type(self).Row.model_validate(row_dict)
            except PydanticValidationError as e:
                # Extract the first error for a cleaner message
                errors = e.errors()
                if errors:
                    first_error = errors[0]
                    field = first_error.get("loc", ("unknown",))[0]
                    msg_detail = first_error.get("msg", str(first_error))
                    msg = f"Row {idx}, field '{field}': {msg_detail} and instead got value '{row_dict.get(field)}'"
                else:
                    msg = f"Row {idx}: validation error"
                raise ValidationError(msg) from e
            except Exception as e:
                msg = f"Row {idx}: {e}"
                raise ValidationError(msg) from e

    def read(self, validate_data: bool = True) -> pd.DataFrame:
        """Read the dataframe from disk with validation.

        Returns:
            pd.DataFrame: The loaded dataframe.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not self.path.exists():
            msg = f"File not found: {self.path}"
            raise FileNotFoundError(msg)

        loaded_df = pd.read_pickle(self.path)
        if validate_data:
            self.validate(loaded_df)
        logger.debug(f"Read {len(loaded_df)} rows from {self.path}")
        return loaded_df

    def read_or_create(self) -> pd.DataFrame:
        """Read the dataframe or return an empty one with the expected schema.

        Returns:
            pd.DataFrame: The loaded dataframe if present, otherwise an empty
                dataframe with expected columns in order.
        """
        if self.path.exists():
            return self.read()

        return pd.DataFrame(columns=self._effective_columns)

    def create_empty(self) -> pd.DataFrame:
        """Create an empty dataframe with the expected schema.

        Returns:
            pd.DataFrame: An empty dataframe with expected columns in order.
        """
        return pd.DataFrame(columns=self._effective_columns)

    def save(self, df: pd.DataFrame) -> None:
        """Save the dataframe to disk with validation.

        Args:
            df: DataFrame to save.

        Raises:
            OutputDirectoryNotFoundError: If output directory does not exist.
        """
        self.validate(df)

        if not self.output_dir.exists():
            raise OutputDirectoryNotFoundError(self.output_dir)

        df.to_pickle(self.path)
        logger.debug(f"Saved {len(df)} rows to {self.path}")

    def exists(self) -> bool:
        """Check if the dataframe file exists on disk.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.path.exists()


class ParsedReports(SavedDataFrame):
    """Manages parsed PDF report text storage.

    Stores the raw text extracted from PDF reports with minimal processing.
    Each row contains a report_id and the full text of the report.
    """

    filename = "parsed_reports.pkl"

    class Row(BaseModel):
        """Schema for a single row in the parsed reports dataframe."""

        report_id: str = Field(..., description="Unique identifier for the report")
        text: str = Field(..., description="Full extracted text from the PDF report")


class ExtractedReports(SavedDataFrame):
    """Manages extracted report data (safety issues, recommendations, sections, metadata).

    Stores structured information extracted from reports using AI and text processing.
    This is the new schema for the report-extracting-rewrite branch.
    """

    filename = "extracted_reports.pkl"

    class Row(BaseModel):
        """Schema for a single row in the extracted reports dataframe."""

        report_id: str = Field(..., description="Unique identifier for the report")
        safety_issues: list[dict[str, Any]] | None = Field(
            default_factory=list,
            description="List of safety issues extracted from the report",
        )
        recommendations: list[dict[str, Any]] | None = Field(
            default_factory=list,
            description="List of recommendations extracted from the report",
        )
        metadata: dict[str, Any] = Field(
            default_factory=dict,
            description="Occurrence metadata including occurrence details and vehicle/vessel/personnel information",
        )
        sections: list[dict[str, str]] = Field(
            default_factory=list,
            description="Report sections chunked by page, with section keys like 'page_1', 'page_2.1', etc.",
        )


class ReportTitles(SavedDataFrame):
    """Manages report metadata (titles, URLs, summaries).

    Stores metadata scraped from agency websites for each report.
    """

    filename = "report_titles.pkl"

    class Row(BaseModel):
        """Schema for a single row in the report titles dataframe."""

        report_id: str
        title: str
        event_type: str | None = None
        investigation_type: Literal["full", "short", "unknown"]
        summary: str | None = None
        misc: dict = Field(default_factory=dict)
        url: str
        agency_id: str


class ATSBWebsiteSafetyIssues(SavedDataFrame):
    """Manages safety issues scraped from the ATSB website.

    Stores safety issues extracted from ATSB's public safety issues database.
    Each row contains a report_id and a safety issue.
    """

    filename = "atsb_website_safety_issues.pkl"

    class Row(BaseModel):
        """Schema for a single row in the ATSB website safety issues dataframe."""

        report_id: str
        safety_issue_id: str
        safety_issue: str
        quality: Literal["exact", "inferred"]


class TSBWebsiteRecommendations(SavedDataFrame):
    """Manages recommendations scraped from the TSB website.

    Stores recommendations extracted from TSB's public recommendations database.
    """

    filename = "tsb_website_recommendations.pkl"

    class Row(BaseModel):
        """Schema for a single row in the TSB website recommendations dataframe."""

        report_id: str
        recommendation_id: str
        recommendation: str | None
        agency_id: str
        current_assessment: Literal[
            "Fully Satisfactory",
            "Satisfactory in Part",
            "Unsatisfactory",
            "Not Yet Assessed",
            "Satisfactory Intent",
            "Unable to Assess",
        ]
        status: Literal["Active", "Closed", "Dormant"]
        watchlist: str
        url: str | None
        made: datetime | None
        recommendation_context: str | None


class TAICWebsiteRecommendations(SavedDataFrame):
    """Manages recommendations scraped from the TAIC website.

    Stores recommendations extracted from TAIC's public recommendations database.
    """

    filename = "taic_website_recommendations.pkl"

    class Row(BaseModel):
        """Schema for a single row in the TAIC website recommendations dataframe."""

        report_id: str
        recommendation_id: str
        made: datetime | None
        agency_id: str
        recipient: str
        recommendation: str
        reply_text: str | None
        url: str


class ATSBWebsiteReportsTable(SavedDataFrame):
    """Manages the ATSB reports table scraped from their website.

    Stores metadata about ATSB reports extracted from their public reports listing.
    """

    filename = "atsb_website_reports_table.pkl"

    class Row(BaseModel):
        """Schema for a single row in the ATSB website reports table dataframe. This is not the direct html table and instead has had the columsn renamed."""

        title: str
        agency_id: str
        url: str
        occurrence_date: str
        year: int
        report_id: str


class TAICWebsiteReportsTable(SavedDataFrame):
    """Manages the TAIC reports table scraped from their website.

    Stores metadata about TAIC reports extracted from their public reports listing.
    """

    filename = "taic_website_reports_table.pkl"

    class Row(BaseModel):
        """Schema for a single row in the TAIC website reports table dataframe."""

        id: str
        year: int


class VectorDBDocumentIDs(SavedDataFrame):
    """Manages the tracking file for documents embedded in the vector database.

    Stores document IDs that have been successfully embedded and added to the
    vector database to avoid re-embedding on subsequent runs.
    """

    filename = "vector_db_document_ids.pkl"

    class Row(BaseModel):
        """Schema for a single row in the vector database document IDs dataframe."""

        document_id: str = Field(
            ..., description="Document ID embedded in vector database"
        )


class DataForVectorDB(SavedDataFrame):
    """SavedDataFrame for the long-format canonical document rows."""

    filename = "complete_data.pkl"

    class Row(BaseModel):
        """Schema for a single row in the complete data for vector database dataframe."""

        report_id: str
        document_id: str
        document: str
        document_type: str
        url: str | None = None
        location: str | None = None
        occurrence_date: datetime | None = None
        occurrence_type: str | None = None
        fatalities: int | None = None
        injuries: int | None = None
        damage: str | None = None
        who_may_benefit: str | None = None
        agency_id: str
        mode: str
        year: int
        agency: str
