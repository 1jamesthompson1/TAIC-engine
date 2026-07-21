"""Tests for Embedding module."""

import hashlib
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine import Embedding, SavedDataFrames


def test_upload_report_text_table(tmp_path: object) -> None:
    """Local LanceDB: upload report_text, verify table exists and is queryable."""
    dc = SavedDataFrames.DataForVectorDB(tmp_path)
    test_report_texts = pd.DataFrame(
        [
            {
                "report_id": "TAIC_m_2024_001",
                "document_id": "TAIC_m_2024_001_text_0",
                "document": "Very long report body…"
                * 10_000,  # trying to simluate a 50 page report
                "document_type": "report_text",
                "url": "https://example.com/r/1",
                "metadata_json": "{}",
                "location": "Auckland",
                "occurrence_date": datetime(2024, 1, 15),
                "occurrence_type": "Accident",
                "fatalities": 0,
                "injuries": 1,
                "agency_id": "TAIC",
                "publication_date": datetime(2024, 6, 1),
                "mode": "m",
                "year": 2024,
                "agency": "TAIC",
            }
        ],
        columns=dc._effective_columns,
    )
    dc.save(test_report_texts)

    vdb = Embedding.VectorDB(
        db_uri=str(tmp_path / "lancedb"),
        model_name=pytest.vector_config["model"]["name"],
        context_limit=pytest.vector_config["model"]["context_limit"],
        table_name="main_test",
    )
    try:
        ids = vdb.upload_report_text_table(dc)
        assert ids is not None and len(ids) == 1

        # Call again with same data — should not duplicate rows
        vdb.upload_report_text_table(dc)
        t = vdb.db.open_table(vdb.report_text_table_name)
        assert t.count_rows() == 1, "Duplicate rows were inserted on second call"

        result = t.to_pandas()
        assert result["document_id"].iloc[0] == "TAIC_m_2024_001_text_0"
        assert result["document"].iloc[0] == "Very long report body…" * 10_000
        assert "vector" not in result.columns
    finally:
        vdb.db.drop_all_tables()


def test_basic_embedding(tmp_path: object) -> None:
    """Test the embedding pipeline using the saved dataframe flow."""
    generated_embeddings: list[list[float]] = []

    def mock_generate_embeddings(
        self: object, texts: object, *args: object, **kwargs: object
    ) -> list[list[float]]:
        if isinstance(texts, np.ndarray):
            if texts.dtype != object:
                msg = "AzureAIEmbeddingFunction only supports input of strings for numpy arrays."
                raise ValueError(msg)
            texts = texts.tolist()

        result = []
        dim = self.ndims()
        for text in texts:
            seed = int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16) % (
                2**32
            )
            rng = np.random.default_rng(seed)
            emb = rng.random(dim).astype(np.float32).tolist()
            result.append(emb)
            generated_embeddings.append(emb)
        return result

    complete_data_dc = SavedDataFrames.DataForVectorDB(tmp_path)

    complete_data = pd.DataFrame(
        [
            {
                "report_id": "TAIC_m_2024_001",
                "document_id": "TAIC_m_2024_001_sum_0",
                "document": "A short report text about an incident.",
                "document_type": "summary",
                "url": "https://example.com/report/1",
                "metadata_json": '{"occurrence": {"location": {"standardized_location": "Auckland, Auckland, Auckland, New Zealand"}, "occurrence_datetime": {"local_datetime": "2024-01-15T10:00"}, "occurrence_type": "Accident", "fatalities": 0, "injuries": 1, "damage_description": "Minor damage", "who_may_benefit": "Operators"}}',
                "location": "Auckland",
                "occurrence_date": datetime(2024, 1, 15),
                "occurrence_type": "Accident",
                "fatalities": 0,
                "injuries": 1,
                "agency_id": "TAIC",
                "publication_date": datetime(2024, 6, 1),
                "mode": "m",
                "year": 2024,
                "agency": "TAIC",
            },
            {
                "report_id": "TAIC_a_2024_002",
                "document_id": "TAIC_a_2024_002_sum_0",
                "document": "Another short report text about a second incident.",
                "document_type": "summary",
                "url": "https://example.com/report/2",
                "metadata_json": '{"occurrence": {"location": {"standardized_location": "Wellington, Wellington, Wellington, New Zealand"}, "occurrence_datetime": {"local_datetime": "2024-02-20T14:00"}, "occurrence_type": "Incident", "fatalities": 0, "injuries": 0, "damage_description": "No damage", "who_may_benefit": "Investigators"}}',
                "location": "Wellington",
                "occurrence_date": datetime(2024, 2, 20),
                "occurrence_type": "Incident",
                "fatalities": 0,
                "injuries": 0,
                "agency_id": "TAIC",
                "publication_date": datetime(2024, 7, 15),
                "mode": "a",
                "year": 2024,
                "agency": "TAIC",
            },
        ],
        columns=complete_data_dc._effective_columns,
    )
    complete_data_dc.save(complete_data)

    test_db_uri = str(tmp_path / "vectordb")
    vector_db = None

    try:
        with patch.object(
            Embedding.AzureAITextEmbeddingFunction,
            "generate_embeddings",
            new=mock_generate_embeddings,
        ):
            vector_db = Embedding.VectorDB(
                db_uri=test_db_uri,
                model_name=pytest.vector_config["model"]["name"],
                context_limit=pytest.vector_config["model"]["context_limit"],
                table_name="all_document_types_test",
            )

            vector_db.process_extracted_reports(
                complete_data_dc,
            )

        assert vector_db.table_name in vector_db.db.list_tables().tables
        assert vector_db.table.count_rows() == len(complete_data)

        stored_ids = (
            vector_db.table.search()
            .select(["document_id"])
            .to_pandas()["document_id"]
            .tolist()
        )
        assert sorted(stored_ids) == sorted(complete_data["document_id"].tolist())

        azure_embeddings = (
            Embedding.EmbeddingFunctionRegistry.get_instance()
            .get("azure-ai-text")
            .create(name=pytest.vector_config["model"]["name"])
        )
        expected_dim = azure_embeddings.ndims()
        assert generated_embeddings, "No embeddings were generated by the mock."
        assert all(
            len(e) == expected_dim for e in generated_embeddings
        ), f"Expected embeddings of length {expected_dim}, got {[len(e) for e in generated_embeddings][:5]}"

    finally:
        if vector_db is not None:
            vector_db.db.drop_all_tables()
