"""Vector database embedding functionality for the TAIC engine.

This module is responsible for uploading all document types to the vector
database and embedding them for search and analysis.
"""

import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import ClassVar

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import tiktoken
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import TEXT
from lancedb.index import FTS, BTree
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from tqdm import tqdm

from engine.Logging import get_logger
from engine.SavedDataFrames import DataForVectorDB

logger = get_logger(__name__)


@register("azure-ai-text")
class AzureAITextEmbeddingFunction(TextEmbeddingFunction):
    """An embedding function that uses the AzureAI API.

    https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme?view=azure-python-preview

    - AZURE_AI_ENDPOINT: The endpoint URL for the AzureAI service.
    - AZURE_AI_API_KEY: The API key for the AzureAI service.

    Parameters
    ----------
    - name: str
        The name of the model you want to use from the model catalog.


    Examples:
    import lancedb
    import pandas as pd
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import get_registry

    model = get_registry().get("azure-ai-text").create(name="embed-v-4-0")

    class TextModel(LanceModel):
        text: str = model.SourceField()
        vector: Vector(model.ndims()) = model.VectorField()

    df = pd.DataFrame({"text": ["hello world", "goodbye world"]})
    db = lancedb.connect("lance_example")
    tbl = db.create_table("test", schema=TextModel, mode="overwrite")

    tbl.add(df)
    rs = tbl.search("hello").limit(1).to_pandas()
    #           text                                             vector  _distance
    # 0  hello world  [-0.018188477, 0.0134887695, -0.013000488, 0.0...   0.841431
    """

    name: str
    client: ClassVar = None

    def ndims(self) -> int:
        """Get the embedding dimensions for the model.

        Returns:
            int: The number of dimensions for embeddings.

        Raises:
            ValueError: If model name is not recognized.
        """
        if self.name == "embed-v-4-0":
            return 1536
        if self.name in {"Cohere-embed-v3-english", "Cohere-embed-v3-multilingual"}:
            return 1024
        if self.name == "text-embedding-ada-002":
            return 1536
        if self.name == "text-embedding-3-large":
            return 3072
        if self.name == "text-embedding-3-small":
            return 1536
        msg = f"Unknown model name: {self.name}"
        raise ValueError(msg)

    def compute_query_embeddings(self, query: str, **kwargs: object) -> list[np.array]:
        """Compute embeddings for a query.

        Args:
            query: The query string to embed.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[np.array]: List of embedding arrays for the query.
        """
        return self.compute_source_embeddings(query, input_type="query")

    def compute_source_embeddings(
        self, texts: TEXT, **kwargs: object
    ) -> list[np.array]:
        """Compute embeddings for source texts.

        Args:
            texts: The texts to embed.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[np.array]: List of embedding arrays.
        """
        texts = self.sanitize_input(texts)
        input_type = (
            kwargs.get("input_type") or "document"
        )  # assume source input type if not passed by `compute_query_embeddings`
        return self.generate_embeddings(texts, input_type=input_type)

    def generate_embeddings(
        self, texts: list[str] | np.ndarray, **kwargs: object
    ) -> list[np.array]:
        """Get the embeddings for the given texts.

        Args:
            texts: list[str] or np.ndarray of strings to embed.
            **kwargs: Arbitrary keyword arguments including:
                input_type: Optional input type for embeddings.
                truncation: Optional boolean to truncate texts.

        Returns:
            list[np.array]: List of embedding arrays for the provided texts.

        Raises:
            ValueError: If input is numpy array with non-string dtype.
        """
        AzureAITextEmbeddingFunction._init_client()

        if isinstance(texts, np.ndarray):
            if texts.dtype != object:
                msg = "AzureAIEmbeddingFunction only supports input of strings for numpy arrays."
                raise ValueError(msg)
            texts = texts.tolist()

        batch_size = 96
        embeddings = []
        for i in range(0, len(texts), batch_size):
            rs = AzureAITextEmbeddingFunction.client.embed(
                input=texts[i : i + batch_size],
                model=self.name,
                dimensions=self.ndims(),
                **kwargs,
            )
            embeddings.extend(emb.embedding for emb in rs.data)
        return embeddings

    @staticmethod
    def _init_client() -> None:
        """Initialize the Azure AI embeddings client.

        Raises:
            ValueError: If required environment variables are not set.
        """
        if AzureAITextEmbeddingFunction.client is None:
            if os.environ.get("AZURE_AI_API_KEY") is None:
                msg = "AZURE_AI_API_KEY not found in environment variables"
                raise ValueError(msg)
            if os.environ.get("AZURE_AI_ENDPOINT") is None:
                msg = "AZURE_AI_ENDPOINT not found in environment variables"
                raise ValueError(msg)

            AzureAITextEmbeddingFunction.client = EmbeddingsClient(
                endpoint=os.environ["AZURE_AI_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_AI_API_KEY"]),
            )


class VectorDB:
    """Vector database for managing document embeddings.

    This class handles adding documents to a vector database, managing
    embeddings, and providing search functionality.
    """

    def __init__(
        self,
        db_uri: str,
        model_name: str,
        context_limit: int,
        table_name: str,
        report_text_table_name: str | None = None,
    ) -> None:
        """Initialize the VectorDB.

        Args:
            db_uri: URI for the LanceDB database.
            model_name: Name of the embedding model to use.
            context_limit: Maximum context limit for the model.
            table_name: Name of the table in the database.
            report_text_table_name: Name of the separate table for raw report_text
                (no embeddings). Defaults to "{table_name}_report_text".

        Raises:
            ValueError: If the existing table's embedding function does not match the specified model.
        """
        self.model_context_limit = context_limit
        self.table_name = table_name
        self.report_text_table_name = (
            report_text_table_name or f"{table_name}_report_text"
        )
        self.db = lancedb.connect(db_uri)
        azure_embeddings = (
            EmbeddingFunctionRegistry.get_instance()
            .get("azure-ai-text")
            .create(name=model_name)
        )

        # This is effectively a rewrite of the SavedDataFrame 'DataForVectorDB'.
        class VectorDBSchema(LanceModel):
            vector: Vector(azure_embeddings.ndims()) = azure_embeddings.VectorField()
            document: str = azure_embeddings.SourceField()
            document_id: str
            report_id: str
            year: int
            mode: str
            agency: str
            agency_id: str
            url: str | None = None
            document_type: str
            location: str | None = None
            occurrence_date: datetime | None = None
            occurrence_type: str | None = None
            fatalities: int | None = None
            injuries: int | None = None
            metadata_json: str | None = None
            publication_date: datetime | None = None

        self.VectorDBSchema = VectorDBSchema

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.table = self.get_or_create_table(
            self.table_name, schema=self.VectorDBSchema
        )
        schema = pa.schema([f for f in self.table.schema if f.name != "vector"])
        self.report_text_table = self.get_or_create_table(
            self.report_text_table_name, schema=schema
        )
        self.current_table_model = self.table.embedding_functions[
            "vector"
        ].function.name
        if self.current_table_model != model_name:
            msg = f"Existing table {self.table_name} has embedding function {self.current_table_model} which does not match the specified model {model_name}. Please specify a different table name or delete the existing table if you want to use a different embedding model."
            raise ValueError(msg)

    @staticmethod
    def _create_indices(table: Table) -> None:
        """Create FTS and BTree indices on the document and document_id columns.

        Args:
            table: The LanceDB table to create indices on.
        """
        try:
            table.create_index(
                "document",
                config=FTS(
                    with_position=True,
                    language="English",
                    ascii_folding=True,
                    remove_stop_words=False,
                ),
                replace=True,
            )
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

        try:
            table.create_index("document_id", config=BTree())
        except Exception as e:
            logger.warning(f"Could not create scalar index on document_id: {e}")

    def get_or_create_table(
        self, table_name: str, schema: pa.Schema | type[LanceModel]
    ) -> Table:
        """Get existing table or create new one with indices.

        Args:
            table_name: Name of the table in the database.
            schema: PyArrow schema or LanceModel schema for the table.

        Returns:
            The LanceDB table object.
        """
        if table_name in self.db.list_tables().tables:
            return self.db.open_table(table_name)
        table = self.db.create_table(
            table_name, data=None, schema=schema, mode="create"
        )
        self._create_indices(table)
        return table

    def tokenize_documents(
        self, df: pd.DataFrame, document_column_name: str, tokenization_column_name: str
    ) -> pd.DataFrame:
        """Tokenize documents and add token counts to dataframe.

        Args:
            df: The dataframe containing documents.
            document_column_name: Name of the column with documents.
            tokenization_column_name: Name of the column to add with token counts.

        Returns:
            pd.DataFrame: The dataframe with token counts added.
        """
        if tokenization_column_name not in df.columns:
            df[tokenization_column_name] = df[document_column_name].apply(
                lambda x: len(self.tokenizer.encode(x))
            )
        else:
            df[tokenization_column_name] = df.apply(
                lambda x: (
                    len(self.tokenizer.encode(x[document_column_name]))
                    if not isinstance(x[tokenization_column_name], int)
                    else x[tokenization_column_name]
                ),
                axis=1,
            )

        return df

    def upload_report_text_table(
        self,
        complete_data_dc: DataForVectorDB,
    ) -> pd.Series | None:
        """Upload report_text documents to a separate table without embeddings.

        Reads the complete data from the given dataclass, filters for report_text
        documents, and upserts them into a non-vector LanceDB table using the same
        schema as the main vector table (minus the vector column).

        Uses merge insert on ``document_id`` so new docs are inserted and existing
        ones are updated in place.

        Args:
            complete_data_dc: DataForVectorDB dataclass holding all documents.

        Returns:
            pd.Series: Series of document IDs that were added/updated, or None if empty.
        """
        logger.welcome(
            "Uploading report_text documents to separate table",
            {
                "complete_data_dc": complete_data_dc.path,
                "report_text_table_name": self.report_text_table_name,
                "db_uri": self.db.uri,
            },
        )

        complete_data = complete_data_dc.read()
        report_text_docs = complete_data[
            complete_data["document_type"] == "report_text"
        ]

        if report_text_docs.empty:
            logger.info("No report_text documents to upload.")
            return None

        cols = [f.name for f in self.report_text_table.schema]
        to_upsert = report_text_docs[cols].copy()
        pa_table = pa.Table.from_pandas(to_upsert, schema=self.report_text_table.schema)

        pre_row_count = self.report_text_table.count_rows()

        self.report_text_table.merge_insert(
            "document_id"
        ).when_not_matched_insert_all().execute(pa_table)

        inserted_rows_count = self.report_text_table.count_rows()
        logger.info(
            f"Inserted {inserted_rows_count - pre_row_count} new report_text documents out of a total of {len(to_upsert)} documents into {self.report_text_table_name}."
        )
        return to_upsert["document_id"]

    def add_documents(
        self, documents_df: pd.DataFrame, document_column_name: str = "document"
    ) -> Generator[pd.Series, None, None]:
        """Add documents to the vector database, yielding IDs batch by batch.

        Processes the dataframe in token-balanced batches, adding each to the
        LanceDB table and yielding the document IDs as each batch is saved.
        This lets callers persist progress incrementally so a failure mid-way
        only loses the current batch, not everything.

        Args:
            documents_df: The dataframe containing documents to add.
            document_column_name: The name of the column that contains the documents.

        Yields:
            pd.Series: Document IDs successfully added for each batch.

        Raises:
            ValueError: If dataframe columns don't match the expected schema.
        """
        expected_cols = list(self.VectorDBSchema.model_fields.keys())[1:]
        missing = [c for c in expected_cols if c not in documents_df.columns.tolist()]
        if missing:
            msg = (
                f"Dataframe is missing required columns {missing}. "
                f"Expected at least: {expected_cols}"
            )
            raise ValueError(msg)

        documents_df = documents_df.copy()

        token_length_column_name = f"{document_column_name}_token_length"
        self.tokenize_documents(
            documents_df, document_column_name, token_length_column_name
        )

        logger.info(
            f"There are a total of {documents_df[token_length_column_name].sum()} tokens in {len(documents_df)} documents"
        )

        within_limit = (
            documents_df[token_length_column_name] < self.model_context_limit * 2
        )
        documents_df = documents_df.loc[within_limit]

        logger.info(
            f"Dropping documents with more than {self.model_context_limit * 2} tokens "
            f"which is {len(within_limit) - within_limit.sum()} documents"
        )

        # Filter out empty document strings — Azure AI rejects empty inputs
        empty_docs = documents_df[
            documents_df[document_column_name].isna()
            | (documents_df[document_column_name].str.strip() == "")
        ]
        if not empty_docs.empty:
            logger.warning(
                f"Dropping {len(empty_docs)} documents with empty text before embedding:\n"
                f"{empty_docs[['report_id', 'document_id', 'document_type']].to_csv(index=False)}"
            )
            documents_df = documents_df.drop(empty_docs.index)

        if documents_df.empty:
            return

        documents_df.loc[:, document_column_name] = documents_df.apply(
            lambda x: x[document_column_name][: self.model_context_limit - 50],
            axis=1,
        )

        num_batches = min(os.cpu_count() or 1, len(documents_df))

        df_sorted = documents_df.sort_values(
            token_length_column_name, ascending=False
        ).reset_index(drop=True)

        batches: list[list[int]] = [[] for _ in range(num_batches)]
        batch_token_counts = [0] * num_batches

        for idx, row in df_sorted.iterrows():
            min_batch_idx = min(range(num_batches), key=lambda i: batch_token_counts[i])
            batches[min_batch_idx].append(idx)
            batch_token_counts[min_batch_idx] += row[token_length_column_name]

        batch_dfs = [
            df_sorted.iloc[batch_indices].drop(token_length_column_name, axis=1)
            for batch_indices in batches
        ]

        for i, (batch_df, token_count) in enumerate(
            zip(batch_dfs, batch_token_counts, strict=False)
        ):
            logger.debug(f"Batch {i}: {len(batch_df)} documents, {token_count} tokens")

        lance_schema = pa.schema(
            [field for field in self.table.schema if field.name != "vector"]
        )

        def add_batch(batch_df: pd.DataFrame) -> object:
            pa_table = pa.Table.from_pandas(batch_df, schema=lance_schema)
            return (
                self.table.merge_insert("document_id")
                .when_not_matched_insert_all()
                .execute(pa_table)
            )

        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            future_to_ids = {
                executor.submit(add_batch, batch_df): batch_df["document_id"]
                for batch_df in batch_dfs
            }

            for future in tqdm(as_completed(future_to_ids), total=len(future_to_ids)):
                future.result()
                yield future_to_ids[future]

    def process_extracted_reports(
        self,
        complete_data_dc: DataForVectorDB,
    ) -> None:
        """Process extracted reports and generate embeddings.

        Embeds all document types (except report_text, which is filtered out) into
        the main vector table. report_text documents should be uploaded separately
        via upload_report_text_table() before calling this.

        Uses the vector table itself to track which document IDs have already been
        embedded, so it's safe to call multiple times.

        Args:
            complete_data_dc: The complete data for vector database insertion.
        """
        logger.welcome(
            "Embedding Reports",
            {
                "complete_data_dc": complete_data_dc.path,
                "table_name": self.table_name,
                "db_uri": self.db.uri,
                "model_name": self.current_table_model,
                "context_limit": self.model_context_limit,
            },
        )

        complete_data = complete_data_dc.read()

        dupe_ids = complete_data["document_id"].value_counts()
        dupe_ids = dupe_ids[dupe_ids > 1]
        if len(dupe_ids) > 0:
            logger.warning(
                f"complete_data has {len(dupe_ids)} duplicate document_ids "
                f"({dupe_ids.sum() - len(dupe_ids)} extra rows). "
                "Keeping last occurrence per document_id."
            )
            complete_data = (
                complete_data.groupby("document_id", sort=False)
                .tail(1)
                .reset_index(drop=True)
            )

        already_embedded_ids = pd.Series(
            self.table.search()
            .select(["document_id"])
            .to_pandas()["document_id"]
            .unique()
        )

        if len(already_embedded_ids) == 0:
            logger.warning(
                "No documents found in vector table, starting fresh embedding process."
            )

        data_to_add = complete_data[
            ~complete_data["document_id"].isin(already_embedded_ids)
        ]

        # Exclude report_text — too long to embed, stored separately
        data_to_embed = data_to_add[data_to_add["document_type"] != "report_text"]

        if data_to_embed.empty:
            logger.info("No new documents to embed.")
            return

        all_added_ids: list[pd.Series] = []

        try:
            for batch_ids in self.add_documents(data_to_embed):
                all_added_ids.append(batch_ids)
                logger.info(
                    f"Added {len(batch_ids)} documents to the vector database table {self.table_name}."
                )
        except Exception:
            saved_count = len(pd.concat(all_added_ids)) if all_added_ids else 0
            logger.exception(
                "Error during adding of documents. "
                f"{saved_count} documents were already saved."
            )
            raise

        total_added = len(pd.concat(all_added_ids))
        logger.info(
            f"Added {total_added} documents to the vector database table {self.table_name}."
        )

        logger.info("Finished embedding all reports.")
        tag_name = f"engine-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.table.tags.create(tag_name, self.table.version)
        logger.info(f"Tagged version {self.table.version} as '{tag_name}'.")
        # Using these deprecated methods as bug in 0.34: https://github.com/lance-format/lance/issues/7653 wiating for 0.35 release to fix it
        self.table.cleanup_old_versions()
        self.table.compact_files()
        # self.table.optimize(cleanup_older_than=timedelta(days=14))  # noqa: ERA001
        logger.info(
            f"Optimized table {self.table_name} by cleaning up old versions and compacting files."
        )
