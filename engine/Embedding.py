"""Vector database embedding functionality for the TAIC engine.

This module is responsible for uploading all document types to the vector
database and embedding them for search and analysis.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from tqdm import tqdm

from engine.Logging import get_logger
from engine.SavedDataFrames import DataForVectorDB, VectorDBDocumentIDs

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
    ) -> None:
        """Initialize the VectorDB.

        Args:
            db_uri: URI for the LanceDB database.
            model_name: Name of the embedding model to use.
            context_limit: Maximum context limit for the model.
            table_name: Name of the table in the database.

        Raises:
            ValueError: If the existing table's embedding function does not match the specified model.
        """
        self.model_context_limit = context_limit
        self.table_name = table_name
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
            damage: str | None = None
            who_may_benefit: str | None = None

        self.VectorDBSchema = VectorDBSchema

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.table = self._get_or_create_table()

        self.current_table_model = self.table.embedding_functions[
            "vector"
        ].function.name
        if self.current_table_model != model_name:
            msg = f"Existing table {self.table_name} has embedding function {self.current_table_model} which does not match the specified model {model_name}. Please specify a different table name or delete the existing table if you want to use a different embedding model."
            raise ValueError(msg)

    def _get_or_create_table(self) -> Table:
        """Get existing table or create new one.

        Returns:
            The LanceDB table object.
        """
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)
        table = self.db.create_table(
            self.table_name, data=None, schema=self.VectorDBSchema, mode="create"
        )

        # Create FTS index for text search
        try:
            table.create_fts_index(
                field_names="document",
                use_tantivy=False,
                language="English",
                ascii_folding=True,
                with_position=True,
                remove_stop_words=False,
                replace=True,
            )
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

        try:
            table.create_scalar_index("document_id")
        except Exception as e:
            logger.warning(f"Could not create scalar index on document_id: {e}")

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

    def add_documents(
        self, documents_df: pd.DataFrame, document_column_name: str = "document"
    ) -> pd.Series | None:
        """Add documents to the vector database with generated embeddings.

        This processes a dataframe of documents and generates embeddings for all
        documents that don't have embeddings in the dataframe. It uses
        multithreading to speed up the process.

        Args:
            documents_df: The dataframe containing documents to add.
            document_column_name: The name of the column that contains the documents.

        Returns:
            pd.Series: Series of document IDs that were added, or None if empty.

        Raises:
            ValueError: If dataframe columns don't match the expected schema.
        """
        # Validate dataframe has required columns (order not required)
        expected_cols = list(self.VectorDBSchema.model_fields.keys())[1:]
        missing = [c for c in expected_cols if c not in documents_df.columns.tolist()]
        if missing:
            msg = (
                f"Dataframe is missing required columns {missing}. "
                f"Expected at least: {expected_cols}"
            )
            raise ValueError(msg)

        # Get document lengths
        token_length_column_name = f"{document_column_name}_token_length"
        documents_df = self.tokenize_documents(
            documents_df, document_column_name, token_length_column_name
        )

        logger.info(
            f"There are a total of {documents_df[token_length_column_name].sum()} tokens in {len(documents_df)} documents"
        )

        to_drop = pd.Series(
            documents_df[token_length_column_name] < self.model_context_limit * 2
        )

        documents_df = documents_df.loc[to_drop]

        logger.info(
            f"Dropping documents with more than {self.model_context_limit * 2} tokens which is {len(to_drop) - sum(to_drop)} documents"
        )

        if documents_df.empty:
            return None

        # Truncate all documents to just below the context limit
        documents_df.loc[:, document_column_name] = documents_df.apply(
            lambda x: x[document_column_name][: self.model_context_limit - 50],
            axis=1,
        )

        num_batches = min(os.cpu_count() or 1, len(documents_df))

        # Split the dataframe into batches based on token length
        # Reorder columns so they match the table schema (token length kept for batching)
        documents_df = documents_df.copy()

        df_sorted = documents_df.sort_values(
            token_length_column_name, ascending=False
        ).reset_index(drop=True)

        batches = [[] for _ in range(num_batches)]
        batch_token_counts = [0] * num_batches

        for idx, row in df_sorted.iterrows():
            min_batch_idx = min(range(num_batches), key=lambda i: batch_token_counts[i])

            batches[min_batch_idx].append(idx)
            batch_token_counts[min_batch_idx] += row[token_length_column_name]

        # Convert to DataFrames and drop token length column
        batches = [
            df_sorted.iloc[batch_indices].drop(token_length_column_name, axis=1)
            for batch_indices in batches
        ]

        for i, (batch, token_count) in enumerate(
            zip(batches, batch_token_counts, strict=False)
        ):
            logger.debug(f"Batch {i}: {len(batch)} documents, {token_count} tokens")

        def add_documents_to_db(batch: pd.DataFrame) -> object:
            pa_table = pa.Table.from_pandas(
                batch,
                schema=pa.schema(
                    [field for field in self.table.schema if field.name != "vector"]
                ),
            )

            return self.table.add(pa_table, mode="append")

        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = {
                executor.submit(add_documents_to_db, batch): i
                for i, batch in enumerate(batches)
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

        return documents_df["document_id"]

    def process_extracted_reports(
        self,
        complete_data_dc: DataForVectorDB,
        already_embedded_ids_dc: VectorDBDocumentIDs,
    ) -> None:
        """Process extracted reports and generate embeddings.

        Args:
            complete_data_dc: The complete data for vector database insertion.
            already_embedded_ids_dc: A list of document IDs that have already been embedded.
        """
        logger.welcome(
            "Embedding Reports",
            {
                "complete_data_dc": complete_data_dc.path,
                "already_embedded_ids_dc": already_embedded_ids_dc.path,
                "table_name": self.table_name,
                "db_uri": self.db.uri,
                "model_name": self.current_table_model,
                "context_limit": self.model_context_limit,
            },
        )

        complete_data = complete_data_dc.read()

        already_embedded_ids = already_embedded_ids_dc.read_or_create()["document_id"]

        if len(already_embedded_ids) == 0:
            logger.warning(
                f"No already embedded document IDs found, starting fresh embedding process. If you have previously embedded documents, make sure to save the document IDs to {already_embedded_ids_dc.path} to avoid re-embedding."
            )

        data_to_add = complete_data[
            ~complete_data["document_id"].isin(already_embedded_ids)
        ]

        current_table_version = self.table.version

        try:
            added_document_ids = self.add_documents(
                data_to_add,
            )

            logger.info(
                f"Added {len(added_document_ids)} documents to the vector database table {self.table_name}."
            )
            already_embedded_ids_dc.save(
                pd.DataFrame(
                    {
                        "document_id": pd.concat(
                            [already_embedded_ids, added_document_ids]
                        )
                    }
                )
            )

        except Exception:
            logger.exception(
                "Error during adding of documents, going to restore previous state of the table"
            )
            self.table = self.table.restore(current_table_version)
            raise

        logger.info("Finished embedding all reports.")
        self.table.optimize(cleanup_older_than=timedelta(days=14))
        logger.info("Optimized the table and cleaned up older entries.")
