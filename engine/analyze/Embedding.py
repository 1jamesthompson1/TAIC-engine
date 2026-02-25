"""Vector database embedding functionality for the TAIC engine.

This module is responsible for uploading all document types to the vector
database and embedding them for search and analysis.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
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
from tqdm import tqdm

from engine.utils.logging_config import get_logger

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

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> list[np.array]:
        """Compute embeddings for a query.

        Args:
            query: The query string to embed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            list[np.array]: List of embedding arrays for the query.
        """
        return self.compute_source_embeddings(query, input_type="query")

    def compute_source_embeddings(self, texts: TEXT, *args, **kwargs) -> list[np.array]:
        """Compute embeddings for source texts.

        Args:
            texts: The texts to embed.
            *args: Variable length argument list.
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
        self, texts: list[str] | np.ndarray, *args, **kwargs
    ) -> list[np.array]:
        """Get the embeddings for the given texts.

        Args:
            texts: list[str] or np.ndarray of strings to embed.
            *args: Variable length argument list.
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

        # batch process so that no more than 96 texts are sent at once.
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
        local_embedded_ids_path: Path,
        db_uri: str,
        model_name: str,
        context_limit: int,
        table_name: str,
    ) -> None:
        """Initialize the VectorDB.

        Args:
            local_embedded_ids_path: Path to store locally embedded document IDs.
            db_uri: URI for the LanceDB database.
            model_name: Name of the embedding model to use.
            context_limit: Maximum context limit for the model.
            table_name: Name of the table in the database.
        """
        self.local_embedded_ids_path = local_embedded_ids_path
        self.model_context_limit = context_limit
        self.table_name = table_name
        self.db = lancedb.connect(db_uri)
        azure_embeddings = (
            EmbeddingFunctionRegistry.get_instance()
            .get("azure-ai-text")
            .create(name=model_name)
        )

        class VectorDBSchema(LanceModel):
            vector: Vector(azure_embeddings.ndims()) = azure_embeddings.VectorField()
            document: str = azure_embeddings.SourceField()
            document_id: str
            report_id: str
            year: int
            mode: str
            agency: str
            type: str
            agency_id: str
            url: str
            document_type: str

        self.VectorDBSchema = VectorDBSchema

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
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
        self, df, document_column_name, tokenization_column_name
    ) -> pd.DataFrame:
        """Tokenize documents and add token counts to dataframe.

        Args:
            df: The dataframe containing documents.
            document_column_name: Name of the column with documents.
            tokenization_column_name: Name of the column to store token counts.

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
        self, documents_df, document_column_name="document"
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
        # Check if the df follows the schema
        if (
            documents_df.columns.tolist()
            != list(self.VectorDBSchema.model_fields.keys())[1:]
        ):
            msg = f"Dataframe columns {documents_df.columns.tolist()} do not match the expected schema {list(self.VectorDBSchema.model_fields.keys())[1:]}"
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
        documents_df[document_column_name] = documents_df.apply(
            lambda x: x[document_column_name][: self.model_context_limit - 50],
            axis=1,
        )

        num_batches = min(os.cpu_count() or 1, len(documents_df))

        # Split the dataframe into batches based on token length
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

        def add_documents_to_db(batch: pd.DataFrame):
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

    def clean_dataframes(
        self, dataframe_to_embed, dataframe_column_name, document_column_name
    ) -> pd.DataFrame:
        """Clean and prepare the dataframe for embedding.

        Cleans the dataframe to all have the right column names, formats,
        and data types.

        Args:
            dataframe_to_embed: The dataframe to clean.
            dataframe_column_name: Name of the document column in the dataframe.
            document_column_name: Name to use for the document column.

        Returns:
            pd.DataFrame: The cleaned and formatted dataframe.

        Raises:
            ValueError: If the document column name is unknown.
        """
        match dataframe_column_name:
            case "recommendations":
                dataframe_to_embed = dataframe_to_embed.rename(
                    columns={"recommendation": "document"}
                )
                dataframe_to_embed["document_id"] = dataframe_to_embed.apply(
                    lambda row: (
                        f"{row['recommendation_id']}_{'rec'}_{row['report_id']}"
                    ),
                    axis=1,
                )
                dataframe_to_embed["document_type"] = "recommendation"
            case "sections":
                dataframe_to_embed = dataframe_to_embed.rename(
                    columns={"section_text": "document"}
                )
                dataframe_to_embed["document_id"] = dataframe_to_embed.apply(
                    lambda row: f"{row['section']}_{'sec'}_{row['report_id']}",
                    axis=1,
                )
                dataframe_to_embed["document_type"] = "section"
            case "safety_issues":
                dataframe_to_embed = dataframe_to_embed.rename(
                    columns={"safety_issue": "document"}
                )
                dataframe_to_embed["document_id"] = dataframe_to_embed.apply(
                    lambda row: f"{row['safety_issue_id']}_{'si'}_{row['report_id']}",
                    axis=1,
                )
                dataframe_to_embed["document_type"] = "safety_issue"
            case "summary":
                dataframe_to_embed = dataframe_to_embed.rename(
                    columns={dataframe_column_name: "document"}
                )
                dataframe_to_embed["document_type"] = dataframe_column_name

                dataframe_to_embed["document_id"] = dataframe_to_embed.apply(
                    lambda row: f"sum_{row['report_id']}", axis=1
                )
            case _:
                msg = f"Unknown document column name: {document_column_name}"
                raise ValueError(msg)

        dataframe_to_embed = dataframe_to_embed[
            list(self.VectorDBSchema.model_fields.keys())[1:]
        ]
        # Drop unmatched
        # Drop unmatched documents and track count
        unmatched_count = dataframe_to_embed["report_id"].str.contains("nmatched").sum()
        dataframe_to_embed = dataframe_to_embed[
            ~dataframe_to_embed["report_id"].str.contains("nmatched")
        ]
        logger.info(f"Dropped {unmatched_count} unmatched documents")

        # Drop columns that are none or are empty strings and track count
        initial_count = len(dataframe_to_embed)
        dataframe_to_embed = dataframe_to_embed.dropna(subset=["document"])
        dataframe_to_embed = dataframe_to_embed[
            dataframe_to_embed["document"].str.strip() != ""
        ]
        empty_document_count = initial_count - len(dataframe_to_embed)
        logger.info(f"Dropped {empty_document_count} documents with empty/null content")

        # Check for missing values
        max_sample_rows = 200
        if dataframe_to_embed.isna().any(axis=1).any():
            missing_values = dataframe_to_embed[
                dataframe_to_embed.isna().any(axis=1)
            ].drop(labels="document", axis=1)
            if missing_values.shape[0] > max_sample_rows:
                missing_values = missing_values.sample(n=max_sample_rows)
            logger.warning(
                f"Dataframe {dataframe_column_name} has {missing_values.shape[0]} missing values. No missing values should be present at this point. They will be ignored but they should be checked on. Rows with missing values are:\n{missing_values.to_csv()}"
            )
            dataframe_to_embed = dataframe_to_embed.dropna()

        if dataframe_to_embed.duplicated(keep=False).any():
            duplicated_rows = dataframe_to_embed[
                dataframe_to_embed.duplicated(keep=False)
            ].drop(labels="document", axis=1)
            logger.warning(
                f"Dataframe {dataframe_column_name} has {duplicated_rows.shape[0]} duplicated rows. Duplicated rows will be ignored but they should be checked on. Rows with duplicates are:\n{duplicated_rows.to_csv()}"
            )
            dataframe_to_embed = dataframe_to_embed.drop_duplicates()

        dataframe_to_embed["agency_id"] = dataframe_to_embed["agency_id"].astype(str)
        dataframe_to_embed["mode"] = dataframe_to_embed["mode"].astype(str)
        dataframe_to_embed["type"] = dataframe_to_embed["type"].astype(str)

        return dataframe_to_embed

    def process_extracted_reports(self, extracted_df_path, embeddings_config) -> None:
        """Process extracted reports and generate embeddings.

        Args:
            extracted_df_path: Path to the extracted reports dataframe.
            embeddings_config: Configuration for which embeddings to generate.
        """
        logger.info("Embedding reports")
        logger.info(f"Extracted reports: {extracted_df_path}")
        logger.info(f"Embeddings {len(embeddings_config)} dataframes")
        logger.info(
            f"Embeddings config: {chr(10).join([str(config) for config in embeddings_config])}"
        )

        extracted_df = pd.read_pickle(extracted_df_path)

        for dataframe_column_name, document_column_name in (
            pbar := tqdm(embeddings_config)
        ):
            pbar.set_description(f"Embedding {dataframe_column_name}")
            dataframe_to_embed = None
            if isinstance(
                extracted_df[dataframe_column_name].dropna().iloc[0], pd.DataFrame
            ):
                filtered_extracted_df = extracted_df.dropna(
                    subset=[dataframe_column_name]
                )
                dataframe_to_embed = pd.concat(
                    [
                        df.assign(
                            report_id=report_id,
                            type=report_type,
                            mode=mode,
                            year=year,
                            agency=agency,
                            agency_id=agency_id,
                            url=url,
                        )
                        for df, report_id, report_type, mode, year, agency, agency_id, url in zip(
                            filtered_extracted_df[dataframe_column_name],
                            filtered_extracted_df["report_id"],
                            filtered_extracted_df["type"],
                            filtered_extracted_df["mode"],
                            filtered_extracted_df["year"],
                            filtered_extracted_df["agency"],
                            filtered_extracted_df["agency_id"],
                            filtered_extracted_df["url"],
                            strict=False,
                        )
                    ],
                    ignore_index=True,
                )
            else:
                dataframe_to_embed = extracted_df[
                    [
                        dataframe_column_name,
                        *list(self.VectorDBSchema.model_fields.keys())[3:-1],
                    ]
                ].dropna()

            cleaned_df = self.clean_dataframes(
                dataframe_to_embed,
                dataframe_column_name,
                document_column_name,
            )

            if self.local_embedded_ids_path.exists():
                local_embedded_ids = pd.read_pickle(self.local_embedded_ids_path)
                already_completed = cleaned_df["document_id"].isin(local_embedded_ids)
                cleaned_df = cleaned_df[~already_completed]
                logger.info(
                    f"Filtered out {sum(already_completed)} already embedded documents"
                )
            else:
                local_embedded_ids = pd.Series(dtype=str)

            if cleaned_df.empty:
                logger.info(
                    f"No new documents to embed for {dataframe_column_name}. Skipping."
                )
                continue

            current_table_version = self.table.version
            try:
                added_document_ids = self.add_documents(
                    cleaned_df,
                )
                if added_document_ids is None:
                    logger.info(f"No new documents added for {dataframe_column_name}")
                    continue
                logger.info(
                    f"Added {len(added_document_ids)} documents to the database for {dataframe_column_name}"
                )
                pd.concat(
                    [local_embedded_ids, added_document_ids],
                ).to_pickle(self.local_embedded_ids_path)
                logger.info(
                    f"Saved updated local embedded IDs to {self.local_embedded_ids_path}"
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
