import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import ClassVar, List, Union

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import TEXT
from lancedb.pydantic import LanceModel, Vector
from tqdm import tqdm
from transformers import AutoTokenizer


@register("azure-ai-text")
class AzureAITextEmbeddingFunction(TextEmbeddingFunction):
    """
    An embedding function that uses the AzureAI API

    https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme?view=azure-python-preview

    - AZURE_AI_ENDPOINT: The endpoint URL for the AzureAI service.
    - AZURE_AI_API_KEY: The API key for the AzureAI service.

    Parameters
    ----------
    - name: str
        The name of the model you want to use from the model catalog.


    Examples
    --------
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

    def ndims(self):
        if self.name == "embed-v-4-0":
            return 1536
        elif self.name == "Cohere-embed-v3-english":
            return 1024
        elif self.name == "Cohere-embed-v3-multilingual":
            return 1024
        elif self.name == "text-embedding-ada-002":
            return 1536
        elif self.name == "text-embedding-3-large":
            return 3072
        elif self.name == "text-embedding-3-small":
            return 1536
        else:
            raise ValueError(f"Unknown model name: {self.name}")

    def compute_query_embeddings(self, query: str, *args, **kwargs) -> List[np.array]:
        return self.compute_source_embeddings(query, input_type="query")

    def compute_source_embeddings(self, texts: TEXT, *args, **kwargs) -> List[np.array]:
        texts = self.sanitize_input(texts)
        input_type = (
            kwargs.get("input_type") or "document"
        )  # assume source input type if not passed by `compute_query_embeddings`
        return self.generate_embeddings(texts, input_type=input_type)

    def generate_embeddings(
        self, texts: Union[List[str], np.ndarray], *args, **kwargs
    ) -> List[np.array]:
        """
        Get the embeddings for the given texts

        Parameters
        ----------
        texts: list[str] or np.ndarray (of str)
            The texts to embed
        input_type: Optional[str]

        truncation: Optional[bool]
        """
        AzureAITextEmbeddingFunction._init_client()

        if isinstance(texts, np.ndarray):
            if texts.dtype != object:
                raise ValueError(
                    "AzureAIEmbeddingFunction only supports input of strings for numpy \
                        arrays."
                )
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
    def _init_client():
        if AzureAITextEmbeddingFunction.client is None:
            if os.environ.get("AZURE_AI_API_KEY") is None:
                raise ValueError("AZURE_AI_API_KEY not found in environment variables")
            if os.environ.get("AZURE_AI_ENDPOINT") is None:
                raise ValueError("AZURE_AI_ENDPOINT not found in environment variables")

            AzureAITextEmbeddingFunction.client = EmbeddingsClient(
                endpoint=os.environ["AZURE_AI_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_AI_API_KEY"]),
            )


class VectorDB:
    """
    This is a VectorDB class that is used to add documents to the vector database.
    """

    def __init__(
        self,
        local_embedded_ids_path: str,
        db_uri: str,
        model_name: str,
        context_limit: int,
        table_name: str,
    ):
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

        # using Cohere tokenizer for tokenization as a proxy to give me a rough guage.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Cohere/Cohere-embed-english-v3.0",
            )
        except Exception:
            print("Could not download the tokenizer. Will try again")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Cohere/Cohere-embed-english-v3.0",
                force_download=True,
            )

        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        """Get existing table or create new one."""
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)
        else:
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
                print(f"Warning: Could not create FTS index: {e}")

            try:
                table.create_scalar_index("document_id")
            except Exception as e:
                print(f"Warning: Could not create scalar index on document_id: {e}")

            return table

    def tokenize_documents(self, df, document_column_name, tokenization_column_name):
        if tokenization_column_name not in df.columns:
            df[tokenization_column_name] = df[document_column_name].apply(
                lambda x: len(self.tokenizer.tokenize(x))
            )
        else:
            df[tokenization_column_name] = df.apply(
                lambda x: len(self.tokenizer.tokenize(x[document_column_name]))
                if not isinstance(x[tokenization_column_name], int)
                else x[tokenization_column_name],
                axis=1,
            )

        return df

    def add_documents(self, df, document_column_name="document"):
        """
        Given a dataframe with atleast the document column and embedding column name it will generate embeddings for all of the documents that dont have embeddings in the dataframe.
        It does this by calling the embedding_function on batches of the documents.
        There is multithreading to speed up the process.

        Args:
            df: The dataframe
            embedding_function: The function that will be called to generate embeddings for documents. It must be`f([embedding]) -> [embedding]`
            batch_size: The size of the batches that will be passed to the embedding function
            document_column_name: The name of the column that contains the documents
            embedding_column_name: The name of the column that will contain the embeddings
        Returns the dataframe with the missing embeddings filled in.
        """

        # Check if the df follows the schema
        if df.columns.tolist() != list(self.VectorDBSchema.model_fields.keys())[1:]:
            raise ValueError(
                f"Dataframe columns {df.columns.tolist()} do not match the expected schema {list(self.VectorDBSchema.model_fields.keys())[1:]}"
            )

        # Get document lengths
        token_length_column_name = f"{document_column_name}_token_length"
        df = self.tokenize_documents(df, document_column_name, token_length_column_name)

        print(
            f"There are a total of {df[token_length_column_name].sum()} tokens in {len(df)} documents"
        )

        to_drop = pd.Series(df[token_length_column_name] < self.model_context_limit * 2)

        df = df.loc[to_drop]

        print(
            f"Dropping documents with more than {self.model_context_limit * 2} tokens which is {len(to_drop) - sum(to_drop)} documents"
        )

        if df.empty:
            return None

        # Truncate all documents to just below the context limit
        df[document_column_name] = df.apply(
            lambda x: x[document_column_name][: self.model_context_limit - 50],
            axis=1,
        )

        num_batches = min(os.cpu_count() or 1, len(df))

        # Split the dataframe into batches based on token length
        df_sorted = df.sort_values(
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

        for i, (batch, token_count) in enumerate(zip(batches, batch_token_counts)):
            print(f"Batch {i}: {len(batch)} documents, {token_count} tokens")

        def add_documents_to_db(batch: pd.DataFrame):
            pa_table = pa.Table.from_pandas(
                batch,
                schema=pa.schema(
                    [field for field in self.table.schema if field.name != "vector"]
                ),
            )

            results = self.table.add(pa_table, mode="append")

            return results

        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            futures = {
                executor.submit(add_documents_to_db, batch): i
                for i, batch in enumerate(batches)
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

                # As I we are not using merge_insert there is no way of knowing what rows were inserted. Instead we simply add all the documents even if it creates duplicates.
                # if merge_result.num_inserted_rows != batches[batch_index].shape[0]:
                #     raise RuntimeError(
                #         f"Warning: Batch {batch_index} inserted {merge_result.num_inserted_rows} rows, expected {batches[batch_index].shape[0]} rows."
                #     )

        return df["document_id"]

    def clean_dataframes(
        self, dataframe_to_embed, dataframe_column_name, document_column_name
    ):
        """
        Cleans the dataframe to all have the right column names and formats.
        """
        match dataframe_column_name:
            case "recommendations":
                dataframe_to_embed = dataframe_to_embed.rename(
                    columns={"recommendation": "document"}
                )
                dataframe_to_embed["document_id"] = dataframe_to_embed.apply(
                    lambda row: f"{row['recommendation_id']}_{'rec'}_{row['report_id']}",
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
                raise ValueError(
                    f"Unknown document column name: {document_column_name}"
                )

        dataframe_to_embed = dataframe_to_embed[
            list(self.VectorDBSchema.model_fields.keys())[1:]
        ]
        # Drop unmatched
        # Drop unmatched documents and track count
        unmatched_count = dataframe_to_embed["report_id"].str.contains("nmatched").sum()
        dataframe_to_embed = dataframe_to_embed[
            ~dataframe_to_embed["report_id"].str.contains("nmatched")
        ]
        print(f"Dropped {unmatched_count} unmatched documents")

        # Drop columns that are none or are empty strings and track count
        initial_count = len(dataframe_to_embed)
        dataframe_to_embed = dataframe_to_embed.dropna(subset=["document"])
        dataframe_to_embed = dataframe_to_embed[
            dataframe_to_embed["document"].str.strip() != ""
        ]
        empty_document_count = initial_count - len(dataframe_to_embed)

        print(f"Dropped {empty_document_count} documents with empty/null content")

        if dataframe_to_embed.isna().any(axis=1).any():
            missing_values = dataframe_to_embed[
                dataframe_to_embed.isna().any(axis=1)
            ].drop(labels="document", axis=1)
            if missing_values.shape[0] > 200:
                missing_values = missing_values.sample(n=200)
            print(
                f"====WARNING====\nDataframe {dataframe_column_name} has {missing_values.shape[0]} missing values. No missing values should be  present at this point. They will be ignored but they should be checked on. Rows with missing values are:\n{missing_values.to_csv()}\n{'=' * 50}\n"
            )
            dataframe_to_embed = dataframe_to_embed.dropna()

        if dataframe_to_embed.duplicated(keep=False).any():
            duplicated_rows = dataframe_to_embed[
                dataframe_to_embed.duplicated(keep=False)
            ].drop(labels="document", axis=1)
            print(
                f"====WARNING====\nDataframe {dataframe_column_name} has {duplicated_rows.shape[0]} duplicated rows. Duplicated rows will be ignored but they should be checked on. Rows with duplicates are:\n{duplicated_rows.to_csv()}\n{'=' * 50}"
            )
            dataframe_to_embed = dataframe_to_embed.drop_duplicates()

        dataframe_to_embed["agency_id"] = dataframe_to_embed["agency_id"].astype(str)
        dataframe_to_embed["mode"] = dataframe_to_embed["mode"].astype(str)
        dataframe_to_embed["type"] = dataframe_to_embed["type"].astype(str)

        return dataframe_to_embed

    def process_extracted_reports(self, extracted_df_path, embeddings_config):
        print("==================================================")
        print("---------------  Embedding reports  --------------")
        print("   Extracted reports: ", extracted_df_path)
        print(f"   Embeddings {len(embeddings_config)} dataframes")
        print(
            f"   Embeddings config: \n{chr(10).join([str(config) for config in embeddings_config])}"
        )
        print("==================================================")

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
                            type=type,
                            mode=mode,
                            year=year,
                            agency=agency,
                            agency_id=agency_id,
                            url=url,
                        )
                        for df, report_id, type, mode, year, agency, agency_id, url in zip(
                            filtered_extracted_df[dataframe_column_name],
                            filtered_extracted_df["report_id"],
                            filtered_extracted_df["type"],
                            filtered_extracted_df["mode"],
                            filtered_extracted_df["year"],
                            filtered_extracted_df["agency"],
                            filtered_extracted_df["agency_id"],
                            filtered_extracted_df["url"],
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

            if os.path.exists(self.local_embedded_ids_path):
                local_embedded_ids = pd.read_pickle(self.local_embedded_ids_path)
                already_completed = cleaned_df["document_id"].isin(local_embedded_ids)
                cleaned_df = cleaned_df[~already_completed]
                print(
                    f"Filtered out {sum(already_completed)} already embedded documents"
                )
            else:
                local_embedded_ids = pd.Series(dtype=str)

            if cleaned_df.empty:
                print(
                    f"No new documents to embed for {dataframe_column_name}. Skipping."
                )
                continue

            current_table_version = self.table.version
            try:
                added_document_ids = self.add_documents(
                    cleaned_df,
                )
                if added_document_ids is None:
                    print(f"No new documents added for {dataframe_column_name}")
                    continue
                print(
                    f"Added {len(added_document_ids)} documents to the database for {dataframe_column_name}"
                )
                pd.concat(
                    [local_embedded_ids, added_document_ids],
                ).to_pickle(self.local_embedded_ids_path)
                print(
                    f"Saved updated local embedded IDs to {self.local_embedded_ids_path}"
                )

            except Exception as e:
                print(
                    f"Error during adding of documents: {e}, going to restore previous state of the table"
                )

                self.table = self.table.restore(current_table_version)

                raise e

        print("Finished embedding all reports.")
        self.table.optimize(cleanup_older_than=timedelta(days=14))
        print("Optimized the table and cleaned up older entries.")
