import os
from typing import ClassVar, Literal, NamedTuple

import lancedb
import numpy as np
import pandas as pd
import plotly.express as px
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from lancedb.embeddings.base import TextEmbeddingFunction
from lancedb.embeddings.registry import register
from lancedb.embeddings.utils import TEXT
from rich import print, table  # noqa: A004


# This has to be added in manually unless https://github.com/lancedb/lancedb/issues/2518 is resolved
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

    def compute_query_embeddings(self, query: str, *_args, **_kwargs) -> list[np.array]:
        return self.compute_source_embeddings(query, input_type="query")

    def compute_source_embeddings(
        self,
        texts: TEXT,
        *_args,
        **kwargs,
    ) -> list[np.array]:
        texts = self.sanitize_input(texts)
        input_type = (
            kwargs.get("input_type") or "document"
        )  # assume source input type if not passed by `compute_query_embeddings`
        return self.generate_embeddings(texts, input_type=input_type)

    def generate_embeddings(
        self,
        texts: list[str] | np.ndarray,
        *_args,
        **kwargs,
    ) -> list[np.array]:
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
                msg = (
                    "AzureAITextEmbeddingFunction only supports input of strings for numpy \
                        arrays."
                )
                raise ValueError(
                    msg,
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
                msg = "AZURE_AI_API_KEY not found in environment variables"
                raise ValueError(msg)
            if os.environ.get("AZURE_AI_ENDPOINT") is None:
                msg = "AZURE_AI_ENDPOINT not found in environment variables"
                raise ValueError(msg)

            AzureAITextEmbeddingFunction.client = EmbeddingsClient(
                endpoint=os.environ["AZURE_AI_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_AI_API_KEY"]),
            )


class SearchParams(NamedTuple):
    query: str
    search_type: Literal["fts", "vector"] | None
    year_range: tuple[int, int]
    document_type: list[str]
    modes: list[str]
    agencies: list[str]


class Searcher:
    def __init__(self, db_uri, table_name):
        print("[bold]Creating searcher[/bold]")
        print(f"connecting to database at {db_uri}")
        self.vector_db = lancedb.connect(db_uri)
        try:
            self.all_document_types_table = self.vector_db.open_table(table_name)
        except ValueError as e:
            print(f"[bold red]Error opening table {table_name}[/bold red]")
            print(f"Error: {e}")
            print(f"Only {self.vector_db.table_names()} exist")
            raise

        self.last_updated = self.all_document_types_table.list_versions()[-1][
            "timestamp"
        ].strftime("%Y-%m-%d")
        self.db_version = self.all_document_types_table.version
        searcher_config = table.Table(title="🔍 Searcher Config", show_header=True)
        searcher_config.add_column("Name")
        searcher_config.add_column("Value")
        searcher_config.add_row("Database URI", db_uri)
        searcher_config.add_row("Table Name", table_name)
        searcher_config.add_row(
            "Table Version",
            str(self.all_document_types_table.version),
        )
        searcher_config.add_row("Last updated", self.last_updated)
        searcher_config.add_row(
            "Table Size",
            f"{self.all_document_types_table.count_rows()} rows",
        )
        searcher_config.add_row(
            "Columns",
            ", ".join(self.all_document_types_table.schema.names),
        )
        print(searcher_config)

        if "agency" not in self.all_document_types_table.schema.names:
            msg = "agency column not found in table"
            raise ValueError(msg)

    def __get_where_statement(
        self,
        year_range: tuple[int, int],
        document_type: list[str],
        modes: list[str],
        agencies: list[str],
    ):
        where_statement = []
        if year_range:
            where_statement.append(
                f"year >= {int(year_range[0])} and year <= {int(year_range[1])}",
            )
        if document_type:
            document_types = ", ".join(f'"{dt}"' for dt in document_type)
            where_statement.append(f"document_type in ({document_types})")
        if modes and len(modes) > 1:
            where_statement.append(f"mode in {tuple([str(mode) for mode in modes])}")
        elif modes and len(modes) == 1:
            where_statement.append(f"mode = '{modes[0]!s}'")
        if agencies and len(agencies) > 1:
            where_statement.append(f"agency in {tuple(agencies)}")
        elif agencies and len(agencies) == 1:
            where_statement.append(f"agency = '{agencies[0]}'")

        return " AND ".join(where_statement)

    def __print_search_query(
        self,
        query: str,
        final_query: str | list[float] | None,
        where_statement: str,
    ):
        query_table = table.Table(
            title="🔍 Conducting search with 🔍",
            show_header=True,
            title_style="bold blue",
        )
        query_table.add_column("Parameter")
        query_table.add_column("Value")
        if final_query is None:
            query_table.add_row("Query", "None")
        else:
            query_table.add_row(
                "Query",
                final_query
                if isinstance(final_query, str)
                else "vector embeddings of " + query,
            )
        if where_statement:
            query_table.add_row("Filters", where_statement)
        print(query_table)

    def knowledge_search(
        self,
        params: SearchParams,
        limit: int = 150,
        relevance: float = 0,
    ):
        info = {
            "info_message": "",
        }

        # Add info message
        if "TSB" in params.agencies and "summary" in params.document_type:
            info["info_message"] += (
                "Summaries are only available for ATSB and TAIC reports, not TSB reports.\n"
            )

        where_statement = self.__get_where_statement(
            year_range=params.year_range,
            document_type=params.document_type,
            modes=params.modes,
            agencies=params.agencies,
        )

        final_query: list[float] | str | None = None
        if params.query == "" or params.query is None:
            final_query = None
            # Fix up error with LLM not providing the right parameters
            params = params._replace(search_type=None)
        elif params.search_type in ["fts", "vector"]:
            final_query = params.query
        else:
            msg = f"type must be 'fts' or 'vector' not {params.search_type}"
            raise ValueError(msg)

        self.__print_search_query(params.query, final_query, where_statement)

        search = self.all_document_types_table.search(
            final_query,
            query_type=params.search_type,
        )

        if params.search_type == "vector":
            search = search.metric("cosine")

        results = (
            search.where(where_statement, prefilter=True).limit(limit).to_pandas()
        ).drop(columns=["vector"])

        print(
            f"[bold green]Found {len(results)} results for query: {params.query}[/bold green]",
        )

        info["total_results"] = len(results)

        # Clean up the relevance score column so that it is always sorted in descending order
        if final_query is not None:
            if "_distance" in results.columns:
                results["_distance"] = 1 - results["_distance"]
            results = results.rename(
                columns={
                    "_relevance_score": "relevance",
                    "_score": "relevance",
                    "_distance": "relevance",
                },
            )
            results = results.sort_values(by=["relevance"], ascending=False)
            results = results.reset_index(drop=True)

            cols = ["relevance"] + [
                col for col in results.columns if col != "relevance"
            ]
            results = results[cols]

            if relevance > 0:
                print(
                    f"[bold yellow]Filtering results to only include relevance >= {relevance}[/bold yellow]",
                )
                results = results[results["relevance"] >= relevance]
                info["relevant_results"] = len(results)
                print(
                    f"[bold green]Found {len(results)} relevant results for query: {params.query}[/bold green]",
                )

        # Convert mode back to strings
        results["mode"] = results["mode"].apply(
            lambda x: ["aviation", "rail", "maritime"][int(x)],
        )

        graph_maker = GraphMaker(results)

        plots = {
            "document_type": graph_maker.get_document_type_pie_chart(),
            "mode": graph_maker.get_mode_pie_chart(),
            "year": graph_maker.get_year_histogram(),
            "event_type": graph_maker.get_most_common_event_types(),
            "agency": graph_maker.get_agency_pie_chart(),
        }

        return results, info, plots


class GraphMaker:
    def __init__(self, context):
        self.context = context

    def add_visual_layout(self, fig):
        fig = fig.update_layout(width=310)

        # If fig a pie chart
        if fig.data[0].type == "pie":
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                insidetextorientation="radial",
            )

            # Remove legend
            fig.update_layout(showlegend=False)

        return fig

    def get_document_type_pie_chart(self):
        context_df = self.context["document_type"].value_counts().reset_index()
        context_df.columns = ["document_type", "count"]
        fig = px.pie(
            context_df,
            values="count",
            names="document_type",
            title="Document type distribution",
        )

        return self.add_visual_layout(fig)

    def get_mode_pie_chart(self):
        context_df = self.context["mode"].value_counts().reset_index()
        context_df.columns = ["mode", "count"]
        fig = px.pie(
            context_df,
            values="count",
            names="mode",
            title="Mode distribution",
        )

        return self.add_visual_layout(fig)

    def get_year_histogram(self):
        context_df = self.context
        fig = px.histogram(
            context_df,
            x="year",
            title="Year distributions",
        )
        return self.add_visual_layout(fig)

    def get_most_common_event_types(self):
        context_df = self.context
        type_counts = context_df.groupby("type")["document"].count()

        top_5_types = type_counts.nlargest(5).reset_index()

        others_count = type_counts[~type_counts.index.isin(top_5_types["type"])].sum()

        combined_df = pd.concat(
            [
                top_5_types,
                pd.DataFrame([["Others", others_count]], columns=["type", "document"]),
            ],
            ignore_index=True,
        )

        combined_df.columns = ["Event type", "Count"]

        fig = px.pie(
            combined_df,
            values="Count",
            names="Event type",
            title="Top 5 most common event types",
        )

        return self.add_visual_layout(fig)

    def get_agency_pie_chart(self):
        context_df = self.context["agency"].value_counts().reset_index()
        context_df.columns = ["agency", "count"]
        fig = px.pie(
            context_df,
            values="count",
            names="agency",
            title="Agency distribution",
        )

        return self.add_visual_layout(fig)
