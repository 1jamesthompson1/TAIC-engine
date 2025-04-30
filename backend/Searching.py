from rich import print, table
import openai
import voyageai
import lancedb
import plotly.express as px
import pandas as pd
from typing import Optional, Union

class Searcher:
    def __init__(self, openai_api_key, voyageai_api_key, db_uri):
        print("[bold]Creating searcher[/bold]")
        print(f"connecting to database at {db_uri}")
        self.vector_db = lancedb.connect(db_uri)
        table_name = "all_document_types"
        self.all_document_types_table = self.vector_db.open_table(table_name)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.voyageai_client = voyageai.Client(api_key=voyageai_api_key) # type: ignore

        self.last_updated = self.all_document_types_table.list_versions()[-1][
            "timestamp"
        ].strftime("%Y-%m-%d")
        searcher_config = table.Table(title="🔍 Searcher Config", show_header=True)
        searcher_config.add_column("Name")
        searcher_config.add_column("Value")
        searcher_config.add_row("Database URI", db_uri)
        searcher_config.add_row("Table Name", table_name)
        searcher_config.add_row("Last updated", self.last_updated)
        searcher_config.add_row(
            "Table Size", f"{self.all_document_types_table.count_rows()} rows"
        )
        searcher_config.add_row(
            "Columns", ", ".join(self.all_document_types_table.schema.names)
        )
        print(searcher_config)

        if "agency" not in self.all_document_types_table.schema.names:
            raise ValueError("agency column not found in table")

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
                f"year >= {int(year_range[0])} and year <= {int(year_range[1])}"
            )
        if document_type:
            document_types = ", ".join(f'"{dt}"' for dt in document_type)
            where_statement.append(f"document_type in ({document_types})")
        if modes and len(modes) > 1:
            where_statement.append(f"mode in {tuple([str(mode) for mode in modes])}")
        elif modes and len(modes) == 1:
            where_statement.append(f"mode = '{str(modes[0])}'")
        if agencies and len(agencies) > 1:
            where_statement.append(f"agency in {tuple(agencies)}")
        elif agencies and len(agencies) == 1:
            where_statement.append(f"agency = '{agencies[0]}'")

        return " AND ".join(where_statement)

    def __print_search_query(
        self,
        query: str,
        final_query: Union[str, list[float], None],
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
        query: str,
        type: Optional[str],
        year_range: tuple[int, int],
        document_type: list[str],
        modes: list[str],
        agencies: list[str],
        limit: int = 100,
        relevance: float = 0,
    ):
        info = {}

        where_statement = self.__get_where_statement(
            year_range=year_range,
            document_type=document_type,
            modes=modes,
            agencies=agencies,
        )

        final_query: Union[list[float], Optional[str]] = None
        if query == "" or query is None:
            final_query = None
            type = None # Fix up error with LLM not providing the right paramters
        elif type == "fts":
            final_query = query
        elif type == "vector":
            final_query = self.embed_query(query)
        else:
            raise ValueError(f"type must be 'fts' or 'vector' not {type}")

        self.__print_search_query(query, final_query, where_statement)


        search = self.all_document_types_table.search(final_query, query_type=type)

        if type == "vector":
            search = search.metric("cosine")

        results = (
            search
            .where(where_statement, prefilter=True)
            .limit(limit)
            .to_pandas()
        ).drop(columns=["vector"])

        print(
            f"[bold green]Found {len(results)} results for query: {query}[/bold green]"
        )

        info["total_results"] = len(results)

        # Clean up the relevance score column so that it is always sorted in descending order
        if final_query is not None:
            if "_distance" in results.columns:
                results["_distance"] = 1 - results["_distance"]
            results.rename(
                columns={
                    "_relevance_score": "relevance",
                    "_score": "relevance",
                    "_distance": "relevance",
                },
                inplace=True,
            )
            results.sort_values(by=["relevance"], ascending=False, inplace=True)
            results.reset_index(drop=True, inplace=True)

            cols = ["relevance"] + [
                col for col in results.columns if col != "relevance"
            ]
            results = results[cols]

            if relevance > 0:
                print(
                    f"[bold yellow]Filtering results to only include relevance >= {relevance}[/bold yellow]"
                )
                results = results[results["relevance"] >= relevance]
                info['relevant_results'] = len(results)
                print(
                    f"[bold green]Found {len(results)} relevant results for query: {query}[/bold green]"
                )

        # Convert mode back to strings
        results["mode"] = results["mode"].apply(
            lambda x: ["aviation", "rail", "maritime"][int(x)]
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

    def embed_query(self, query: str):
        return self.voyageai_client.embed(
            query, model="voyage-large-2-instruct", input_type="query", truncation=False, output_dtype="float"
        ).embeddings[0]




class GraphMaker():
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
        context_df = (
            self.context["document_type"].value_counts().reset_index()
        )
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
