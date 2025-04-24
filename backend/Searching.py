from rich import print, table
import openai
import voyageai
import lancedb

class Searcher:
    def __init__(self, openai_api_key, voyageai_api_key, db_uri):
        print("[bold]Creating searcher[/bold]")
        print(f"connecting to database at {db_uri}")
        self.vector_db = lancedb.connect(db_uri)
        table_name = "all_document_types"
        self.all_document_types_table = self.vector_db.open_table(table_name)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.voyageai_client = voyageai.Client(api_key=voyageai_api_key)

        searcher_config = table.Table(title="🔍 Searcher Config", show_header=True)
        searcher_config.add_column("Name")
        searcher_config.add_column("Value")
        searcher_config.add_row("Database URI", db_uri)
        searcher_config.add_row("Table Name", table_name)
        searcher_config.add_row(
            "Table Size", f"{self.all_document_types_table.count_rows()} rows"
        )
        searcher_config.add_row(
            "Columns", ", ".join(self.all_document_types_table.schema.names)
        )
        print(searcher_config)

        if "agency" not in self.all_document_types_table.schema.names:
            raise ValueError("agency column not found in table")
    
    def knowledge_search(
        self,
        query: str,
        type: str,
        year_range: tuple[int, int],
        document_type: list[str],
        modes: list[str],
        agencies: list[str],
    ):
        limit = 100
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
            where_statement.append(f"mode = '{int(modes[0])}'")
        if agencies and len(agencies) > 1:
            where_statement.append(f"agency in {tuple(map(int, agencies))}")
        elif agencies and len(agencies) == 1:
            where_statement.append(f"agency = '{agencies[0]}'")

        where_statement = " AND ".join(where_statement)

        if query == "" or query is None:
            final_query = None
        elif type == "fts":
            final_query = query
        elif type == "vector":
            final_query = self.embed_query(query)
        else:
            raise ValueError(f"type must be 'fts' or 'vector' not {type}")

        query_table = table.Table(
            title="🔍 Conducting search with 🔍",
            show_header=True,
            title_style="bold blue",
        )
        query_table.add_column("Parameter")
        query_table.add_column("Value")
        query_table.add_row(
            "Query",
            final_query
            if isinstance(final_query, str)
            else "vector embeddings of " + query,
        )
        if where_statement:
            query_table.add_row("Filters", where_statement)
        print(query_table)

        results = (
            self.all_document_types_table.search(final_query, query_type=type)
            .where(where_statement, prefilter=True)
            .limit(limit)
            .to_pandas()
        ).drop(columns=["vector"])

        print(
            f"[bold green]Found {len(results)} results for query: {query}[/bold green]"
        )
        return results
    
    def embed_query(self, query: str):
        return self.voyageai_client.embed(
            query, model="voyage-large-2-instruct", input_type="query", truncation=False
        ).embeddings[0]