import json
import openai
import voyageai
import lancedb
from datetime import datetime
from rich import print, table


class assistant:
    def __init__(self, openai_api_key, voyageai_api_key, db_uri):
        print("[bold]Creating Chatbot[/bold]")
        print(f"connecting to database at {db_uri}")
        self.vector_db = lancedb.connect(db_uri)
        table_name = "all_document_types"
        self.all_document_types_table = self.vector_db.open_table(table_name)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.voyageai_client = voyageai.Client(api_key=voyageai_api_key)

        print("[bold]Created Chatbot[/bold]")
        chatbot_config = table.Table(title="🛠️  Chatbot Configuration 🛠️")

        chatbot_config.add_column("Name")
        chatbot_config.add_column("Value")
        chatbot_config.add_row("Database URI", db_uri)
        chatbot_config.add_row("Table Name", table_name)
        chatbot_config.add_row(
            "Table Size", f"{self.all_document_types_table.count_rows()} rows"
        )
        chatbot_config.add_row(
            "Columns", ", ".join(self.all_document_types_table.schema.names)
        )
        print(chatbot_config)

        if "agency" not in self.all_document_types_table.schema.names:
            raise ValueError("agency column not found in table")

    def embed_query(self, query: str):
        return self.voyageai_client.embed(
            query, model="voyage-large-2-instruct", input_type="query", truncation=False
        ).embeddings[0]

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
                f"year >= {year_range[0]} and year <= {year_range[1]}"
            )
        if document_type:
            document_types = ", ".join(f'"{dt}"' for dt in document_type)
            where_statement.append(f"document_type in ({document_types})")
        if modes and len(modes) > 1:
            where_statement.append(f"mode in {tuple(modes)}")
        elif modes and len(modes) == 1:
            where_statement.append(f"mode = '{modes[0]}'")
        if agencies and len(agencies) > 1:
            where_statement.append(f"agency in {tuple(agencies)}")
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

        return results

    def provide_conversation_title(self, history=[]):
        if history == []:
            raise ValueError("history is empty")

        return (
            self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                    You are part of a chatbot that help users add titles to their conversation. You will recieve the conversation and you are too response with no more than 5 works summary of the conversation.
                    Just respond with the title and nothing else.
                    """,
                    },
                    history[-1],
                ],
            )
            .choices[0]
            .message.content
        )

    def process_input(self, history=[]):
        system_message = {
            "role": "system",
            "content": f"""
You are a expert working at the New Zealand transport accident investigation commision. Your job is to assistant employees of TAIC with their queries. The day is {datetime.now()}.
You will be provided the conversation history and a query from the user. You can either respond directly or can call a function that searches a database of accident reports.
For each report you should provide use the report IDs, if you reference any other document you should provide the document type and document ID.

Here is some more dataset information
There are {len(self.all_document_types_table.schema.names)} columns with {self.all_document_types_table.count_rows()} rows.
The columns available are: {"".join(self.all_document_types_table.schema.names)}

Here are some definitions from TAIC:

Safety factor - Any (non-trivial) events or conditions, which increases safety risk. If they occurred in the future, these would
increase the likelihood of an occurrence, and/or the
severity of any adverse consequences associated with the
occurrence.

Safety issue - A safety factor that:
• can reasonably be regarded as having the
potential to adversely affect the safety of future
operations, and
• is characteristic of an organisation, a system, or an
operational environment at a specific point in time.
Safety Issues are derived from safety factors classified
either as Risk Controls or Organisational Influences.

Safety theme - Indication of recurring circumstances or causes, either across transport modes or over time. A safety theme may
cover a single safety issue, or two or more related safety
issues.
""",
        }

        response_stream = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message] + history,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": """Search for safety issues and recommendations from the New Zealand Transport Accident Investigation Commission. This function searches a vector database.
Eample function calls:
```json
{
    "query": "What are common elements in floatation devices and fires?",
    "type": "vector",
    "year_range": [2020, 2023],
    "document_type": ["safety_issue", "recommendation"],
    "modes": [0, 1, 2],
    "agencies": ["TSB", "ATSB", "TAIC"]
}
```

```json
{
    "query": "",
    "type": "vector",
    "year_range": [2020, 2023],
    "document_type": ["safety_issue"],
    "modes": [2],
    "agencies": ["TAIC"]
}
```

""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query to search for. If left as an empty string it will return all results that match the other paramters.",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["fts", "vector"],
                                    "description": "The type of search to perform. fts should be used if the query is asking a specific question about a organisation, organisation etc. Otherwsie for more general information use vector, it will embed your query and search the vector database.",
                                },
                                "year_range": {
                                    "type": "array",
                                    "description": "An array specifying the start and end years for filtering results. Valid range is 2000-2023.",
                                    "items": {"type": "number"},
                                },
                                "document_type": {
                                    "type": "array",
                                    "description": "A list of document types to filter the search results. Valid types are 'safety_issue', 'recommendation', 'report_section'.",
                                    "items": {"type": "string"},
                                },
                                "modes": {
                                    "type": "array",
                                    "description": "A list of modes to filter the search results. Valid modes are 0, 1, and 2. Which are aviation, rail, and marine respectively.",
                                    "items": {"type": "string"},
                                },
                                "agencies": {
                                    "type": "array",
                                    "description": "A list of agencies to filter the search results. Valid agencies are TSB, ATSB, and TAIC. These are Transport Safety Board (Canada), Australian Transport Safety Board, and Transport Accident Investigation Commission (New Zealand) respectively.",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "query",
                                "type",
                                "year_range",
                                "document_type",
                                "modes",
                                "agencies",
                            ],
                            "additionalProperties": False,
                            "strict": True,
                        },
                    },
                }
            ],
            stream=True,
        )
        function_arguments_str = ""
        function_name = ""
        tool_call_id = ""
        is_collecting_function_args = False

        history.append({"role": "assistant", "content": ""})

        for part in response_stream:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history

            if delta.tool_calls:
                is_collecting_function_args = True
                tool_call = delta.tool_calls[0]

                if tool_call.id:
                    tool_call_id = tool_call.id
                if tool_call.function.name:
                    function_name = tool_call.function.name

                # Process function arguments delta
                if tool_call.function.arguments:
                    function_arguments_str += tool_call.function.arguments

            # Process tool call with complete arguments
            if finish_reason == "tool_calls" and is_collecting_function_args:
                break

        if not is_collecting_function_args:
            return history

        function_arguments = json.loads(function_arguments_str)

        history[-1]["metadata"] = {
            "title": "🔍 Searching database for more information"
        }
        history[-1]["content"] += (
            f"Using these parameters to search the database: {function_arguments_str}"
        )
        yield history

        results = self.knowledge_search(**function_arguments)
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_arguments_str,
                    },
                }
            ],
        }

        messages = (
            [system_message]
            + history
            + [tool_call_message]
            + [
                {
                    "role": "tool",
                    "content": results.to_json(orient="records"),
                    "tool_call_id": tool_call_id,
                },
            ]
        )

        history.append(
            {
                "role": "assistant",
                "content": results.to_html(index=False),
                "metadata": {"title": f"📖 Reading {results.shape[0]} results"},
            }
        )
        yield history

        rag_response = self.openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, stream=True
        )

        history.append({"role": "assistant", "content": ""})

        for part in rag_response:
            delta = part.choices[0].delta
            finish_reason = part.choices[0].finish_reason

            # Process assistant content
            if delta.content:
                history[-1]["content"] += delta.content
                yield history

        return history
