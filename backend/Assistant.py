import json
from collections.abc import Generator
from datetime import datetime, timezone

import openai
from rich import print  # noqa: A004

from .AssistantTools import SearchTool


class CompleteHistory(list):
    """
    This is a modified list that holds the complete history of the conversation.

    It stores all the information needed for both OpenAI and Gradio formats.
    """

    def __init__(self, *args):
        super().__init__(*args)

        if len(args) != 1:
            msg = "CompleteHistory must be initialized with a single iterable argument"
            raise ValueError(
                msg,
            )

        if not isinstance(args[0], list):
            msg = "CompleteHistory must be initialized with a list"
            raise TypeError(
                msg,
            )

        try:
            self.format_check()
        except ValueError as e:
            print(f"[red]Warning: History format issue on init: {e}[/red]")
            self.format()

    def format(self):
        """
        Pre to 0.3.0 the message was justa  list of dict with "role", "content" and optional "metadata".
        Need to expand this out into the full format with "display" and "ai" keys. TO do this just copy the message to both and only have role and content for the ai key.
        """
        new_history = []

        try:
            new_history = [
                {
                    "display": curr,
                    "ai": {
                        "role": curr["role"],
                        "content": curr["content"],
                    },
                }
                for curr in self
            ]
        except KeyError as e:
            msg = f"Failed to format history: {e}"
            raise ValueError(msg) from e

        self.clear()
        self.extend(new_history)

    def format_check(self):
        """
        Check that the history is in the correct format.
        Each message must be a dict with "display" and "ai" keys.
        The "display" key must have "role" and "content".
        The "ai" key must have either "role" and "content" or "type", "output", "call_id".
        """
        for i, message in enumerate(self):
            # Check roughly in the right format
            if not isinstance(message, dict):
                msg = f"Message {i} is not a dict"
                raise TypeError(msg)
            if "display" not in message or "ai" not in message:
                msg = f"Message {i} must have 'display' and 'ai' keys"
                raise ValueError(msg)
            display = message["display"]
            ai = message["ai"]

            # Check display format
            if not isinstance(display, dict):
                msg = f"Message {i} 'display' must be a dict, got {type(display)}"
                raise TypeError(
                    msg,
                )
            if "role" not in display or "content" not in display:
                msg = f"Message {i} 'display' must have 'role' and 'content', got {display.keys()}"
                raise ValueError(
                    msg,
                )

            # Check ai format
            if not isinstance(ai, dict):
                msg = f"Message {i} 'ai' must be a dict, got {type(ai)}"
                raise TypeError(msg)
            if ("role" in ai and "content" in ai) or (
                "type" in ai and "output" in ai and "call_id" in ai
            ):  # chat messsage
                continue
            if "type" in ai and "name" in ai and "arguments" in ai:  # Function call
                continue

            msg = f"Message {i} 'ai' must be either function call, message or function output, got {ai.keys()}"
            raise ValueError(
                msg,
            )

    def add_message(self, role, content, metadata=None):
        message = {
            "role": role,
            "content": content,
        }

        display_message = message.copy()
        if metadata:
            display_message["metadata"] = metadata

        self.append(
            {
                "display": display_message,
                "ai": message.copy(),
            },
        )

    def start_thought(self, content=""):
        """
        Start a new assistant thought message.
        """
        self.append(
            {
                "display": {
                    "role": "assistant",
                    "content": content,
                    "metadata": {
                        "title": "🧠 Orienting and planning",
                        "status": "pending",
                    },
                },
                "ai": {
                    "role": "assistant",
                    "content": content,
                },
            },
        )

    def end_thought(self):
        """
        End the current assistant thought message.
        """
        if len(self) == 0 or self[-1]["display"]["role"] != "assistant":
            msg = "No assistant message to end"
            raise ValueError(msg)

        self[-1]["display"]["metadata"]["status"] = "done"

    def update_last_message(self, delta_content):
        """
        Used for streaming updates to the last message.
        """

        if len(self) == 0:
            msg = "No messages to update"
            raise ValueError(msg)

        if self[-1]["display"]["role"] != "assistant":
            msg = "Can only update the last assistant message"
            raise ValueError(msg)

        self[-1]["display"]["content"] += delta_content
        self[-1]["ai"]["content"] += delta_content

    def add_function_call(self, ai_call):
        self.append(
            {
                "display": {
                    "role": "assistant",
                    "content": f"Executing {ai_call['name']} with parameters: {ai_call['arguments']}",
                    "metadata": {
                        "title": f"🔧 Executing {ai_call['name']}",
                        "status": "pending",
                    },
                },
                "ai": ai_call,
            },
        )

    def complete_function_call(self, output, call_id):
        """
        Complete a function call by setting the previous message to done and adding the output message if provided.
        """

        if len(self) == 0 or self[-1]["ai"].get("type") != "function_call":
            msg = "No function call to complete"
            raise ValueError(msg)

        self[-1]["display"]["metadata"]["status"] = "done"

        self.append(
            {
                "display": {
                    "role": "assistant",
                    "content": output,
                    "metadata": {
                        "title": f"📖 Result from {self[-1]['ai']['name']}",
                        "status": "done",
                    },
                },
                "ai": {
                    "type": "function_call_output",
                    "output": output,
                    "call_id": call_id,
                },
            },
        )

    def undo(self, index):
        """
        Undo to a specific index in the history.
        This removes all messages after the specified index. Reverts to a previous state

        Args:
            index (int): The index to revert to. Must be a user message.
        Returns:
            str: The content of the last user message after undo.
        """
        if index < 0 or index >= len(self):
            msg = f"Index out of range for undo: {index}"
            raise ValueError(msg)

        last_message = self[index]["display"]["content"]

        del self[index:]

        return last_message

    def edit(self, index, new_content):
        """
        Edit the content of a specific message in the history.
        Will only let you edit user message.

        Args:
            index (int): The index of the message to edit.
            new_content (str): The new content for the message.
        """
        if index < 0 or index >= len(self):
            msg = f"Index out of range for edit: {index}"
            raise ValueError(msg)

        if self[index]["display"]["role"] != "user":
            msg = "Can only edit user messages"
            raise ValueError(msg)

        self[index]["display"]["content"] = new_content
        self[index]["ai"]["content"] = new_content

        del self[index + 1 :]

    def openai_format(self):
        """
        Convert to OpenAI message format.
        This means the only two formats are:
        messages with a "role" and "content"
        or function calls with "type", "output", "call_id"
        """
        return [msg["ai"] for msg in self]

    def gradio_format(self):
        """
        Convert to Gradio message format.
        All formats are the same, they must have atleast role and content, but could also have metadata and status.
        """
        return [msg["display"] for msg in self]


class AssistantPrompts:
    """
    A collection of prompt templates for the Assistant.
    All methods are static since they don't require instance state.
    """

    @staticmethod
    def conversation_title():
        return """
You are part of a chatbot assistant at the Transport Accident Investigation Commission that help users add titles to their conversation. You will receive the conversation and you are too response with a 5 word summary of the conversation.
Provide a title that will best help the user identify what conversation it was.
Just respond with the title and nothing else.
        """

    @staticmethod
    def general_info(columns, rows, last_updated):
        return f"""
Below is general information to help you contextualise the user's query.

**Dataset Information:**
The core of your tools are built around a vector database that contains accident reports from various transport accident investigation commissions, including:
- New Zealand (TAIC)
- Australia (ATSB)
- Canada (TSB)
There are {len(columns)} columns with {rows} rows.
The columns available are: {", ".join(columns)}
The data was last updated on {last_updated}.

**Key Definitions:**

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
"""

    @staticmethod
    def orient_plan_system(general_info):
        return f"""
You are a expert working at the New Zealand transport accident investigation commission. Your job is to assistant employees of TAIC with their queries. The day is {datetime.now(timezone.utc)}. You should respond as if you are a senior accident investigator/research who is speaking to your colleagues.

You will be provided the conversation history including any function calls and output you have made. You are to orient yourself to the user's query and provide a plan for how you will react to the user's query. If you need more information you should call functions to get that information. If you have enough information to respond to the user, you should provide a short guideline for how you will respond to the user (you will be acting on this plan momentarily).

{general_info}

"""

    @staticmethod
    def act_system(general_info):
        return f"""
You are a expert working at the New Zealand transport accident investigation commission. Your job is to assistant employees of TAIC with their queries. The day is {datetime.now(timezone.utc)}. You should respond as if you are a senior accident investigator/research who is speaking to your colleagues. Keep your responses short and to the point.

You will be provided the conversation history including the plan you have made.
You are to act on your plan, this may involve calling functions to get more information or providing a response for the user.

If you choose to respond to the user, ensure you provide a concise and accurate answer based on the information available. If you reference any reports, ensure you provide the report IDs. If you reference any other document you should provide the document type and document ID.

{general_info}
"""


class Assistant:
    def __init__(
        self,
        searcher,
        openai_api_key,
        openai_endpoint=None,
    ):
        print("[bold]Creating Chatbot[/bold]")
        self.searcher = searcher
        if openai_endpoint:
            self.openai_client = openai.AzureOpenAI(
                api_version="2025-04-01-preview",
                api_key=openai_api_key,
                azure_endpoint=openai_endpoint,
            )
        else:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)

        # Initialize tools
        self.tools = [
            SearchTool(searcher),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}

        print("[bold]Chatbot created[/bold]")

    def provide_conversation_title(
        self,
        history=None,
        conversation_context_length=5,
    ) -> str:
        """
        Read the conversation history and provide a short title for the conversation.
        """
        if history is None:
            history = []
        if history == []:
            msg = "history is empty"
            raise ValueError(msg)

        # Clear history to only include the last 5 user or assistant messages, no system messages or tool calls
        msg = [
            m["display"]
            for m in history
            if m["display"]["role"] in ["user", "assistant"]
            and m["display"].get("metadata") is None
        ]

        if len(msg) > conversation_context_length:
            msg = msg[-conversation_context_length:]

        return (
            self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": AssistantPrompts.conversation_title(),
                    },
                    *msg,
                ],
            )
            .choices[0]
            .message.content
        )

    def complete_tool_use(
        self,
        history: CompleteHistory,
        function_call,
        chunk,
    ) -> Generator[tuple[CompleteHistory, list[dict], bool], None, None]:
        """
        Complete a tool use by executing the tool and returning the result.
        """
        # Add function call to history
        history.add_function_call(chunk.item.to_dict())
        yield history, history.gradio_format(), True

        # Execute the function
        tool_name = function_call.name
        tool_args = json.loads(function_call.arguments)

        tool = self.tool_map.get(tool_name)
        if not tool:
            result = f"Error: Unknown tool {tool_name}"
        else:
            result = tool.execute(**tool_args)

        # Add function result to history
        history.complete_function_call(
            output=result,
            call_id=chunk.item.call_id,
        )
        yield history, history.gradio_format(), True

    def process_streamed_input(
        self,
        response,
        history: CompleteHistory,
    ) -> Generator[tuple[CompleteHistory, list[dict], bool], None, None]:
        """
        Process a streamed response from OpenAI, handling both text and function calls.

        Args:
            response: Streamed response from OpenAI API
            history: The conversation history to update

        Yields:
            tuple: (updated_history, gradio_formatted_history, has_function_calls)

        Returns:
            bool: True if any function calls were made (indicating we need another loop iteration)
        """
        function_calls = {}
        has_function_calls = False

        for chunk in response:
            # Prepare to collect either function calls or text deltas
            if chunk.type == "response.output_item.added":
                if chunk.item.type == "function_call":
                    function_calls[chunk.output_index] = chunk.item
                    has_function_calls = True
                elif chunk.item.type == "message":
                    history.add_message("assistant", "")
                    yield history, history.gradio_format(), has_function_calls

            # Collect the function call arguments as they stream in
            elif chunk.type == "response.function_call_arguments.delta":
                index = chunk.output_index
                if function_calls.get(index):
                    function_calls[index].arguments += chunk.delta

            # Collect the message text as it streams in
            elif chunk.type == "response.output_text.delta":
                history.update_last_message(chunk.delta)
                yield history, history.gradio_format(), has_function_calls

            # Handle completed function call - execute it
            elif (
                chunk.type == "response.output_item.done"
                and chunk.item.type == "function_call"
            ):
                # Use yield from to delegate to complete_tool_use generator
                # This allows the UI to update as the tool is executed
                yield from self.complete_tool_use(
                    history=history,
                    function_call=function_calls[chunk.output_index],
                    chunk=chunk,
                )
            # Handle message done and yield final message
            elif (
                chunk.type == "response.output_item.done"
                and chunk.item.type == "message"
            ):
                yield history, history.gradio_format(), has_function_calls

        return has_function_calls

    def process_input(
        self,
        history: CompleteHistory,
    ) -> Generator[tuple[CompleteHistory, list[dict]], None, None]:
        """
        Process user input and generate a response using the orient/plan/act loop.

        This is a generator that yields the updated history after each step, allowing
        the UI to update in real-time as the assistant thinks and responds.

        Flow:
        1. Orient/Plan: Assistant analyzes the query and plans its response
        2. Act: Assistant executes the plan (may call tools or respond directly)
        3. Loop: If tools were called, repeat from step 1 to process tool results

        Args:
            history: The conversation history containing the user's latest message

        Yields:
            tuple: (updated_history, gradio_formatted_history) after each update

        Returns:
            tuple: Final (history, gradio_formatted_history) when complete
        """
        # Validate inputs
        if not isinstance(history, CompleteHistory):
            msg = "history must be a CompleteHistory instance"
            raise TypeError(msg)
        if len(history) == 0:
            msg = "history is empty"
            raise ValueError(msg)

        print(f"[bold]Processing user input {history[-1]['display']['content']}[/bold]")

        # Prepare system messages with context
        general_info = AssistantPrompts.general_info(
            columns=self.searcher.all_document_types_table.schema.names,
            rows=self.searcher.all_document_types_table.count_rows(),
            last_updated=self.searcher.last_updated,
        )
        orient_plan_system_message = AssistantPrompts.orient_plan_system(general_info)
        act_system_message = AssistantPrompts.act_system(general_info)

        # Orient/Plan/Act loop - continue until no more function calls are needed
        while True:
            # STEP 1: Orient and Plan
            # Assistant thinks about the query and plans how to respond
            orient_plan_response = self.openai_client.responses.create(
                model="gpt-4.1",
                instructions=orient_plan_system_message,
                input=history.openai_format(),
                tools=[tool.to_openai_format() for tool in self.tools],
                store=False,
                stream=True,
                tool_choice="none",  # No tool calls in planning phase
            )

            # Stream the planning thoughts to the UI
            history.start_thought()
            for chunk in orient_plan_response:
                if chunk.type == "response.output_text.delta":
                    history.update_last_message(chunk.delta)
                    yield history, history.gradio_format()

            history.end_thought()
            yield history, history.gradio_format()

            # STEP 2: Act on the Plan
            # Assistant executes the plan (may call tools or provide final response)
            act_response = self.openai_client.responses.create(
                model="gpt-4.1",
                instructions=act_system_message,
                input=history.openai_format(),
                tools=[tool.to_openai_format() for tool in self.tools],
                parallel_tool_calls=True,
                store=False,
                stream=True,
            )

            # Process the action response (handles both text and function calls)
            has_function_calls = False
            for history_update, gradio_update, had_calls in self.process_streamed_input(
                act_response,
                history,
            ):
                history = history_update
                has_function_calls = had_calls
                yield history, gradio_update

            # STEP 3: Decide whether to loop
            # If function calls were made, loop again to let assistant process results
            # Otherwise, we're done - assistant has provided final response
            if not has_function_calls:
                break

        return history, history.gradio_format()
