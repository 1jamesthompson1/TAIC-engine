import logging
import os
import re
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import dotenv
import gradio as gr
import pandas as pd
import uvicorn
from authlib.integrations.starlette_client import OAuth, OAuthError
from azure.data.tables import TableServiceClient
from fastapi import Depends, FastAPI, HTTPException, Request
from gradio_rangeslider import RangeSlider
from rich import print  # noqa: A004
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.status import HTTP_302_FOUND

from backend import Assistant, Searching, Storage, Version

logging.basicConfig(level=logging.WARNING)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Removing excessive logging from azure sdk
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING,
)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.data.tables").setLevel(logging.WARNING)
dotenv.load_dotenv(override=True)

# Setup the storage connection
print("[bold green]✓ Initializing Azure Storage connection[/bold green]")
connection_string = f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
client = TableServiceClient.from_connection_string(conn_str=connection_string)
knowledgesearchlogs_table_name = os.getenv("KNOWLEDGE_SEARCH_LOGS_TABLE_NAME")
conversation_metadata_table_name = os.getenv("CONVERSATION_METADATA_TABLE_NAME")
conversation_container_name = os.getenv("CONVERSATION_CONTAINER_NAME")
knowledge_search_container_name = os.getenv(
    "KNOWLEDGE_SEARCH_CONTAINER_NAME",
    "knowledgesearchlogs",
)

# Check all environment variables are set
if not all(
    [
        knowledgesearchlogs_table_name,
        conversation_metadata_table_name,
        conversation_container_name,
    ],
):
    missing_vars = [
        var
        for var in [
            "KNOWLEDGE_SEARCH_LOGS_TABLE_NAME",
            "CONVERSATION_METADATA_TABLE_NAME",
            "CONVERSATION_CONTAINER_NAME",
        ]
        if not locals()[var]
    ]
    msg = f"Missing environment variables: {', '.join(missing_vars)}"
    raise ValueError(msg)

# Initialize Table Storage clients
knowledgesearchlogs = client.create_table_if_not_exists(
    table_name=knowledgesearchlogs_table_name,
)
chatlogs = client.create_table_if_not_exists(
    table_name=conversation_metadata_table_name,
)

# Initialize Blob Storage clients
conversation_blob_store = Storage.ConversationBlobStore(
    connection_string,
    conversation_container_name,
)
knowledge_search_blob_store = Storage.KnowledgeSearchBlobStore(
    connection_string,
    knowledge_search_container_name,
)

# Initialize metadata stores
conversation_store = Storage.ConversationMetadataStore(
    chatlogs,
    conversation_blob_store,
)
knowledge_search_store = Storage.KnowledgeSearchMetadataStore(
    knowledgesearchlogs,
    knowledge_search_blob_store,
)

print("[bold green]✓ All storage systems initialized[/bold green]")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Azure AD OAuth settings
AZURE_CLIENT_ID = os.getenv("CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("TENANT_ID")
SECRET_KEY = os.urandom(24)

config_data = {
    "AZURE_CLIENT_ID": AZURE_CLIENT_ID,
    "AZURE_CLIENT_SECRET": AZURE_CLIENT_SECRET,
}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name="azure",
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    server_metadata_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0/.well-known/openid-configuration",
    authorize_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/authorize",
    authorize_params=None,
    access_token_url=f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token",
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri="/auth",
    client_kwargs={"scope": "openid profile email"},
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# Dependency to get the current user
def get_user(request: Request):
    # Developer no-login mode: return a dev username and set session
    if os.getenv("NO_LOGIN", "false").lower() == "true":
        dev_username = os.getenv("DEV_USERNAME", "devuser")
        # Try to ensure session user is available for code paths that read it
        request.session["user"] = {"name": dev_username}
        return dev_username

    user = request.session.get("user")
    if user:
        return user["name"]
    raise HTTPException(
        status_code=HTTP_302_FOUND,
        detail="Not authenticated",
        headers={"Location": "/login-page"},
    )


@app.get("/")
def public(user: dict = Depends(get_user)):  # noqa: B008
    if user:
        return RedirectResponse(url="/tools")
    return RedirectResponse(url="/login-page")


@app.route("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


@app.route("/login")
async def login(request: Request):
    # Forcing deployed url to use https. This is because it seems to default to using http.
    if "localhost" not in request.url.hostname and "https" not in request.url.scheme:
        redirect_uri = "https://" + request.url.hostname + request.url_for("auth").path
    else:
        redirect_uri = request.url_for("auth")
    return await oauth.azure.authorize_redirect(request, redirect_uri)


@app.route("/auth")
async def auth(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user = await oauth.azure.parse_id_token(token, token["userinfo"]["nonce"])
    except OAuthError:
        return RedirectResponse(url="/")
    request.session["user"] = user
    return RedirectResponse(url="/")


# =====================================================================
#
# Assistant UI functions
#
# =====================================================================
def handle_undo(history: Assistant.CompleteHistory, undo_data: gr.UndoData):
    last_message = history.undo(undo_data.index)
    return history, history.gradio_format(), last_message


def handle_edit(history: Assistant.CompleteHistory, edit_data: gr.EditData):
    history.edit(edit_data.index, edit_data.value)
    return history, history.gradio_format(), None, edit_data.value


def handle_example_select(
    selection: gr.SelectData,
    current_conversation: Assistant.CompleteHistory,
):
    return handle_submit(selection.value["text"], current_conversation)


def handle_submit(
    user_input,
    history: Assistant.CompleteHistory = None,
    conversation_id=None,
):
    if history is None:
        history = Assistant.CompleteHistory([])
    history.add_message("user", user_input)

    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    return (
        gr.Textbox(interactive=False, value=None),
        user_input,
        history,
        history.gradio_format(),
        conversation_id,
    )


def create_or_update_conversation(
    request: gr.Request,
    conversation_id,
    conversation_title,
    history: Assistant.CompleteHistory,
):
    if history == []:  # ignore instance when history is empty
        return None

    username = request.username

    # Generate conversation title only every 3rd after the first message that
    # the user sends
    if (
        len(
            [msg for msg in history.gradio_format() if msg["role"] == "user"],
        )
        % 3
        == 1
    ):
        conversation_title = assistant_instance.provide_conversation_title(history)

    print(f"Looking at updating conversation {conversation_id} for user {username}")

    if os.getenv("NO_LOGS", "false").lower() == "true":
        print(
            "[orange]⚠ NO_LOGS is set to true, skipping storing conversation[/orange]",
        )
        return conversation_title

    # Clean conversation_id to just the UUID part if it has extra formatting
    uuid_regex = (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    match = re.search(uuid_regex, conversation_id)
    cleaned_conversation_id = match.group(0) if match else conversation_id

    # Store using blob storage (JSON in blob + metadata in table)
    success = conversation_store.create_or_update_conversation(
        username=username,
        conversation_id=cleaned_conversation_id,
        history=history,
        conversation_title=conversation_title,
        db_version=searching_instance.db_version,
    )

    if not success:
        print(f"[bold red]✗ Failed to store conversation {conversation_id}[/bold red]")

    return conversation_title


def get_user_conversations_metadata(request: gr.Request):
    username = request.username

    # Get only conversation metadata (no full message history)
    return conversation_store.get_user_conversations_metadata(username)


def load_conversation(request: gr.Request, conversation_id: str):
    username = request.username

    print(
        f"[orange]Loading conversation {conversation_id} for user {username}[/orange]",
    )

    # Load the full conversation with message history
    conversation = conversation_store.load_single_conversation(
        username,
        conversation_id,
    )
    if conversation:
        # Check version compatibility
        stored_version = conversation.get("app_version")
        is_compatible, version_message = Version.is_compatible(stored_version or "")

        if not is_compatible:
            print(
                f"[bold red]⚠ Version incompatibility for conversation {conversation_id}: {version_message}[/bold red]",
            )
            formatted_id = f"ID: `{conversation['id']}`"
            formatted_title = (
                f"{conversation['conversation_title']} ⚠️ *Version Incompatible*"
            )
            # Return empty messages with error indication
            error_message = {
                "display": {
                    "role": "assistant",
                    "content": f"⚠️ **Version Compatibility Error**\n\n{version_message}\n\nThis conversation may not load correctly due to version differences. Consider creating a new conversation.",
                },
                "ai": {
                    "role": "assistant",
                    "content": f"⚠️ **Version Compatibility Error**\n\n{version_message}\n\nThis conversation may not load correctly due to version differences. Consider creating a new conversation.",
                },
            }
            error_messages = [error_message] + conversation["messages"]
            history = Assistant.CompleteHistory(error_messages)
            return (
                history,
                history.gradio_format(),
                formatted_id,
                formatted_title,
                gr.Button(
                    visible=True,
                ),
            )

        if version_message:  # Minor version differences
            print(
                f"[orange]⚠ Version warning for conversation {conversation_id}: {version_message}[/orange]",
            )
            # Add a subtle warning to the title but still load the conversation
            formatted_title = f"{conversation['conversation_title']} ⚠️"
        else:
            formatted_title = conversation["conversation_title"]

        print(f"[bold green]✓ Loaded conversation {conversation_id}[/bold green]")

        history = Assistant.CompleteHistory(conversation["messages"])

        return (
            history,
            history.gradio_format(),
            conversation["id"],
            formatted_title,
            gr.Button(
                visible=True,
            ),
        )
    print(f"[bold red]✗ Failed to load conversation {conversation_id}[/bold red]")
    history = Assistant.CompleteHistory([])
    return history, [], conversation_id, "Failed to load", gr.Button(visible=True)


def delete_conversation(  # noqa: PLR0913
    request: gr.Request,
    current_conv: gr.State,
    chatbot: gr.Chatbot,
    current_conv_id: gr.State,
    current_conv_title: gr.State,
    to_delete: str | None,
):
    """
    Delete a conversation and clear current conversation state if needed.
    Will ask the user for confirmation before calling this function.
    """
    username = request.username

    if current_conv_id is None:
        new_conversation_button = gr.Button(visible=False)
    else:
        new_conversation_button = gr.Button(visible=True)

    if to_delete is None:
        print("[orange]Deletion of conversation cancelled by user[/orange]")
        return (
            current_conv,
            chatbot,
            current_conv_id,
            current_conv_title,
            new_conversation_button,
        )

    if current_conv_id == to_delete:
        # Clear current conversation if it's the one being deleted
        current_conv = Assistant.CompleteHistory([])
        chatbot = gr.Chatbot(value=current_conv.gradio_format(), type="messages")
        current_conv_id = None
        current_conv_title = None
        new_conversation_button = gr.Button(visible=False)

    success = conversation_store.delete_conversation(
        username,
        to_delete,
    )

    if success:
        print(f"[bold green]✓ Deleted conversation {to_delete}[/bold green]")
    else:
        print(f"[bold red]✗ Failed to delete conversation {to_delete}[/bold red]")
        gr.Warning("Failed to delete conversation. Try again later.")

    return (
        current_conv,
        chatbot,
        current_conv_id,
        current_conv_title,
        new_conversation_button,
    )


searching_instance = Searching.Searcher(
    db_uri=os.getenv("VECTORDB_PATH"),
    table_name=os.getenv("VECTORDB_TABLE_NAME"),
)

assistant_instance = Assistant.Assistant(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
    openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    searcher=searching_instance,
)

# =====================================================================
#
# Knowledge search UI functions
#
# =====================================================================


def get_user_search_history(request: gr.Request):
    username = request.username

    # Get only search metadata (no full detailed data)
    searches = knowledge_search_store.get_user_search_history(username, limit=20)
    print(f"[bold green]✓ Retrieved metadata for {len(searches)} searches[/bold green]")
    return searches


def load_previous_search(request: gr.Request, search_id: str):
    username = request.username

    print(f"[orange]Loading search {search_id} for user {username}[/orange]")

    # Load the full search with detailed data
    search_data = knowledge_search_store.load_detailed_search(username, search_id)
    if search_data:
        print(f"[bold green]✓ Loaded search {search_id}[/bold green]")

        # Extract search settings from detailed data
        detailed_data = search_data["detailed_data"]
        search_settings = detailed_data.get("search_settings", {})

        # Return values to populate the search form
        query = search_settings.get("query", "")
        year_range = list(
            search_settings.get(
                "year_range",
                [2007, datetime.now(tz=timezone.utc).year],
            ),
        )

        # Map document types back to UI format
        document_type_reverse_mapping = {
            "safety_issue": "Safety issues",
            "recommendation": "Safety recommendations",
            "report_section": "Report sections",
            "report_text": "Report summaries",
        }
        mapped_document_types = search_settings.get("document_type", [])
        if isinstance(mapped_document_types, list):
            document_type = [
                document_type_reverse_mapping.get(dt, dt)
                for dt in mapped_document_types
            ]
        else:
            document_type = document_type_reverse_mapping.get(
                mapped_document_types,
                mapped_document_types,
            )

        mode_mapping = {
            0: "Aviation",
            1: "Rail",
            2: "Maritime",
        }
        modes = search_settings.get("modes", [])
        if isinstance(modes, list):
            modes = [mode_mapping.get(mode, mode) for mode in modes]
        else:
            modes = [mode_mapping.get(modes, modes)]
        agencies = search_settings.get("agencies", [])
        relevance = search_settings.get("relevance", 0.6)

        return query, year_range, document_type, modes, agencies, relevance
    print(f"[bold red]✗ Failed to load search {search_id}[/bold red]")
    return "", [2007, datetime.now(tz=timezone.utc).year], [], [], [], 0.6


def create_complete_search_params(
    query: str,
    year_range: list,
    document_type: list,
    modes: list,
    agencies: list,
):
    """
    Create complete SearchParams by filling in defaults for any missing fields.
    """
    search_type = (
        "none"
        if (query == "" or query is None)
        else ("fts" if query[0] == '"' and query[-1] == '"' else "vector")
    )
    if search_type == "fts":
        query = query[1:-1]  # Remove quotes for exact match search

    document_type_mapping = {
        "Safety issues": "safety_issue",
        "Safety recommendations": "recommendation",
        "Report sections": "report_section",
        "Report summaries": "summary",
    }

    mapped_document_type = [
        document_type_mapping[dt] for dt in document_type if dt in document_type_mapping
    ]

    return Searching.SearchParams(
        query=query,
        search_type=search_type,
        year_range=year_range,
        document_type=mapped_document_type,
        modes=modes,
        agencies=agencies,
    )


def format_search_results(  # noqa: PLR0913
    results: pd.DataFrame,
    plots: dict,
    info: dict,
    search_settings: Searching.SearchParams,
    search_start_time: datetime,
    username: str,
):
    """
    Format the search results for display and download.
    """
    results_to_download = results.copy()
    # Format the results to be displayed in the dataframe
    if not results.empty:
        results["agency_id"] = results.apply(
            lambda x: f"<a href='{x['url']}' style='color: #1a73e8; text-decoration-line: underline;'>{x['agency_id']}</a>",
            axis=1,
        )
        results = results.drop(columns=["url"])
        message = f"""Found {info["relevant_results"]} results from database.
_These are the relevant results (out of {info["total_results"]}) from the search of the database, there is a no guarantee of its completeness._"""

    else:
        message = "No results found for the given criteria."
        # Ensure plots are None if no results
        plots = dict.fromkeys(
            ["document_type", "mode", "year", "agency", "event_type"],
        )

    download_dict = {
        "settings": search_settings,
        "results": results_to_download,
        "search_start_time": search_start_time,
        "username": username,
    }

    # Prepare results information
    results_info = {
        "total_results": info.get("total_results", 0),
        "relevant_results": info.get("relevant_results", 0),
        "has_results": results is not None,
        "plots_generated": plots is not None,
    }

    # Add trimmed results data for storage (if available)
    if results is not None:
        # Prepare results for storage (trimmed version)
        trimmed_results = results.head(100).drop(columns=["document"], errors="ignore")
        results_info["sample_results"] = trimmed_results.to_dict(orient="records")

    return results, results_info, message, download_dict, plots


def perform_actual_search(
    settings: Searching.SearchParams,
    relevance: float,
    limit: int = 5000,
):
    results, info, plots = searching_instance.knowledge_search(
        settings,
        relevance=relevance,
        limit=limit,
    )

    return results, info, plots


def perform_search(  # noqa: PLR0913
    username: str,
    query: str,
    year_range: list,
    document_type: list,
    modes: list,
    agencies: list,
    relevance: float,
):
    search_start_time = datetime.now(tz=timezone.utc)
    error_info = None
    try:
        # Build complete search parameters
        search_settings = create_complete_search_params(
            query=query,
            year_range=year_range,
            document_type=document_type,
            modes=modes,
            agencies=agencies,
        )

        results, info, plots = perform_actual_search(
            settings=search_settings,
            relevance=relevance,
        )

        results, results_info, message, download_dict, plots = format_search_results(
            results,
            plots,
            info,
            search_settings,
            search_start_time,
            username,
        )

    except (ValueError, TypeError, KeyError) as e:
        error_info = {
            "error": str(e),
            "error_trace": traceback.format_exc(),
            "occurred_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    try:
        if os.getenv("NO_LOGS", "false").lower() == "true":
            print(
                "[orange]⚠ NO_LOGS is set to true, skipping storing search log[/orange]",
            )
        else:
            search_id = str(uuid.uuid4())
            knowledge_search_store.store_search_log(
                username=username,
                search_id=search_id,
                search_settings=search_settings,
                relevance=relevance,
                results_info=results_info,
                error_info=error_info,
            )
            print(f"✓ Stored search log with ID: {search_id}")
    except Exception as e:  # noqa: BLE001
        print(f"✗ Failed to store search log: {e}")

    if error_info is not None:
        msg = f"An error has occurred during your search, please refresh page and try again.\nError: \n{error_info}"
        raise gr.Error(
            title="Error while conducting search",
            message=msg,
            duration=5,
        )

    # Return plots along with results and message
    return (
        results,
        download_dict,
        message,
        plots.get("document_type"),
        plots.get("mode"),
        plots.get("year"),
        plots.get("agency"),
        plots.get("event_type"),
    )


def update_download_button(download_dict: dict):
    """
    Update the download button to to point to a new temporary file that is the results ready to be downloaded.
    """
    if download_dict["results"] is None or download_dict["results"].empty:
        return gr.DownloadButton(visible=False)
    save_name = f"{download_dict['settings'].query[: min(20, len(download_dict['settings'].query))]}_{download_dict['search_start_time'].strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    temp_dir = Path(tempfile.mkdtemp())
    file_path = temp_dir / save_name

    summary_data = [
        [
            "This spreadsheet contains the results of a search conducted using the TAIC knowledge search tool.",
            "",
        ],
        [
            f"The search was conducted on {download_dict['search_start_time'].strftime('%Y-%m-%d %H:%M:%S')} by {download_dict['username']}. The settings used for the search can be found below and the full results can be found in the 'Results' sheet.",
            "",
        ],
    ]
    for key, value in download_dict["settings"]._asdict().items():
        summary_data.append([f"{key}:", str(value)])
    summary_df = pd.DataFrame(summary_data, columns=["Setting", "Value"])

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, header=False, sheet_name="Summary")
        download_dict["results"].to_excel(writer, index=False, sheet_name="Results")

    return gr.DownloadButton(
        label="Download results (spreadsheet)",
        value=file_path,
        visible=True,
    )


# =====================================================================
#
# Gradio UI
#
# =====================================================================


def get_welcome_message(request: gr.Request):
    return (
        request.username,
        f"**Data:** {searching_instance.last_updated} • **App:** {Version.CURRENT_VERSION} • **DB:** {searching_instance.db_version}",
    )


def get_user_name(request: gr.Request):
    return request.username


TAIC_theme = gr.themes.Default(
    primary_hue=gr.themes.utils.colors.Color(
        name="primary_hue",
        c50="#2e679c",
        **{f"c{n}": "#2e679c" for n in range(100, 901, 100)},
        c950="#2e679c",
    ),
    secondary_hue=gr.themes.utils.colors.Color(
        name="secondary_hue",
        c50="#e6dca1",
        **{f"c{n}": "#e6dca1" for n in range(100, 901, 100)},
        c950="#e6dca1",
    ),
    neutral_hue="gray",
)


def get_footer():
    return gr.HTML(f"""
<style>
    .custom-footer {{
        margin-top: 20px;
        text-align: center;
        padding: 10px;
        background-color: {TAIC_theme.primary_50};
        color: {TAIC_theme.neutral_50};
    }}
</style>
<div class="custom-footer">
    <p>Created by <a href="https://github.com/1jamesthompson1">James Thompson</a> for the <a href="https://www.taic.org.nz">New Zealand Transport Accident Investigation Commission.</a></p>
    <p>Contact directed to <a href="mailto:james.thompson@taic.org.nz">james.thompson@taic.org.nz</a> or for suggestions and/or bug reports please use the provided <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=RmxQlKGu1key34UuP4dPFavrlUtUJCpGvY1oQw3ObrlUQjZSTFRFUDRZRk8wUUxPWkVYVEw1SUVDUy4u" target="_blank">feeback form</a>.</p>
    <p>Project is being developed openly on <a href="https://github.com/1jamesthompson1/TAIC_smart_assistant">https://github.com/1jamesthompson1/TAIC_smart_assistant</a></p>
    <p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="CC logo"><img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="BY logo"></a></p>
</div>
""")


with gr.Blocks(
    title="TAIC smart tools",
    theme=TAIC_theme,
    fill_height=True,
    fill_width=True,
    head='<link rel="icon" href="/static/favicon.png" type="image/png">',
    css="""
footer {visibility: hidden} /* Hides the default gradio footer */
""",  # Hides the footer
) as smart_tools:
    username = gr.State()
    smart_tools.load(get_user_name, inputs=None, outputs=username)

    with gr.Row():
        gr.Markdown("# TAIC smart tools")
        data_update = gr.Markdown("Data last updated: ")
        gr.Markdown("Logged in as:")
        username_display = gr.Markdown()
        logout_button = gr.Button("Logout", link="/logout")

    smart_tools.load(
        get_welcome_message,
        inputs=None,
        outputs=[username_display, data_update],
    )

    with gr.Tabs():
        with gr.TabItem("Assistant") as assistant_tab:
            user_conversations = gr.State([])
            new_input = gr.State(None)
            current_conversation = gr.State(None)
            current_conversation_id = gr.State(None)
            current_conversation_title = gr.State(None)

            chatbot_interface = gr.Chatbot(
                type="messages",
                height="90%",
                min_height=400,
                show_label=False,
                watermark="This paragraph was generated by AI, please check thoroughly before using.",
                editable="user",
                avatar_images=(
                    None,
                    "https://www.taic.org.nz/themes/custom/taic/favicon/android-icon-192x192.png",
                ),
                render=False,
                examples=[
                    {
                        "display_text": "Recent TAIC safety issues",
                        "text": "Can you please provide a summary of recent safety issues identified by TAIC in the last 2 years?",
                    },
                    {
                        "display_text": "Breakdown of common causes of incidents",
                        "text": "What are the common threads that run through aviation safety incidents investigated by TAIC over the last decade? Is this different from what is seen in ATSB aviation incident reports?",
                    },
                    {
                        "display_text": "Mentions of 'International Maritime Organization'",
                        "text": "How many times has the 'International Maritime Organization' been mentioned in TAIC's investigation reports?",
                    },
                ],
                placeholder="#### Welcome to the TAIC smart assistant\nI have access to TAIC's, ATSB's and TSB's investigations reports, safety issues and recommendations from 2000 to present day. Ask me anything in the box below, or try out one of the example questions!\n\n*Please note that while I strive to provide accurate and helpful information, I may occasionally generate incorrect or nonsensical responses. Always verify critical information from authoritative sources.*\n\n##### Examples",
            )

            input_text = gr.Textbox(
                placeholder="Please type your message here...",
                show_label=False,
                submit_btn="Send",
                render=False,
                lines=3,
            )
            new_conversation_button = gr.Button(
                "New conversation",
                visible="hidden",
                render=False,
            )

            smart_tools.load(
                fn=get_user_conversations_metadata,
                inputs=None,
                outputs=user_conversations,
            )

            assistant_process = new_input.change(
                assistant_instance.process_input,
                inputs=[current_conversation],
                outputs=[current_conversation, chatbot_interface],
                trigger_mode="once",
            )
            ui_update = (
                assistant_process.then(
                    create_or_update_conversation,
                    inputs=[
                        current_conversation_id,
                        current_conversation_title,
                        current_conversation,
                    ],
                    outputs=current_conversation_title,
                )
                .then(
                    get_user_conversations_metadata,
                    inputs=None,
                    outputs=user_conversations,
                )
                .then(
                    lambda: gr.Textbox(interactive=True),
                    None,
                    input_text,
                )
                .then(
                    lambda: gr.Button(visible=True),
                    None,
                    new_conversation_button,
                )
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Previous conversations")

                    to_delete = gr.JSON(None, visible="hidden")
                    to_delete.change(
                        fn=delete_conversation,
                        inputs=[
                            current_conversation,
                            chatbot_interface,
                            current_conversation_id,
                            current_conversation_title,
                            to_delete,
                        ],
                        outputs=[
                            current_conversation,
                            chatbot_interface,
                            current_conversation_id,
                            current_conversation_title,
                            new_conversation_button,
                        ],
                    ).then(
                        get_user_conversations_metadata,
                        inputs=None,
                        outputs=user_conversations,
                    )

                    @gr.render(inputs=[user_conversations])
                    def render_conversations(conversations):
                        for conversation in conversations:
                            with gr.Group(), gr.Row():
                                gr.Markdown(
                                    f"### {conversation['conversation_title']}",
                                    container=False,
                                )
                                with gr.Column(min_width=40):

                                    def create_load_function(conv_id):
                                        def load_func(request: gr.Request):
                                            return load_conversation(request, conv_id)

                                        return load_func

                                    gr.Button("load", size="sm").click(
                                        fn=create_load_function(conversation["id"]),
                                        inputs=None,
                                        outputs=[
                                            current_conversation,
                                            chatbot_interface,
                                            current_conversation_id,
                                            current_conversation_title,
                                            new_conversation_button,
                                        ],
                                    )
                                    js = f"""
                                    () => {{
                                        return confirm(`Are you sure you want to delete\n\n{conversation["conversation_title"]}\n\nThis action cannot be undone.`)
                                            ? '{conversation["id"]}'
                                            : null;
                                    }}
                                    """
                                    gr.Button(
                                        "delete conversation",
                                        size="sm",
                                    ).click(
                                        None,
                                        None,
                                        to_delete,
                                        js=js,
                                    )

                with gr.Column(scale=3):
                    with gr.Row():
                        conversation_title = gr.Markdown(None)
                    with gr.Row():
                        conversation_id = gr.Markdown(None)
                        new_conversation_button.render()

                    chatbot_interface.render()

                    # Update the conversation title and id displays when the current conversation state changes
                    current_conversation_title.change(
                        lambda title: f"## {title}" if title else None,
                        inputs=current_conversation_title,
                        outputs=conversation_title,
                    ).then(
                        lambda cid: f"ID: `{cid}`" if cid else None,
                        current_conversation_id,
                        conversation_id,
                    )

                    input_text.render()

            # Handling the clearning of the conversation
            clear_trigger = gr.State(None)
            clear_trigger.change(
                lambda: (
                    None,
                    [],
                    Assistant.CompleteHistory([]),
                    None,
                    None,
                ),
                None,
                [
                    conversation_id,
                    chatbot_interface,
                    current_conversation,
                    current_conversation_id,
                    current_conversation_title,
                ],
                queue=False,
            ).then(
                get_user_conversations_metadata,
                inputs=None,
                outputs=user_conversations,
            ).then(
                lambda: gr.Button(visible=False),
                None,
                new_conversation_button,
            )

            chatbot_interface.clear(
                uuid.uuid4,
                None,
                clear_trigger,
                js=True,
            )

            new_conversation_button.click(
                uuid.uuid4,
                None,
                clear_trigger,
                js=True,
            )

            # Handle undo action
            chatbot_interface.undo(
                fn=handle_undo,
                inputs=current_conversation,
                outputs=[current_conversation, chatbot_interface, input_text],
            )

            # Handle examples, edit and submit actions

            chatbot_interface.edit(
                fn=handle_edit,
                inputs=current_conversation,
                outputs=[
                    current_conversation,
                    chatbot_interface,
                    input_text,
                    new_input,
                ],
            )

            input_text.submit(
                fn=handle_submit,
                inputs=[input_text, current_conversation, current_conversation_id],
                outputs=[
                    input_text,
                    new_input,
                    current_conversation,
                    chatbot_interface,
                    current_conversation_id,
                ],
            )

            chatbot_interface.example_select(
                fn=handle_example_select,
                inputs=[current_conversation],
                outputs=[
                    input_text,
                    new_input,
                    current_conversation,
                    chatbot_interface,
                    current_conversation_id,
                ],
            )

        with gr.TabItem("Knowledge Search"):
            search_results_to_download = gr.State(None)
            user_search_history = gr.State([])
            smart_tools.load(
                get_user_search_history,
                inputs=None,
                outputs=user_search_history,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Previous searches")

                    @gr.render(inputs=[user_search_history])
                    def render_search_history(search_history):
                        for search in search_history:
                            with gr.Row():
                                query_text = search.get("query", "No query")
                                max_query_display_length = 30
                                if len(query_text) > max_query_display_length:
                                    query_text = (
                                        query_text[:max_query_display_length] + "..."
                                    )
                                timestamp = search.get("search_timestamp", "Unknown")
                                results_count = search.get("relevant_results", 0)

                                gr.Markdown(
                                    f"**{query_text}**  \n*{timestamp}* ({results_count} results)",
                                    container=False,
                                )

                                def create_load_search_function(search_id):
                                    def load_func(request: gr.Request):
                                        return load_previous_search(request, search_id)

                                    return load_func

                                gr.Button("load").click(
                                    fn=create_load_search_function(search["search_id"]),
                                    inputs=None,
                                    outputs=[
                                        query,
                                        year_range,
                                        document_type,
                                        modes,
                                        agencies,
                                        relevance,
                                    ],
                                )

                with gr.Column(scale=2):
                    query = gr.Textbox(label="Search Query")
                    search_button = gr.Button("Search")
                    with gr.Row():
                        with gr.Column():
                            search_summary = gr.Markdown()
                        with gr.Column():
                            download_button = gr.DownloadButton(
                                "Download results",
                                visible=False,
                            )
                with (
                    gr.Column(scale=1),
                    gr.Accordion("Advanced Search Options", open=True),
                ):
                    current_year = datetime.now(tz=timezone.utc).year
                    year_range = RangeSlider(
                        label="Year Range",
                        minimum=2000,
                        maximum=current_year,
                        step=1,
                        value=[2007, current_year],
                    )
                    document_type = gr.CheckboxGroup(
                        label="Document Type",
                        choices=[
                            "Safety issues",
                            "Safety recommendations",
                            "Report sections",
                            "Report summaries",
                        ],
                        value=["Safety issues", "Safety recommendations"],
                    )
                    modes = gr.CheckboxGroup(
                        label="Modes of Transport",
                        choices=["Aviation", "Rail", "Maritime"],
                        value=["Aviation", "Rail", "Maritime"],
                        type="index",
                    )
                    agencies = gr.CheckboxGroup(
                        label="Agencies",
                        choices=["TAIC", "ATSB", "TSB"],
                        value=["TAIC"],
                    )
                    gr.Markdown(
                        "*Note: Relevance threshold is highly sensitive and should be tuned if too many or too few results are being returned. Increase relevance to reduce the number of results*",
                    )
                    relevance = gr.Slider(
                        label="Relevance",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.2,
                    )

            with gr.Accordion(label="Result graphs", open=False), gr.Row():
                doc_type_plot = gr.Plot()
                mode_plot = gr.Plot()
                agency_plot = gr.Plot()
                year_hist = gr.Plot()
                event_type_plot = gr.Plot()

            with gr.Row():
                search_results = gr.Dataframe(
                    max_chars=1500,
                    max_height=10000,
                    pinned_columns=1,
                    wrap=True,
                    type="pandas",
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "html",
                        "str",
                    ],
                    show_fullscreen_button=True,
                    show_search="search",
                )

            search_outputs = [
                search_results,
                search_results_to_download,
                search_summary,
                doc_type_plot,
                mode_plot,
                year_hist,
                agency_plot,
                event_type_plot,
            ]

            search_button.click(
                perform_search,
                inputs=[
                    username,
                    query,
                    year_range,
                    document_type,
                    modes,
                    agencies,
                    relevance,
                ],
                outputs=search_outputs,
                scroll_to_output=True,
            ).then(
                update_download_button,
                search_results_to_download,
                download_button,
            ).then(
                get_user_search_history,
                inputs=None,
                outputs=user_search_history,
            )
            query.submit(
                perform_search,
                inputs=[
                    username,
                    query,
                    year_range,
                    document_type,
                    modes,
                    agencies,
                    relevance,
                ],
                outputs=search_outputs,
            ).then(
                update_download_button,
                search_results_to_download,
                download_button,
            ).then(
                get_user_search_history,
                inputs=None,
                outputs=user_search_history,
            )
    footer = get_footer()

app = gr.mount_gradio_app(
    app,
    smart_tools,
    path="/tools",
    auth_dependency=lambda request: get_user(request),
    show_api=False,
)


# Add head parameter to login_page Blocks
with gr.Blocks(
    title="TAIC smart tools login",
    theme=TAIC_theme,
    fill_height=True,
    head='<link rel="icon" href="/static/favicon.png" type="image/png">',
    css="""
.complete-center {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}
""",
) as login_page:
    with gr.Column(elem_classes="complete-center"):
        gr.Markdown("# TAIC smart tools")
        gr.Markdown("Please login to continue:")
        gr.Button("Login", link="/login")

    footer = get_footer()


app = gr.mount_gradio_app(app, login_page, path="/login-page", show_api=False)

if __name__ == "__main__":
    uvicorn.run(app)
