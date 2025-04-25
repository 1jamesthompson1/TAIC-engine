import uuid
from fastapi import FastAPI, Request, Depends, HTTPException
from starlette.config import Config
from starlette.status import HTTP_302_FOUND
from starlette.responses import RedirectResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth, OAuthError
import gradio as gr
from gradio_rangeslider import RangeSlider
import dotenv
import os
import uvicorn
from rich import print
import logging
from azure.data.tables import TableServiceClient
import json
from datetime import datetime

from backend import Assistant, Searching

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv(override=True)

connection_string = f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
client = TableServiceClient.from_connection_string(conn_str=connection_string)
chatlogs = client.create_table_if_not_exists(table_name="chatlogs")

app = FastAPI()

# Mount static files directory *before* mounting Gradio apps
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
    user = request.session.get("user")
    if user:
        return user["name"]
    raise HTTPException(
        status_code=HTTP_302_FOUND,
        detail="Not authenticated",
        headers={"Location": "/login-page"},
    )


@app.get("/")
def public(user: dict = Depends(get_user)):
    if user:
        return RedirectResponse(url="/assistant")
    else:
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
# Assistant / Gradio UI Setup
#
# =====================================================================


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_submit(user_input, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})

    return gr.Textbox(interactive=False, value=None), history


def create_or_update_conversation(request: gr.Request, conversation_id, history):
    if history == []:  # ignore instance when history is empty
        return
    # Split messages into chunks of no more than 60kb
    history_json = json.dumps(history)
    MAX_LEN = 30000
    history_chunks = []
    while len(history_json) > MAX_LEN:
        history_chunks.append(history_json[:MAX_LEN])
        history_json = history_json[MAX_LEN:]
    history_chunks.append(history_json)

    messages = {
        f"messages_{index}": history_chunk
        for index, history_chunk in enumerate(history_chunks)
    }

    username = request.username
    previous_logs = list(
        chatlogs.query_entities(
            f"PartitionKey eq '{username}' and RowKey eq '{conversation_id}'"
        )
    )
    if len(previous_logs) == 1:
        previous_entity = chatlogs.get_entity(
            partition_key=username, row_key=conversation_id
        )
        chatlogs.update_entity(
            entity={
                **{
                    "PartitionKey": username,
                    "RowKey": conversation_id,
                    "conversation_title": previous_entity.get("conversation_title"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                **messages,
            },
        )
    elif len(previous_logs) == 0:
        chatlogs.create_entity(
            entity={
                **messages,
                **{
                    "PartitionKey": username,
                    "RowKey": conversation_id,
                    "conversation_title": assistant_instance.provide_conversation_title(
                        history
                    ),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
        )
    else:
        raise ValueError(
            f"More than one conversation found, found {len(previous_logs)}"
        )


def get_user_conversations(request: gr.Request):
    converstations = chatlogs.query_entities(
        query_filter=f"PartitionKey eq '{request.username}'"
    )

    previous_conversations = list()

    for conversation in converstations:
        all_messages = [
            conversation[message_key]
            for message_key in conversation.keys()
            if message_key.startswith("messages")
        ]

        try:
            messages = json.loads("".join(all_messages))
        except json.JSONDecodeError:
            print(
                f"Failed to decode messages for conversation {conversation['PartitionKey']} and {conversation['RowKey']}"
            )
            continue

        previous_conversations.append(
            {
                "conversation_title": conversation.get("conversation_title"),
                "messages": messages,
                "id": conversation.get("RowKey"),
                "last_updated": conversation.get("last_updated"),
            }
        )

    # Sort by last updated
    previous_conversations = sorted(
        previous_conversations,
        key=lambda x: datetime.strptime(x["last_updated"], "%Y-%m-%d %H:%M:%S"),
        reverse=True,
    )

    print(f"[bold]Found {len(previous_conversations)} user conversations[/bold]")
    return previous_conversations


searching_instance = Searching.Searcher(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    voyageai_api_key=os.getenv("VOYAGEAI_API_KEY"),
    db_uri=os.getenv("VECTORDB_PATH"),
)

assistant_instance = Assistant.Assistant(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    searcher=searching_instance,
)

# =====================================================================
#
# Knowledge search
#
# =====================================================================


def perform_search(
    query: str,
    year_range: list[int],
    document_type: list[str],
    modes: list[str],
    agencies: list[str],
    relevance: float,
):
    search_type = (
        "none"
        if (query == "" or query is None)
        else ("fts" if query[0] == '"' and query[-1] == '"' else "vector")
    )
    if search_type == "fts":
        query = query[1:-1]

    document_type_mapping = {
        "Safety Issues": "safety_issue",
        "Safety Recommendations": "recommendation",
        "Report sections": "report_section",
        "Entire Reports": "report_text",
    }

    mapped_document_type = [
        document_type_mapping[dt] for dt in document_type if dt in document_type_mapping
    ]

    results = searching_instance.knowledge_search(
        query=query,
        year_range=year_range,
        document_type=mapped_document_type,
        modes=modes,
        agencies=agencies,
        type=search_type,
        limit=5000,
        relevance=relevance,
    )

    # Format the results to be displayed in the dataframe
    results["agency_id"] = results.apply(
        lambda x: f"<a href='{x['url']}' style='color: #1a73e8; text-decoration-line: underline;'>{x['agency_id']}</a>",
        axis=1,
    )
    results.drop(columns=["url"], inplace=True)

    message = f"""Found {len(results)} results from database.  
_These are the relevant results from the search of the database, there is a no guarantee of its completeness._"""

    return results, message


def get_welcome_message(request: gr.Request):
    return request.username, f"Data last updated: {searching_instance.last_updated}"

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
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: {TAIC_theme.primary_50};
        color: {TAIC_theme.neutral_50};
    }}
</style>
<div class="custom-footer">
    <p>Created by <a href="https://github.com/1jamesthompson1">James Thompson</a> for the <a href="https://www.taic.org.nz">New Zealand Transport Accident Investigation Commission.</a></p>
    <p>Contact directed to <a href="mailto:james.thompson@taic.org.nz">james.thompson@taic.org.nz</a> or for suggestions and/or bug reports please use the provided <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=RmxQlKGu1key34UuP4dPFavrlUtUJCpGvY1oQw3ObrlUQjZSTFRFUDRZRk8wUUxPWkVYVEw1SUVDUy4u" target="_blank">feeback form</a>.</p>
    <p>Project is being developed openly on <a href="https://github.com/1jamesthompson1/TAIC_smart_assistant">https://github.com/1jamesthompson1/TAIC-report-summary</a></p>
    <p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="CC logo"><img style="height:22px!important;margin-left:3px;vertical-align:middle;display: inline-block;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="BY logo"></a></p>                  
</div>
""")

# Add head parameter to assistant_page Blocks
with gr.Blocks(
    title="TAIC smart tools",
    theme=TAIC_theme,
    fill_height=True,
    fill_width=True,
    head='<link rel="icon" href="/static/favicon.png" type="image/png">'
) as assistant_page:
    user_conversations = gr.State([])

    assistant_page.load(get_user_conversations, inputs=None, outputs=user_conversations)

    with gr.Row():
        gr.Markdown("# TAIC smart tools")
        data_update = gr.Markdown("Data last updated: ")
        gr.Markdown("Logged in as:")
        username = gr.Markdown()
        logout_button = gr.Button("Logout", link="/logout")
    
    # Redirect to login page if not logged in
    assistant_page.load(get_welcome_message, inputs=[], outputs=[username, data_update])

    with gr.Tabs():
        with gr.TabItem("Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Current conversation")
                    conversation_id = gr.Markdown(str(uuid.uuid4()))
                    gr.Markdown("### Previous conversations")

                    @gr.render(inputs=[user_conversations])
                    def render_conversations(conversations):
                        for conversation in conversations:
                            with gr.Row():
                                gr.Markdown(
                                    f"### {conversation['conversation_title']}",
                                    container=False,
                                )

                                def load_conversation(conversation=conversation):
                                    return conversation["messages"], conversation["id"]

                                gr.Button("load").click(
                                    fn=load_conversation,
                                    inputs=None,
                                    outputs=[chatbot_interface, conversation_id],
                                )

                with gr.Column(scale=3):
                    with gr.Row():
                        gr.Markdown("### Chat: ")

                    chatbot_interface = gr.Chatbot(
                        type="messages",
                        height="90%",
                        min_height=400,
                        show_label=False,
                        avatar_images=(
                            None,
                            "https://www.taic.org.nz/themes/custom/taic/favicon/android-icon-192x192.png",
                        ),
                    )

                    input_text = gr.Textbox(
                        placeholder="Please type your message here...",
                        show_label=False,
                        submit_btn="Send",
                    )
            chatbot_interface.undo(
                fn=handle_undo,
                inputs=chatbot_interface,
                outputs=[chatbot_interface, input_text],
            )
            chatbot_interface.clear(
                lambda: (str(uuid.uuid4()), []),
                None,
                [conversation_id, chatbot_interface],
                queue=False,
            ).then(
                get_user_conversations,
                inputs=None,
                outputs=user_conversations,
            )

            input_text.submit(
                fn=handle_submit,
                inputs=[input_text, chatbot_interface],
                outputs=[input_text, chatbot_interface],
                queue=False,
            ).then(
                assistant_instance.process_input,
                inputs=[chatbot_interface],
                outputs=[chatbot_interface],
            ).then(
                create_or_update_conversation,
                inputs=[conversation_id, chatbot_interface],
                outputs=None,
            ).then(
                get_user_conversations,
                inputs=None,
                outputs=user_conversations,
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                input_text,
            )

        with gr.TabItem("Knowledge Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    query = gr.Textbox(label="Search Query")
                    search_button = gr.Button("Search")
                    search_summary = gr.Markdown()
                with gr.Column(scale=1):
                    with gr.Accordion("Advanced Search Options", open=True):
                        current_year = datetime.now().year
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
                                "Safety Issues",
                                "Safety Recommendations",
                                "Report sections",
                                "Entire Reports",
                            ],
                            value=["Safety Issues", "Safety Recommendations"],
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
                        relevance = gr.Slider(
                            label="Relevance",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.6,
                        )

                    search = [
                        query,
                        year_range,
                        document_type,
                        modes,
                        agencies,
                        relevance,
                    ]

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

            search_button.click(
                perform_search, inputs=search, outputs=[search_results, search_summary]
            )
            query.submit(
                perform_search,
                inputs=search,
                outputs=[search_results, search_summary],
            )
    footer = get_footer()

app = gr.mount_gradio_app(
    app, assistant_page, path="/assistant", auth_dependency=get_user, show_api=False
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
"""
) as login_page:
    with gr.Column(elem_classes="complete-center"):
        gr.Markdown("# TAIC smart tools")
        gr.Markdown("Please login to continue:")
        gr.Button("Login", link="/login")

    footer = get_footer()


app = gr.mount_gradio_app(app, login_page, path="/login-page", show_api=False)

if __name__ == "__main__":
    uvicorn.run(app)
