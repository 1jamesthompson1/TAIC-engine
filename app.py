import uuid
from fastapi import FastAPI, Request, Depends
from starlette.config import Config
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
import gradio as gr
import dotenv
import os
import uvicorn
from rich import print
import logging
from azure.data.tables import TableServiceClient
import json
from datetime import datetime

import assistant

logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv(override=True)

connection_string = f"AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
client = TableServiceClient.from_connection_string(conn_str=connection_string)
chatlogs = client.create_table_if_not_exists(table_name="chatlogs")

app = FastAPI()

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
    return None


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


def handle_undo(history, undo_data: gr.UndoData):
    return history[: undo_data.index], history[undo_data.index]["content"]


def handle_submit(user_input, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})

    return "", history


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
        messages = json.loads("".join(all_messages))

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
    print(previous_conversations)
    return previous_conversations


assistant_instance = assistant.assistant(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    voyageai_api_key=os.getenv("VOYAGEAI_API_KEY"),
    db_uri=os.getenv("VECTORDB_PATH"),
)


def get_welcome_message(request: gr.Request):
    return request.username


with gr.Blocks(
    title="TAIC smart assistant",
    theme=gr.themes.Base(),
    fill_height=True,
    fill_width=True,
) as assistant_page:
    user_conversations = gr.State([])

    assistant_page.load(get_user_conversations, inputs=None, outputs=user_conversations)

    with gr.Row():
        gr.Markdown("# TAIC smart assistant demo")
        gr.Markdown("Logged in as:")
        username = gr.Markdown()
        logout_button = gr.Button("Logout", link="/logout")

    # Redirecto login page if not logged in
    assistant_page.load(get_welcome_message, inputs=[], outputs=[username])

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
                            f"### {conversation['conversation_title']}", container=False
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
                avatar_images=(
                    None,
                    "https://www.taic.org.nz/themes/custom/taic/favicon/android-icon-192x192.png",
                ),
            )

            input_text = gr.Textbox(
                placeholder="Type your message here...", show_label=False
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
    )

    chatbot_interface.change(
        create_or_update_conversation,
        inputs=[conversation_id, chatbot_interface],
        outputs=None,
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
    )


app = gr.mount_gradio_app(
    app, assistant_page, path="/assistant", auth_dependency=get_user, show_api=False
)


with gr.Blocks() as login_page:
    gr.Button("Login", link="/login")

app = gr.mount_gradio_app(app, login_page, path="/login-page", show_api=False)

if __name__ == "__main__":
    uvicorn.run(app)
