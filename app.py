import re
from fastapi import FastAPI, Request, Depends
from numpy import isin
from starlette.config import Config
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
import gradio as gr
import dotenv
import os
import uvicorn
from rich import print, table
import logging

logging.basicConfig(level=logging.INFO)

import assistant
dotenv.load_dotenv(override=True)

app = FastAPI()

# Azure AD OAuth settings
AZURE_CLIENT_ID = os.getenv("CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("TENANT_ID")
SECRET_KEY = os.urandom(24)

config_data = {
    'AZURE_CLIENT_ID': AZURE_CLIENT_ID,
    'AZURE_CLIENT_SECRET': AZURE_CLIENT_SECRET
}
starlette_config = Config(environ=config_data)
oauth = OAuth(starlette_config)
oauth.register(
    name='azure',
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    server_metadata_url=f'https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0/.well-known/openid-configuration',
    authorize_url=f'https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/authorize',
    authorize_params=None,
    access_token_url=f'https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token',
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri='http://localhost:8000/auth',
    client_kwargs={'scope': 'openid profile email'},
)

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Dependency to get the current user
def get_user(request: Request):
    user = request.session.get('user')
    if user:
        return user['name']
    return RedirectResponse(url='/login-page')

@app.get('/')
def public(user: dict = Depends(get_user)):
    if isinstance(user, str):
        return RedirectResponse(url='/assistant')
    else:
        return RedirectResponse(url='/login-page')

@app.route('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url='/')

@app.route('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth')
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@app.route('/auth')
async def auth(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user = await oauth.azure.parse_id_token(token, token['userinfo']["nonce"])
    except OAuthError:
        return RedirectResponse(url='/')
    request.session['user'] = user
    return RedirectResponse(url='/')

def handle_undo(self, history, undo_data: gr.UndoData):
    return history[:undo_data.index], history[undo_data.index]['content']

def handle_submit(user_input, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return "", history

chatbot_instance = assistant.assistant(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    voyageai_api_key=os.getenv("VOYAGEAI_API_KEY"),
    db_uri=os.getenv("db_URI")
)

def get_welcome_message(request: gr.Request):
    return "Logged in as: {}".format(request.username)

with gr.Blocks(
    title="TAIC smart assistant",
    theme=gr.themes.Base(),
    fill_height=True,
    fill_width=True
) as assistant_page:

    with gr.Row():
        gr.Markdown("# TAIC smart assistant demo")
        username = gr.Markdown('Logged in as: ')
        gr.Button("Logout", link="/logout")

        assistant_page.load(
            get_welcome_message, None, username
        )

    chatbot_interface = gr.Chatbot(
        type="messages",
        height="90%",
        min_height=400,
        avatar_images=(None, "https://www.taic.org.nz/themes/custom/taic/favicon/android-icon-192x192.png")
    )

    input_text = gr.Textbox(placeholder="Type your message here...", show_label=False)

    chatbot_interface.undo(
        fn=handle_undo,
        inputs=chatbot_interface,
        outputs=[chatbot_interface, input_text],
    )

    input_text.submit(fn=handle_submit, inputs=[input_text, chatbot_interface], outputs=[input_text, chatbot_interface], queue=False).then(
        chatbot_instance.process_input, inputs=[chatbot_interface], outputs=[chatbot_interface]
    )

app = gr.mount_gradio_app(app, assistant_page, path="/assistant", auth_dependency=get_user, show_api=False)

with gr.Blocks() as login_page:
    gr.Button("Login", link="/login")

app = gr.mount_gradio_app(app, login_page, path="/login-page", show_api=False)

if __name__ == '__main__':
    uvicorn.run(app)