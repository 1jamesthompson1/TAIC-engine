import os
import tempfile
from unittest.mock import Mock, patch

import dotenv
import pytest
from fastapi.testclient import TestClient

from app import app as fastapi_app
from backend import Searching
from backend.Assistant import Assistant

dotenv.load_dotenv()  # Load environment variables from .env file if present

# Always set this to true
os.environ["NO_LOGIN"] = "false"
os.environ["NO_LOGS"] = "false"  # Ensure logging is enabled for tests


@pytest.fixture(scope="session")
def use_real_services():
    """Check if tests should use real services or mocks."""
    return os.getenv("TEST_USE_REAL_SERVICES", "false").lower() == "true"


@pytest.fixture(scope="session")
def mock_env_vars(use_real_services):
    """Mock environment variables required for the app."""
    env_vars = {
        "AZURE_STORAGE_ACCOUNT_NAME": os.getenv(
            "AZURE_STORAGE_ACCOUNT_NAME",
            "teststorage",
        ),
        "AZURE_STORAGE_ACCOUNT_KEY": os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "testkey"),
        "KNOWLEDGE_SEARCH_LOGS_TABLE_NAME": os.getenv(
            "KNOWLEDGE_SEARCH_LOGS_TABLE_NAME",
            "testlogs",
        ),
        "CONVERSATION_METADATA_TABLE_NAME": os.getenv(
            "CONVERSATION_METADATA_TABLE_NAME",
            "testmeta",
        ),
        "CONVERSATION_CONTAINER_NAME": os.getenv(
            "CONVERSATION_CONTAINER_NAME",
            "testconv",
        ),
        "VECTORDB_PATH": os.getenv(
            "VECTORDB_PATH",
            tempfile.NamedTemporaryFile(delete=False).name,  # noqa: SIM115
        ),
        "VECTORDB_TABLE_NAME": os.getenv("VECTORDB_TABLE_NAME", "testtable"),
        "CLIENT_ID": os.getenv("CLIENT_ID", "testclient"),
        "CLIENT_SECRET": os.getenv("CLIENT_SECRET", "testsecret"),
        "TENANT_ID": os.getenv("TENANT_ID", "testtenant"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "testkey"),
        "AZURE_OPENAI_ENDPOINT": os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://test.openai.azure.com/",
        ),
        "NO_LOGS": "false",  # Enable logging for real tests
    }

    # Only patch if not using real services
    if not use_real_services:
        with patch.dict(os.environ, env_vars):
            yield
    else:
        # Use real environment variables
        yield


@pytest.fixture(scope="session")
def mock_azure_storage(use_real_services):
    """Mock Azure storage components."""
    if use_real_services:
        # Import and use real components
        # Don't mock, let it use real services
        yield None
    else:
        with (
            patch("app.TableServiceClient") as mock_table_client,
            patch("app.Storage") as mock_storage,
            patch("backend.Storage") as mock_backend_storage,
        ):
            # Mock table client
            mock_client = Mock()
            mock_table_client.from_connection_string.return_value = mock_client
            mock_client.create_table_if_not_exists.return_value = None

            # Mock storage classes
            mock_conv_store = Mock()
            mock_knowledge_store = Mock()
            mock_conv_meta_store = Mock()
            mock_knowledge_meta_store = Mock()

            mock_storage.ConversationBlobStore.return_value = mock_conv_store
            mock_storage.KnowledgeSearchBlobStore.return_value = mock_knowledge_store
            mock_backend_storage.ConversationMetadataStore.return_value = (
                mock_conv_meta_store
            )
            mock_backend_storage.KnowledgeSearchMetadataStore.return_value = (
                mock_knowledge_meta_store
            )

            yield {
                "table_client": mock_client,
                "conv_store": mock_conv_store,
                "knowledge_store": mock_knowledge_store,
                "conv_meta_store": mock_conv_meta_store,
                "knowledge_meta_store": mock_knowledge_meta_store,
            }


@pytest.fixture(scope="session")
def mock_searcher(use_real_services):
    """Mock or use real Searching.Searcher class."""
    if use_real_services:
        searcher = Searching.Searcher(
            db_uri=os.getenv("VECTORDB_PATH"),
            table_name=os.getenv("VECTORDB_TABLE_NAME"),
        )
        yield searcher
    else:
        with patch("backend.Searching.Searcher") as mock_searcher_class:
            mock_searcher = Mock()
            mock_searcher.db_version = "test-v1.0"
            mock_searcher.last_updated = "2024-01-01"
            mock_searcher.all_document_types_table = Mock()
            mock_searcher.all_document_types_table.schema.names = [
                "id",
                "content",
                "type",
            ]
            mock_searcher.all_document_types_table.count_rows.return_value = 1000
            mock_searcher.knowledge_search.return_value = (
                Mock(),
                {"total_results": 10, "relevant_results": 5},
                {},
            )

            mock_searcher_class.return_value = mock_searcher
            yield mock_searcher


@pytest.fixture(scope="session")
def mock_openai(use_real_services):
    """Mock or use real OpenAI client."""
    if use_real_services:
        # Use real OpenAI client with env vars
        yield None
    else:
        with patch("openai.AzureOpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"

            # For the new responses API
            mock_chunk = Mock()
            mock_chunk.type = "response.output_text.delta"
            mock_chunk.delta = "Test response"
            mock_client.responses.create.return_value = [mock_chunk]

            mock_openai_class.return_value = mock_client
            yield mock_client


@pytest.fixture(scope="session")
def client(
    mock_env_vars,  # noqa: ARG001
    mock_azure_storage,  # noqa: ARG001
    mock_searcher,  # noqa: ARG001
    mock_openai,  # noqa: ARG001
):
    """
    Test client for the FastAPI app.
    Uses session scope to load only once for all tests, improving performance.

    Dependencies are explicitly listed to ensure proper initialization order.
    The fixtures are not used in the function body but are required for setup.
    """
    with TestClient(fastapi_app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def mock_assistant(mock_searcher, use_real_services):
    """Mock or real Assistant instance. Session-scoped for performance."""

    if use_real_services:
        assistant = Assistant(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            searcher=mock_searcher,
        )
    else:
        assistant = Assistant(
            openai_api_key="test",
            openai_endpoint="https://test.openai.azure.com/",
            searcher=mock_searcher,
        )
    return assistant
