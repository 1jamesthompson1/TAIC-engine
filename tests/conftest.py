"""Pytest fixtures and session hooks for loading test config and Azure PDF storage cleanup."""

import contextlib
import json
import logging
import os
from pathlib import Path

import dotenv
import pytest

from engine.utils import Config
from engine.utils.AICaller import get_api_costs, print_api_cost_summary
from engine.utils.AzureStorage import PDFStorageManager


@pytest.fixture(scope="session", autouse=True)
def load_test_config():
    """Load test configuration and setup logging for the test session.

    This fixture runs once per test session and configures:
    - Logging at DEBUG level for all tests
    - Test configuration from test_config.yaml
    - Environment variables from .env file

    The loaded configuration is stored in pytest.config and made available
    to all tests in the session.
    """
    # Configure logging for tests.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Silence noisy Azure HTTP logging in test output.
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )

    config = Config.ConfigReader(Path("tests") / "test_config.yaml").get_config()
    pytest.config = config

    pytest.output_config = config["engine"]["output"]
    pytest.vector_config = config["engine"]["vector"]

    dotenv.load_dotenv(override=True)


@pytest.fixture(scope="function")
def test_pdf_storage_manager():
    """Create a PDF storage manager for tests.

    This fixture is available to all tests in the suite and provides access
    to the test Azure storage container for PDF operations.

    Usage in test files:
    ```python
    def test_my_pdf_function(test_pdf_storage_manager):
        # Use the PDF storage manager
        pdf_list = test_pdf_storage_manager.list_pdfs()
        assert len(pdf_list) >= 0

        # Upload a test PDF
        test_pdf_storage_manager.upload_pdf("test_report", b"fake pdf data")
    ```

    The fixture automatically connects to the test container specified in test_config.yaml.

    Returns:
        PDFStorageManager: Configured manager for the test PDF container.
    """
    return PDFStorageManager(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        pytest.output_config["pdf_container_name"],
    )


@pytest.fixture(scope="function")
def stable_pdf_storage_manager():
    """Create a PDF storage manager for stable test PDFs.

    This fixture connects to a separate container with a consistent set of test PDFs
    that are NOT automatically cleaned up. This is useful for tests that need
    reliable, consistent PDF data.

    Usage:
    ```python
    def test_pdf_parsing(stable_pdf_storage_manager):
        # This container has a known set of test PDFs
        pdf_list = stable_pdf_storage_manager.list_pdfs()
        # pdf_list will always contain the same test reports
    ```

    Note: This container is separate from the regular test container and is not
    subject to automatic cleanup.

    Returns:
        PDFStorageManager: Configured manager for the stable PDF container.
    """
    return PDFStorageManager(
        os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
        pytest.output_config["stable_pdf_container_name"],
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_containers():
    """Universal cleanup fixture for Azure test containers.

    This fixture automatically cleans up Azure storage containers after each test
    to prevent accumulation of test data and associated storage costs.
    Available to all tests in the suite.

    NOTE: This does NOT clean up the stable PDF container, which is meant to
    contain consistent test data.
    """
    # This runs before each test
    yield

    # This runs after each test completes
    try:
        # Clean up regular PDF container (but NOT the stable one)
        pdf_storage_manager = PDFStorageManager(
            os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
            os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
            pytest.output_config["pdf_container_name"],
        )

        # Get list of all blobs and delete them silently
        all_blobs = pdf_storage_manager.list_blobs()
        for blob_name in all_blobs:
            with contextlib.suppress(Exception):
                # Silently ignore individual deletion failures
                pdf_storage_manager.delete_blob(blob_name)

    except Exception:
        # Don't fail the test run if cleanup fails
        pass


def pytest_sessionstart(session):
    """Called before tests start. Clean up previous cost files."""
    # Only run on master
    if not hasattr(session.config, "workerinput"):
        # Ensure directory exists
        cache_dir = session.config.cache.makedir("aicosts")
        # Clear existing files
        for f in os.listdir(cache_dir):
            with contextlib.suppress(Exception):
                os.remove(os.path.join(cache_dir, f))


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finishes."""
    # If we are in a worker node (xdist)
    if hasattr(session.config, "workerinput"):
        costs = get_api_costs()
        worker_id = session.config.workerinput["workerid"]
        # Save to cache
        cache_dir = session.config.cache.makedir("aicosts")
        with open(
            os.path.join(str(cache_dir), f"costs_{worker_id}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(costs, f)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Aggregate and print costs from all workers (or just main process if not using xdist)."""
    # Only run on master
    if not hasattr(config, "workerinput"):
        aggregated_costs = {
            "total_cost": 0.0,
            "input_cost": 0.0,
            "cached_input_cost": 0.0,
            "output_cost": 0.0,
            "total_input_tokens": 0,
            "total_cached_tokens": 0,
            "total_output_tokens": 0,
            "calls": 0,
            "by_model": {},
        }

        # Load costs from main process
        main_costs = get_api_costs()
        cache_dir = config.cache.makedir("aicosts")
        if os.path.exists(cache_dir):
            cost_files = [
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.startswith("costs_")
            ]
        else:
            cost_files = []

        all_costs_list = [main_costs]

        for cost_file in cost_files:
            try:
                with open(cost_file, encoding="utf-8") as f:
                    all_costs_list.append(json.load(f))
            except Exception:
                pass  # Ignore errors in reading these files

        # Aggregate
        for cost in all_costs_list:
            aggregated_costs["total_cost"] += cost["total_cost"]
            aggregated_costs["input_cost"] += cost["input_cost"]
            aggregated_costs["cached_input_cost"] += cost["cached_input_cost"]
            aggregated_costs["output_cost"] += cost["output_cost"]
            aggregated_costs["total_input_tokens"] += cost["total_input_tokens"]
            aggregated_costs["total_cached_tokens"] += cost["total_cached_tokens"]
            aggregated_costs["total_output_tokens"] += cost["total_output_tokens"]
            aggregated_costs["calls"] += cost["calls"]

            for model, model_stats in cost["by_model"].items():
                if model not in aggregated_costs["by_model"]:
                    aggregated_costs["by_model"][model] = {
                        "total_cost": 0.0,
                        "input_cost": 0.0,
                        "cached_input_cost": 0.0,
                        "output_cost": 0.0,
                        "input_tokens": 0,
                        "cached_tokens": 0,
                        "output_tokens": 0,
                        "calls": 0,
                    }

                start_model = aggregated_costs["by_model"][model]
                start_model["total_cost"] += model_stats["total_cost"]
                start_model["input_cost"] += model_stats["input_cost"]
                start_model["cached_input_cost"] += model_stats["cached_input_cost"]
                start_model["output_cost"] += model_stats["output_cost"]
                start_model["input_tokens"] += model_stats["input_tokens"]
                start_model["cached_tokens"] += model_stats["cached_tokens"]
                start_model["output_tokens"] += model_stats["output_tokens"]
                start_model["calls"] += model_stats["calls"]

        # Print summary
        if aggregated_costs["calls"] > 0:
            print_api_cost_summary(aggregated_costs)
