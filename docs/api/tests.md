# Tests

This section describes the test modules for each component of the TAIC Report Engine. Each test module covers the corresponding API module's functionality, including edge cases, integration scenarios, and expected error handling.

The tests use `pytest` with fixtures defined in `tests/conftest.py`. Tests that interact with Azure Storage use dedicated test containers to avoid affecting production data.

## Test Modules

| Module | Description |
|--------|-------------|
| [TestAICaller](test_aicaller.md) | Tests for AI model querying, token limits, and structured output |
| [TestAzureStorage](test_azurestorage.md) | Tests for upload/download to Azure Blob Storage |
| [TestCombine](test_combine.md) | Tests for merging scraped and extracted data into long-format DataFrames |
| [TestEmbedding](test_embedding.md) | Tests for vector embedding generation and storage |
| [TestMetadataExtraction](test_metadataextraction.md) | Tests for extracting occurrence metadata, vehicle details, and personnel |
| [TestPDFParser](test_pdfparser.md) | Tests for PDF text extraction using stable test containers |
| [TestReportExtracting](test_reportextracting.md) | Tests for the main extraction pipeline, chunking, and parallel processing |
| [TestWebsiteScraping](test_websitescraping.md) | Tests for agency website scraping and PDF download |

## Running Tests

I would recommend running the tests with vscode or your IDE. If needed though you can do it via CLI:

```bash
# Run all tests
uv run pytest tests/

# Run a specific test module
uv run pytest tests/test_AICaller.py

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=engine --cov-report=html
```

## Test Configuration

The test suite reads configuration from `config.yaml` via the `conftest.py` fixtures. Key test infrastructure:

- **Azure Containers**: Tests use dedicated containers (e.g., `test-stable-reportpdfs`) to isolate test data
- **Test PDFs**: A stable set of PDFs is maintained in the `test-stable-reportpdfs` container
- **Cleanup**: Session-scoped fixtures automatically clean up test containers after runs
