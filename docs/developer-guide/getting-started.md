# Getting Started

This guide will help you set up a local development environment for TAIC Report Engine. First though you should read the [User Guide](../user-guide/overview.md) to understand what the engine does and how to use it.

## Prerequisites

- Linux or WSL (Windows Subsystem for Linux)
- Python 3.11+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/1jamesthompson1/TAIC-engine
cd TAIC-engine
```

### 2. Install uv

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

```bash
uv sync --dev
uv run pre-commit install
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with the Azure Storage account name, Azure OpenAI endpoint, and (when needed) `AZURE_CLIENT_ID`.

Azure authentication uses Microsoft Entra ID only. There are no storage account keys, SAS tokens, or Azure OpenAI API keys. Run `az login` locally so `DefaultAzureCredential` can use your Azure CLI identity. In production and CI, the engine uses a user-assigned managed identity selected with `AZURE_CLIENT_ID`:

- `id-taicengine-prod` — production engine
- `id-taicengine-test` — test pipeline and CI

Assign each identity `Storage Blob Data Contributor` on its containers and `Cognitive Services OpenAI User` on the Azure OpenAI resource. Test resources use the `test` prefix (for example `test-vectordb`, `test-reportpdfs`, `testengineoutput`).

For TAIC developers please see the devops wiki for instructions on how to get access to the Azure resources.

### 5. (Optional) Run the Engine

The engine running will take days to complete and cost hundreds in API costs. Instead one would usually download the latest output from the Azure blob storage and only experiment from there.

Downloading of latest output requires `AZURE_STORAGE_ACCOUNT_NAME` in `.env`, an authenticated Azure CLI session (`az login`), and `Storage Blob Delegator` on the storage account so the helper can create a short-lived user delegation SAS. Then you can run:

```bash
uv run azure --latest-output
```

Afterwards running the engine will only process reports that have not been processed yet. You can run the engine with:

```bash
uv run engine -t all
```

## Project Structure

```
TAIC-report-summary/
├── engine/                 # Core pipeline modules
│   ├── AICaller.py        # AI model interface
│   ├── AzureStorage.py    # Azure blob/table storage
│   ├── Combine.py         # Data combination logic
│   ├── Config.py          # Configuration management
│   ├── Embedding.py       # Vector embedding generation
│   ├── ExtractionModels.py # Pydantic extraction models
│   ├── ExtractionPrompts.py # LLM prompt templates
│   ├── Logging.py         # Centralized logging
│   ├── Modes.py           # Simple transport mode enums
│   ├── PDFParsing.py      # PDF text extraction
│   ├── ReportExtracting.py # Main extraction orchestrator
│   ├── SavedDataFrames.py # DataFrame persistence
│   └── WebsiteScraping.py # Agency website scraping
├── data/                  # Raw data
├── output/                # Extracted datasets it is gitignored
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
└── workbench/             # Experimental code it is gitignored
```

## Running Tests

```bash
uv run pytest
```

## Next Steps

- Read the [Architecture Guide](architecture.md)
- Review the [API Reference](../api/index.md)
- Check the [Contributing Guidelines](contributing.md)
