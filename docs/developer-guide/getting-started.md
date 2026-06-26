# Getting Started

This guide will help you set up a local development environment for TAIC Report Engine.

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

Edit `.env` with your Azure credentials and API keys.

### 5. Run the Pipeline

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
│   ├── Modes.py           # Pipeline execution modes
│   ├── PDFParsing.py      # PDF text extraction
│   ├── ReportExtracting.py # Main extraction orchestrator
│   ├── SavedDataFrames.py # DataFrame persistence
│   └── WebsiteScraping.py # Agency website scraping
├── data/                  # Raw and processed data
├── output/                # Extracted datasets
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
└── workbench/             # Experimental code
```

## Running Tests

```bash
uv run pytest
```

## Next Steps

- Read the [Architecture Guide](architecture.md)
- Review the [API Reference](../api/index.md)
- Check the [Contributing Guidelines](contributing.md)
