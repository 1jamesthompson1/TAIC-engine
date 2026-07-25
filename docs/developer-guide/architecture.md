# Architecture

High-level overview of TAIC Report Engine architecture.

## Components

### Scraping Layer (`WebsiteScraping.py`)
Fetches report listings and PDFs from investigation agency websites. Handles different website structures for TAIC, ATSB, and TSB.

### Parsing Layer (`PDFParsing.py`)
Extracts text content from downloaded PDF reports. Uses PyMuPDF for reliable text extraction across varied report layouts.

### Extraction Layer
The core AI-powered extraction system:

- **`ReportExtracting.py`** — Orchestrates the extraction pipeline
- **`ExtractionPrompts.py`** — Manages LLM prompt templates
- **`ExtractionModels.py`** — Pydantic models for structured output
- **`AICaller.py`** — Interface to Azure OpenAI / AI models
- **`Config.py`** — Pipeline configuration via YAML

### Storage Layer
- **`AzureStorage.py`** — Cloud blob and table storage
- **`SavedDataFrames.py`** — Local DataFrame persistence
- **`Embedding.py`** — Vector embedding creation and storage

### Pipeline Orchestration (`Combine.py`, `Modes.py`, `cli.py`)
Coordinates the end-to-end pipeline execution with different operational modes.

## Technology Stack

- **Python 3.11+**: Runtime
- **Azure OpenAI**: AI model inference
- **LanceDB**: Vector database
- **PyMuPDF**: PDF parsing
- **Pandas**: Data manipulation
- **pytest**: Testing

## Data Flow

```
Agency Websites
      ↓
WebsiteScraping → PDF Downloads, Report Metadata, Safety Issues (ATSB), Recommendations(TSB and TAIC)
      ↓
PDFParsing → Text Content (parsed_reports.pkl)
      ↓
ReportExtracting → Safety Issues, Recommendations, Sections (extracted_reports.pkl)
      ↓
Combine → Merged Dataset (complete_data.pkl)
      ↓
Embedding → LanceDB Vector Database (all_document_types)
```
