# API Reference

This section provides detailed documentation for all modules in TAIC Report Engine.

The API documentation is automatically generated from Python docstrings in the source code.

## Modules

### [AICaller](aicaller.md)
Interface to Azure OpenAI and AI model inference.

### [AzureStorage](azurestorage.md)
Azure Blob and Table Storage interactions.

### [Combine](combine.md)
Data combination and merge logic.

### [Config](config.md)
YAML-based configuration management.

### [Embedding](embedding.md)
Vector embedding generation and storage.

### [ExtractionModels](extractionmodels.md)
Pydantic models for structured extraction output.

### [ExtractionPrompts](extractionprompts.md)
LLM prompt templates for extraction tasks.

### [Logging](logging.md)
Centralized logging configuration.

### [Modes](modes.md)
Pipeline execution modes and orchestration.

### [PDFParsing](pdfparsing.md)
PDF text extraction and parsing.

### [ReportExtracting](reportextracting.md)
Main extraction pipeline orchestrator.

### [SavedDataFrames](saveddataframes.md)
DataFrame persistence utilities.

### [WebsiteScraping](websitescraping.md)
Agency website scraping functionality.

## Contributing

When adding new code:

- Use type hints for all parameters and returns
- Write clear, descriptive docstrings (Google style)
- Include examples in docstrings where helpful

See the [Contributing Guide](../developer-guide/contributing.md) for more details.
