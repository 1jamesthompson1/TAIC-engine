# User Guide Overview

This guide explains how the TAIC Report Engine pipeline works and how to use it effectively.

## What is TAIC Report Engine?

TAIC Report Engine is a data pipeline that:

1. **Scrapes** reports from transport accident investigation agency websites, as well as safety issues and recommendations if available.
2. **Parses** the PDF reports into structured text
3. **Extracts** safety issues, recommendations, and splits the report into sections using AI models
4. **Stores** the results are stored in a structured datasets and embedded in a vector database for retrieval-augmented generation (RAG) applications.

## Supported Agencies

- **TAIC** — Transport Accident Investigation Commission (New Zealand)
- **ATSB** — Australian Transport Safety Bureau (Australia)
- **TSB** — Transportation Safety Board of Canada

## Typical Workflow

1. **Configure** the pipeline via `config.yaml`
2. **Run** the pipeline with `uv run engine`
3. **Review** extracted data in the `output/` directory
4. **Use** the structured data in downstream applications

## Next Steps

- Read about the [Pipeline](pipeline.md) in detail
- Learn about [Report Processing](reports.md)
