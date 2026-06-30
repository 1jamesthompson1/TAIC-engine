# User Guide Overview

This guide explains how the TAIC Report Engine pipeline works and how to use it effectively.  

!!! note
    From a cold start the engine takes a few days to run and costs a few hundred dollars in AI api costs. If you are wanted to get started quickly, we may be able to provide the current 'output' folder with all the data to save you time and money. Please contact TAIC for more information.

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
3. **Use** the structured data from the vector database for downstream applications, such as the [TAIC Smart Tools](https://github.com/1jamesthompson1/TAIC-smart-tools) project.

## Next Steps

- Read about the [Pipeline](pipeline.md) in detail
- Learn about [Report Processing](reports.md)
