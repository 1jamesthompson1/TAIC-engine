# User Guide Overview

This guide explains how the TAIC Report Engine pipeline works and how to use it effectively.  

!!! note
    From a cold start the engine takes a few days to run and costs a few hundred dollars in AI api costs. If you are wanted to get started quickly, we may be able to provide the current 'output' folder with all the data to save you time and money. Please contact TAIC for more information.

## What is TAIC Report Engine?

TAIC Report Engine is a data pipeline that:

1. **Scrapes** report metadata, PDFs, safety issues, and recommendations from agency websites
2. **Parses** PDF reports into structured text
3. **Extracts** safety issues, recommendations, and occurrence metadata from report text using AI models
4. **Stores** the results in structured datasets and a vector database for retrieval-augmented generation (RAG) applications

## Supported Agencies

- **TAIC** — Transport Accident Investigation Commission (New Zealand)
- **ATSB** — Australian Transport Safety Bureau (Australia)
- **TSB** — Transportation Safety Board of Canada

## Next Steps

- Read about the [Pipeline](pipeline.md) in detail
- Then find out about the [data sources](pipeline-inputs.md) for each item
- Learn about [Report Processing](reports.md)
