# Pipeline Inputs & Data Sources

This page explains how each item of data in the pipeline is sourced — whether it is **scraped from agency websites**, **AI-extracted from report text**, or a **combination of both**.

## Report text

**Source:** Scraped — PDF downloaded from each agency's website, then text-extracted using PyMuPDF.

Every available agency local occurrence investigation report PDF published by TAIC, ATSB, and TSB is downloaded and parsed into plain text. This means that special reports (i.e safety studies, research reports, and other non-occurrence investigations, overseas inquiries) are not included in the pipeline.

The full text is stored in `parsed_reports.pkl` and embedded into the vector database for semantic search.

## Report sections

**Source:** Simple overlapping chunking of the report text.

Each report's text is split into overlapping sections based on the PDF page/chapter structure. These sections are the primary unit for vector search — users searching for specific topics find the most relevant section rather than an entire report.

## Safety issues

| Agency | Source | Details |
|---|---|---|
| **ATSB (2008 onwards)** | Scraped from [ATSB safety issues database](https://www.atsb.gov.au/safety-issues) | Includes official safety issue ID, the verbatim text, and its quality designation ("exact" or "inferred"). |
| **ATSB (pre-2008)** | AI-extracted from report text | Safety issues from older reports are extracted by the AI model. |
| **TAIC** | AI-extracted from report text | TAIC does not publish a separate safety issues database, so all TAIC safety issues are AI-extracted. |
| **TSB** | AI-extracted from report text | TSB does not publish a separate safety issues database, so all TSB safety issues are AI-extracted. |

The pipeline combines both sources: scraped safety issues from the ATSB database are merged with AI-extracted safety issues from the same agency, providing both the authoritative record and any issues the AI found in the report text.

## Recommendations

| Agency | Source | Details |
|---|---|---|
| **TAIC** | Scraped from [TAIC recommendations database](https://taic.org.nz/inquiries-recommendations) | Includes recommendation ID, recipient, recommendation text, reply text, and URL. |
| **TSB** | Scraped from TSB recommendations database | Includes recommendation ID, recipient, recommendation text, current assessment status, and whether it is active/closed/dormant. |
| **ATSB** | AI-extracted from report text | ATSB does not publish a separate recommendations database, so all ATSB recommendations are AI-extracted by reading the report text. |

The pipeline combines scraped recommendations from the TAIC and TSB databases with AI-extracted recommendations, merging by agency. This provides both the in-context extraction (including surrounding context from the report) and the authoritative record from the agency.

## Summaries

| Agency | Source | Details |
|---|---|---|
| **TAIC** | Scraped from report page | The executive summary section is extracted from each TAIC report's webpage. |
| **ATSB** | Scraped from report page | The "Executive summary", "Investigation summary", or "Safety summary" section is extracted from the ATSB report webpage. Falls back to the full content div if no specific heading is found. |
| **TSB** | Not available | TSB report pages do not include summary text — only press releases which are not equivalent. TSB reports contribute to the vector database via their report sections, safety issues, and recommendations only. |

## Occurrence metadata

**Source:** AI-extracted from report text.

The AI model extracts detailed metadata about each occurrence, including:

- **Date, time, and timezone** of the occurrence
- **Location** — both a raw description and a standardised 4-part location
- **Occurrence type** — classified according to the mode-specific taxonomy (e.g. collision, derailment, ditching)
- **Persons involved** — total persons, fatalities, and injuries
- **Damage description**
- **Who may benefit** from the report's findings

### Mode-specific vehicle metadata

Depending on the transport mode, the AI also extracts:

- **Air (`a`):** Aircraft details (type, registration, make, model, engines, operator, flight type, persons on board, damage) and pilot details (role, licence, age, experience)
- **Rail (`r`):** Train details (type, number, length, weight, classification, operator, crew)
- **Marine (`m`):** Vessel details (name, type, classification, dimensions, tonnage, propulsion, owner, port of registry)

The full metadata dict (all of the above) is stored as a JSON string in the `metadata_json` column of the vector database, alongside flat columns for the most commonly filtered fields (e.g. `location`, `occurrence_date`, `fatalities`, `injuries`).

## Report titles & URLs

**Source:** Scraped from each agency's website.

Basic listing metadata is scraped for every report: title, URL, agency ID, occurrence date, publication date, and event type. This is stored in `report_titles.pkl` and used to enrich all other data items.

## Vector database

All of the above is consolidated into a [LanceDB vector database](vector-database.md). Each row in the table corresponds to one document (a section, summary, full report text, safety issue, or recommendation) enriched with metadata. See the [Vector Database](vector-database.md) page for the table schema and query examples.
