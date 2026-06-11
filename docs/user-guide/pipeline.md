# Pipeline

The TAIC Report Engine pipeline processes reports through several stages.

## Stage 1: Scraping

`WebsiteScraping.py` fetches report listings and PDFs from each agency's website.

- Discovers available reports
- Downloads PDF files
- Extracts metadata (title, date, agency, mode)

## Stage 2: Parsing

`PDFParsing.py` converts PDF reports into structured text using PyMuPDF.

- Extracts text content page by page
- Preserves document structure
- Handles complex PDF layouts

## Stage 3: Extraction

`ReportExtracting.py` uses AI models to extract structured information from report text.

- Safety issues identification
- Recommendation extraction
- Report summarisation
- Structured data output

## Stage 4: Combination

`Combine.py` merges extracted data with scraped safety issues and recommendations into a single canonical dataset.

- Combines AI-extracted data with website-scraped data
- Outputs `complete_data.pkl` for vector DB ingestion

## Stage 5: Embedding

`Embedding.py` creates vector embeddings and stores them directly in **LanceDB**.

- Reads from `complete_data.pkl`
- Generates embeddings using Azure OpenAI
- Stores vectors in a LanceDB table (`all_document_types`)
- Tracks already-embedded documents in `vector_db_document_ids.pkl` to avoid re-embedding

## Data Storage

Each pipeline stage saves its results as pickle (`.pkl`) files in the `output/` directory:

| File | Content |
|---|---|
| `parsed_reports.pkl` | Raw PDF text per report |
| `report_titles.pkl` | Scraped report titles, URLs, summaries |
| `atsb_website_safety_issues.pkl` | Scraped safety issues from ATSB |
| `tsb_website_recommendations.pkl` | Scraped recommendations from TSB |
| `taic_website_recommendations.pkl` | Scraped recommendations from TAIC |
| `extracted_reports.pkl` | AI-extracted safety issues, recommendations, sections |
| `complete_data.pkl` | Merged canonical dataset for vector DB ingestion |
| `vector_db_document_ids.pkl` | Document IDs already embedded (dedup) |

Vector embeddings are stored externally in **LanceDB** (configured via `VECTORDB_URI`), not in the `output/` directory.
