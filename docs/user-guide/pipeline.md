# Pipeline

The TAIC Report Engine pipeline processes reports through several stages.

In its entirety it takes a day or so to run (i.e scraping all the websites and processing the PDF), it also costs about $150 to run the full AI extraction section. Therefore it is normally only run once fully and then incrementally updated to only process the new reports.

## Stage 0: Downloading previous output

The [output](#data-storage) of each stage is saved to disk in the `output/` directory. When running the pipeline, it will first grab teh most recent output data from an Azure blob storage container. This allows the pipeline to be run incrementally, only processing new reports and not reprocessing old reports.

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

## Stage 6: Uploading output

The final output of each stage is uploaded to Azure blob storage for persistence and future incremental runs. See [Data Storage](#data-storage) for details on the output files.

## Data Storage

Each pipeline stage saves its results as pickle (`.pkl`) files in the `output/` directory. There are more files however they are not too interesting so omitted here. See [SavedDataFrames](../api/saveddataframes.md) for the full list of output files and their corresponding DataFrame classes.

| File | Content | DataFrame Class |
|---|---|---|
| `parsed_reports.pkl` | Raw PDF text per report | [`ParsedReports`](../api/saveddataframes.md#engine.SavedDataFrames.ParsedReports) |
| `report_titles.pkl` | Scraped report titles, URLs, summaries | [`ReportTitles`](../api/saveddataframes.md#engine.SavedDataFrames.ReportTitles) |
| `atsb_website_safety_issues.pkl` | Scraped safety issues from ATSB | [`ATSBWebsiteSafetyIssues`](../api/saveddataframes.md#engine.SavedDataFrames.ATSBWebsiteSafetyIssues) |
| `tsb_website_recommendations.pkl` | Scraped recommendations from TSB | [`TSBWebsiteRecommendations`](../api/saveddataframes.md#engine.SavedDataFrames.TSBWebsiteRecommendations) |
| `taic_website_recommendations.pkl` | Scraped recommendations from TAIC | [`TAICWebsiteRecommendations`](../api/saveddataframes.md#engine.SavedDataFrames.TAICWebsiteRecommendations) |
| `extracted_reports.pkl` | AI-extracted safety issues, recommendations, sections | [`ExtractedReports`](../api/saveddataframes.md#engine.SavedDataFrames.ExtractedReports) |
| `complete_data.pkl` | Merged canonical dataset for vector DB ingestion | [`DataForVectorDB`](../api/saveddataframes.md#engine.SavedDataFrames.DataForVectorDB) |
| `vector_db_document_ids.pkl` | Document IDs already embedded (dedup) | [`VectorDBDocumentIDs`](../api/saveddataframes.md#engine.SavedDataFrames.VectorDBDocumentIDs) |

Vector embeddings are stored externally in **LanceDB** (configured via `VECTORDB_URI`), not in the `output/` directory.
