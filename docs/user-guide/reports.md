# Report Processing

How TAIC Report Engine processes individual investigation reports.

## Report Structure

Investigation reports typically contain:

- **Synopsis** — Brief overview of the occurrence
- **Factual Information** — Detailed findings
- **Analysis** — Examination of contributing factors
- **Findings** — Determined causes and contributing factors
- **Safety Issues** — Identified hazards
- **Recommendations** — Safety recommendations
- **Safety Actions** — Actions already taken

## Extraction Output Model

The AI extraction produces a single structured output per report via `ExtractedReport` (dynamically built by `build_extraction_output_model`). Depending on the pipeline configuration, it includes some or all of the following:

### Safety Issues

Each safety issue has:

| Field | Type | Description |
|---|---|---|
| `safety_issue` | `str` | The text of the actual safety issue |
| `safety_issue_id` | `str \| None` | Unique identifier if present in the report |
| `quality` | `"exact" \| "inferred"` | Whether the issue is verbatim from the report or implied |

```json
{
  "safety_issue": "Inadequate crew resource management",
  "safety_issue_id": "SI-2023-001",
  "quality": "exact"
}
```

### Recommendations

Each recommendation has:

| Field | Type | Description |
|---|---|---|
| `recommendation` | `str` | The verbatim recommendation text |
| `recommendation_id` | `str \| None` | Unique identifier if present |
| `recipient` | `str \| None` | Who the recommendation was addressed to |
| `recommendation_context` | `str \| None` | Background info, safety issue addressed, recipient's response |
| `made` | `str \| None` | Date the recommendation was made (ISO 8601) |

```json
{
  "recommendation": "Review crew training procedures...",
  "recommendation_id": "R-2023-002",
  "recipient": "Operator",
  "recommendation_context": "The safety issue regarding...",
  "made": "2023-06-15"
}
```

### Occurrence Metadata

Structured metadata about the occurrence itself:

| Field | Type | Description |
|---|---|---|
| `occurrence_datetime` | object | Local datetime + UTC offset + timezone source |
| `location` | object | Raw description + standardized 4-part location |
| `occurrence_type` | `str \| None` | Type/classification (mode-specific taxonomy) |
| `total_persons_involved` | `int \| None` | Total persons involved |
| `fatalities` | `int` | Number of fatalities (0 if none) |
| `injuries` | `int` | Number of injuries (0 if none) |
| `damage_description` | `str` | Summary of damage to equipment/property |
| `who_may_benefit` | `str \| None` | Explicit "who may benefit" text if present |

```json
{
  "occurrence_datetime": {
    "local_datetime": "2023-06-10T14:30",
    "time_zone": "UTC+12:00",
    "time_zone_source": "explicit_in_report"
  },
  "location": {
    "description": "Near Wellington Airport",
    "standardized_location": "Wellington Airport, Wellington, Wellington, New Zealand"
  },
  "fatalities": 0,
  "injuries": 2,
  "damage_description": "substantial damage: left wing damaged; engine detached."
}
```

### Mode-Specific Vehicle Metadata

Depending on the report mode, additional vehicle/personnel metadata is extracted:

**Aviation** — `aircraft: list[AircraftMetadata]`

Each aircraft includes: type, registration, make, model, number of engines, engine type, year manufactured, operator, flight type, persons on board (total/crew/passengers), damage, and a list of pilots with their role, licence, age, and experience.

```json
{
  "aircraft": [{
    "aircraft_type": "Aeroplane",
    "registration": "ZK-ABC",
    "make": "Cessna",
    "model": "172",
    "number_of_engines": 1,
    "type_of_engines": "piston",
    "operator": "Flying School Ltd",
    "flight_type": "training",
    "persons_on_board_total": 2,
    "pilots": [{
      "role": "Instructor",
      "licence": "Commercial Pilot Licence (Aeroplane)",
      "age": 35,
      "total_flying_experience": 1500
    }]
  }]
}
```

**Rail** — `trains: list[TrainMetadata]`

Each train includes: type, number, length, weight, classification, year manufactured, operator, and crew count.

```json
{
  "trains": [{
    "train_type": "freight",
    "train_number": "1234",
    "length": 450.0,
    "weight": 3110.0,
    "operator": "KiwiRail",
    "operating_crew": 2
  }]
}
```

**Marine** — `vessels: list[VesselMetadata]`

Each vessel includes: name, type, classification society, length, breadth, gross tonnage, manufacturer, year built, propulsion, power, speed, owner/operator, and port of registry.

```json
{
  "vessels": [{
    "vessel_name": "MV Example",
    "vessel_type": "fishing",
    "length": 25.0,
    "propulsion": "diesel",
    "port_of_registry": "Auckland"
  }]
}
```

## Pipeline Flow

Reports flow through the pipeline as follows:

1. **Scraping** → PDFs downloaded, metadata stored in `report_titles.pkl` and agency-specific pickle files
2. **Parsing** → PDF text extracted and saved to `parsed_reports.pkl`
3. **Extraction** → AI extracts safety issues, recommendations, and metadata using the models above, stored in `extracted_reports.pkl`
4. **Combination** → All data merged into `complete_data.pkl`
5. **Embedding** → Content from `complete_data.pkl` is embedded and stored in **LanceDB** (`all_document_types` table)

## Output Files

All structured data is stored as pickle (`.pkl`) files in the `output/` directory:

| File | Content |
|---|---|
| `parsed_reports.pkl` | Raw extracted text from PDFs |
| `extracted_reports.pkl` | AI-extracted safety issues, recommendations, and metadata |
| `complete_data.pkl` | Merged canonical rows for vector DB ingestion |
| `report_titles.pkl` | Report metadata from website scraping |
| `atsb_website_safety_issues.pkl` | Safety issues scraped directly from ATSB |
| `tsb_website_recommendations.pkl` | Recommendations scraped from TSB |
| `taic_website_recommendations.pkl` | Recommendations scraped from TAIC |

Vector embeddings are stored externally in **LanceDB** (configured via `VECTORDB_URI`), not in the `output/` directory.
