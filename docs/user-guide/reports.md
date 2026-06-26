# Report Processing

How TAIC Report Engine processes individual investigation reports. This is all conducted in the `engine.ReportExtracting` module. It is done with a single API request to a state of the art LLM which reads the report text and produces a structured output. This may include safety issues and/or recommendations, along with metadata about the occurrence and involved vehicles/personnel.

## Extraction quality

!!! Important

    The AI extraction is not perfect. There are no guarantees that it is perfect. However different types of data come with different levels of confidence.

There are two types of extraction quality:

- **Confident** - The extracted data is expected to be accurate and is tested on being exactly correct.
- **Best effort** - The extracted data is expected to be mostly accurate, however the testing is more lenient and allows for some errors.

In each of the following sections it is stated whether the extraction is *confident* or *best effort*.

## Extraction items

Three different types of items are extracted from reports: [safety issues](#safety-issues), [recommendations](#recommendations), and [metadata](#metadata). Each of these items is extracted with different levels of confidence, as described in the following sections.

### Safety Issues

*See the [`SafetyIssueItem`](../api/extractionmodels.md#engine.ExtractionModels.SafetyIssueItem) Pydantic model for the full definition.*

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

Safety issues are extracted from TAIC reports with *confident* quality, most of these will be exact safety issues.

Safety issues are extracted from TSB reports with *confident* quality however note that they are all treated as inferred safety issues, as TSB reports do not explicitly label safety issues (instead we are treating 'findings as to risk' as safety issues).

All ATSB reports after 2008 do not have safety issues extracted from reports and instead are scraped directly from the ATSB website. Pre to 2008 ATSB reports have safety issues extracted from the report with *best effort* quality, as these reports are known to be inconsistent in how they present safety issues.


### Recommendations

*See the [`RecommendationItem`](../api/extractionmodels.md#engine.ExtractionModels.RecommendationItem) Pydantic model for the full definition.*

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

For TSB and TAIC reports, recommendations are not extracted by AI and are instead scraped directly from the respective websites.

For ATSB reports, recommendations are extracted from the report with *confident* quality. However the recommendation_context, made, and recipient fields are extracted with *best effort* quality, as these fields are not always present in the report.

### Metadata

Metadata is extracted from all the reports. There are two sorts of metadata [occurrence metadata](#occurrence-metadata) and [mode-specific vehicle/personnel metadata](#mode-specific-vehicle-metadata).

#### Occurrence Metadata

*See the [`OccurrenceMetadata`](../api/extractionmodels.md#engine.ExtractionModels.OccurrenceMetadata) Pydantic model for the full definition.*

Structured metadata about the occurrence itself:

| Field | Type | Description |
|---|---|---|
| `occurrence_datetime` | `OccurrenceDateTime` | Local datetime + UTC offset + timezone source |
| `location` | `OccurrenceLocation` | Raw description + standardized 4-part location |
| `occurrence_type` | `str \| None` | Type/classification (mode-specific taxonomy) |
| `total_persons_involved` | `int \| None` | Total persons involved |
| `fatalities` | `int` | Number of fatalities (0 if none) |
| `injuries` | `int` | Number of injuries (0 if none) |
| `damage_description` | `Literal["nil"] \| str` | Summary of damage to equipment/property. Use `"nil"` for no damage |
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

For all reports the only items that are extracted with *confident* quality is the fatalities, injuries, occurrence_datetime, and occurrence_type fields. All other fields are extracted with *best effort* quality.

#### Mode-Specific Vehicle Metadata

Depending on the report mode, additional vehicle/personnel metadata is extracted. Note that all of this data is extracted with *best effort* quality, as the categories are not always well defined.

**Aviation** — `aircraft: list[AircraftMetadata]`

*See the [`AircraftMetadata`](../api/extractionmodels.md#engine.ExtractionModels.AircraftMetadata) and [`PilotMetadata`](../api/extractionmodels.md#engine.ExtractionModels.PilotMetadata) Pydantic models for the full definitions.*

Each aircraft includes: type, registration, make, model, number of engines, engine type, year manufactured, operator, flight type, persons on board (total/crew/passengers), damage, and a list of pilots with their role, responsibility, licence, age, and experience (total and on type).

```json
{
  "aircraft": [{
    "aircraft_type": "Aeroplane",
    "registration": "ZK-ABC",
    "make": "Cessna",
    "model": "172",
    "number_of_engines": 1,
    "type_of_engines": "piston",
    "year_manufactured": 1995,
    "operator": "Flying School Ltd",
    "flight_type": "training",
    "persons_on_board_total": 2,
    "persons_on_board_crew": 1,
    "persons_on_board_passengers": 1,
    "damage": "substantial damage",
    "pilots": [{
      "role": "Instructor",
      "responsibility": "Pilot monitoring",
      "licence": "Commercial Pilot Licence (Aeroplane)",
      "age": 35,
      "total_flying_experience": 1500,
      "experience_on_type": 500
    }]
  }]
}
```

**Rail** — `trains: list[TrainMetadata]`

*See the [`TrainMetadata`](../api/extractionmodels.md#engine.ExtractionModels.TrainMetadata) Pydantic model for the full definition.*

Each train includes: type, number, length, weight, classification, year manufactured, operator, and crew count.

```json
{
  "trains": [{
    "train_type": "freight",
    "train_number": "1234",
    "length": 450.0,
    "weight": 3110.0,
    "classification": "manifest",
    "year_manufactured": 1998,
    "operator": "KiwiRail",
    "operating_crew": 2
  }]
}
```

**Marine** — `vessels: list[VesselMetadata]`

*See the [`VesselMetadata`](../api/extractionmodels.md#engine.ExtractionModels.VesselMetadata) Pydantic model for the full definition.*

Each vessel includes: name, type, classification society, length, breadth, gross tonnage, manufacturer, year built, propulsion, power, speed, owner/operator, and port of registry.

```json
{
  "vessels": [{
    "vessel_name": "MV Example",
    "vessel_type": "fishing",
    "classification": "unclassed",
    "length": 25.0,
    "breadth": 6.5,
    "gross_tonnage": 80.0,
    "manufacturer": "Whangarei Engineering",
    "year_built": 2005,
    "propulsion": "diesel",
    "total_power": 500.0,
    "service_speed": 12.0,
    "owner_operator": "Coastal Fishing Ltd",
    "port_of_registry": "Auckland"
  }]
}
```