This folder is intended to provide a spot to put data that is used by this engine that can't be retrieved any other way.

It was introduced when the recommendation extraction was replaced with a dataset provided by TAIC (https://github.com/1jamesthompson1/TAIC-engine/issues/130#issuecomment-2041618860).

Here is a list of the files with their purpose or a bit of description.

| file | description |
| --- | ---- |
| event_types.csv | Canonical taxonomy of event types used by extraction to constrain `metadata.occurrence.occurrence_type` by report mode.
| atsb_historic_aviation_investigations.csv | This file is to speed up the web scraping process. This is because there are about 350 pages to the ATSB aviation report table (https://www.atsb.gov.au/aviation-investigation-reports). Therefore I have done a complete run through and `WebsiteScraping.py` uses this file to not scrape more pages then it has too. Potentially this could be moved to the output file and each agency could maintain its own table within the output folder. Currently this is not done and is discussed here: https://github.com/1jamesthompson1/TAIC-engine/issues/259