# TAIC Report Engine

> An AI-powered data pipeline that transforms publicly available transport accident investigation reports into structured datasets for [downstream RAG applications](https://github.com/1jamesthompson1/TAIC-smart-tools).

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tests](https://github.com/1jamesthompson1/TAIC-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/1jamesthompson1/TAIC-engine/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/1jamesthompson1/TAIC-engine/graph/badge.svg?token=3IMJCA4B49)](https://codecov.io/gh/1jamesthompson1/TAIC-engine)
[![version](https://img.shields.io/github/v/tag/1jamesthompson1/TAIC-engine?sort=semver&label=version)](https://github.com/1jamesthompson1/TAIC-engine/tags)

## Overview

TAIC Report Engine processes reports from transport accident investigation organisations (TAIC, ATSB, TSB) and extracts safety issues, recommendations, summaries, and chunked reports — all stored in a vector database for RAG like applications (i.e AI smart assistants). The pipeline broadly consists of three stages:  
1. Scrape all of the reports from the agency websites  
2. Parse the PDF reports into structured text (using an LLM to read the text and extract the relevant information)  
3. Merge the structured data with scraped data and store in a vector database for downstream applications.  

Running from scratch (scraping all websites and processing all reports) can take multiple days and cost a few hundred dollars in AI API costs. If you want to get started quickly, we may be able to provide the current 'output' folder with all the data to save you time and money. Please contact TAIC for more information. TAIC staff will find information in internal wiki pages on how to get access to the Azure resources and current output.

## Documentation

Full documentation is available at the [docs site](https://1jamesthompson1.github.io/TAIC-engine/) or can be served locally with `uv run mkdocs serve`.

## About project



This project started as a university project for James' final semester of his BSc. The university work was completed in July-October 2023 and finished with a basic engine and viewer app. Since then work has been completed directly with TAIC to bring the engine and viewer from POC to Prototype to Production. The legacy viewer app was released in late 2024, with a new [smart tools](https://github.com/1jamesthompson1/TAIC-smart-tools) app developed and deployed in November 2025.

### More information

Most of the work is organised inside a private Azure DevOps repository for TAIC. However GitHub is used for all code storing and PR resolutions. Contact James Thompson for more information.
