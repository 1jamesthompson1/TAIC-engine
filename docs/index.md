# TAIC Report Engine Documentation

Welcome to the documentation for TAIC Report Engine — an AI-powered data pipeline that transforms publicly available transport accident investigation reports into structured datasets for [downstream applications](https://github.com/1jamesthompson1/TAIC-smart-tools).

## Overview

TAIC Report Engine processes reports from transport accident investigation organisations (TAIC, ATSB, TSB) and extracts:

- **Safety Issues** — Identified hazards and risks
- **Recommendations** — Safety recommendations issued
- **Summaries** — Concise report summaries
- **Chunked reports** — Reports split into sections that are easily searchable

All of this information is stored in a vector database, enabling powerful retrieval-augmented generation (RAG) applications.

## Quick Links

- [User Guide](user-guide/overview.md) — Learn about the pipeline
- [Developer Guide](developer-guide/getting-started.md) — Set up local development
- [API Reference](api/index.md) — Detailed code documentation

## Project Status

Current version: **0.1.0**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
