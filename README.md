# TAIC report analysis

> An AI powered data pipeline that ingests transport accident investigation reports and outputs structured datasets to be used downstream in RAG like applications, like [TAIC smart tools](https://github.com/1jamesthompson1/TAIC_smart_assistant)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![tests](https://github.com/1jamesthompson1/TAIC-report-summary/actions/workflows/ci.yml/badge.svg)](https://github.com/1jamesthompson1/TAIC-report-summary/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/1jamesthompson1/TAIC-report-summary/graph/badge.svg?token=3IMJCA4B49)](https://codecov.io/gh/1jamesthompson1/TAIC-report-summary)


## What

This github repository contains code for a data pipeline that ingests publicly available transport accident investigation reports from the [TAIC](https://taic.org.nz/reports), [ATSB](https://www.atsb.gov.au/publications/investigation-reports/) and [TSB](https://www.tsb.gc.ca/eng/rapports-reports.html) and outputs structured datasets of safety issues, recommendations, report paragraphs and more.

There is also a [collection](https://github.com/1jamesthompson1/TAIC-report-summary/tree/d735c0f3a50f4ef24f1e7198730c984fdb3446c7/notebooks) of jupyter notebooks that show the development process of some of the more complex features in the engine.


### About

This project started as a university project for James' final semester of his BSc. The university work was completed in July-October 2023 and finished with a basic engine and viewer app. Since then work has been completed directly with TAIC to bring the engine and viewer from POC -> Prototype -> Production. The legacy 'viewer' app was released in later 2024. With a new [smart tools](https://github.com/1jamesthompson1/TAIC_smart_assistant) app developed and deployed in November 2025

### More information

Most of the work is orgnaised inside a private Azure DevOps repository for TAIC. However GitHub is used for all code storing and PR resolutions. Contact James Thompson for more information.