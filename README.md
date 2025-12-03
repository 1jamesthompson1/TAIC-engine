# TAIC smart tools

> A platform for smart tools for internal use at the New Zealand Transport Accident Investigation Commission. Featured tools are smart assistant and smart document search.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Continuous Integration](https://github.com/1jamesthompson1/TAIC-smart-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/1jamesthompson1/TAIC-smart-tools/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/Version-0.6.2-blue)

## Preview

![TAIC Smart Assistant Demo](https://github.com/1jamesthompson1/TAIC-smart-tools/releases/download/v0.4.0/demo.gif)

---

## Features

🤖 **AI-Powered Assistant**
- Interactive chat interface with conversational AI
- Context-aware responses using TAIC's knowledge base
- Conversation history and management

🔍 **Advanced Knowledge Search**
- Vector-based semantic search across TAIC, ATSB and TSB documents (database created by [TAIC-engine](https://github.com/1jamesthompson1/TAIC-engine)
- Filter by document type, agency, year, and transport mode
- Visual analytics and result export capabilities

---

## Project Status

In November 2025 this app was deployed organisation wide at TAIC. There is no public access at the moment.

The app is currently deployed internally and privately, however it is feasible to create a new deployment that is completely separate from this one to be used by other organisations. Any interest in your own deployment or a tool like this should be directed to James Thompson (1jamesthompson1 [at] gmail dot com)

## Contributing

### Setup for Local Development

_Note that this has only ever been 'completed' on Linux machines. For ease of use I would recommend [WSL](https://learn.microsoft.com/en-us/windows/wsl/about) for windows users_

1. **Install dependencies**

```bash
# Get the code
git clone https://github.com/1jamesthompson1/TAIC-smart-tools
cd TAIC-smart-tools

# Install uv if you haven't already
curl -Ls https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync --dev

# Setup pre-commits
uv run pre-commit install
```

2. **Get the vector database**
*This is simply so that when you want to test the webapp you can use the searcher quickly as it is locally fetching data.*

Install `azcopy` by following: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=apt
Install `az cli` by following: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

You will then need to login with `az login`.

```bash
# Download and setup the vector database
uv run python working_files/download_vector_db.py
```

3. **Environment configuration**

```bash
# Copy and configure your environment variables
cp .env.example .env
# Edit .env with your Azure credentials and API keys
```

4. **Run the app locally**

```bash
uv run uvicorn app:app --host localhost --port 7860 --reload --timeout-graceful-shutdown 1
```

5. **Access the application**
- Open your browser to `http://localhost:7860`
- For the tools interface: `http://localhost:7860/tools`

### Development Workflow

1. Create a feature branch from `main`
2. Make your changes with appropriate commit messages
3. Create a pull request with a prefix of the type of change (`major:`, `minor:`, `patch:`)
4. When merging, use squash-and-merge to keep a clean history

#### Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):

**PR auto versioning**
PR titles should include one of the following keywords to indicate the type of version bump required:

- `major:` → Major version bump
- `minor:` → Minor version bump  
- `patch:` → Patch version bump

### CI/CD Configuration

This project uses GitHub Actions for automated testing and deployment. The following secrets need to be configured in your GitHub repository:

#### Required GitHub Secrets

| Secret Name | Description |
|-------------|-------------|
| `AZURE_STORAGE_SAS_TOKEN` | Azure storage account SAS token to allow reading of the `envs` container |
| `AZURE_STORAGE_ACCOUNT_NAME` | Azure Storage account name where test config is stored |

#### Test Environment Setup

The CI pipeline automatically downloads a test configuration file (`test.env` saved as `.env`) from Azure Blob Storage during test runs. This file should contain all necessary environment variables for testing with real services. This setup is to reduce the coupling between the code repository and the TAIC internal deployment.
