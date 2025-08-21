---
title: TAIC smart assistant
emoji: 💻
colorFrom: purple
colorTo: gray
sdk: docker
app_file: app.py
---
This is a quick chatbot gradio app. 

It is being built seperately from the main project repo: https://github.com/1jamesthompson1/TAIC-report-summary to keep oveheads low.

It is likely following dicussions at TAIC that this chat interface is alot better for the end user. Therefore it may be implemented and merged with the current flask viewer app. The easiest way seems to be do doing this via embedding. However long term it would be better to have it all on the same framework.

## Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):

- **Patch** versions are auto-bumped on every push to main
- **Minor/Major** versions can be manually triggered using GitHub Actions
- All conversations and searches store the app version they were created with
- Version compatibility checking prevents loading incompatible data

### Auto-versioning Rules

Commit messages can trigger different version bumps:

- `BREAKING:` or `breaking change:` → Major version bump
- `feat:` or `feature:` or `minor:` → Minor version bump  
- Everything else → Patch version bump

### Manual Version Bumping

Use GitHub Actions workflow "Auto Version Bump" with manual trigger to specify version bump type.

**Note for Pull Requests**: When using squash-and-merge, ensure your final squashed commit message contains the appropriate keywords above, as the GitHub Action analyzes the squashed commit message for version bumping.

## Contributing

### Setup for Local Development

_Note that this has only ever been 'completed' on Linux machines. For ease of use I would recommend [WSL](https://learn.microsoft.com/en-us/windows/wsl/about) for windows users_

1. **Install dependencies**

```bash
# Install uv if you haven't already
curl -Ls https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

2. **Get the vector database**

Install `azcopy` by following: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=apt
Install `az cli` by following: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

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
uv run uvicorn app:app --host localhost --port 7860 --reload
```

5. **Access the application**
- Open your browser to `http://localhost:7860`
- For the tools interface: `http://localhost:7860/tools`

### Development Workflow

1. Create a feature branch from `main`
2. Make your changes with appropriate commit messages
3. Create a pull request
4. When merging, use squash-and-merge with conventional commit keywords in the final commit message for automatic version bumping