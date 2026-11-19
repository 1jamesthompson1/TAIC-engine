# Contributing

## Workflow

1. Create feature branch from `main`
2. Make changes with clear [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) messages, note that pre-commit hooks are setup to enforce this as well as linting and formatting
3. Run tests
4. Push branch to remote
5. Create a pull request

## Documentation

Documentation is built with **MkDocs + Material theme**.

```bash
# Dev mode with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

**Writing docs:**
- User guides in `docs/user-guide/`
- Developer guides in `docs/developer-guide/`
- API reference pages in `docs/api/`
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public APIs
- Update `mkdocs.yml` nav when adding pages

## Code Style

This project uses:
- **Ruff** for linting and formatting
- **Google-style docstrings**
- **Type hints** for all functions

## Testing

You can do it with cli however I would recommend using the `pytest` plugin for your ide.

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=engine --cov-report=html
```

## Azure DevOps Pipelines (TAIC deployment specific!)

The actual running of the engine is done using a azure devops pipeline.

### Setup (one-time)

```bash
# Enable auto-install of Azure DevOps extension
az config set extension.use_dynamic_install=yes_without_prompt

# Configure defaults
az devops configure --defaults \
    organization=https://dev.azure.com/NZ-TAIC \
    project="AI at TAIC"
```

### Running pipelines

```bash
# Run the sync pipeline
az pipelines run --name "sync"

# Run the engine pipeline with the version of the code that you are in locally.
az pipelines run --name "engine"   --variables "run-branch=$(git branch --show-current)" --open

```

The results will open in your browser automatically.