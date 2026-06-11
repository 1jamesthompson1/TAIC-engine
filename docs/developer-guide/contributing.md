# Contributing

## Workflow

1. Create feature branch from `main`
2. Make changes with clear commit messages
3. Run tests: `uv run pytest`
4. Run linting: `uv run pre-commit run --all-files`
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

**Callout boxes:**
```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.

!!! tip
    This is a tip.
```

## Code Style

This project uses:
- **Ruff** for linting and formatting
- **Google-style docstrings**
- **Type hints** for all functions

Run linting before committing:

```bash
uv run pre-commit run --all-files
```

## Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=engine --cov-report=html
```
