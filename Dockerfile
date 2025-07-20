FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Install necessary packages to allow download of azcopy
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https

# Download and install the Microsoft package repository
RUN curl -sSL -O https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update

# Install azcopy
RUN apt-get install -y azcopy

# Download the vector database
RUN --mount=type=secret,id=SAS_TOKEN,mode=0444,required=true \
    azcopy cp "https://taicdocumentsearcherdata.blob.core.windows.net/vectordb/prod/*?$(cat /run/secrets/SAS_TOKEN)" vectordb --recursive

# Expose the port that the app runs on
EXPOSE 7860

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
