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

# Install necessary packages for Azure CLI and runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Download and install the Microsoft package repository for azcopy
RUN curl -sSL -O https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update

# Install azcopy
RUN apt-get install -y azcopy && rm -rf /var/lib/apt/lists/*

# Copy and make startup scripts executable
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Set the port for the application
ENV PORT=8080

# Expose the port that the app runs on
EXPOSE ${PORT}

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Command to run the startup script (which handles vectordb download and starts the app)
CMD ["/app/startup.sh"]
