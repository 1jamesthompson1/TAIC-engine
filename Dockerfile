# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /

# Copy the pyproject.toml and poetry.lock files into the container
COPY . .

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install the dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]