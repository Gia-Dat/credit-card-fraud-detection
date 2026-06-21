# Use a lightweight, official Python base image
FROM python:3.12-slim

# Set system environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set our working folder inside the container
WORKDIR /app

# Install uv globally inside the container for lightning-fast dependency loading
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only dependency files first to maximize Docker caching speed
COPY pyproject.toml uv.lock ./

# Install project dependencies strictly matching our lockfile
RUN uv sync --frozen --no-cache

# Copy your source code and exported model binaries into the image
COPY src/ ./src/
COPY models/ ./models/

# Expose the API port to your local machine
EXPOSE 8000

# Fire up the production web server using our virtual environment launcher
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]