# DeepEval Local LLM Test Runner
# Multi-stage build for efficient image size

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies (without dev dependencies for smaller image)
RUN poetry install --no-interaction --no-ansi --no-root

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e . --no-deps

# Default environment variables
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2
ENV PYTHONUNBUFFERED=1

# Health check to verify connection to Ollama
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('${OLLAMA_BASE_URL}/api/tags', timeout=5)" || exit 1

# Default command runs the local LLM tests
CMD ["deepeval", "test", "run", "test_local_llm.py", "-v"]
