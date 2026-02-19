# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Playwright: install Chromium for html_to_pdf
RUN python -m playwright install chromium --with-deps

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p results/plots results/data

# Default command (can be overridden in docker-compose)
# CMD will be set in docker-compose.yml or can be overridden
CMD ["python", "--version"]

