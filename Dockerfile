FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal set for PyTorch and PostgreSQL)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Note: Using pandas-ta instead of ta-lib (no C compilation needed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/parquet \
    /app/checkpoints /app/logs /app/outputs

# Set random seeds for reproducibility
ENV PYTHONHASHSEED=42

# Default command
CMD ["python", "-m", "src.training.train"]
