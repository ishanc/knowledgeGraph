FROM python:3.8.16-slim

WORKDIR /app

# Install system dependencies and cleanup in one layer
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    tesseract-ocr \
    libgomp1 \
    dos2unix \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first to avoid dependency issues
RUN pip install --no-cache-dir numpy==1.24.3

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /app/
RUN dos2unix /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

# Create app user and set up permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/torch/sentence_transformers && \
    mkdir -p /app/uploads /app/processed_files /app/static/graph /app/templates && \
    chown -R appuser:appuser /home/appuser/.cache && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/uploads /app/processed_files /app/static/graph

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV PYTHONPATH=/app
ENV HOME=/home/appuser
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/torch/sentence_transformers

EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
