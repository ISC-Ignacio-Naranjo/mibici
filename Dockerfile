# Base Python image
FROM python:3.11-slim

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install pinned dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY mibici/ ./mibici/

# Copy exported champion model
COPY models/mibici_champion_model/ ./models/mibici_champion_model/

# Configure local model URI
ENV MODEL_URI=/app/models/mibici_champion_model

# API port
EXPOSE 8000

# Start the inference service
CMD ["python", "-m", "uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000"]