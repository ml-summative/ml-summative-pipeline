# Dockerfile for Traffic-Net ML Application

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     libgomp1 \
#     libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy requirements
COPY models .

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/train data/test data/retrain uploads logs

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# Run the application
CMD ["python", "app.py"]