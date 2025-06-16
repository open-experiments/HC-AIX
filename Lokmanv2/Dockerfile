# OpenShift compatible Dockerfile for Lokman-v2: CT Breast Cancer Detection
FROM python:3.9-slim

# Create non-root user for OpenShift
RUN useradd -m -u 1001 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-cors

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER 1001

# Expose port 30080 (Custom service port)
EXPOSE 30080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web/ocp_app.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:30080/api/health || exit 1

# Run the application
CMD ["python", "web/ocp_app.py", "--host", "0.0.0.0", "--port", "30080"]