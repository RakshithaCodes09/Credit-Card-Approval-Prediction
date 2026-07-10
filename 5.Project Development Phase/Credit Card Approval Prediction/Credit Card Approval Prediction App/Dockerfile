# Build Stage: Use lightweight official Python image
FROM python:3.11-slim as builder

WORKDIR /app

# Setup environmental values for build performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install compiler and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into local directory
RUN pip install --no-cache-dir --user -r requirements.txt


# Production Stage: Minimize image size
FROM python:3.11-slim as runner

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY requirements.txt .

ENV PATH=/root/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

# Copy application directories
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY model/ ./model/

# Create application logs directory
RUN mkdir -p logs temp && chmod -R 755 logs temp

# Expose server port
EXPOSE 5000

# Run with secure, non-root user for security compliance
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Use Gunicorn as the production Web Server Gateway Interface (WSGI)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "60", "app:app"]
