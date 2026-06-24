FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/

# Expose port (Zeabur will use PORT env variable)
EXPOSE 8080

# Run the application - use PORT env variable for Zeabur compatibility
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
