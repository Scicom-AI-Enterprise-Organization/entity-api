FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Flash Attention
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install other requirements
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    pydantic>=2.0.0 \
    accelerate>=0.25.0

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Default command
CMD ["python3", "-m", "app.main", "--host", "0.0.0.0", "--port", "8000"]
