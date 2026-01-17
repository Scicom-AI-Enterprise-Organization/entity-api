FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install FlashInfer
RUN pip3 install --no-cache-dir flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/

# Install other requirements
RUN pip3 install --no-cache-dir \
    transformers>=4.40.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    accelerate>=0.25.0

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Default command
CMD ["python3", "-m", "app.main", "--host", "0.0.0.0", "--port", "8000"]
