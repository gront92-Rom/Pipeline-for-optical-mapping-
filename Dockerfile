FROM python:3.12-slim-bookworm

# Build tools needed for optimap C++ extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY requirements.txt install_optimap_dev.sh ./

# Install optimap dev build (the longest step) before other deps
RUN chmod +x install_optimap_dev.sh && ./install_optimap_dev.sh

# Install remaining pinned Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the entire pipeline
COPY . .

# Default: run a sample if data is mounted, otherwise print help
ENTRYPOINT ["/app/run_cardiac.sh"]
CMD ["--help"]
