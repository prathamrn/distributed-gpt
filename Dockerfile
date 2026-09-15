# Worker node image: CPU-only PyTorch + tc (netem) for network shaping.
# Code is bind-mounted at runtime (see docker-compose.yml) so edits need no rebuild;
# only requirements.txt changes do.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 iputils-ping curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 \
    && pip install --no-cache-dir numpy fastapi uvicorn requests pydantic

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "node.py"]
