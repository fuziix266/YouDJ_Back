FROM python:3.11-slim

# Instalar ffmpeg y deno (runtime JS para yt-dlp)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl unzip && \
    curl -fsSL https://deno.land/install.sh | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Agregar deno al PATH
ENV DENO_DIR=/root/.deno
ENV PATH="${DENO_DIR}/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear directorio de cache con permisos
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Health check para Dokploy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
