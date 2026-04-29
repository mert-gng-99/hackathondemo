# NeuroBridge Enterprise — multi-stage build, FastAPI + pipeline runtime image.
# Python 3.12 because RDKit / scikit-learn / numpy pins ship cp310-cp312 wheels only.
FROM python:3.12-slim AS runtime

# System deps required by RDKit (libxrender, libxext) and nibabel/MNE
# (libgomp). Slim base lacks them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer is cached when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY AGENTS.md README.md ./

# Determinism env vars baked in (the pipelines re-pin defensively but
# baking them avoids a brief race on container start).
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
