FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "pyspark>=4.0.0" "numpy>=2.0" "pandas>=2.2,<3.0" "pyarrow>=17.0" "matplotlib>=3.9" \
    "pytest>=8.0" jupyter nbconvert ipykernel

COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY data/sample/ data/sample/

CMD ["pytest", "-q"]
