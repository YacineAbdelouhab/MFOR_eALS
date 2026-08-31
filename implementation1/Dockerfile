FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYSPARK_PYTHON=/usr/local/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/local/bin/python \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir ".[test,figures]"

CMD ["python", "scripts/smoke_job.py"]
