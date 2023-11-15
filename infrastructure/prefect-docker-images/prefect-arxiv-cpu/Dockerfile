FROM prefecthq/prefect:2-python3.11
LABEL maintainer="Filippo Broggini <https://github.com/filippo82>"

WORKDIR /opt/arxiv

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt .

RUN pip install --no-cache-dir --requirement requirements.txt
