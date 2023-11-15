FROM prefecthq/prefect:2-python3.11
LABEL maintainer="Filippo Broggini <https://github.com/filippo82>"

WORKDIR /opt/arxiv

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements-torch-cpu.txt .

RUN pip install --no-cache-dir --requirement requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir --requirement requirements.txt

COPY requirements-optimum.txt .

RUN pip install --no-cache-dir --requirement requirements-optimum.txt --no-deps
