FROM prefecthq/prefect:2-python3.10

WORKDIR /opt/arxiv

RUN pip install -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt
